from pprint import pprint
from numbers import Integral

from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_data_metrics,
    compute_timing_metrics,
)

from orchrl.trainer.mate.runtime import MateRuntime
from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry
from orchrl.trainer.training_step_executor import TrainingStepExecutor
from orchrl.trainer.validation_runner import ValidationRunner
from orchrl.utils.clean_up import cleanup_old_image_folders, run_async_cleanup
from orchrl.utils.performance import simple_timer, colorful_print


class MultiAgentsPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer_dict,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        agent_policy_mapping: dict = None,
    ):
        self.config = config
        self.tokenizer_dict = tokenizer_dict
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls

        self.best_validation_reward = float("-inf")
        self._ppo_trainer_config_dict = {}
        self._ppo_trainer_dict = {}
        self.agent_policy_mapping = agent_policy_mapping
        self.training_step_executor = None
        self.validation_runner = None

        self.agent_untrained = []
        if hasattr(config, 'multi_agent_interaction') and hasattr(config.multi_agent_interaction, 'agent_untrained'):
            self.agent_untrained = config.multi_agent_interaction.agent_untrained
            colorful_print(f"Agents excluded from training: {self.agent_untrained}", "yellow")

        self.policy_trainer_registry = PolicyTrainerRegistry(
            config=self.config,
            tokenizer_dict=self.tokenizer_dict,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
            ray_worker_group_cls=self.ray_worker_group_cls,
        )
        self.mate_runtime = MateRuntime(
            config=self.config,
            agent_policy_mapping=self.agent_policy_mapping,
        )
        self.validation_runner = ValidationRunner(
            config=self.config,
            policy_trainer_registry=self.policy_trainer_registry,
            mate_runtime=self.mate_runtime,
            agent_policy_mapping=self.agent_policy_mapping,
        )
        self.training_step_executor = TrainingStepExecutor(
            config=self.config,
            policy_trainer_registry=self.policy_trainer_registry,
            mate_runtime=self.mate_runtime,
            agent_policy_mapping=self.agent_policy_mapping,
            agent_untrained=self.agent_untrained,
        )
        self.best_validation_reward = self.validation_runner.best_validation_reward
        self._initialize_ppo_trainers()

    @property
    def ppo_trainer_config_dict(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is not None and hasattr(registry, "ppo_trainer_config_dict"):
            return registry.ppo_trainer_config_dict
        return self._ppo_trainer_config_dict

    @ppo_trainer_config_dict.setter
    def ppo_trainer_config_dict(self, value):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is not None and hasattr(registry, "ppo_trainer_config_dict"):
            registry.ppo_trainer_config_dict = value
            return
        self._ppo_trainer_config_dict = value

    @property
    def ppo_trainer_dict(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is not None and hasattr(registry, "ppo_trainer_dict"):
            return registry.ppo_trainer_dict
        return self._ppo_trainer_dict

    @ppo_trainer_dict.setter
    def ppo_trainer_dict(self, value):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is not None and hasattr(registry, "ppo_trainer_dict"):
            registry.ppo_trainer_dict = value
            return
        self._ppo_trainer_dict = value

    @property
    def mate_config(self):
        return getattr(getattr(self, "mate_runtime", None), "mate_config", None)

    @property
    def monitor_pool_manager(self):
        return getattr(
            getattr(self, "mate_runtime", None), "monitor_pool_manager", None
        )

    @property
    def mate_train_prompt_loader(self):
        return getattr(
            getattr(self, "mate_runtime", None), "mate_train_prompt_loader", None
        )

    @property
    def mate_val_prompt_loader(self):
        return getattr(
            getattr(self, "mate_runtime", None), "mate_val_prompt_loader", None
        )

    @property
    def mate_reward_provider(self):
        return getattr(
            getattr(self, "mate_runtime", None), "mate_reward_provider", None
        )

    @property
    def mate_rollout_adapter(self):
        return getattr(
            getattr(self, "mate_runtime", None), "mate_rollout_adapter", None
        )

    @property
    def mate_val_rollout_adapter(self):
        return getattr(
            getattr(self, "mate_runtime", None), "mate_val_rollout_adapter", None
        )

    @property
    def async_rollout_manager_dict(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is None or not hasattr(registry, "get_async_rollout_managers"):
            return {}
        return registry.get_async_rollout_managers()

    @property
    def checkpoint_manager_dict(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is None or not hasattr(registry, "get_checkpoint_managers"):
            return {}
        return registry.get_checkpoint_managers()

    @property
    def server_handle_dict(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is None or not hasattr(registry, "get_server_handles"):
            return {}
        return registry.get_server_handles()

    @property
    def policy_server_name_mapping(self):
        registry = getattr(self, "policy_trainer_registry", None)
        if registry is None or not hasattr(registry, "get_policy_server_names"):
            return {}
        return registry.get_policy_server_names()

    def _initialize_ppo_trainers(self):
        self.policy_trainer_registry.initialize_ppo_trainers()

    def init_mate_rollout_runtime(self):
        self.policy_trainer_registry.collect_runtime_handles()
        self.tokenizer_dict = self.policy_trainer_registry.get_tokenizers()
        server_handle_dict = self.policy_trainer_registry.get_server_handles()
        policy_server_name_mapping = (
            self.policy_trainer_registry.get_policy_server_names()
        )
        self.mate_runtime.initialize(
            tokenizer_dict=self.tokenizer_dict,
            async_rollout_manager_dict=self.policy_trainer_registry.get_async_rollout_managers(),
            server_handle_dict=server_handle_dict,
            policy_server_name_mapping=policy_server_name_mapping,
        )

    def _has_real_batch(self, batch) -> bool:
        return self.training_step_executor.has_real_batch(batch)

    def fit_one_collect_phase_for_test(self):
        return self.training_step_executor.collect_mate_step_batches(
            step_idx=self.global_steps
        )

    def init_workers(self):
        self.policy_trainer_registry.init_workers()

    @staticmethod
    def _build_tracking_config(config):
        tracking_config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(tracking_config, dict):
            return tracking_config

        trainer_cfg = tracking_config.get("trainer")
        if not isinstance(trainer_cfg, dict):
            tracking_config["trainer"] = {}
        return tracking_config

    def _initialize_logger_safely(self):
        from verl.utils.tracking import Tracking
        from datetime import datetime
        import os
        from pathlib import Path

        # Resolve logger output against the configured run directory so Ray remote
        # tasks do not depend on their process working directory.
        current_time = datetime.now()
        date_str = current_time.strftime("%m-%d")
        time_str = current_time.strftime("%H-%M-%S")

        experiment_name = self.config.training.experiment_name
        run_dir = getattr(self.config.training, "run_dir", None)
        if run_dir:
            run_dir_path = Path(str(run_dir)).expanduser()
            if not run_dir_path.is_absolute():
                run_dir_path = Path.cwd() / run_dir_path
            log_dir = run_dir_path / "logs" / date_str / time_str
        else:
            log_dir = (Path.cwd() / "outputs" / "logs" / experiment_name / date_str / time_str)

        log_dir = log_dir.resolve()
        os.makedirs(log_dir, exist_ok=True)

        logger = Tracking(
            project_name=self.config.training.project_name,
            experiment_name=experiment_name,
            default_backend=self.config.training.logger,
            config=self._build_tracking_config(self.config),
        )

        colorful_print(f"Logger initialized with log_dir: {log_dir}", "cyan")
        return logger

    @staticmethod
    def _resolve_loaded_checkpoint_step(trainer, loaded_step) -> int:
        if isinstance(loaded_step, Integral):
            return int(loaded_step)
        trainer_global_steps = getattr(trainer, "global_steps", 0)
        if isinstance(trainer_global_steps, Integral):
            return int(trainer_global_steps)
        return 0

    def _restore_global_steps_from_checkpoints(self) -> int:
        resolved_steps = {}

        for model_name, trainer in self.ppo_trainer_dict.items():
            loaded_step = trainer._load_checkpoint()
            resolved_steps[model_name] = self._resolve_loaded_checkpoint_step(
                trainer, loaded_step
            )

        resumed_steps = {
            model_name: step
            for model_name, step in resolved_steps.items()
            if step > 0
        }
        if not resumed_steps:
            return 0

        unique_steps = set(resumed_steps.values())
        if len(unique_steps) != 1 or len(resumed_steps) != len(resolved_steps):
            raise RuntimeError(
                "Inconsistent checkpoint steps across policies: "
                + ", ".join(
                    f"{model_name}={step}"
                    for model_name, step in resolved_steps.items()
                )
            )

        resumed_step = unique_steps.pop()
        colorful_print(
            f"Resumed training from global step {resumed_step}", "green"
        )
        return resumed_step

    def fit(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        logger = self._initialize_logger_safely()

        # Load checkpoint if resume is enabled
        # This must be done after init_workers() and before training loop
        self.global_steps = self._restore_global_steps_from_checkpoints()
        for trainer in self.ppo_trainer_dict.values():
            trainer.global_steps = self.global_steps

        self.total_training_steps = self.config.training.total_training_steps
        progress_bar = tqdm(range(self.total_training_steps), desc="Training Progress", position=0, leave=True)

        while self.global_steps < self.total_training_steps:
            progress_bar.update(1)
            progress_bar.set_description(f"Step {self.global_steps}")
            pprint(f"step {self.global_steps} started")

            batch_per_trainer: dict[str, DataProto] = {}
            present_policy_names = []

            metrics = {}
            timing_raw = {}

            with simple_timer("step", timing_raw):
                step_result = self.training_step_executor.execute_training_step(
                    step_idx=self.global_steps
                )
                batch_per_trainer = step_result.batch_per_trainer
                present_policy_names = step_result.present_policy_names
                metrics.update(step_result.metrics)
                timing_raw = step_result.timing_raw

            # TODO: collect metrics
            # Use the first trainer's batch for metrics calculation
            for model_name, batch in batch_per_trainer.items():
                if not self._has_real_batch(batch):
                    continue
                for metric_name, metric_value in compute_data_metrics(batch=batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict.values())).items():
                    metric_name_policy= model_name + "_" + metric_name
                    metrics[metric_name_policy] = metric_value

                for metric_name, metric_value in compute_timing_metrics(batch=batch, timing_raw=timing_raw).items():
                    metric_name_policy= model_name + "_" + metric_name
                    metrics[metric_name_policy] = metric_value

            # Add training step metrics
            metrics.update({
                "training/global_step": self.global_steps,

            })


            if self.global_steps % self.config.training.val_freq == 0 and self.global_steps != 0:
                val_metrics = self._validate(global_steps=self.global_steps)
                metrics.update(val_metrics)
            self.global_steps += 1
            for ppo_trainer in self.ppo_trainer_dict.values():
                ppo_trainer.global_steps = self.global_steps
            try:
                logger.log(data=metrics, step=self.global_steps)
            except Exception as e:
                pprint(f"Warning: Failed to log metrics to logger: {type(e).__name__}: {e}")
                pprint(f"Metrics that failed to log: {list(metrics.keys())}")

            # Clean up old image folders if multimodal is enabled
            enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
            if enable_multimodal:
                try:
                    # Get image save directory from config
                    image_save_dir = "tmp_image"  # default
                    if hasattr(self.config, 'env') and hasattr(self.config.env, 'image_save_dir'):
                        image_save_dir = self.config.env.image_save_dir
                    elif hasattr(self.config.training, 'image_save_dir'):
                        image_save_dir = self.config.training.image_save_dir

                    # Get max subfolders from config (default: 20)
                    max_image_steps = getattr(self.config.training, 'max_image_steps', 20)

                    # Clean up old image folders
                    cleanup_old_image_folders(
                        base_dir=image_save_dir,
                        max_subfolders=max_image_steps,
                        verbose=True
                    )
                except Exception as e:
                    pprint(f"Warning: Failed to clean up image folders: {type(e).__name__}: {e}")

            # Check if any trainer has reached its total training steps
            if self.global_steps >= self.total_training_steps:
                progress_bar.close()

                # perform final validation and print summary

                return

        progress_bar.close()

    def _validate(self, global_steps=0):
        validation_metrics = self.validation_runner.validate(global_steps=global_steps)
        self.best_validation_reward = self.validation_runner.best_validation_reward
        return validation_metrics

    def cleanup(self):
        """Clean up all resources including trainers and resource pools"""
        try:
            colorful_print("Starting MultiAgentsPPOTrainer cleanup...", "yellow")

            # Clean up MATE monitor pool actors first.
            if self.monitor_pool_manager is not None and hasattr(
                self.monitor_pool_manager, "close"
            ):
                if run_async_cleanup(
                    self.monitor_pool_manager.close(),
                    label="MATE monitor pool",
                ):
                    colorful_print("Cleaned up MATE monitor pool", "yellow")

            # Clean up PPO trainers
            if hasattr(self, 'ppo_trainer_dict'):
                colorful_print(f"Cleaning up {len(self.ppo_trainer_dict)} PPO trainers...", "yellow")
                for model_name, trainer in self.ppo_trainer_dict.items():
                    try:
                        # Call the trainer's cleanup method
                        if hasattr(trainer, 'cleanup'):
                            trainer.cleanup()
                        colorful_print(f"Cleaned up trainer for model: {model_name}", "yellow")
                    except Exception as e:
                        colorful_print(f"Error cleaning up trainer for {model_name}: {e}", "red")
                self.ppo_trainer_dict.clear()

            # Clean up resource pool managers
            if hasattr(self, 'resource_pool_manager') and self.resource_pool_manager is not None:
                try:
                    if isinstance(self.resource_pool_manager, list):
                        colorful_print(f"Cleaning up {len(self.resource_pool_manager)} resource pool managers...", "yellow")
                        for i, manager in enumerate(self.resource_pool_manager):
                            try:
                                if hasattr(manager, 'cleanup'):
                                    manager.cleanup()
                                colorful_print(f"Cleaned up resource pool manager {i}", "yellow")
                            except Exception as e:
                                colorful_print(f"Error cleaning up resource pool manager {i}: {e}", "red")
                    else:
                        if hasattr(self.resource_pool_manager, 'cleanup'):
                            self.resource_pool_manager.cleanup()
                        colorful_print("Cleaned up resource_pool_manager", "yellow")
                except Exception as e:
                    colorful_print(f"Error cleaning up resource_pool_manager: {e}", "red")

            colorful_print("Multi-agent trainer cleanup completed", "green")
        except Exception as e:
            colorful_print(f"Error during cleanup: {e}", "red")
