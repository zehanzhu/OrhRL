from __future__ import annotations

import os

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
)
from orchrl.trainer.v1_tq_adapter import (
    ensure_v1_ppo_config,
    get_v1_trainer_cls,
    is_v1_tq_backend,
)
from orchrl.utils.performance import colorful_print
from orchrl.utils.served_model_name import resolve_policy_server_name



class PolicyTrainerRegistry:
    def __init__(
        self,
        *,
        config,
        tokenizer_dict,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        ppo_trainer_cls=RayPPOTrainer,
        trainer_backend=None,
    ):
        self.config = config
        self.input_tokenizer_dict = tokenizer_dict
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.ppo_trainer_cls = ppo_trainer_cls
        self.trainer_backend = trainer_backend

        self.ppo_trainer_config_dict = {}
        self.ppo_trainer_dict = {}
        self.async_rollout_manager_dict = {}
        self.checkpoint_manager_dict = {}
        self.tokenizer_dict = {}
        self.server_handle_dict = {}
        self.policy_server_name_mapping = {}

    def _uses_v1_tq_backend(self):
        if self.trainer_backend is not None:
            return str(self.trainer_backend).lower() in {
                "v1",
                "v1_tq",
                "ppo_v1",
                "ppo_v1_tq",
                "transfer_queue",
            }
        return is_v1_tq_backend(self.config)

    def _resolve_trainer_cls(self, ppo_config):
        if self._uses_v1_tq_backend() and self.ppo_trainer_cls is RayPPOTrainer:
            return get_v1_trainer_cls(ppo_config)
        return self.ppo_trainer_cls

    def _create_ppo_trainer(self, *, model_name, ppo_config, resource_pool_manager):
        if self._uses_v1_tq_backend():
            ensure_v1_ppo_config(ppo_config)
            trainer_cls = self._resolve_trainer_cls(ppo_config)
            ppo_trainer = trainer_cls(config=ppo_config)
        else:
            trainer_cls = self._resolve_trainer_cls(ppo_config)
            ppo_trainer = trainer_cls(
                config=ppo_config,
                tokenizer=self.input_tokenizer_dict[model_name],
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=self.ray_worker_group_cls,
            )
        ppo_trainer.global_steps = 0
        return ppo_trainer

    def _resource_pool_at(self, index: int):
        if isinstance(self.resource_pool_manager, (list, tuple)):
            if index < len(self.resource_pool_manager):
                return self.resource_pool_manager[index]
            return None
        return self.resource_pool_manager

    def initialize_ppo_trainers(self):
        specialization = validate_specialization_mode(self.config.specialization)
        if specialization == ROLE_SHARING:
            self.create_single_ppo_trainer()
            return
        if specialization == ROLE_SPECIFIC:
            self.create_multiple_ppo_trainers()
            return
        raise ValueError(f"Unsupported specialization '{specialization}'")

    def _resolve_default_local_dir(self, model_name: str, *, multi_policy: bool):
        training_cfg = getattr(self.config, "training", None)
        checkpoint_root = (
            getattr(training_cfg, "model_checkpoints_dir", None)
            if training_cfg is not None
            else None
        )
        if not checkpoint_root:
            return None
        if multi_policy:
            return os.path.join(checkpoint_root, model_name)
        return checkpoint_root

    def create_single_ppo_trainer(self):
        config = self.config
        model_key = list(config.models.keys())[0]
        model_config = config.models[model_key]
        model_name = model_config.name

        if not hasattr(model_config, "ppo_trainer_config"):
            raise ValueError(f"Model '{model_name}' missing ppo_trainer_config")

        ppo_config = model_config.ppo_trainer_config
        self.ppo_trainer_config_dict[model_name] = ppo_config
        ppo_config.data["train_batch_size"] = config.training.train_batch_size
        default_local_dir = self._resolve_default_local_dir(
            model_name, multi_policy=False
        )
        if default_local_dir is not None:
            ppo_config.trainer.default_local_dir = default_local_dir

        ppo_trainer = self._create_ppo_trainer(
            model_name=model_name,
            ppo_config=ppo_config,
            resource_pool_manager=self._resource_pool_at(0),
        )
        self.ppo_trainer_dict[model_name] = ppo_trainer

    def create_multiple_ppo_trainers(self):
        config = self.config
        multi_policy = sum(
            1
            for model_config in config.models.values()
            if hasattr(model_config, "ppo_trainer_config")
        ) > 1

        for i, (_, model_config) in enumerate(config.models.items()):
            model_name = model_config.name

            if not hasattr(model_config, "ppo_trainer_config"):
                continue

            ppo_config = model_config.ppo_trainer_config
            self.ppo_trainer_config_dict[model_name] = ppo_config
            ppo_config.data["train_batch_size"] = config.training.train_batch_size
            ppo_config.trainer.experiment_name = config.training.experiment_name
            default_local_dir = self._resolve_default_local_dir(
                model_name, multi_policy=multi_policy
            )
            if default_local_dir is not None:
                ppo_config.trainer.default_local_dir = default_local_dir

            ppo_trainer = self._create_ppo_trainer(
                model_name=model_name,
                ppo_config=ppo_config,
                resource_pool_manager=self._resource_pool_at(i),
            )
            self.ppo_trainer_dict[model_name] = ppo_trainer

    def init_workers(self):
        colorful_print("Initializing workers for all PPO trainers...", "cyan")
        if not self.ppo_trainer_dict:
            colorful_print("No PPO trainers to initialize", "yellow")
            return

        total_trainers = len(self.ppo_trainer_dict)
        colorful_print(
            f"Initializing {total_trainers} trainers sequentially (each trainer spawns workers in parallel)...",
            "blue",
        )

        for idx, (model_name, trainer) in enumerate(self.ppo_trainer_dict.items(), 1):
            colorful_print(
                f"[{idx}/{total_trainers}] Initializing workers for: {model_name}",
                "blue",
            )
            if hasattr(trainer, "init_workers"):
                trainer.init_workers()
            elif hasattr(trainer, "init"):
                trainer.init()
            else:
                raise AttributeError(
                    f"PPO trainer for '{model_name}' has neither init_workers() nor init()"
                )
            colorful_print(
                f"✓ [{idx}/{total_trainers}] Successfully initialized: {model_name}",
                "green",
            )

        colorful_print(f"All {total_trainers} trainers initialized successfully!", "green")

    def collect_runtime_handles(self):
        self.async_rollout_manager_dict = {}
        self.checkpoint_manager_dict = {}
        self.tokenizer_dict = {}
        self.server_handle_dict = {}
        self.policy_server_name_mapping = {}

        for model_name, trainer in self.ppo_trainer_dict.items():
            rollout_manager = getattr(
                trainer,
                "async_rollout_manager",
                getattr(trainer, "llm_server_manager", None),
            )
            self.async_rollout_manager_dict[model_name] = rollout_manager
            self.checkpoint_manager_dict[model_name] = getattr(
                trainer,
                "checkpoint_manager",
                None,
            )
            self.tokenizer_dict[model_name] = getattr(
                trainer,
                "tokenizer",
                self.input_tokenizer_dict.get(model_name),
            )
            server_handle_list = getattr(rollout_manager, "server_handles", [])
            self.server_handle_dict[model_name] = server_handle_list
            self.policy_server_name_mapping[model_name] = resolve_policy_server_name(
                model_name, self.ppo_trainer_config_dict.get(model_name)
            )

    def get_async_rollout_managers(self):
        return self.async_rollout_manager_dict

    def get_checkpoint_managers(self):
        return self.checkpoint_manager_dict

    def get_tokenizers(self):
        return self.tokenizer_dict

    def get_server_handles(self):
        return self.server_handle_dict

    def get_policy_server_names(self):
        return self.policy_server_name_mapping
