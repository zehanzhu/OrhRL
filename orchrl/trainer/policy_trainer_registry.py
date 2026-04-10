from __future__ import annotations

import os

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
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
    ):
        self.config = config
        self.input_tokenizer_dict = tokenizer_dict
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.ppo_trainer_cls = ppo_trainer_cls

        self.ppo_trainer_config_dict = {}
        self.ppo_trainer_dict = {}
        self.async_rollout_manager_dict = {}
        self.checkpoint_manager_dict = {}
        self.tokenizer_dict = {}
        self.server_handle_dict = {}
        self.policy_server_name_mapping = {}

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

        ppo_trainer = self.ppo_trainer_cls(
            config=ppo_config,
            tokenizer=self.input_tokenizer_dict[model_name],
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager[0],
            ray_worker_group_cls=self.ray_worker_group_cls,
        )
        ppo_trainer.global_steps = 0
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

            ppo_trainer = self.ppo_trainer_cls(
                config=ppo_config,
                tokenizer=self.input_tokenizer_dict[model_name],
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager[i],
                ray_worker_group_cls=self.ray_worker_group_cls,
            )
            ppo_trainer.global_steps = 0
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
            trainer.init_workers()
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
            self.async_rollout_manager_dict[model_name] = trainer.async_rollout_manager
            self.checkpoint_manager_dict[model_name] = trainer.checkpoint_manager
            self.tokenizer_dict[model_name] = trainer.tokenizer
            server_handle_list = getattr(trainer.async_rollout_manager, "server_handles", [])
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
