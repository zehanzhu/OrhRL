from __future__ import annotations

import asyncio

from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
)
from orchrl.utils.performance import colorful_print


class ValidationRunner:
    def __init__(
        self,
        *,
        config,
        policy_trainer_registry,
        mate_runtime,
        agent_policy_mapping,
    ):
        self.config = config
        self.policy_trainer_registry = policy_trainer_registry
        self.mate_runtime = mate_runtime
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.best_validation_reward = float("-inf")

    def iter_validation_episode_batches(self):
        validate_batch_size = int(
            getattr(
                self.config.training,
                "validate_batch_size",
                self.config.training.train_batch_size,
            )
        )
        if validate_batch_size < 1:
            raise ValueError("training.validate_batch_size must be >= 1")

        checkpoint_managers = self.policy_trainer_registry.get_checkpoint_managers()
        for checkpoint_manager in checkpoint_managers.values():
            checkpoint_manager.update_weights()

        try:
            for prompt_batch in self.mate_runtime.mate_val_prompt_loader.iter_batches(
                batch_size=validate_batch_size
            ):
                yield asyncio.run(
                    self.mate_runtime.mate_val_rollout_adapter.collect_prompt_batch_rollouts(
                        prompts=prompt_batch,
                        n_samples_per_prompt=1,
                    )
                )
        finally:
            for checkpoint_manager in checkpoint_managers.values():
                checkpoint_manager.sleep_replicas()

    @staticmethod
    def init_validation_stats():
        return {
            "sample_count": 0,
            "reward_sum": 0.0,
        }

    @staticmethod
    def normalize_validation_reward(value):
        if value is None:
            return 0.0
        if isinstance(value, (list, tuple)):
            return float(sum(float(item) for item in value))
        return float(value)

    def accumulate_validation_episode_batch(self, stats, episodes):
        for episode in episodes:
            stats["sample_count"] += 1
            stats["reward_sum"] += self.normalize_validation_reward(
                getattr(episode, "final_reward", 0.0)
            )

    @staticmethod
    def build_validation_metrics(stats):
        sample_count = stats["sample_count"]
        sample_avg_reward = (
            stats["reward_sum"] / sample_count if sample_count > 0 else 0.0
        )
        return {"validation/sample_avg_reward": float(sample_avg_reward)}

    def save_best_checkpoint(self, sample_avg_reward):
        if_save = getattr(self.config.training, "if_save", True)

        if not if_save:
            colorful_print(
                (
                    "Checkpoint saving disabled (if_save=False). "
                    f"Current validation reward: {sample_avg_reward:.4f}"
                ),
                "yellow",
            )
            return

        if sample_avg_reward <= self.best_validation_reward:
            colorful_print(
                (
                    f"Current validation reward: {sample_avg_reward:.4f} "
                    f"(best: {self.best_validation_reward:.4f})"
                ),
                "yellow",
            )
            return

        self.best_validation_reward = sample_avg_reward
        colorful_print(
            f"New best validation reward: {sample_avg_reward:.4f}, saving checkpoint...",
            "green",
        )

        spec = validate_specialization_mode(self.config.specialization)
        save_jobs = []
        ppo_trainer_dict = self.policy_trainer_registry.ppo_trainer_dict

        if spec == ROLE_SHARING:
            for trainer in ppo_trainer_dict.values():
                save_jobs.append(("shared_model", trainer))
        elif spec == ROLE_SPECIFIC:
            num_base_models = (
                len(self.config.base_models)
                if hasattr(self.config, "base_models")
                else 0
            )
            if num_base_models == 1:
                for agent_name, policy_name in self.agent_policy_mapping.items():
                    trainer = ppo_trainer_dict[policy_name]
                    save_jobs.append((agent_name, trainer))
            else:
                for model_name, trainer in ppo_trainer_dict.items():
                    save_jobs.append((model_name, trainer))

        for _, trainer in save_jobs:
            trainer._save_checkpoint()

    def validate(self, global_steps=0):
        stats = self.init_validation_stats()

        for episode_batch in self.iter_validation_episode_batches():
            self.accumulate_validation_episode_batch(stats, episode_batch)

        validation_metrics = self.build_validation_metrics(stats)
        sample_avg_reward = validation_metrics["validation/sample_avg_reward"]

        if global_steps > 0:
            self.save_best_checkpoint(sample_avg_reward)

        return validation_metrics
