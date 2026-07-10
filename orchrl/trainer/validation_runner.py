from __future__ import annotations

import asyncio

from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
)
from orchrl.trainer.mate.mas_metrics import (
    accumulate_mas_metric_stats,
    finalize_mas_metrics,
    init_mas_metric_stats,
)
from orchrl.trainer.policy_backend import build_policy_backends_from_trainers
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

    def _get_policy_backends(self):
        get_policy_backends = getattr(
            self.policy_trainer_registry,
            "get_policy_backends",
            None,
        )
        if callable(get_policy_backends):
            policy_backends = get_policy_backends()
            if isinstance(policy_backends, dict):
                return policy_backends
        return build_policy_backends_from_trainers(
            getattr(self.policy_trainer_registry, "ppo_trainer_dict", {})
        )

    def _get_native_policy_backends(self):
        get_policy_backends = getattr(
            self.policy_trainer_registry,
            "get_policy_backends",
            None,
        )
        if not callable(get_policy_backends):
            return None
        policy_backends = get_policy_backends()
        return policy_backends if isinstance(policy_backends, dict) else None

    def _update_rollout_weights(self):
        policy_backends = self._get_native_policy_backends()
        if policy_backends is not None:
            for policy_backend in policy_backends.values():
                policy_backend.update_rollout_weights()
            return
        checkpoint_managers = self.policy_trainer_registry.get_checkpoint_managers()
        for checkpoint_manager in checkpoint_managers.values():
            checkpoint_manager.update_weights()

    def _sleep_rollout_replicas(self):
        policy_backends = self._get_native_policy_backends()
        if policy_backends is not None:
            for policy_backend in policy_backends.values():
                policy_backend.sleep_rollout_replicas()
            return
        checkpoint_managers = self.policy_trainer_registry.get_checkpoint_managers()
        for checkpoint_manager in checkpoint_managers.values():
            checkpoint_manager.sleep_replicas()

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

        self._update_rollout_weights()

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
            self._sleep_rollout_replicas()

    @staticmethod
    def init_validation_stats():
        return {
            "expected_sample_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "reward_sum": 0.0,
            "correct_count": 0,
            "mas_metric_stats": init_mas_metric_stats(),
        }

    @staticmethod
    def normalize_validation_reward(value):
        if value is None:
            return 0.0
        if isinstance(value, (list, tuple)):
            return float(sum(float(item) for item in value))
        return float(value)

    def accumulate_validation_episode_batch(
        self,
        stats,
        episodes,
        *,
        expected_sample_count: int | None = None,
        failed_count: int = 0,
    ):
        if expected_sample_count is None:
            expected_sample_count = len(episodes)
        stats["expected_sample_count"] += int(expected_sample_count)
        stats["success_count"] += len(episodes)
        stats["failed_count"] += int(failed_count)
        accumulate_mas_metric_stats(stats["mas_metric_stats"], episodes)
        for episode in episodes:
            reward_value = self.normalize_validation_reward(
                getattr(episode, "final_reward", 0.0)
            )
            stats["reward_sum"] += reward_value
            if reward_value >= 1.0:
                stats["correct_count"] += 1

    @staticmethod
    def build_validation_metrics(stats):
        expected_sample_count = stats["expected_sample_count"]
        sample_avg_reward = (
            stats["reward_sum"] / expected_sample_count if expected_sample_count > 0 else 0.0
        )
        accuracy = (
            stats["correct_count"] / expected_sample_count if expected_sample_count > 0 else 0.0
        )
        failed_count = int(stats["failed_count"])
        failed_rate = failed_count / expected_sample_count if expected_sample_count > 0 else 0.0
        metrics = {
            "validation/sample_avg_reward": float(sample_avg_reward),
            "validation/accuracy": float(accuracy),
            "validation/failed_sample_count": failed_count,
            "validation/failed_sample_rate": float(failed_rate),
        }
        metrics.update(
            finalize_mas_metrics(
                stats=stats.get("mas_metric_stats", {}),
                expected_sample_count=expected_sample_count,
                failed_count=failed_count,
                prefix="mas/validation",
            )
        )
        return metrics

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
        policy_backends = self._get_policy_backends()

        if spec == ROLE_SHARING:
            for policy_backend in policy_backends.values():
                save_jobs.append(("shared_model", policy_backend))
        elif spec == ROLE_SPECIFIC:
            num_base_models = (
                len(self.config.base_models)
                if hasattr(self.config, "base_models")
                else 0
            )
            if num_base_models == 1:
                for agent_name, policy_name in self.agent_policy_mapping.items():
                    policy_backend = policy_backends[policy_name]
                    save_jobs.append((agent_name, policy_backend))
            else:
                for model_name, policy_backend in policy_backends.items():
                    save_jobs.append((model_name, policy_backend))

        for _, policy_backend in save_jobs:
            policy_backend.save_checkpoint()

    def validate(self, global_steps=0):
        stats = self.init_validation_stats()

        for rollout_batch in self.iter_validation_episode_batches():
            episodes = self._extract_rollout_episodes(rollout_batch)
            self.accumulate_validation_episode_batch(
                stats,
                episodes,
                expected_sample_count=self._rollout_expected_job_count(
                    rollout_batch,
                    episodes,
                ),
                failed_count=self._rollout_failed_count(rollout_batch),
            )

        validation_metrics = self.build_validation_metrics(stats)
        sample_avg_reward = validation_metrics["validation/sample_avg_reward"]

        if global_steps > 0:
            self.save_best_checkpoint(sample_avg_reward)

        return validation_metrics

    @staticmethod
    def _extract_rollout_episodes(rollout_result):
        if hasattr(rollout_result, "episodes"):
            return list(getattr(rollout_result, "episodes") or [])
        return list(rollout_result or [])

    @staticmethod
    def _rollout_expected_job_count(rollout_result, episodes) -> int:
        if hasattr(rollout_result, "expected_job_count"):
            return int(getattr(rollout_result, "expected_job_count", 0))
        return len(episodes)

    @staticmethod
    def _rollout_failed_count(rollout_result) -> int:
        if hasattr(rollout_result, "failed_count"):
            return int(getattr(rollout_result, "failed_count", 0))
        return 0
