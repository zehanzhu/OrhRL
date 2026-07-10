from __future__ import annotations

from typing import Any, Protocol


class PolicyTrainerBackend(Protocol):
    """Stable policy update boundary used by OrchRL multi-agent orchestration."""

    model_name: str

    @property
    def config(self) -> Any: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def global_steps(self) -> int: ...

    @global_steps.setter
    def global_steps(self, value: int) -> None: ...

    @property
    def async_rollout_manager(self) -> Any: ...

    @property
    def checkpoint_manager(self) -> Any: ...

    @property
    def server_handles(self) -> list[Any]: ...

    @property
    def policy_server_name(self) -> str: ...

    def init_workers(self) -> None: ...

    def load_checkpoint(self) -> Any: ...

    def save_checkpoint(self) -> None: ...

    def cleanup(self) -> None: ...

    def update_rollout_weights(self) -> None: ...

    def sleep_rollout_replicas(self) -> None: ...

    def data_parallel_world_size(self) -> int: ...

    def uses_critic(self) -> bool: ...

    def needs_ref_log_prob(self) -> bool: ...

    def compute_old_log_prob(self, batch: Any) -> Any: ...

    def compute_ref_log_prob(self, batch: Any) -> Any: ...

    def compute_values(self, batch: Any) -> Any: ...

    def update_critic(self, batch: Any) -> Any: ...

    def update_actor(self, batch: Any) -> Any: ...

    def dump_generations(
        self,
        *,
        inputs: list[str],
        outputs: list[str],
        scores: list[float],
        reward_extra_infos_dict: dict[str, list],
        dump_path: str,
    ) -> None: ...


class VerlPPOPolicyBackend:
    """Adapter that hides VERL RayPPOTrainer internals behind OrchRL's boundary."""

    def __init__(
        self,
        *,
        model_name: str,
        trainer: Any,
        policy_server_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.trainer = trainer
        self._policy_server_name = policy_server_name or model_name

    @property
    def config(self) -> Any:
        return self.trainer.config

    @property
    def tokenizer(self) -> Any:
        return self.trainer.tokenizer

    @property
    def global_steps(self) -> int:
        return int(getattr(self.trainer, "global_steps", 0))

    @global_steps.setter
    def global_steps(self, value: int) -> None:
        self.trainer.global_steps = int(value)

    @property
    def async_rollout_manager(self) -> Any:
        return getattr(self.trainer, "async_rollout_manager", None)

    @property
    def checkpoint_manager(self) -> Any:
        return getattr(self.trainer, "checkpoint_manager", None)

    @property
    def server_handles(self) -> list[Any]:
        rollout_manager = self.async_rollout_manager
        return list(getattr(rollout_manager, "server_handles", []) or [])

    @property
    def policy_server_name(self) -> str:
        return self._policy_server_name

    def init_workers(self) -> None:
        self.trainer.init_workers()

    def load_checkpoint(self) -> Any:
        return self.trainer._load_checkpoint()

    def save_checkpoint(self) -> None:
        self.trainer._save_checkpoint()

    def cleanup(self) -> None:
        if hasattr(self.trainer, "cleanup"):
            self.trainer.cleanup()

    def update_rollout_weights(self) -> None:
        checkpoint_manager = self.checkpoint_manager
        if checkpoint_manager is not None:
            checkpoint_manager.update_weights()

    def sleep_rollout_replicas(self) -> None:
        checkpoint_manager = self.checkpoint_manager
        if checkpoint_manager is not None:
            checkpoint_manager.sleep_replicas()

    def data_parallel_world_size(self) -> int:
        try:
            return int(self.trainer.actor_rollout_wg.world_size)
        except Exception:
            return 1

    def uses_critic(self) -> bool:
        return bool(getattr(self.trainer, "use_critic", False))

    def needs_ref_log_prob(self) -> bool:
        use_reference_policy = bool(getattr(self.trainer, "use_reference_policy", False))
        use_kl_in_reward = bool(getattr(self.config.algorithm, "use_kl_in_reward", False))
        return use_reference_policy or use_kl_in_reward

    def compute_old_log_prob(self, batch: Any) -> Any:
        return self.trainer.actor_rollout_wg.compute_log_prob(batch)

    def compute_ref_log_prob(self, batch: Any) -> Any:
        if not getattr(self.trainer, "ref_in_actor", False):
            return self.trainer.ref_policy_wg.compute_ref_log_prob(batch)
        return self.trainer.actor_rollout_wg.compute_ref_log_prob(batch)

    def compute_values(self, batch: Any) -> Any:
        return self.trainer.critic_wg.compute_values(batch)

    def update_critic(self, batch: Any) -> Any:
        return self.trainer.critic_wg.update_critic(batch)

    def update_actor(self, batch: Any) -> Any:
        return self.trainer.actor_rollout_wg.update_actor(batch)

    def dump_generations(
        self,
        *,
        inputs: list[str],
        outputs: list[str],
        scores: list[float],
        reward_extra_infos_dict: dict[str, list],
        dump_path: str,
    ) -> None:
        self.trainer._dump_generations(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
        )


def build_policy_backends_from_trainers(
    trainer_dict: dict[str, Any],
    *,
    existing_backends: dict[str, PolicyTrainerBackend] | None = None,
    policy_server_name_mapping: dict[str, str] | None = None,
) -> dict[str, PolicyTrainerBackend]:
    backends: dict[str, PolicyTrainerBackend] = dict(existing_backends or {})
    server_names = dict(policy_server_name_mapping or {})
    for model_name, trainer in trainer_dict.items():
        if model_name not in backends:
            backends[model_name] = VerlPPOPolicyBackend(
                model_name=model_name,
                trainer=trainer,
                policy_server_name=server_names.get(model_name),
            )
    return backends
