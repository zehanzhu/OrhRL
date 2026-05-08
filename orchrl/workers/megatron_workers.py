from __future__ import annotations

from orchrl.utils.mcore_compat import bootstrap_mcore_runtime_compat
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.megatron_workers import (
    AsyncActorRolloutRefWorker as VerlAsyncActorRolloutRefWorker,
    CriticWorker as VerlCriticWorker,
)


class AsyncActorRolloutRefWorker(VerlAsyncActorRolloutRefWorker):
    @staticmethod
    def _bootstrap_orchrl_runtime_compat() -> None:
        bootstrap_mcore_runtime_compat()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._bootstrap_orchrl_runtime_compat()
        return super().init_model()

    @staticmethod
    def _resolve_actor_dispatch_mode():
        # Keep wrapper import-light for unit tests that mock verl.workers.megatron_workers.
        try:
            from verl.workers.megatron_workers import make_nd_compute_dataproto_dispatch_fn

            return make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")
        except Exception:
            return Dispatch.ONE_TO_ALL

    @register(dispatch_mode=_resolve_actor_dispatch_mode.__func__())
    def update_actor(self, data):
        """Guard mini-batch divisibility to avoid VERL assertion failures.

        VERL requires `batch_size % ppo_mini_batch_size == 0` for megatron actor
        updates. In OrchRL MATE mode, per-step sample count can vary with turn
        expansion and may not match static config exactly.
        """
        actor_cfg = getattr(self.config, "actor", None)
        if actor_cfg is not None:
            mini_batch_size = int(getattr(actor_cfg, "ppo_mini_batch_size", 0) or 0)
            batch_size = int(getattr(getattr(data, "batch", None), "batch_size", [0])[0] or 0)
            if batch_size > 0 and mini_batch_size > 0 and batch_size % mini_batch_size != 0:
                adjusted = 1
                for candidate in range(min(mini_batch_size, batch_size), 0, -1):
                    if batch_size % candidate == 0:
                        adjusted = candidate
                        break
                actor_cfg.ppo_mini_batch_size = adjusted
                inner_actor_cfg = getattr(getattr(self, "actor", None), "config", None)
                if inner_actor_cfg is not None:
                    inner_actor_cfg.ppo_mini_batch_size = adjusted
                print(
                    "[orchrl/megatron] adjusted ppo_mini_batch_size "
                    f"{mini_batch_size} -> {adjusted} for batch_size={batch_size}"
                )
        return super().update_actor(data)


class CriticWorker(VerlCriticWorker):
    @staticmethod
    def _bootstrap_orchrl_runtime_compat() -> None:
        bootstrap_mcore_runtime_compat()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._bootstrap_orchrl_runtime_compat()
        return super().init_model()
