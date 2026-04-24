from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import ray

from ._support.renderer import ChatRenderer
from ._support.replay_cache import ReplayCache
from .backend import RolloutBackend
from .datatypes import ModelMappingEntry
from .monitor_actor import MonitorActor


@dataclass(slots=True)
class MonitorLease:
    actor: Any
    actor_index: int
    lease_id: str
    base_url: str
    episode_id: str


class MonitorPoolManager:
    def __init__(
        self,
        *,
        backend: RolloutBackend,
        model_mapping: dict[str, ModelMappingEntry],
        size: int,
        host: str = "127.0.0.1",
        base_port: int = 19000,
        acquire_timeout_sec: float = 300.0,
        actor_resources: dict[str, float] | None = None,
        renderer: ChatRenderer | dict[str, ChatRenderer] | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("monitor pool size must be >= 1")
        if base_port < 1:
            raise ValueError("monitor pool base_port must be >= 1")
        if acquire_timeout_sec <= 0:
            raise ValueError("monitor pool acquire_timeout_sec must be > 0")

        self._backend = backend
        self._model_mapping = model_mapping
        self._size = size
        self._host = host
        self._base_port = base_port
        self._acquire_timeout_sec = acquire_timeout_sec
        self._actor_resources = dict(actor_resources or {})
        self._renderer = renderer

        self._actors: dict[int, Any] = {}
        self._available: Queue[int] = Queue()
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            for actor_index in range(self._size):
                actor = self._create_actor(actor_index)
                self._actors[actor_index] = actor
                self._available.put(actor_index)
            self._started = True

    async def acquire(
        self,
        *,
        episode_id: str | None = None,
        replay_cache: ReplayCache | None = None,
    ) -> MonitorLease:
        return await asyncio.to_thread(
            self._acquire_sync,
            episode_id,
            replay_cache,
        )

    async def release(self, lease: MonitorLease) -> None:
        await asyncio.to_thread(self._release_sync, lease)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _acquire_sync(
        self,
        episode_id: str | None,
        replay_cache: ReplayCache | None,
    ) -> MonitorLease:
        self.start()
        resolved_episode_id = episode_id or uuid.uuid4().hex
        try:
            actor_index = self._available.get(timeout=self._acquire_timeout_sec)
        except Empty as exc:
            raise TimeoutError(
                f"monitor pool acquire timed out after {self._acquire_timeout_sec} seconds"
            ) from exc

        with self._lock:
            actor = self._actors[actor_index]

        try:
            lease_id = ray.get(
                actor.begin_run.remote(
                    episode_id=resolved_episode_id,
                    replay_cache=replay_cache,
                )
            )
            base_url = ray.get(actor.get_lease_base_url.remote(lease_id))
        except Exception:
            rebuilt_actor = self._replace_actor(actor_index)
            self._available.put(actor_index)
            raise RuntimeError(f"failed to acquire monitor lease from actor {actor_index}")

        return MonitorLease(
            actor=actor,
            actor_index=actor_index,
            lease_id=lease_id,
            base_url=base_url,
            episode_id=resolved_episode_id,
        )

    def _release_sync(self, lease: MonitorLease) -> None:
        try:
            ray.get(lease.actor.reset_run_state.remote(lease.lease_id))
        except Exception:
            self._replace_actor(lease.actor_index)
        finally:
            self._available.put(lease.actor_index)

    def _close_sync(self) -> None:
        with self._lock:
            actors = list(self._actors.values())
            self._actors = {}
            self._available = Queue()
            self._started = False

        for actor in actors:
            try:
                ray.get(actor.stop_server.remote())
            except Exception:
                pass
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

    def _replace_actor(self, actor_index: int):
        with self._lock:
            stale_actor = self._actors.get(actor_index)
            if stale_actor is not None:
                try:
                    ray.kill(stale_actor, no_restart=True)
                except Exception:
                    pass
            actor = self._create_actor(actor_index)
            self._actors[actor_index] = actor
            return actor

    def _create_actor(self, actor_index: int):
        actor_cls = MonitorActor.options(resources=self._actor_resources) if self._actor_resources else MonitorActor
        actor = actor_cls.remote(
            backend=self._backend,
            model_mapping=self._model_mapping,
            host=self._host,
            port=self._base_port + actor_index,
            renderer=self._renderer,
        )
        ray.get(actor.start_server_once.remote())
        return actor
