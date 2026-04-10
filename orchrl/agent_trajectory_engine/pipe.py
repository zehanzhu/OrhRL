from __future__ import annotations

import asyncio
import copy
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray

from ._support.collector import TrajectoryCollector
from ._support.launcher import MASLauncher
from .datatypes import EpisodeResult, InteractionRecord, ModelMappingEntry
from .monitor_pool import MonitorLease
from .reward import RewardProvider, RewardWorker


@dataclass
class AgentPipeConfig:
    mas_command_template: str
    config_template: dict[str, Any]
    model_mapping: dict[str, ModelMappingEntry]
    timeout: float = 300.0
    mas_work_dir: str | Path | None = None


class AgentPipe:
    def __init__(
        self,
        config: AgentPipeConfig,
    ) -> None:
        self._config = config
        self._collector = TrajectoryCollector()
        self._reward_worker = RewardWorker()
        self._last_buffer: list[InteractionRecord] = []

    def last_buffer(self) -> list[InteractionRecord]:
        return copy.deepcopy(self._last_buffer)

    async def run(
        self,
        prompt: str,
        reward_provider: RewardProvider,
        allow_partial: bool = False,
        *,
        monitor_lease: MonitorLease,
    ) -> EpisodeResult:
        episode_id = monitor_lease.episode_id
        launcher = MASLauncher(work_dir=self._config.mas_work_dir)
        primary_error: BaseException | None = None
        partial_result: EpisodeResult | None = None
        self._last_buffer = []

        try:
            monitor_url = monitor_lease.base_url
            config_path = await asyncio.to_thread(
                launcher.prepare_config,
                config_template=self._config.config_template,
                monitor_url=monitor_url,
                agent_roles=list(self._config.model_mapping.keys()),
            )
            command = self._config.mas_command_template.format(
                config_path=shlex.quote(str(config_path)),
                prompt=shlex.quote(prompt),
            )
            process = await asyncio.to_thread(launcher.launch, command=command)
            exit_code = await asyncio.to_thread(
                launcher.wait,
                process,
                self._config.timeout,
            )
            if exit_code != 0:
                self._last_buffer = await self._get_buffer_snapshot(monitor_lease)
                if allow_partial:
                    trajectory = self._collector.build(
                        buffer=self._last_buffer,
                        episode_id=episode_id,
                    )
                    partial_result = EpisodeResult(
                        trajectory=trajectory,
                        rewards={},
                        final_reward=None,
                        metadata={"exit_code": exit_code},
                        status="failed",
                        failure_info={
                            "exit_code": exit_code,
                            "reason": "MAS non-zero exit",
                        },
                    )
                    return partial_result
                raise RuntimeError(f"MAS process exited with non-zero exit code {exit_code}")

            self._last_buffer = await self._get_buffer_snapshot(monitor_lease)
            trajectory = self._collector.build(buffer=self._last_buffer, episode_id=episode_id)
            result = await asyncio.to_thread(
                self._reward_worker.compute,
                trajectory,
                reward_provider,
            )
            result.metadata["exit_code"] = exit_code
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: Exception | None = None
            self._last_buffer = await self._get_buffer_snapshot(monitor_lease, suppress_inactive_lease=True)

            try:
                launcher.cleanup()
            except Exception as exc:
                cleanup_error = exc

            if primary_error is None and partial_result is None:
                if cleanup_error is not None:
                    raise cleanup_error

    async def _get_buffer_snapshot(
        self,
        monitor_lease: MonitorLease,
        *,
        suppress_inactive_lease: bool = False,
    ) -> list[InteractionRecord]:
        try:
            return await asyncio.to_thread(
                ray.get,
                monitor_lease.actor.snapshot_buffer.remote(monitor_lease.lease_id),
            )
        except Exception:
            if suppress_inactive_lease:
                return list(self._last_buffer)
            raise
