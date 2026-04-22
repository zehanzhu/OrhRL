from __future__ import annotations

import asyncio
import logging

from ._support.replay_cache import ReplayCache
from .datatypes import BranchResult, InteractionRecord, TreeEpisodeResult
from .monitor_pool import MonitorPoolManager
from .pipe import AgentPipe, AgentPipeConfig
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def tree_rollout(
    prompt: str,
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    k_branches: int = 3,
    max_concurrent_branches: int | None = None,
    *,
    monitor_pool_manager: MonitorPoolManager,
) -> TreeEpisodeResult:
    if k_branches < 1:
        raise ValueError("k_branches must be >= 1")
    if max_concurrent_branches is not None and max_concurrent_branches < 1:
        raise ValueError("max_concurrent_branches must be >= 1 when provided")

    pilot_pipe = AgentPipe(config=config)
    pilot_result = await _run_pipe_with_monitor(
        pipe=pilot_pipe,
        prompt=prompt,
        reward_provider=reward_provider,
        monitor_pool_manager=monitor_pool_manager,
    )
    pilot_buffer = _sorted_buffer(pilot_pipe.last_buffer())
    pilot_total_turns = len(pilot_buffer)

    if pilot_result.status != "success" or not pilot_buffer:
        expected_branch_count = 0
        return TreeEpisodeResult(
            pilot_result=pilot_result,
            branch_results=[],
            prompt=prompt,
            tree_metadata={
                "n_branch_points": 0,
                "k_branches": k_branches,
                "expected_branch_count": expected_branch_count,
                "failed_branch_count": 0,
                "total_branches_collected": 0,
                "pilot_total_turns": pilot_total_turns,
            },
        )

    semaphore = (
        asyncio.Semaphore(max_concurrent_branches)
        if max_concurrent_branches is not None
        else None
    )

    async def run_branch(record: InteractionRecord, global_position: int) -> BranchResult | None:
        cache = ReplayCache.from_buffer(
            pilot_buffer,
            branch_at_global_position=global_position,
        )
        branch_pipe = AgentPipe(config=config)

        async def execute() -> BranchResult | None:
            try:
                branch_result = await _run_pipe_with_monitor(
                    pipe=branch_pipe,
                    prompt=prompt,
                    reward_provider=reward_provider,
                    allow_partial=True,
                    monitor_pool_manager=monitor_pool_manager,
                    replay_cache=cache,
                )
            except Exception as exc:
                _LOGGER.warning(
                    "tree_rollout dropped failed branch at position %s for %s[%s]: %s",
                    global_position,
                    record.agent_role,
                    record.turn_index,
                    exc,
                )
                return None

            if branch_result.status != "success":
                return None

            _annotate_branch_result(branch_result, branch_turn=global_position)

            return BranchResult(
                episode_result=branch_result,
                branch_turn=global_position,
                branch_agent_role=record.agent_role,
                parent_episode_id=pilot_result.trajectory.episode_id,
            )

        if semaphore is None:
            return await execute()

        async with semaphore:
            return await execute()

    tasks = [
        run_branch(record, global_position)
        for global_position, record in enumerate(pilot_buffer)
        for _ in range(k_branches)
    ]
    expected_branch_count = len(tasks)
    branch_results = [result for result in await asyncio.gather(*tasks) if result is not None]

    return TreeEpisodeResult(
        pilot_result=pilot_result,
        branch_results=branch_results,
        prompt=prompt,
        tree_metadata={
            "n_branch_points": pilot_total_turns,
            "k_branches": k_branches,
            "expected_branch_count": expected_branch_count,
            "failed_branch_count": expected_branch_count - len(branch_results),
            "total_branches_collected": len(branch_results),
            "pilot_total_turns": pilot_total_turns,
        },
    )


async def _run_pipe_with_monitor(
    *,
    pipe: AgentPipe,
    prompt: str,
    reward_provider: RewardProvider,
    monitor_pool_manager: MonitorPoolManager,
    allow_partial: bool = False,
    replay_cache: ReplayCache | None = None,
):
    lease = await monitor_pool_manager.acquire(replay_cache=replay_cache)
    try:
        return await pipe.run(
            prompt=prompt,
            reward_provider=reward_provider,
            allow_partial=allow_partial,
            monitor_lease=lease,
        )
    finally:
        await monitor_pool_manager.release(lease)


def _sorted_buffer(buffer: list[InteractionRecord]) -> list[InteractionRecord]:
    return sorted(buffer, key=lambda record: record.timestamp)


def _annotate_branch_result(result, branch_turn: int) -> None:
    all_turns = [
        turn
        for turns in result.trajectory.agent_trajectories.values()
        for turn in turns
    ]
    ordered_turns = sorted(all_turns, key=lambda turn: turn.timestamp)

    for idx, turn in enumerate(ordered_turns):
        if idx < branch_turn:
            replayed = True
            branch_phase = "replay_prefix"
        elif idx == branch_turn:
            replayed = False
            branch_phase = "branch_point"
        else:
            replayed = False
            branch_phase = "post_branch"

        turn.replayed = replayed
        turn.branch_phase = branch_phase
        turn.metadata["replayed"] = replayed
        turn.metadata["branch_phase"] = branch_phase
