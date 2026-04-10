from __future__ import annotations

import asyncio
import logging

from .datatypes import EpisodeResult
from .monitor_pool import MonitorPoolManager
from .pipe import AgentPipe, AgentPipeConfig
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def parallel_rollout(
    prompts: list[str],
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    n_samples_per_prompt: int = 1,
    max_concurrent: int | None = None,
    *,
    monitor_pool_manager: MonitorPoolManager,
) -> list[EpisodeResult]:
    """
    Sample n_samples_per_prompt episodes in parallel for each prompt.
    max_concurrent limits the number of AgentPipes running simultaneously (None = unlimited).
    """
    if n_samples_per_prompt < 1:
        raise ValueError("n_samples_per_prompt must be >= 1")
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1 when provided")
    if not prompts:
        return []

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None

    async def run_one(prompt: str) -> EpisodeResult:
        pipe = AgentPipe(config=config)

        async def run_with_monitor() -> EpisodeResult:
            lease = await monitor_pool_manager.acquire()
            try:
                return await pipe.run(
                    prompt=prompt,
                    reward_provider=reward_provider,
                    monitor_lease=lease,
                )
            finally:
                await monitor_pool_manager.release(lease)

        if semaphore is None:
            return await run_with_monitor()
        async with semaphore:
            return await run_with_monitor()

    tasks = [
        run_one(prompt)
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[EpisodeResult] = []
    for item in gathered:
        if isinstance(item, Exception):
            _LOGGER.warning("parallel_rollout dropped failed episode: %s", item)
            continue
        if isinstance(item, BaseException):
            raise item
        results.append(item)
    return results
