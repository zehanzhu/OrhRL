from __future__ import annotations

from typing import Any

from orchrl.agent_trajectory_engine import FunctionRewardProvider
from orchrl.utils.imports import import_callable


def build_reward_provider(reward_cfg: dict[str, Any]):
    provider_path = reward_cfg.get("provider")
    if not isinstance(provider_path, str) or not provider_path:
        raise ValueError("mate.reward.provider must be a non-empty import path")
    func = import_callable(provider_path)
    return FunctionRewardProvider(func)
