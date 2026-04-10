from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

_DEFAULT_MONITOR_POOL = {
    "size": 4,
    "host": "127.0.0.1",
    "base_port": 19000,
    "acquire_timeout_sec": 300.0,
}


def to_plain_dict(mate_cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(mate_cfg):
        resolved = OmegaConf.to_container(mate_cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("mate config must resolve to a dict")
        return resolved
    if isinstance(mate_cfg, Mapping):
        return dict(mate_cfg)
    raise TypeError("mate config must be a dict")


def _normalize_monitor_pool_config(raw_monitor_pool: Any) -> dict[str, Any]:
    if raw_monitor_pool is None:
        monitor_pool = {}
    elif isinstance(raw_monitor_pool, Mapping):
        monitor_pool = dict(raw_monitor_pool)
    else:
        raise TypeError("mate.monitor_pool must be a dict when provided")

    merged = dict(_DEFAULT_MONITOR_POOL)
    merged.update(monitor_pool)

    host = merged.get("host")
    if not isinstance(host, str) or not host:
        raise ValueError("mate.monitor_pool.host must be a non-empty string")

    size = int(merged.get("size", _DEFAULT_MONITOR_POOL["size"]))
    if size < 1:
        raise ValueError("mate.monitor_pool.size must be >= 1")

    base_port = int(merged.get("base_port", _DEFAULT_MONITOR_POOL["base_port"]))
    if base_port < 1:
        raise ValueError("mate.monitor_pool.base_port must be >= 1")

    acquire_timeout_sec = float(
        merged.get("acquire_timeout_sec", _DEFAULT_MONITOR_POOL["acquire_timeout_sec"])
    )
    if acquire_timeout_sec <= 0:
        raise ValueError("mate.monitor_pool.acquire_timeout_sec must be > 0")

    return {
        "size": size,
        "host": host,
        "base_port": base_port,
        "acquire_timeout_sec": acquire_timeout_sec,
    }


def validate_mate_config(mate_cfg: Any, agent_policy_mapping: Mapping[str, str] | None) -> dict[str, Any]:
    config_dict = to_plain_dict(mate_cfg)
    roles = config_dict.get("roles")
    role_policy_mapping = config_dict.get("role_policy_mapping")
    rollout_mode = config_dict.get("rollout_mode", "parallel")

    if not isinstance(roles, list) or not roles:
        raise ValueError("mate.roles must be a non-empty list")
    if not isinstance(role_policy_mapping, dict) or not role_policy_mapping:
        raise ValueError("mate.role_policy_mapping must be a non-empty dict")
    if rollout_mode not in {"parallel", "tree"}:
        raise ValueError("mate.rollout_mode must be either 'parallel' or 'tree'")

    known_policies = set((agent_policy_mapping or {}).values())
    for role in roles:
        if role not in role_policy_mapping:
            raise ValueError(f"mate.role_policy_mapping missing role '{role}'")
        policy_name = role_policy_mapping[role]
        if not isinstance(policy_name, str) or not policy_name:
            raise ValueError(f"mate.role_policy_mapping for role '{role}' must be a non-empty string")
        if policy_name not in known_policies:
            raise ValueError(f"unknown policy in mate.role_policy_mapping: {policy_name}")

    tree_cfg = config_dict.get("tree", {})
    k_branches = tree_cfg.get("k_branches", config_dict.get("k_branches"))
    max_concurrent_branches = tree_cfg.get(
        "max_concurrent_branches",
        config_dict.get("max_concurrent_branches"),
    )
    if k_branches is not None and int(k_branches) < 1:
        raise ValueError("mate.tree.k_branches must be >= 1")
    if max_concurrent_branches is not None and int(max_concurrent_branches) < 1:
        raise ValueError("mate.tree.max_concurrent_branches must be >= 1")

    config_dict["rollout_mode"] = rollout_mode
    config_dict["monitor_pool"] = _normalize_monitor_pool_config(config_dict.get("monitor_pool"))
    return config_dict
