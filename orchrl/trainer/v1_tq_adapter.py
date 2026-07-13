from __future__ import annotations

from collections import defaultdict
from typing import Any
import uuid

import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

V1_TQ_BACKENDS = {"v1", "v1_tq", "ppo_v1", "ppo_v1_tq", "transfer_queue"}

_TRANSFER_QUEUE_INITIALIZED = False


DEFAULT_V1_CONFIG = {
    "trainer_mode": "sync",
    "sync": {},
    "colocate_async": {"num_warmup_batches": 1},
    "separate_async": {"num_warmup_batches": 1, "parameter_sync_step": 4},
    "sampler": {
        "max_off_policy_threshold": 8,
        "max_off_policy_strategy": "drop",
        "custom_sampler": {"path": None, "name": None},
        "sampler_kwargs": {},
    },
}

DEFAULT_TRANSFER_QUEUE_CONFIG = {
    "enable": False,
    "metrics": {"enabled": False, "port": 0},
    "backend": {
        "storage_backend": "SimpleStorage",
        "SimpleStorage": {
            "total_storage_size": 100000,
            "num_data_storage_units": 8,
        },
        "MooncakeStore": {
            "auto_init": False,
            "metadata_server": "localhost:50123",
            "master_server_address": "localhost:50124",
            "local_hostname": "localhost",
            "protocol": "tcp",
            "global_segment_size": 4294967296,
            "local_buffer_size": 1073741824,
            "device_name": "",
        },
    },
}


def is_v1_tq_backend(config: Any) -> bool:
    training_cfg = getattr(config, "training", None)
    for owner in (training_cfg, config):
        if owner is None:
            continue
        backend = getattr(owner, "ppo_backend", None)
        if backend is not None and str(backend).lower() in V1_TQ_BACKENDS:
            return True
    return False


def get_v1_trainer_cls(ppo_config: DictConfig):
    from verl.trainer.ppo.v1 import get_trainer_cls

    ensure_v1_ppo_config(ppo_config)
    return get_trainer_cls(str(ppo_config.trainer.v1.trainer_mode))


def ensure_v1_ppo_config(ppo_config: DictConfig) -> None:
    _ensure_child_config(ppo_config, "trainer", {})
    _set_config_value(ppo_config.trainer, "use_v1", True)
    _merge_missing(ppo_config.trainer, "v1", DEFAULT_V1_CONFIG)
    trainer_mode = str(ppo_config.trainer.v1.trainer_mode)
    if trainer_mode != "sync":
        raise ValueError(
            "OrchRL V1 TransferQueue backend currently supports "
            "trainer.v1.trainer_mode=sync only; "
            f"got {trainer_mode!r}."
        )
    _merge_missing(ppo_config, "transfer_queue", DEFAULT_TRANSFER_QUEUE_CONFIG)
    ppo_config.transfer_queue.enable = True


def ensure_top_level_transfer_queue_config(config: DictConfig) -> None:
    _merge_missing(config, "transfer_queue", DEFAULT_TRANSFER_QUEUE_CONFIG)
    config.transfer_queue.enable = True


def initialize_transfer_queue_for_v1(config: DictConfig) -> None:
    global _TRANSFER_QUEUE_INITIALIZED
    ensure_top_level_transfer_queue_config(config)
    if _TRANSFER_QUEUE_INITIALIZED:
        return
    tq.init(config.transfer_queue)
    _TRANSFER_QUEUE_INITIALIZED = True


def close_transfer_queue_for_v1() -> None:
    global _TRANSFER_QUEUE_INITIALIZED
    if not _TRANSFER_QUEUE_INITIALIZED:
        return
    tq.close()
    _TRANSFER_QUEUE_INITIALIZED = False


def prepare_dataproto_for_v1_transfer_queue(
    *,
    data_proto,
    trainer,
    global_steps: int,
    partition_id: str = "train",
) -> int:
    if data_proto is None or getattr(data_proto, "batch", None) is None:
        return 0
    if len(data_proto) == 0:
        return 0

    tokenizer = getattr(trainer, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
    if pad_token_id is None:
        pad_token_id = 0

    records = _build_v1_records(
        data_proto=data_proto,
        pad_token_id=int(pad_token_id),
        global_steps=int(global_steps),
    )
    if not records:
        return 0

    group_to_prompt_key: dict[str, str] = {}
    group_to_session_idx: dict[str, int] = defaultdict(int)
    prompt_keys: list[str] = []
    prompt_tags: list[dict[str, Any]] = []
    trajectory_keys: list[str] = []
    trajectory_tags: list[dict[str, Any]] = []
    trajectory_fields: list[dict[str, Any]] = []

    for record in records:
        group_key = str(record.pop("_group_key"))
        prompt_key = group_to_prompt_key.get(group_key)
        if prompt_key is None:
            prompt_key = f"orchrl{uuid.uuid4().hex}"
            group_to_prompt_key[group_key] = prompt_key
            prompt_keys.append(prompt_key)
            prompt_tags.append(
                {
                    "is_prompt": True,
                    "status": "finished",
                    "global_steps": int(global_steps),
                }
            )

        session_idx = group_to_session_idx[group_key]
        group_to_session_idx[group_key] += 1
        trajectory_keys.append(f"{prompt_key}_{session_idx}_0")
        trajectory_tags.append(
            {
                "status": "success",
                "prompt_len": int(record["prompts"].numel()),
                "response_len": int(record["responses"].numel()),
                "seq_len": int(record["input_ids"].numel()),
                "global_steps": int(global_steps),
                "min_global_steps": int(global_steps),
                "max_global_steps": int(global_steps),
            }
        )
        trajectory_fields.append(record)

    tq.kv_batch_put(
        keys=prompt_keys,
        partition_id=partition_id,
        tags=prompt_tags,
    )
    tq.kv_batch_put(
        keys=trajectory_keys,
        partition_id=partition_id,
        fields=list_of_dict_to_tensordict(trajectory_fields),
        tags=trajectory_tags,
    )
    return len(prompt_keys)


def _build_v1_records(*, data_proto, pad_token_id: int, global_steps: int):
    batch = data_proto.batch
    non_tensors = getattr(data_proto, "non_tensor_batch", {}) or {}
    rewards = _as_list(non_tensors.get("reward", [0.0] * len(data_proto)))
    group_keys = _resolve_group_keys(data_proto)

    records = []
    for index in range(len(data_proto)):
        prompt_ids = _trim_left_padding(
            _tensor_to_list(batch["prompts"][index]),
            pad_token_id=pad_token_id,
        )
        response_mask = batch.get("response_mask", None)
        if response_mask is not None:
            response_len = int(batch["response_mask"][index].sum().item())
        else:
            response_len = _infer_response_length(
                _tensor_to_list(batch["responses"][index]),
                pad_token_id=pad_token_id,
            )
        if response_len <= 0:
            continue

        response_ids = _tensor_to_list(batch["responses"][index])[:response_len]
        prompts = torch.tensor(prompt_ids, dtype=torch.int64)
        responses = torch.tensor(response_ids, dtype=torch.int64)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        position_ids = torch.arange(input_ids.numel(), dtype=torch.int64)
        response_mask_tensor = torch.ones(responses.numel(), dtype=torch.int64)
        rm_scores = torch.zeros(responses.numel(), dtype=torch.float32)
        rm_scores[-1] = float(rewards[index])
        group_key = str(group_keys[index])

        records.append(
            {
                "_group_key": group_key,
                "uid": group_key,
                "global_steps": int(global_steps),
                "session_id": int(index),
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask_tensor,
                "loss_mask": response_mask_tensor.clone(),
                "rm_scores": rm_scores,
                "token_level_rewards": rm_scores.clone(),
                "rollout_log_probs": torch.zeros_like(rm_scores),
                "ref_log_prob": torch.zeros_like(rm_scores),
                "values": torch.zeros_like(rm_scores),
                "num_turns": int(_resolve_num_turns(non_tensors, index)),
                "data_source": "orchrl_mate",
                "reward_model": {"ground_truth": None},
                "extra_fields": {
                    "agent_name": _optional_non_tensor(non_tensors, "agent_name", index),
                    "episode_id": _optional_non_tensor(non_tensors, "episode_id", index),
                    "prompt_group_id": _optional_non_tensor(
                        non_tensors,
                        "prompt_group_id",
                        index,
                    ),
                },
            }
        )

    return records


def _resolve_group_keys(data_proto) -> list[str]:
    non_tensors = getattr(data_proto, "non_tensor_batch", {}) or {}
    uids = _as_list(non_tensors.get("uid", [str(i) for i in range(len(data_proto))]))
    group_ids = non_tensors.get("group_id")
    if group_ids is None:
        return [str(uid) for uid in uids]
    resolved = []
    for uid, group_id in zip(uids, _as_list(group_ids), strict=False):
        resolved.append(str(group_id) if group_id is not None else str(uid))
    return resolved


def _resolve_num_turns(non_tensors, index: int) -> int:
    turn_indices = non_tensors.get("turn_idx")
    if turn_indices is None:
        return 1
    try:
        return int(_as_list(turn_indices)[index]) + 1
    except Exception:
        return 1


def _optional_non_tensor(non_tensors, key: str, index: int):
    if key not in non_tensors:
        return None
    values = _as_list(non_tensors[key])
    if index >= len(values):
        return None
    value = values[index]
    return value.item() if hasattr(value, "item") else value


def _tensor_to_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().tolist()]
    if hasattr(value, "tolist"):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def _trim_left_padding(tokens: list[int], *, pad_token_id: int) -> list[int]:
    index = 0
    while index < len(tokens) - 1 and tokens[index] == pad_token_id:
        index += 1
    return tokens[index:]


def _infer_response_length(tokens: list[int], *, pad_token_id: int) -> int:
    length = len(tokens)
    while length > 0 and tokens[length - 1] == pad_token_id:
        length -= 1
    return length


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _merge_missing(config: DictConfig, key: str, default: dict) -> None:
    if not hasattr(config, key) or getattr(config, key) is None:
        _set_config_value(config, key, OmegaConf.create(default))
        return
    merged = OmegaConf.merge(OmegaConf.create(default), getattr(config, key))
    _set_config_value(config, key, merged)


def _ensure_child_config(config: DictConfig, key: str, default: dict) -> None:
    if hasattr(config, key) and getattr(config, key) is not None:
        return
    _set_config_value(config, key, OmegaConf.create(default))


def _set_config_value(config: DictConfig, key: str, value) -> None:
    if OmegaConf.is_config(config):
        with open_dict(config):
            config[key] = value
        return
    setattr(config, key, value)
