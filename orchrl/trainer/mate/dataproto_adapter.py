from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from verl import DataProto

from orchrl.agent_trajectory_engine import TreeEpisodeResult



def episodes_to_policy_batches(
    *,
    episodes,
    tokenizer_dict,
    role_policy_mapping,
    max_prompt_length,
    max_response_length,
    role_index_mapping=None,
    credit_assignment="all_turns",
):
    records_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    role_index_mapping = dict(role_index_mapping or {role: idx for idx, role in enumerate(role_policy_mapping.keys())})

    for env_idx, episode in enumerate(episodes):
        _append_episode_records(
            records_by_policy=records_by_policy,
            env_idx=env_idx,
            episode=episode,
            role_policy_mapping=role_policy_mapping,
            role_index_mapping=role_index_mapping,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            credit_assignment=credit_assignment,
            uid_factory=lambda *, prompt_group_id, agent_idx, **_: f"{prompt_group_id}:{agent_idx}",
        )

    return _build_batches(records_by_policy=records_by_policy, tokenizer_dict=tokenizer_dict)


def tree_episodes_to_decision_point_batches(
    *,
    episodes: list[TreeEpisodeResult],
    tokenizer_dict,
    role_policy_mapping,
    max_prompt_length,
    max_response_length,
    role_index_mapping=None,
    credit_assignment="all_turns",
):
    records_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    role_index_mapping = dict(role_index_mapping or {role: idx for idx, role in enumerate(role_policy_mapping.keys())})

    for env_idx, tree_episode in enumerate(episodes):
        pilot_episode = tree_episode.pilot_result
        successful_branches = [branch for branch in tree_episode.branch_results if branch.episode_result.status == "success"]
        pilot_reward_lookup = _build_turn_reward_lookup(
            episode=pilot_episode,
            credit_assignment=credit_assignment,
        )
        branch_reward_lookups = {
            id(branch): _build_turn_reward_lookup(
                episode=branch.episode_result,
                credit_assignment=credit_assignment,
            )
            for branch in successful_branches
        }

        for role, turn, global_turn_index in _iter_global_turns(pilot_episode):
            _append_decision_point_record(
                records_by_policy=records_by_policy,
                env_idx=env_idx,
                episode=pilot_episode,
                role=role,
                turn=turn,
                global_turn_index=global_turn_index,
                role_policy_mapping=role_policy_mapping,
                role_index_mapping=role_index_mapping,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                reward_lookup=pilot_reward_lookup,
                source="pilot",
            )

            for branch in successful_branches:
                if branch.branch_turn != global_turn_index:
                    continue
                branch_role, branch_turn = _select_branch_decision_turn(branch)
                _append_decision_point_record(
                    records_by_policy=records_by_policy,
                    env_idx=env_idx,
                    episode=branch.episode_result,
                    role=branch_role,
                    turn=branch_turn,
                    global_turn_index=global_turn_index,
                    role_policy_mapping=role_policy_mapping,
                    role_index_mapping=role_index_mapping,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                    reward_lookup=branch_reward_lookups[id(branch)],
                    source="branch",
                )

    return _build_batches(records_by_policy=records_by_policy, tokenizer_dict=tokenizer_dict)


def _resolve_turn_rewards(*, reward_value, turn_count: int, credit_assignment: str) -> list[float]:
    if isinstance(reward_value, list):
        if len(reward_value) != turn_count:
            raise ValueError("per-turn reward list must match turn count")
        return [float(item) for item in reward_value]
    scalar_reward = float(reward_value)
    if credit_assignment == "last_turn":
        rewards = [0.0] * turn_count
        rewards[-1] = scalar_reward
        return rewards
    if credit_assignment == "all_turns":
        return [scalar_reward] * turn_count
    raise ValueError(f"unsupported credit_assignment: {credit_assignment}")


def _append_episode_records(
    *,
    records_by_policy,
    env_idx: int,
    episode,
    role_policy_mapping,
    role_index_mapping,
    max_prompt_length: int,
    max_response_length: int,
    credit_assignment: str,
    uid_factory,
    skip_turn_predicate=None,
    global_turn_index_lookup=None,
):
    episode_id = episode.trajectory.episode_id
    prompt_group_id = str(episode.metadata.get("prompt_group_id", episode_id))
    sample_idx = int(episode.metadata.get("sample_idx", 0))

    for role, turns in episode.trajectory.agent_trajectories.items():
        if not turns:
            continue
        policy_name = role_policy_mapping[role]
        agent_idx = int(role_index_mapping[role])
        turn_rewards = _resolve_turn_rewards(
            reward_value=episode.rewards.get(role, 0.0),
            turn_count=len(turns),
            credit_assignment=credit_assignment,
        )

        for turn, reward_value in zip(turns, turn_rewards):
            global_turn_index = None
            if global_turn_index_lookup is not None:
                global_turn_index = global_turn_index_lookup[id(turn)]
            if skip_turn_predicate is not None and skip_turn_predicate(
                turn,
                role=role,
                global_turn_index=global_turn_index,
            ):
                continue

            prompt_ids = _resolve_prompt_ids(
                turn=turn,
                max_prompt_length=max_prompt_length,
            )
            response_ids = _normalize_response_ids(turn.token_ids, max_response_length)
            records_by_policy[policy_name].append(
                {
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_mask": [1] * len(response_ids),
                    "agent_name": role,
                    "agent_idx": agent_idx,
                    "turn_idx": turn.turn_index,
                    "env_idx": env_idx,
                    "episode_id": episode_id,
                    "prompt_group_id": prompt_group_id,
                    "sample_idx": sample_idx,
                    "reward": float(reward_value),
                    "uid": uid_factory(
                        prompt_group_id=prompt_group_id,
                        agent_idx=agent_idx,
                        role=role,
                        turn=turn,
                        global_turn_index=global_turn_index,
                    ),
                }
            )


def _append_decision_point_record(
    *,
    records_by_policy,
    env_idx: int,
    episode,
    role: str,
    turn,
    global_turn_index: int,
    role_policy_mapping,
    role_index_mapping,
    max_prompt_length: int,
    max_response_length: int,
    reward_lookup: dict[int, float],
    source: str,
):
    episode_id = episode.trajectory.episode_id
    prompt_group_id = str(episode.metadata.get("prompt_group_id", episode_id))
    sample_idx = int(episode.metadata.get("sample_idx", 0))
    policy_name = role_policy_mapping[role]
    agent_idx = int(role_index_mapping[role])
    group_id = _decision_point_group_id(
        prompt_group_id=prompt_group_id,
        global_turn_index=global_turn_index,
        agent_idx=agent_idx,
    )
    prompt_ids = _resolve_prompt_ids(
        turn=turn,
        max_prompt_length=max_prompt_length,
    )
    response_ids = _normalize_response_ids(turn.token_ids, max_response_length)
    records_by_policy[policy_name].append(
        {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": [1] * len(response_ids),
            "agent_name": role,
            "agent_idx": agent_idx,
            "turn_idx": turn.turn_index,
            "env_idx": env_idx,
            "episode_id": episode_id,
            "prompt_group_id": prompt_group_id,
            "sample_idx": sample_idx,
            "reward": float(reward_lookup[id(turn)]),
            "uid": _decision_point_uid(group_id=group_id, source=source, episode_id=episode_id),
            "group_id": group_id,
        }
    )


def _build_turn_reward_lookup(*, episode, credit_assignment: str) -> dict[int, float]:
    reward_lookup: dict[int, float] = {}
    for role, turns in episode.trajectory.agent_trajectories.items():
        if not turns:
            continue
        turn_rewards = _resolve_turn_rewards(
            reward_value=episode.rewards.get(role, 0.0),
            turn_count=len(turns),
            credit_assignment=credit_assignment,
        )
        for turn, reward_value in zip(turns, turn_rewards):
            reward_lookup[id(turn)] = float(reward_value)
    return reward_lookup


def _flatten_global_turns(episode) -> list[tuple[float, int, str, Any]]:
    flattened_turns = []
    for role, turns in episode.trajectory.agent_trajectories.items():
        for turn in turns:
            flattened_turns.append((turn.timestamp, turn.turn_index, role, turn))
    flattened_turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return flattened_turns


def _iter_global_turns(episode):
    for index, (_, _, role, turn) in enumerate(_flatten_global_turns(episode)):
        yield role, turn, index


def _select_branch_decision_turn(branch) -> tuple[str, Any]:
    for role, turn, _ in _iter_global_turns(branch.episode_result):
        branch_phase = getattr(turn, "branch_phase", None) or turn.metadata.get("branch_phase")
        if branch_phase == "branch_point":
            return role, turn

    for role, turn, global_turn_index in _iter_global_turns(branch.episode_result):
        if global_turn_index == branch.branch_turn:
            return role, turn

    raise ValueError(
        f"Unable to locate decision-point turn for branch_turn={branch.branch_turn} "
        f"episode_id={branch.episode_result.trajectory.episode_id}"
    )


def _turn_global_positions(episode) -> dict[int, int]:
    return {id(turn): index for index, (_, _, _, turn) in enumerate(_flatten_global_turns(episode))}


def _tree_uid(*, prompt_group_id: str, agent_idx: int, branch_turn: int, global_turn_index: int) -> str:
    if global_turn_index == branch_turn:
        return f"{prompt_group_id}:{agent_idx}:b{branch_turn}"
    return f"{prompt_group_id}:{agent_idx}:b{branch_turn}:t{global_turn_index}"


def _decision_point_group_id(*, prompt_group_id: str, global_turn_index: int, agent_idx: int) -> str:
    return f"{prompt_group_id}:turn{global_turn_index}:agent{agent_idx}"


def _decision_point_uid(*, group_id: str, source: str, episode_id: str) -> str:
    return f"{group_id}:{source}:{episode_id}"


def _resolve_prompt_ids(*, turn, max_prompt_length: int) -> list[int]:
    prompt_ids = _normalize_prompt_ids(turn.prompt_ids, max_prompt_length)
    if prompt_ids is not None:
        return prompt_ids
    raise ValueError(
        "TurnData.prompt_ids must be populated for prompt-id MATE training "
        f"(agent_role={turn.agent_role}, turn_index={turn.turn_index})"
    )


def _normalize_prompt_ids(prompt_ids, max_prompt_length: int) -> list[int] | None:
    if prompt_ids is None:
        return None
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    elif hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if not isinstance(prompt_ids, list):
        raise TypeError("TurnData.prompt_ids must be a list of token ids when provided")
    normalized = [int(token_id) for token_id in prompt_ids]
    if len(normalized) > max_prompt_length:
        raise ValueError(
            "TurnData.prompt_ids is longer than configured max_prompt_length; "
            "runtime prompt truncation and trainer prompt limits are inconsistent"
        )
    if not normalized:
        raise ValueError("TurnData.prompt_ids must contain at least one token when provided")
    return normalized


def _normalize_response_ids(token_ids, max_response_length: int) -> list[int]:
    if token_ids is None:
        raise ValueError("TurnData.token_ids must not be None for MATE training")
    response_ids = [int(token_id) for token_id in token_ids]
    if len(response_ids) > max_response_length:
        raise ValueError(
            "TurnData.token_ids is longer than configured max_response_length; "
            "runtime response truncation and trainer response limits are inconsistent"
        )
    if not response_ids:
        raise ValueError("TurnData.token_ids must contain at least one token")
    return response_ids


def _pad_sequences(sequences: list[list[int]], *, pad_value: int, left_pad: bool) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        pad_length = max_len - len(sequence)
        if left_pad:
            padded.append([pad_value] * pad_length + sequence)
        else:
            padded.append(sequence + [pad_value] * pad_length)
    return torch.tensor(padded, dtype=torch.long)


def _build_batches(*, records_by_policy, tokenizer_dict):
    batches = {}
    for policy_name, records in records_by_policy.items():
        tokenizer = tokenizer_dict[policy_name]
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        non_tensors = {
            "agent_name": [record["agent_name"] for record in records],
            "agent_idx": [record["agent_idx"] for record in records],
            "turn_idx": [record["turn_idx"] for record in records],
            "env_idx": [record["env_idx"] for record in records],
            "episode_id": [record["episode_id"] for record in records],
            "prompt_group_id": [record["prompt_group_id"] for record in records],
            "sample_idx": [record["sample_idx"] for record in records],
            "reward": [record["reward"] for record in records],
            "uid": [record["uid"] for record in records],
        }
        if any("group_id" in record for record in records):
            non_tensors["group_id"] = [record.get("group_id") for record in records]
        batches[policy_name] = DataProto.from_dict(
            tensors={
                "prompts": _pad_sequences([record["prompt_ids"] for record in records], pad_value=pad_token_id, left_pad=True),
                "responses": _pad_sequences([record["response_ids"] for record in records], pad_value=pad_token_id, left_pad=False),
                "response_mask": _pad_sequences([record["response_mask"] for record in records], pad_value=0, left_pad=False),
            },
            non_tensors=non_tensors,
        )
    return batches
