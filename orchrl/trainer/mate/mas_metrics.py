from __future__ import annotations

import re
from typing import Any, Iterable

from orchrl.agent_trajectory_engine.datatypes import BranchResult, EpisodeResult, TreeEpisodeResult


_SEARCH_TAG_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_VERIFY_TAG_RE = re.compile(r"<verify>\s*(.*?)\s*</verify>", re.IGNORECASE | re.DOTALL)


def build_mas_metrics(
    *,
    episodes: Iterable[Any],
    expected_sample_count: int,
    failed_count: int,
    prefix: str,
) -> dict[str, float]:
    """Build MAS-level outcome and behavior metrics.

    Outcome metrics use expected_sample_count as denominator so failed rollouts
    count as zero reward. Behavior metrics are averaged over successful episodes.
    """
    stats = init_mas_metric_stats()
    accumulate_mas_metric_stats(stats, episodes)
    return finalize_mas_metrics(
        stats=stats,
        expected_sample_count=expected_sample_count,
        failed_count=failed_count,
        prefix=prefix,
    )


def init_mas_metric_stats() -> dict[str, float]:
    return {
        "success_count": 0.0,
        "correct_count": 0.0,
        "reward_sum": 0.0,
        "turn_sum": 0.0,
        "search_call_sum": 0.0,
        "search_episode_count": 0.0,
        "answer_episode_count": 0.0,
        "verifier_turn_count": 0.0,
        "verifier_yes_count": 0.0,
        "verifier_no_count": 0.0,
    }


def accumulate_mas_metric_stats(stats: dict[str, float], episodes: Iterable[Any]) -> None:
    for episode in _iter_episode_results(episodes):
        stats["success_count"] += 1.0
        reward = _normalize_reward(getattr(episode, "final_reward", 0.0))
        stats["reward_sum"] += reward
        if _is_correct_reward(reward):
            stats["correct_count"] += 1.0
        behavior = _episode_behavior(episode)
        stats["turn_sum"] += float(behavior["turn_count"])
        stats["search_call_sum"] += float(behavior["search_calls"])
        if float(behavior["search_calls"]) > 0:
            stats["search_episode_count"] += 1.0
        if behavior["has_answer"]:
            stats["answer_episode_count"] += 1.0
        stats["verifier_turn_count"] += float(behavior["verifier_turns"])
        stats["verifier_yes_count"] += float(behavior["verifier_yes"])
        stats["verifier_no_count"] += float(behavior["verifier_no"])


def finalize_mas_metrics(
    *,
    stats: dict[str, float],
    expected_sample_count: int,
    failed_count: int,
    prefix: str,
) -> dict[str, float]:
    success_count = float(stats.get("success_count", 0.0))
    return {
        f"{prefix}/sample_avg_reward": _safe_div(
            float(stats.get("reward_sum", 0.0)),
            expected_sample_count,
        ),
        f"{prefix}/accuracy": _safe_div(
            float(stats.get("correct_count", 0.0)),
            expected_sample_count,
        ),
        f"{prefix}/success_rate": _safe_div(success_count, expected_sample_count),
        f"{prefix}/failed_rate": _safe_div(failed_count, expected_sample_count),
        f"{prefix}/avg_turns": _safe_div(float(stats.get("turn_sum", 0.0)), success_count),
        f"{prefix}/avg_search_calls": _safe_div(
            float(stats.get("search_call_sum", 0.0)),
            success_count,
        ),
        f"{prefix}/search_call_rate": _safe_div(
            float(stats.get("search_episode_count", 0.0)),
            success_count,
        ),
        f"{prefix}/answer_rate": _safe_div(
            float(stats.get("answer_episode_count", 0.0)),
            success_count,
        ),
        f"{prefix}/verifier_yes_rate": _safe_div(
            float(stats.get("verifier_yes_count", 0.0)),
            float(stats.get("verifier_turn_count", 0.0)),
        ),
        f"{prefix}/verifier_no_rate": _safe_div(
            float(stats.get("verifier_no_count", 0.0)),
            float(stats.get("verifier_turn_count", 0.0)),
        ),
    }


def _iter_episode_results(episodes: Iterable[Any]):
    for episode in episodes:
        if isinstance(episode, TreeEpisodeResult):
            yield episode.pilot_result
            for branch in episode.branch_results:
                if isinstance(branch, BranchResult):
                    yield branch.episode_result
                else:
                    branch_episode = getattr(branch, "episode_result", None)
                    if branch_episode is not None:
                        yield branch_episode
            continue
        yield episode


def _episode_behavior(episode: EpisodeResult) -> dict[str, float | bool]:
    turn_count = 0
    search_calls = 0
    has_answer = False
    verifier_turns = 0
    verifier_yes = 0
    verifier_no = 0

    trajectory = getattr(episode, "trajectory", None)
    agent_trajectories = getattr(trajectory, "agent_trajectories", {}) or {}
    for role, turns in agent_trajectories.items():
        for turn in turns or []:
            turn_count += 1
            response_text = str(getattr(turn, "response_text", "") or "")
            role_name = str(role or getattr(turn, "agent_role", ""))
            if role_name == "searcher" and _extract_tag_value(_SEARCH_TAG_RE, response_text):
                search_calls += 1
            if role_name == "answerer" and _extract_tag_value(_ANSWER_TAG_RE, response_text):
                has_answer = True
            if role_name == "verifier":
                verifier_turns += 1
                verify_value = _extract_tag_value(_VERIFY_TAG_RE, response_text).lower()
                if verify_value == "yes":
                    verifier_yes += 1
                elif verify_value == "no":
                    verifier_no += 1

    return {
        "turn_count": float(turn_count),
        "search_calls": float(search_calls),
        "has_answer": bool(has_answer),
        "verifier_turns": float(verifier_turns),
        "verifier_yes": float(verifier_yes),
        "verifier_no": float(verifier_no),
    }


def _extract_tag_value(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _normalize_reward(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        return float(sum(float(item) for item in value))
    return float(value)


def _is_correct_reward(value: float) -> bool:
    return float(value) >= 1.0


def _safe_div(numerator: float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator
