from __future__ import annotations

import re
import unicodedata
from typing import Any


_ANSWER_KEYS = (
    "expected_answer",
    "answer",
    "expected_answers",
    "gold_answer",
    "golden_answers",
    "target",
    "label",
)


def _extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _normalize_text(text: Any) -> str:
    raw = "" if text is None else str(text)
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _parse_stringified_candidates(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    stripped = value.strip()
    if not stripped:
        return []
    if not ((stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("(") and stripped.endswith(")"))):
        return []
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", stripped)
    candidates = [left or right for left, right in matches if left or right]
    return [candidate for candidate in candidates if candidate.strip()]


def _expand_expected_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "target" in value:
            return _expand_expected_value(value["target"])
        return []
    if isinstance(value, (list, tuple, set)):
        candidates: list[str] = []
        for item in value:
            candidates.extend(_expand_expected_value(item))
        return candidates
    parsed_candidates = _parse_stringified_candidates(value)
    if parsed_candidates:
        return parsed_candidates
    return [str(value)]


def extract_expected_answers(metadata: dict[str, Any] | None) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    raw_candidates: list[str] = []

    expected = metadata.get("expected")
    if expected is not None:
        raw_candidates.extend(_expand_expected_value(expected))

    prompt_row = metadata.get("prompt_row")
    if isinstance(prompt_row, dict):
        ground_truth_raw = prompt_row.get("ground_truth_raw")
        if ground_truth_raw is not None:
            raw_candidates.extend(_expand_expected_value(ground_truth_raw))
        for key in _ANSWER_KEYS:
            if key in prompt_row and prompt_row[key] is not None:
                raw_candidates.extend(_expand_expected_value(prompt_row[key]))

    expected_answers: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        text = str(candidate).strip()
        normalized = _normalize_text(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            expected_answers.append(text)
    return expected_answers


def extract_predicted_answer(response_text: Any) -> str:
    text = "" if response_text is None else str(response_text).strip()
    if not text:
        return ""
    return _extract_tag(text, "answer") or text


def build_answer_stats(*, response_text: Any, metadata: dict[str, Any] | None) -> dict[str, Any]:
    predicted_answer = extract_predicted_answer(response_text)
    expected_answers = extract_expected_answers(metadata)
    is_correct = _is_correct(
        predicted_answer,
        [_normalize_text(candidate) for candidate in expected_answers],
    )
    return {
        "predicted_answer": predicted_answer,
        "expected_answers": expected_answers,
        "is_correct": is_correct,
    }


def build_trajectory_answer_stats(trajectory: Any) -> dict[str, Any]:
    answer_turns = getattr(trajectory, "agent_trajectories", {}).get("answerer", [])
    response_text = getattr(answer_turns[-1], "response_text", "") if answer_turns else ""
    metadata = getattr(trajectory, "metadata", {})
    return build_answer_stats(
        response_text=response_text,
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _is_correct(predicted: str, expected_candidates: list[str]) -> bool:
    normalized_predicted = _normalize_text(predicted)
    if not normalized_predicted or not expected_candidates:
        return False
    return any(
        normalized_predicted == candidate
        or normalized_predicted in candidate
        or candidate in normalized_predicted
        for candidate in expected_candidates
    )


def compute_reward(trajectory: Any) -> dict[str, Any]:
    answer_stats = build_trajectory_answer_stats(trajectory)
    final_reward = 1.0 if answer_stats["is_correct"] else 0.0

    return {
        "agent_rewards": {role: final_reward for role in trajectory.agent_trajectories},
        "final_reward": final_reward,
    }
