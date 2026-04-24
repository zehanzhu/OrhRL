from __future__ import annotations

import re
import unicodedata
from typing import Any

MATCH_MODE_EXACT = "exact"
MATCH_MODE_SUBSTRING = "substring"
_VALID_MATCH_MODES = {MATCH_MODE_EXACT, MATCH_MODE_SUBSTRING}


def normalize_text(text: Any) -> str:
    raw = "" if text is None else str(text)
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def resolve_match_mode(
    match_mode: str | None = None,
    *,
    use_substring_em: bool | None = None,
) -> str:
    if match_mode is None:
        return MATCH_MODE_SUBSTRING if use_substring_em else MATCH_MODE_EXACT
    resolved = str(match_mode).strip().lower()
    if resolved not in _VALID_MATCH_MODES:
        raise ValueError(f"unsupported match_mode: {match_mode}")
    return resolved


def extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def is_correct_answer(
    predicted: Any,
    expected_candidates: list[Any],
    *,
    match_mode: str,
) -> bool:
    normalized_prediction = normalize_text(predicted)
    normalized_candidates = [normalize_text(candidate) for candidate in expected_candidates]
    normalized_candidates = [candidate for candidate in normalized_candidates if candidate]
    if not normalized_prediction or not normalized_candidates:
        return False
    if match_mode == MATCH_MODE_EXACT:
        return any(normalized_prediction == candidate for candidate in normalized_candidates)
    if match_mode == MATCH_MODE_SUBSTRING:
        return any(
            normalized_prediction == candidate
            or normalized_prediction in candidate
            or candidate in normalized_prediction
            for candidate in normalized_candidates
        )
    raise ValueError(f"unsupported match_mode: {match_mode}")
