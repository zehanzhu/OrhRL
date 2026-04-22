from __future__ import annotations

import re
import string
from typing import Any

from .matching import is_correct_answer, resolve_match_mode


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def extract_answer(solution_str: str) -> str | None:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_expected_answers(value: Any) -> list[str]:
    if isinstance(value, dict) and "target" in value:
        value = value["target"]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def em_check(prediction: str | None, golden_answers: Any) -> bool:
    return is_correct_answer(
        prediction,
        normalize_expected_answers(golden_answers),
        match_mode="exact",
    )


def subem_check(prediction: str | None, golden_answers: Any) -> bool:
    return is_correct_answer(
        prediction,
        normalize_expected_answers(golden_answers),
        match_mode="substring",
    )


def is_search_answer_correct(
    prediction: str | None,
    golden_answers: Any,
    *,
    match_mode: str | None = None,
    use_substring: bool = False,
) -> bool:
    resolved_match_mode = resolve_match_mode(
        match_mode,
        use_substring_em=use_substring,
    )
    return is_correct_answer(
        prediction,
        normalize_expected_answers(golden_answers),
        match_mode=resolved_match_mode,
    )
