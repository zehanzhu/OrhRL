from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ChatRenderer:
    def __init__(
        self,
        tokenizer: Any,
        *,
        model_name: str | None = None,
        max_prompt_length: int | None = None,
        truncation: str = "error",
    ) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._max_prompt_length = (
            int(max_prompt_length) if max_prompt_length is not None else None
        )
        if self._max_prompt_length is not None and self._max_prompt_length < 1:
            raise ValueError("max_prompt_length must be >= 1 when provided")
        if truncation not in {"left", "right", "middle", "error"}:
            raise ValueError(f"Unsupported truncation mode: {truncation}")
        self._truncation = truncation

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        model_name: str | None = None,
        *,
        max_prompt_length: int | None = None,
        truncation: str = "error",
    ) -> "ChatRenderer":
        return cls(
            tokenizer,
            model_name=model_name,
            max_prompt_length=max_prompt_length,
            truncation=truncation,
        )

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[list[int] | None, dict[str, Any]]:
        try:
            prompt_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
            )
        except TypeError:
            prompt_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )

        return self._truncate_ids(self._normalize_ids(prompt_ids)), {
            "model_name": self._model_name,
            "add_generation_prompt": add_generation_prompt,
            "tokenizer_class": type(self._tokenizer).__name__,
            "max_prompt_length": self._max_prompt_length,
            "truncation": self._truncation,
        }

    @staticmethod
    def _normalize_ids(token_ids: Any) -> list[int] | None:
        if token_ids is None:
            return None
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            return None

        ids: list[int] = []
        for token_id in token_ids:
            if isinstance(token_id, bool):
                return None
            if not isinstance(token_id, int):
                return None
            ids.append(int(token_id))
        return ids if ids else None

    def _truncate_ids(self, token_ids: list[int] | None) -> list[int] | None:
        if token_ids is None or self._max_prompt_length is None:
            return token_ids
        if len(token_ids) <= self._max_prompt_length:
            return token_ids

        if self._truncation == "left":
            return token_ids[-self._max_prompt_length :]
        if self._truncation == "right":
            return token_ids[: self._max_prompt_length]
        if self._truncation == "middle":
            left_half = self._max_prompt_length // 2
            right_half = self._max_prompt_length - left_half
            return token_ids[:left_half] + token_ids[-right_half:]

        raise RuntimeError(
            f"Prompt length {len(token_ids)} is longer than {self._max_prompt_length}."
        )
