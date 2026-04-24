from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class MatePromptLoader:
    def __init__(
        self,
        source_type: str,
        path: str | Path,
        prompt_keys: list[str],
        expected_keys: list[str] | None = None,
        *,
        repeat: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if not prompt_keys:
            raise ValueError("prompt_keys must be a non-empty list")
        self._rows = self._load_rows(source_type=source_type, path=path)
        self._prompt_keys = prompt_keys
        self._expected_keys = expected_keys or []
        self._repeat = bool(repeat)
        self._shuffle = bool(shuffle)
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._indices = list(range(len(self._rows)))
        self._cursor = 0
        if self._shuffle:
            self._rng.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self._rows)

    def get_step_batch(self, step_idx: int, batch_size: int) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self._rows:
            return []

        selected_rows = []
        while len(selected_rows) < batch_size:
            if self._cursor >= len(self._indices):
                if not self._repeat:
                    break
                self._cursor = 0
                if self._shuffle:
                    self._rng.shuffle(self._indices)

            row = self._rows[self._indices[self._cursor]]
            selected_rows.append(self._normalize_row(row))
            self._cursor += 1

        return selected_rows

    def iter_batches(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, len(self._rows), batch_size):
            rows = self._rows[start:start + batch_size]
            yield [self._normalize_row(row) for row in rows]

    def _load_rows(self, source_type: str, path: str | Path) -> list[dict[str, Any]]:
        data_path = Path(path)
        if source_type == "jsonl":
            return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if source_type == "parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("pandas is required to read parquet prompt sources") from exc
            return pd.read_parquet(data_path).to_dict(orient="records")
        raise ValueError(f"unsupported mate prompt source_type: {source_type}")

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": self._extract_value(row, self._prompt_keys),
            "expected": self._extract_value(row, self._expected_keys),
            "raw": row,
        }

    @staticmethod
    def _extract_value(row: dict[str, Any], keys: list[str]) -> Any:
        if not keys:
            return None
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        raise KeyError(keys[0])
