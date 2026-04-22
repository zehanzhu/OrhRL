from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from orchrl.utils.imports import import_callable


class PromptTrajectoryExporter:
    def __init__(
        self,
        *,
        root_dir: str | Path,
        write_json: bool = True,
        write_txt: bool = True,
        include_token_details: bool = True,
        answer_stats_builder: Callable[..., Any] | None = None,
    ):
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._write_json = bool(write_json)
        self._write_txt = bool(write_txt)
        self._include_token_details = bool(include_token_details)
        self._answer_stats_builder = answer_stats_builder

    def export_step(self, *, step_idx: int, episodes: list[Any], rollout_mode: str) -> None:
        if not episodes:
            return
        step_dir = self._root_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for item in episodes:
            payload = self._normalize_jsonable(
                self._serialize_prompt_result(item=item, step_idx=step_idx, rollout_mode=rollout_mode)
            )
            basename = self._prompt_basename(payload)
            json_name = f"{basename}.json"
            txt_name = f"{basename}.txt"
            json_path = step_dir / json_name
            txt_path = step_dir / txt_name
            if self._write_json:
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if self._write_txt:
                txt_path.write_text(self._render_text(payload), encoding="utf-8")

            answer_stats = dict(payload.get("answer_stats") or {})
            summary_rows.append(
                {
                    "step_idx": step_idx,
                    "prompt_group_id": payload["prompt_group_id"],
                    "sample_idx": payload["sample_idx"],
                    "prompt": payload["prompt"],
                    "rollout_mode": payload["rollout_mode"],
                    "status": payload["status"],
                    "pilot_reward": payload.get("pilot", {}).get("final_reward"),
                    "branches_collected": len(payload.get("branches", [])),
                    "predicted_answer": answer_stats.get("predicted_answer", ""),
                    "expected_answers": list(answer_stats.get("expected_answers") or []),
                    "is_correct": answer_stats.get("is_correct", False),
                    "json_path": json_name if self._write_json else None,
                    "txt_path": txt_name if self._write_txt else None,
                }
            )

        summary_path = step_dir / "summary.jsonl"
        summary_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary_rows),
            encoding="utf-8",
        )

    def _serialize_prompt_result(self, *, item: Any, step_idx: int, rollout_mode: str) -> dict[str, Any]:
        if self._is_tree_result(item):
            pilot = item.pilot_result
            pilot_payload = self._serialize_episode(pilot)
            prompt_group_id, sample_idx = self._prompt_keys(pilot)
            return {
                "step_idx": step_idx,
                "prompt_group_id": prompt_group_id,
                "sample_idx": sample_idx,
                "prompt": getattr(item, "prompt", ""),
                "rollout_mode": rollout_mode,
                "status": getattr(pilot, "status", "success"),
                "answer_stats": dict(pilot_payload.get("answer_stats") or {}),
                "tree_metadata": dict(getattr(item, "tree_metadata", {}) or {}),
                "pilot": pilot_payload,
                "branches": [self._serialize_branch(branch) for branch in getattr(item, "branch_results", [])],
            }

        prompt_group_id, sample_idx = self._prompt_keys(item)
        pilot_payload = self._serialize_episode(item)
        return {
            "step_idx": step_idx,
            "prompt_group_id": prompt_group_id,
            "sample_idx": sample_idx,
            "prompt": self._infer_prompt_from_episode(item),
            "rollout_mode": rollout_mode,
            "status": getattr(item, "status", "success"),
            "answer_stats": dict(pilot_payload.get("answer_stats") or {}),
            "pilot": pilot_payload,
            "branches": [],
        }

    @staticmethod
    def _is_tree_result(item: Any) -> bool:
        return hasattr(item, "pilot_result") and hasattr(item, "branch_results") and hasattr(item, "prompt")

    def _prompt_keys(self, result: Any) -> tuple[str, int]:
        metadata = dict(getattr(result, "metadata", {}) or {})
        episode_id = getattr(getattr(result, "trajectory", None), "episode_id", "prompt")
        prompt_group_id = str(metadata.get("prompt_group_id") or episode_id)
        sample_idx = int(metadata.get("sample_idx", 0))
        return prompt_group_id, sample_idx

    def _serialize_branch(self, branch: Any) -> dict[str, Any]:
        episode_result = getattr(branch, "episode_result")
        return {
            "branch_turn": getattr(branch, "branch_turn", None),
            "branch_agent_role": getattr(branch, "branch_agent_role", None),
            "parent_episode_id": getattr(branch, "parent_episode_id", None),
            **self._serialize_episode(episode_result),
        }

    def _serialize_episode(self, result: Any) -> dict[str, Any]:
        trajectory = getattr(result, "trajectory")
        return {
            "episode_id": getattr(trajectory, "episode_id", None),
            "status": getattr(result, "status", "success"),
            "final_reward": getattr(result, "final_reward", None),
            "answer_stats": self._build_episode_answer_stats(result),
            "rewards": dict(getattr(result, "rewards", {}) or {}),
            "metadata": dict(getattr(result, "metadata", {}) or {}),
            "failure_info": getattr(result, "failure_info", None),
            "turns": [self._serialize_turn(turn) for turn in self._sorted_turns(trajectory)],
        }

    def _build_episode_answer_stats(self, result: Any) -> dict[str, Any]:
        if self._answer_stats_builder is None:
            return {}
        trajectory = getattr(result, "trajectory")
        answer_turns = getattr(trajectory, "agent_trajectories", {}).get("answerer", [])
        response_text = getattr(answer_turns[-1], "response_text", "") if answer_turns else ""
        metadata = self._episode_metadata(result, trajectory)
        answer_stats = self._answer_stats_builder(response_text=response_text, metadata=metadata)
        if not isinstance(answer_stats, dict):
            raise TypeError("trajectory export answer_stats_provider must return a dict")
        return dict(answer_stats)

    @staticmethod
    def _episode_metadata(result: Any, trajectory: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        trajectory_metadata = getattr(trajectory, "metadata", {}) or {}
        if isinstance(trajectory_metadata, dict):
            merged.update(trajectory_metadata)
        result_metadata = getattr(result, "metadata", {}) or {}
        if isinstance(result_metadata, dict):
            merged.update(result_metadata)
        return merged

    def _serialize_turn(self, turn: Any) -> dict[str, Any]:
        payload = {
            "agent_role": getattr(turn, "agent_role", None),
            "turn_index": getattr(turn, "turn_index", None),
            "timestamp": getattr(turn, "timestamp", None),
            "messages": list(getattr(turn, "messages", []) or []),
            "response_text": getattr(turn, "response_text", ""),
            "finish_reason": getattr(turn, "finish_reason", None),
            "replayed": bool(getattr(turn, "replayed", False)),
            "branch_phase": getattr(turn, "branch_phase", None),
            "metadata": dict(getattr(turn, "metadata", {}) or {}),
        }
        if self._include_token_details:
            payload["prompt_ids"] = self._normalize_sequence(getattr(turn, "prompt_ids", None))
            payload["token_ids"] = self._normalize_sequence(getattr(turn, "token_ids", None))
            payload["logprobs"] = self._normalize_float_sequence(getattr(turn, "logprobs", None))
        return payload

    @staticmethod
    def _normalize_sequence(values: Any) -> list[int] | None:
        if values is None:
            return None
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [int(value) for value in values]

    @staticmethod
    def _normalize_float_sequence(values: Any) -> list[float] | None:
        if values is None:
            return None
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [float(value) for value in values]

    @classmethod
    def _normalize_jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._normalize_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize_jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            return cls._normalize_jsonable(value.tolist())
        if hasattr(value, "item"):
            try:
                return cls._normalize_jsonable(value.item())
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _sorted_turns(trajectory: Any) -> list[Any]:
        turns = [turn for turn_list in getattr(trajectory, "agent_trajectories", {}).values() for turn in turn_list]
        return sorted(turns, key=lambda turn: (getattr(turn, "timestamp", 0.0), getattr(turn, "turn_index", 0), getattr(turn, "agent_role", "")))

    def _infer_prompt_from_episode(self, result: Any) -> str:
        for turn in self._sorted_turns(getattr(result, "trajectory")):
            for message in getattr(turn, "messages", []) or []:
                if message.get("role") == "user":
                    return str(message.get("content", ""))
        return ""

    def _prompt_basename(self, payload: dict[str, Any]) -> str:
        group = self._sanitize_filename(str(payload["prompt_group_id"]))
        sample_idx = int(payload["sample_idx"])
        return f"prompt_{group}_{sample_idx}"

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
        return sanitized or "prompt"

    def _render_text(self, payload: dict[str, Any]) -> str:
        answer_stats = dict(payload.get("answer_stats") or {})
        lines = [
            f"Step: {payload['step_idx']}",
            f"Prompt Group: {payload['prompt_group_id']}",
            f"Sample Index: {payload['sample_idx']}",
            f"Rollout Mode: {payload['rollout_mode']}",
            f"Status: {payload['status']}",
            f"Predicted Answer: {answer_stats.get('predicted_answer', '')}",
            f"Expected Answers: {self._format_expected_answers(answer_stats.get('expected_answers'))}",
            f"Is Correct: {answer_stats.get('is_correct', False)}",
            "Prompt:",
            str(payload.get("prompt", "")),
            "",
            "=== Pilot ===",
            self._render_episode_text(payload["pilot"]),
        ]
        branches = payload.get("branches", [])
        lines.extend(["", f"=== Branches ({len(branches)}) ==="])
        for index, branch in enumerate(branches):
            lines.extend(
                [
                    f"-- Branch {index} --",
                    f"branch_turn: {branch.get('branch_turn')}",
                    f"branch_agent_role: {branch.get('branch_agent_role')}",
                    f"parent_episode_id: {branch.get('parent_episode_id')}",
                    self._render_episode_text(branch),
                ]
            )
        return "\n".join(lines) + "\n"

    def _render_episode_text(self, payload: dict[str, Any]) -> str:
        answer_stats = dict(payload.get("answer_stats") or {})
        lines = [
            f"episode_id: {payload.get('episode_id')}",
            f"status: {payload.get('status')}",
            f"final_reward: {payload.get('final_reward')}",
            f"Predicted Answer: {answer_stats.get('predicted_answer', '')}",
            f"Expected Answers: {self._format_expected_answers(answer_stats.get('expected_answers'))}",
            f"Is Correct: {answer_stats.get('is_correct', False)}",
        ]
        failure_info = payload.get("failure_info")
        if failure_info:
            lines.append(f"failure_info: {json.dumps(failure_info, ensure_ascii=False)}")
        lines.append("turns:")
        for index, turn in enumerate(payload.get("turns", []), start=1):
            lines.append(f"  [{index}] agent={turn.get('agent_role')} turn_index={turn.get('turn_index')} timestamp={turn.get('timestamp')}")
            lines.append(f"      finish_reason: {turn.get('finish_reason')}")
            lines.append(f"      replayed: {turn.get('replayed')}")
            lines.append(f"      branch_phase: {turn.get('branch_phase')}")
            lines.append("      messages:")
            for message in turn.get("messages", []):
                lines.append(f"        - {message.get('role')}: {message.get('content')}")
            lines.append("      response_text:")
            response = str(turn.get("response_text", ""))
            if response:
                for line in response.splitlines() or [response]:
                    lines.append(f"        {line}")
            else:
                lines.append("        ")
            if self._include_token_details:
                lines.append(f"      prompt_ids: {self._format_sequence(turn.get('prompt_ids'))}")
                lines.append(f"      token_ids: {self._format_sequence(turn.get('token_ids'))}")
                lines.append(f"      logprobs: {self._format_float_sequence(turn.get('logprobs'))}")
        return "\n".join(lines)

    @staticmethod
    def _format_sequence(values: list[int] | None) -> str:
        if values is None:
            return "None"
        return f"len={len(values)} {values}"

    @staticmethod
    def _format_float_sequence(values: list[float] | None) -> str:
        if values is None:
            return "None"
        return f"len={len(values)} {[round(value, 4) for value in values]}"

    @staticmethod
    def _format_expected_answers(values: Any) -> str:
        return str(list(values or []))


def maybe_export_prompt_trajectories(*, episodes: list[Any], step_idx: int, rollout_mode: str, mate_config: dict[str, Any]) -> None:
    export_cfg = dict(mate_config.get("trajectory_export") or {})
    if not export_cfg.get("enable", False):
        return
    try:
        output_dir = export_cfg["output_dir"]
    except KeyError as exc:
        raise ValueError("trajectory export requires trajectory_export.output_dir") from exc
    if not output_dir:
        raise ValueError("trajectory export requires trajectory_export.output_dir")
    answer_stats_provider = export_cfg.get("answer_stats_provider")
    answer_stats_kwargs = dict(export_cfg.get("answer_stats_kwargs") or {})
    answer_stats_builder = None
    if answer_stats_provider:
        if not isinstance(answer_stats_provider, str):
            raise TypeError("trajectory_export.answer_stats_provider must be a non-empty import path")
        answer_stats_func = import_callable(answer_stats_provider)
        answer_stats_builder = (
            answer_stats_func
            if not answer_stats_kwargs
            else partial(answer_stats_func, **answer_stats_kwargs)
        )
    exporter = PromptTrajectoryExporter(
        root_dir=output_dir,
        write_json=export_cfg.get("write_json", True),
        write_txt=export_cfg.get("write_txt", True),
        include_token_details=export_cfg.get("include_token_details", True),
        answer_stats_builder=answer_stats_builder,
    )
    exporter.export_step(step_idx=step_idx, episodes=episodes, rollout_mode=rollout_mode)
