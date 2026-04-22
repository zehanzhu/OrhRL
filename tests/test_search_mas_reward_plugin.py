from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orchrl.trainer.mate.reward_bridge import build_reward_provider
from orchrl.trainer.mate.trajectory_export import maybe_export_prompt_trajectories


def _make_episode_result(*, predicted_answer: str = "Paris", expected_answer: str = "Paris"):
    turn = SimpleNamespace(
        agent_role="answerer",
        turn_index=0,
        timestamp=1.0,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_text=f"<answer>{predicted_answer}</answer>",
        finish_reason="stop",
        replayed=False,
        branch_phase=None,
        metadata={},
    )
    trajectory = SimpleNamespace(
        episode_id="episode-1",
        metadata={
            "prompt_group_id": "prompt-0",
            "sample_idx": 0,
            "prompt_row": {"expected_answer": expected_answer},
        },
        agent_trajectories={
            "verifier": [],
            "searcher": [],
            "answerer": [turn],
        },
    )
    return SimpleNamespace(
        trajectory=trajectory,
        status="success",
        final_reward=1.0,
        rewards={"verifier": 1.0, "searcher": 1.0, "answerer": 1.0},
        metadata={},
        failure_info=None,
    )


class SearchMASRewardPluginTests(unittest.TestCase):
    def test_search_mas_reward_provider_uses_plugin_namespace(self):
        provider = build_reward_provider(
            {"provider": "task_plugins.search_mas.reward:compute_reward"}
        )
        reward_payload = provider.compute(_make_episode_result().trajectory)

        self.assertEqual(reward_payload["final_reward"], 1.0)
        self.assertEqual(reward_payload["agent_rewards"]["answerer"], 1.0)

    def test_reward_plugin_defaults_to_exact_match(self):
        from task_plugins.search_mas.reward import compute_reward

        reward_payload = compute_reward(
            _make_episode_result(
                predicted_answer="York", expected_answer="New York"
            ).trajectory,
            match_mode="exact",
        )
        self.assertEqual(reward_payload["final_reward"], 0.0)

    def test_reward_plugin_supports_explicit_substring_mode(self):
        from task_plugins.search_mas.reward import compute_reward

        reward_payload = compute_reward(
            _make_episode_result(
                predicted_answer="New York", expected_answer="York"
            ).trajectory,
            match_mode="substring",
        )
        self.assertEqual(reward_payload["final_reward"], 1.0)

    def test_build_reward_provider_passes_match_mode_into_compute_reward(self):
        provider = build_reward_provider(
            {
                "provider": "task_plugins.search_mas.reward:compute_reward",
                "match_mode": "substring",
            }
        )

        reward_payload = provider.compute(
            _make_episode_result(
                predicted_answer="New York",
                expected_answer="York",
            ).trajectory
        )

        self.assertEqual(reward_payload["final_reward"], 1.0)

    def test_reward_answer_stats_and_evaluator_share_same_exact_result(self):
        from mas_apps.search.search_mas.apps.search.evaluator import (
            is_search_answer_correct,
        )
        from task_plugins.search_mas.reward import build_answer_stats

        metadata = {"prompt_row": {"expected_answer": "Paris, Texas"}}
        stats = build_answer_stats(
            response_text="<answer>Paris</answer>",
            metadata=metadata,
            match_mode="exact",
        )

        self.assertFalse(stats["is_correct"])
        self.assertFalse(
            is_search_answer_correct(
                "Paris",
                ["Paris, Texas"],
                match_mode="exact",
            )
        )

    def test_build_answer_stats_supports_explicit_substring_mode(self):
        from task_plugins.search_mas.reward import build_answer_stats

        stats = build_answer_stats(
            response_text="<answer>New York</answer>",
            metadata={"prompt_row": {"expected_answer": "York"}},
            match_mode="substring",
        )

        self.assertTrue(stats["is_correct"])

    def test_is_search_answer_correct_supports_explicit_substring_mode(self):
        from mas_apps.search.search_mas.apps.search.evaluator import (
            is_search_answer_correct,
        )

        self.assertTrue(
            is_search_answer_correct(
                "New York",
                ["York"],
                match_mode="substring",
            )
        )

    def test_legacy_use_substring_em_maps_to_substring_mode(self):
        from mas_apps.search.search_mas.apps.search.evaluator import (
            is_search_answer_correct,
        )

        self.assertTrue(
            is_search_answer_correct(
                "New York",
                ["York"],
                use_substring=True,
            )
        )

    def test_search_experiment_config_points_to_plugin_reward(self):
        repo_root = Path(__file__).resolve().parents[1]
        train_config_text = (repo_root / "experiments/search_mas/train.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "task_plugins.search_mas.reward:compute_reward", train_config_text
        )
        self.assertIn(
            "task_plugins.search_mas.reward:build_answer_stats", train_config_text
        )
        self.assertIn("match_mode: exact", train_config_text)
        self.assertNotIn("orchrl.plugins.search_mas", train_config_text)

    def test_evaluator_normalize_answer_preserves_legacy_semantics(self):
        from mas_apps.search.search_mas.apps.search.evaluator import normalize_answer

        self.assertEqual(
            normalize_answer(" The, Quick! Brown fox "),
            "quick brown fox",
        )

    def test_trajectory_export_requires_output_dir_when_enabled(self):
        episode = _make_episode_result()
        mate_config = {
            "trajectory_export": {
                "enable": True,
                "output_dir": None,
                "write_json": True,
                "write_txt": False,
                "include_token_details": False,
            }
        }

        with self.assertRaisesRegex(
            ValueError, "trajectory_export.output_dir"
        ):
            maybe_export_prompt_trajectories(
                episodes=[episode],
                step_idx=0,
                rollout_mode="parallel",
                mate_config=mate_config,
            )

    def test_trajectory_export_without_provider_writes_generic_summary(self):
        episode = _make_episode_result()

        with tempfile.TemporaryDirectory() as tmp_dir:
            mate_config = {
                "trajectory_export": {
                    "enable": True,
                    "output_dir": str(Path(tmp_dir) / "trajectories"),
                    "write_json": True,
                    "write_txt": False,
                    "include_token_details": False,
                }
            }

            maybe_export_prompt_trajectories(
                episodes=[episode],
                step_idx=0,
                rollout_mode="parallel",
                mate_config=mate_config,
            )

            summary_path = Path(tmp_dir) / "trajectories" / "step_000000" / "summary.jsonl"
            rows = [
                json.loads(line)
                for line in summary_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predicted_answer"], "")
        self.assertEqual(rows[0]["expected_answers"], [])
        self.assertFalse(rows[0]["is_correct"])

    def test_trajectory_export_uses_same_match_mode_when_provider_enabled(self):
        episode = _make_episode_result(
            predicted_answer="New York",
            expected_answer="York",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mate_config = {
                "trajectory_export": {
                    "enable": True,
                    "output_dir": str(Path(tmp_dir) / "trajectories"),
                    "write_json": True,
                    "write_txt": False,
                    "include_token_details": False,
                    "answer_stats_provider": "task_plugins.search_mas.reward:build_answer_stats",
                    "answer_stats_kwargs": {
                        "match_mode": "substring",
                    },
                }
            }

            maybe_export_prompt_trajectories(
                episodes=[episode],
                step_idx=0,
                rollout_mode="parallel",
                mate_config=mate_config,
            )

            summary_path = Path(tmp_dir) / "trajectories" / "step_000000" / "summary.jsonl"
            rows = [
                json.loads(line)
                for line in summary_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_correct"])


if __name__ == "__main__":
    unittest.main()
