from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf

from orchrl.agent_trajectory_engine.datatypes import EpisodeResult, EpisodeTrajectory, TurnData
from orchrl.trainer.mate.prompt_loader import MatePromptLoader
from orchrl.trainer.validation_runner import ValidationRunner
from orchrl.utils.output_paths import prepare_training_output_dirs


class TrainingOutputLayoutTests(unittest.TestCase):
    def test_search_mas_training_config_defines_run_scoped_output_layout(self):
        repo_root = Path(__file__).resolve().parents[1]
        train_config_text = (repo_root / "experiments/search_mas/train.yaml").read_text(
            encoding="utf-8"
        )
        ppo_base_text = (repo_root / "orchrl/config/ppo_trainer/base.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("output_root_dir:", train_config_text)
        self.assertIn("output_root_dir: outputs/training_runs", train_config_text)
        self.assertIn("specialization: role_specific", train_config_text)
        self.assertNotIn("specialization: full", train_config_text)
        self.assertIn("data_root_dir:", train_config_text)
        self.assertIn("train_data_path:", train_config_text)
        self.assertIn("val_data_path:", train_config_text)
        self.assertIn(
            "train_data_path: ${training.data_root_dir}/train.parquet",
            train_config_text,
        )
        self.assertIn(
            "val_data_path: ${training.data_root_dir}/test.parquet",
            train_config_text,
        )
        self.assertIn("validate_batch_size:", train_config_text)
        self.assertNotIn("validate_sample_num:", train_config_text)
        self.assertIn("run_name:", train_config_text)
        self.assertIn("run_id:", train_config_text)
        self.assertIn("run_dir:", train_config_text)
        self.assertIn(
            "run_dir: ${training.output_root_dir}/${training.run_name}/${training.run_id}",
            train_config_text,
        )
        self.assertIn(
            "path: ${training.train_data_path}", train_config_text
        )
        self.assertNotIn("val_prompt_loader:", train_config_text)
        self.assertIn(
            "model_checkpoints_dir: ${training.run_dir}/checkpoints", train_config_text
        )
        self.assertIn(
            "output_dir: ${training.run_dir}/trajectories", train_config_text
        )
        self.assertIn(
            "default_local_dir: ${training.model_checkpoints_dir}", ppo_base_text
        )

    def test_search_mas_launcher_logs_under_outputs_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        launcher_text = (
            repo_root / "experiments/search_mas/run_train_e2e.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'DEFAULT_LOG_PATH="$REPO_ROOT/outputs/logs/search_mas_train_e2e_${TIMESTAMP}.log"',
            launcher_text,
        )
        self.assertIn('mkdir -p "$REPO_ROOT/outputs/logs"', launcher_text)

    def test_trainer_internal_logs_under_outputs_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_text = (
            repo_root / "orchrl/trainer/multi_agents_ppo_trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'log_dir = os.path.join("outputs", "logs", experiment_name, date_str, time_str)',
            trainer_text,
        )


    def test_trainer_references_refactor_collaborators(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_text = (
            repo_root / "orchrl/trainer/multi_agents_ppo_trainer.py"
        ).read_text(encoding="utf-8")

        references = [
            ("policy_trainer_registry", "registry module"),
            ("mate.runtime", "runtime coordinator"),
            ("training_step_executor", "training step executor"),
            ("validation_runner", "validation runner"),
        ]

        for ref, desc in references:
            self.assertIn(
                ref,
                trainer_text,
                f"Trainer should reference the soon-to-be extracted {desc}",
            )

    def test_trajectory_export_requires_explicit_output_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        export_text = (
            repo_root / "orchrl/trainer/mate/trajectory_export.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'output_dir = export_cfg["output_dir"]',
            export_text,
        )
        self.assertIn(
            "root_dir=output_dir",
            export_text,
        )
        self.assertNotIn(
            "export_root_dir = output_dir or checkpoints_dir",
            export_text,
        )
        self.assertNotIn(
            "checkpoints_dir: str | Path | None",
            export_text,
        )

    def test_multi_agents_trainer_builds_validation_parallel_single_trajectory_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        runtime_text = (
            repo_root / "orchrl/trainer/mate/runtime.py"
        ).read_text(encoding="utf-8")
        adapter_text = (
            repo_root / "orchrl/trainer/mate/rollout_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("self.mate_train_prompt_loader", runtime_text)
        self.assertIn("self.mate_val_prompt_loader", runtime_text)
        self.assertIn("self.config.training.val_data_path", runtime_text)
        self.assertIn("prompt_loader=self.mate_train_prompt_loader", runtime_text)
        self.assertIn("prompt_loader=self.mate_val_prompt_loader", runtime_text)
        self.assertIn('validation_mate_config["rollout_mode"] = "parallel"', runtime_text)
        self.assertIn('validation_mate_config["n_samples_per_prompt"] = 1', runtime_text)
        self.assertIn("parallel_rollout(", adapter_text)

    def test_mate_prompt_loader_iter_batches_covers_full_dataset(self):
        rows = [
            {"prompt": "q0", "expected": "a0"},
            {"prompt": "q1", "expected": "a1"},
            {"prompt": "q2", "expected": "a2"},
            {"prompt": "q3", "expected": "a3"},
            {"prompt": "q4", "expected": "a4"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "val.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
                encoding="utf-8",
            )

            loader = MatePromptLoader(
                source_type="jsonl",
                path=data_path,
                prompt_keys=["prompt"],
                expected_keys=["expected"],
            )

            batches = list(loader.iter_batches(batch_size=2))

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        prompts = [item["prompt"] for batch in batches for item in batch]
        expected = [item["expected"] for batch in batches for item in batch]
        self.assertEqual(prompts, ["q0", "q1", "q2", "q3", "q4"])
        self.assertEqual(expected, ["a0", "a1", "a2", "a3", "a4"])

    def test_validation_metrics_are_sample_average_reward_only(self):
        runner = ValidationRunner(
            config=OmegaConf.create(
                {"training": {"validate_batch_size": 2, "train_batch_size": 2, "if_save": True}, "specialization": "role_sharing"}
            ),
            policy_trainer_registry=None,
            mate_runtime=None,
            agent_policy_mapping={},
        )
        stats = runner.init_validation_stats()

        batch_0 = [
            self._build_validation_episode(
                episode_id="ep-0",
                rewards={"verifier": 0.5, "searcher": 0.0},
                final_reward=0.5,
                turn_counts={"verifier": 2},
            ),
        ]
        batch_1 = [
            self._build_validation_episode(
                episode_id="ep-1",
                rewards={"verifier": 0.0, "searcher": 1.0},
                final_reward=1.0,
                turn_counts={"searcher": 3},
            ),
            self._build_validation_episode(
                episode_id="ep-2",
                rewards={"verifier": 0.0, "searcher": 0.0},
                final_reward=0.0,
                turn_counts={},
            ),
        ]

        runner.accumulate_validation_episode_batch(stats, batch_0)
        runner.accumulate_validation_episode_batch(stats, batch_1)
        metrics = runner.build_validation_metrics(stats)

        self.assertEqual(set(metrics.keys()), {"validation/sample_avg_reward"})
        self.assertAlmostEqual(metrics["validation/sample_avg_reward"], 0.5)

    def test_save_best_checkpoint_compares_sample_average_reward(self):
        trainer = unittest.mock.Mock()
        registry = unittest.mock.Mock()
        registry.ppo_trainer_dict = {"policy_a": trainer}
        runner = ValidationRunner(
            config=OmegaConf.create({"training": {"if_save": True}, "specialization": "role_sharing"}),
            policy_trainer_registry=registry,
            mate_runtime=None,
            agent_policy_mapping={},
        )
        runner.best_validation_reward = 0.4

        runner.save_best_checkpoint(0.3)
        self.assertEqual(runner.best_validation_reward, 0.4)

        runner.save_best_checkpoint(0.6)
        self.assertEqual(runner.best_validation_reward, 0.6)
        trainer._save_checkpoint.assert_called_once_with()

    def _build_validation_episode(self, *, episode_id, rewards, final_reward, turn_counts):
        return EpisodeResult(
            trajectory=EpisodeTrajectory(
                episode_id=episode_id,
                agent_trajectories={
                    role: [
                        TurnData(
                            agent_role=role,
                            turn_index=turn_idx,
                            messages=[],
                            response_text=f"{role}-{turn_idx}",
                            token_ids=None,
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=float(turn_idx),
                        )
                        for turn_idx in range(count)
                    ]
                    for role, count in turn_counts.items()
                },
            ),
            rewards=rewards,
            final_reward=final_reward,
        )

    def test_prepare_training_output_dirs_creates_run_scoped_directories(self):
        config = OmegaConf.create(
            {
                "training": {
                    "experiment_name": "search_mas_app",
                    "output_root_dir": "outputs/training_runs",
                    "data_root_dir": "/tmp/search_data",
                    "train_data_path": "/tmp/search_data/train.parquet",
                    "val_data_path": "/tmp/search_data/test.parquet",
                    "validate_batch_size": 2,
                    "run_name": "search_mas_app",
                    "run_id": "20260409_000000",
                    "run_dir": "outputs/training_runs/search_mas_app/20260409_000000",
                    "model_checkpoints_dir": "outputs/training_runs/search_mas_app/20260409_000000/checkpoints",
                    "mate": {
                        "prompt_loader": {
                            "path": "/tmp/search_data/train.parquet",
                            "prompt_keys": ["prompt"],
                            "expected_keys": ["expected"],
                        },
                        "trajectory_export": {
                            "enable": True,
                            "output_dir": "outputs/training_runs/search_mas_app/20260409_000000/trajectories",
                        }
                    },
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = prepare_training_output_dirs(config, base_dir=tmp_dir)

            self.assertTrue((Path(tmp_dir) / "outputs/training_runs/search_mas_app/20260409_000000").is_dir())
            self.assertTrue((Path(tmp_dir) / "outputs/training_runs/search_mas_app/20260409_000000/checkpoints").is_dir())
            self.assertTrue((Path(tmp_dir) / "outputs/training_runs/search_mas_app/20260409_000000/trajectories").is_dir())
            self.assertEqual(
                paths["trajectory_output_dir"],
                Path(tmp_dir) / "outputs/training_runs/search_mas_app/20260409_000000/trajectories",
            )


if __name__ == "__main__":
    unittest.main()
