from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from orchrl.agent_trajectory_engine.datatypes import EpisodeResult, EpisodeTrajectory, TurnData
from orchrl.trainer.mate.prompt_loader import MatePromptLoader
from orchrl.trainer.validation_runner import ValidationRunner
from orchrl.utils.output_paths import prepare_training_output_dirs


class TrainingOutputLayoutTests(unittest.TestCase):
    def test_search_mas_megatron_smoke_config_exists(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml"
        self.assertTrue(config_path.is_file(), f"Missing config: {config_path}")

    def test_search_mas_megatron_smoke_config_sets_grpo_megatron_and_disables_critic(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_text = (
            repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("specialization: role_sharing", config_text)
        self.assertIn("strategy: megatron", config_text)
        self.assertIn("critic:", config_text)
        self.assertIn("enable: false", config_text)
        self.assertIn("expert_model_parallel_size:", config_text)
        self.assertIn("tensor_model_parallel_size:", config_text)
        self.assertIn("pipeline_model_parallel_size:", config_text)
        self.assertIn("param_offload: true", config_text)
        self.assertIn("grad_offload: true", config_text)
        self.assertIn("optimizer_offload: true", config_text)

    def test_search_mas_megatron_smoke_uses_full_8_gpu_model_parallel_topology(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

        gpus_per_policy = int(config["resource"]["gpus_per_policy"])
        actor_megatron = (
            config["models"]["model_0"]["ppo_trainer_config"]["actor_rollout_ref"]["actor"]["megatron"]
        )

        tensor_parallel = int(actor_megatron["tensor_model_parallel_size"])
        pipeline_parallel = int(actor_megatron["pipeline_model_parallel_size"])
        context_parallel = int(actor_megatron["context_parallel_size"])
        expert_parallel = int(actor_megatron["expert_model_parallel_size"])
        expert_tensor_parallel = int(
            actor_megatron.get("expert_tensor_parallel_size") or tensor_parallel
        )

        self.assertEqual(
            tensor_parallel * pipeline_parallel * context_parallel,
            gpus_per_policy,
            "8-card Megatron smoke should consume all policy GPUs for attention-model parallelism "
            "to avoid large DP grad buffers on 40GB cards",
        )
        self.assertEqual(
            expert_tensor_parallel * expert_parallel * pipeline_parallel,
            gpus_per_policy,
            "8-card Megatron smoke should consume all policy GPUs for expert parallelism "
            "to avoid MoE DP replication on 40GB cards",
        )

    def test_search_mas_megatron_smoke_explicitly_loads_actor_and_ref_weights(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

        actor_rollout_ref = config["models"]["model_0"]["ppo_trainer_config"]["actor_rollout_ref"]
        self.assertIs(
            actor_rollout_ref["actor"].get("load_weight"),
            True,
            "Megatron smoke actor config must explicitly define load_weight to satisfy OmegaConf struct access",
        )
        self.assertIs(
            actor_rollout_ref["ref"].get("load_weight"),
            True,
            "Megatron smoke ref config must explicitly define load_weight to satisfy OmegaConf struct access",
        )

    def test_search_mas_megatron_smoke_registers_all_mas_roles_on_shared_policy(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

        mate_cfg = config["training"]["mate"]
        self.assertEqual(
            mate_cfg["roles"],
            ["verifier", "searcher", "answerer"],
            "Role-sharing smoke must still register every MAS role with the monitor pipeline",
        )
        self.assertEqual(
            mate_cfg["role_policy_mapping"],
            {
                "verifier": "verifier_model",
                "searcher": "verifier_model",
                "answerer": "verifier_model",
            },
            "Role-sharing smoke should route all MAS roles to the single shared policy",
        )

    def test_search_mas_megatron_smoke_defines_mcore_actor_optimizer_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

        actor_optim = (
            config["models"]["model_0"]["ppo_trainer_config"]["actor_rollout_ref"]["actor"]["optim"]
        )

        self.assertEqual(
            actor_optim["_target_"],
            "verl.workers.config.McoreOptimizerConfig",
            "Megatron smoke actor optimizer must use McoreOptimizerConfig instead of inheriting FSDP-only fields",
        )
        self.assertIn(
            "min_lr",
            actor_optim,
            "Megatron smoke actor optimizer must define min_lr for verl Megatron optimizer init",
        )
        self.assertIn(
            "lr_decay_style",
            actor_optim,
            "Megatron smoke actor optimizer must define lr_decay_style for Megatron scheduler construction",
        )
        self.assertIn(
            "lr_warmup_init",
            actor_optim,
            "Megatron smoke actor optimizer must define lr_warmup_init for Megatron scheduler construction",
        )
        self.assertNotIn(
            "min_lr_ratio",
            actor_optim,
            "Megatron smoke actor optimizer should not rely on FSDP-only min_lr_ratio field",
        )
        override_optim = actor_optim["override_optimizer_config"]
        self.assertIs(
            override_optim.get("use_precision_aware_optimizer"),
            True,
            "Qwen3-30B MoE Megatron smoke should enable precision-aware optimizer to avoid GPU fp32 main-param clones",
        )
        self.assertIs(
            override_optim.get("optimizer_cpu_offload"),
            True,
            "Qwen3-30B MoE Megatron smoke should initialize optimizer via CPU offload to reduce GPU optimizer peak memory",
        )
        self.assertAlmostEqual(
            float(override_optim.get("optimizer_offload_fraction")),
            1.0,
            places=6,
            msg="Qwen3-30B MoE Megatron smoke should fully offload optimizer params during initialization",
        )

    def test_search_mas_megatron_smoke_composed_config_drops_fsdp_only_actor_fields(self):
        from orchrl.trainer.train import _normalize_megatron_train_configs

        repo_root = Path(__file__).resolve().parents[1]
        config_dir = str(repo_root / "experiments" / "search_mas")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="train_megatron_moe_smoke")

        _normalize_megatron_train_configs(config)

        actor = config.models.model_0.ppo_trainer_config.actor_rollout_ref.actor
        actor_optim = actor.optim

        self.assertNotIn(
            "fsdp_config",
            actor,
            "Composed Megatron actor config should not retain FSDP engine settings from base.yaml",
        )
        self.assertNotIn(
            "grad_clip",
            actor,
            "Composed Megatron actor config should rely on actor.optim.clip_grad instead of actor.grad_clip",
        )
        self.assertNotIn(
            "ulysses_sequence_parallel_size",
            actor,
            "Composed Megatron actor config should not retain deprecated FSDP ulysses sequence parallel settings",
        )
        self.assertNotIn(
            "optimizer_impl",
            actor_optim,
            "Composed Megatron optimizer config should not retain FSDP-only optimizer_impl",
        )
        self.assertNotIn(
            "min_lr_ratio",
            actor_optim,
            "Composed Megatron optimizer config should not retain FSDP-only min_lr_ratio",
        )
        self.assertNotIn(
            "num_cycles",
            actor_optim,
            "Composed Megatron optimizer config should not retain FSDP-only cosine scheduler fields",
        )
        self.assertNotIn(
            "warmup_style",
            actor_optim,
            "Composed Megatron optimizer config should not retain deprecated FSDP warmup_style",
        )

    def test_search_mas_megatron_smoke_uses_conservative_vllm_rollout_topology(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (repo_root / "experiments/search_mas/train_megatron_moe_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

        rollout = config["models"]["model_0"]["ppo_trainer_config"]["actor_rollout_ref"]["rollout"]

        self.assertEqual(
            int(rollout["tensor_model_parallel_size"]),
            8,
            "Qwen3-30B MoE smoke should use full-node vLLM TP to minimize per-rank weight residency",
        )
        self.assertEqual(
            int(rollout["expert_parallel_size"]),
            1,
            "Qwen3-30B MoE smoke should avoid vLLM expert parallel in the conservative rollout preset",
        )
        self.assertEqual(
            int(rollout["max_model_len"]),
            2560,
            "Qwen3-30B MoE smoke should cap rollout max_model_len near prompt+response budget",
        )
        self.assertEqual(
            int(rollout["max_num_batched_tokens"]),
            2560,
            "Qwen3-30B MoE smoke should cap batched tokens to the same conservative rollout budget",
        )
        self.assertEqual(
            int(rollout["max_num_seqs"]),
            2,
            "Qwen3-30B MoE smoke should keep concurrent vLLM sequences low in hybrid mode",
        )
        self.assertAlmostEqual(
            float(rollout["gpu_memory_utilization"]),
            0.45,
            places=6,
            msg="Qwen3-30B MoE smoke should reserve headroom for colocated hybrid-engine training workers",
        )

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
        self.assertIn("- wandb", train_config_text)
        self.assertIn("if_save: false", train_config_text)
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
        self.assertIn('DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"', launcher_text)
        self.assertIn('DEFAULT_MCORE_PYDEPS_HOME="/mnt/bn/chenghao1026/resouces/libs/verl-mcore-pydeps-0131"', launcher_text)
        self.assertIn('DEFAULT_TRANSFORMER_ENGINE_HOME="/mnt/bn/chenghao1026/resouces/libs/transformer-engine-cu128-torch290"', launcher_text)
        self.assertIn('ORCHRL_TRANSFORMER_ENGINE_HOME="${ORCHRL_TRANSFORMER_ENGINE_HOME:-$DEFAULT_TRANSFORMER_ENGINE_HOME}"', launcher_text)
        self.assertIn('export PYTHONPATH="$ORCHRL_TRANSFORMER_ENGINE_HOME${PYTHONPATH:+:$PYTHONPATH}"', launcher_text)
        self.assertIn('export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"', launcher_text)
        self.assertIn('export PYTHONPATH="$ORCHRL_MCORE_PYDEPS_HOME${PYTHONPATH:+:$PYTHONPATH}"', launcher_text)
        self.assertIn('export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"', launcher_text)
        self.assertIn('export WANDB_MODE="${WANDB_MODE:-online}"', launcher_text)
        self.assertNotIn("export WANDB_MODE=offline", launcher_text)

    def test_trainer_internal_logs_under_outputs_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_text = (
            repo_root / "orchrl/trainer/multi_agents_ppo_trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'log_dir = run_dir_path / "logs" / date_str / time_str',
            trainer_text,
        )
        self.assertIn(
            'Path.cwd() / "outputs" / "logs" / experiment_name / date_str / time_str',
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

    def test_training_loader_repeats_across_epochs_instead_of_exhausting(self):
        rows = [
            {"prompt": "q0", "expected": "a0"},
            {"prompt": "q1", "expected": "a1"},
            {"prompt": "q2", "expected": "a2"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "train.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
                encoding="utf-8",
            )
            loader = MatePromptLoader(
                source_type="jsonl",
                path=data_path,
                prompt_keys=["prompt"],
                expected_keys=["expected"],
                repeat=True,
                shuffle=False,
                seed=123,
            )

            batch_0 = loader.get_step_batch(step_idx=0, batch_size=2)
            batch_1 = loader.get_step_batch(step_idx=1, batch_size=2)
            batch_2 = loader.get_step_batch(step_idx=2, batch_size=2)

        self.assertEqual([item["prompt"] for item in batch_0], ["q0", "q1"])
        self.assertEqual([item["prompt"] for item in batch_1], ["q2", "q0"])
        self.assertEqual([item["prompt"] for item in batch_2], ["q1", "q2"])

    def test_training_loader_shuffle_is_seeded_and_reproducible(self):
        rows = [{"prompt": f"q{i}", "expected": f"a{i}"} for i in range(4)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "train.jsonl"
            data_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=True) for row in rows),
                encoding="utf-8",
            )
            loader_a = MatePromptLoader(
                source_type="jsonl",
                path=data_path,
                prompt_keys=["prompt"],
                expected_keys=["expected"],
                repeat=True,
                shuffle=True,
                seed=7,
            )
            loader_b = MatePromptLoader(
                source_type="jsonl",
                path=data_path,
                prompt_keys=["prompt"],
                expected_keys=["expected"],
                repeat=True,
                shuffle=True,
                seed=7,
            )

        self.assertEqual(
            [item["prompt"] for item in loader_a.get_step_batch(step_idx=0, batch_size=4)],
            [item["prompt"] for item in loader_b.get_step_batch(step_idx=0, batch_size=4)],
        )

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

        self.assertNotIn("episodes", stats)
        self.assertTrue(
            {
                "validation/sample_avg_reward",
                "validation/accuracy",
                "validation/failed_sample_count",
                "validation/failed_sample_rate",
                "mas/validation/sample_avg_reward",
                "mas/validation/accuracy",
                "mas/validation/success_rate",
                "mas/validation/failed_rate",
            }.issubset(metrics.keys())
        )
        self.assertAlmostEqual(metrics["validation/sample_avg_reward"], 0.5)
        self.assertAlmostEqual(metrics["validation/accuracy"], 1 / 3)
        self.assertEqual(metrics["validation/failed_sample_count"], 0)
        self.assertAlmostEqual(metrics["validation/failed_sample_rate"], 0.0)

    def test_validation_metrics_use_expected_sample_count_denominator(self):
        runner = ValidationRunner(
            config=OmegaConf.create(
                {"training": {"validate_batch_size": 2, "train_batch_size": 2, "if_save": True}, "specialization": "role_sharing"}
            ),
            policy_trainer_registry=None,
            mate_runtime=None,
            agent_policy_mapping={},
        )
        stats = runner.init_validation_stats()

        runner.accumulate_validation_episode_batch(
            stats,
            episodes=[
                self._build_validation_episode(
                    episode_id="ep-0",
                    rewards={},
                    final_reward=1.0,
                    turn_counts={},
                )
            ],
            expected_sample_count=2,
            failed_count=1,
        )
        metrics = runner.build_validation_metrics(stats)

        self.assertEqual(metrics["validation/sample_avg_reward"], 0.5)
        self.assertEqual(metrics["validation/accuracy"], 0.5)
        self.assertEqual(metrics["validation/failed_sample_count"], 1)
        self.assertAlmostEqual(metrics["validation/failed_sample_rate"], 0.5)

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
