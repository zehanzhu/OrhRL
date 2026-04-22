from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import asyncio
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf

from orchrl.agent_trajectory_engine.datatypes import EpisodeResult, EpisodeTrajectory, TurnData
from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer


class MultiAgentsTrainingRefactorSafetyTests(unittest.TestCase):
    def test_required_trainer_modules_exist(self):
        repo_root = Path(__file__).resolve().parents[1]
        required_modules = [
            "orchrl/trainer/policy_trainer_registry.py",
            "orchrl/trainer/mate/runtime.py",
            "orchrl/trainer/training_step_executor.py",
            "orchrl/trainer/validation_runner.py",
        ]

        missing = [path for path in required_modules if not (repo_root / path).is_file()]
        self.assertFalse(
            missing,
            f"The refactor safety guard expects extracted modules to exist: {missing}",
        )

    def test_trainer_references_extracted_collaborators(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_path = repo_root / "orchrl/trainer/multi_agents_ppo_trainer.py"
        trainer_text = trainer_path.read_text(encoding="utf-8")

        references = [
            "policy_trainer_registry",
            "trainer.mate.runtime",
            "training_step_executor",
            "validation_runner",
        ]

        for ref in references:
            self.assertIn(
                ref,
                trainer_text,
                f"MultiAgentsPPOTrainer should reference extractable collaborator {ref}",
            )

    def test_trainer_imports_refactor_collaborators(self):
        repo_root = Path(__file__).resolve().parents[1]
        trainer_path = repo_root / "orchrl/trainer/multi_agents_ppo_trainer.py"
        trainer_text = trainer_path.read_text(encoding="utf-8")

        import_targets = [
            ("policy_trainer_registry", "PolicyTrainerRegistry"),
            ("mate.runtime", "MateRuntime"),
            ("training_step_executor", "TrainingStepExecutor"),
            ("validation_runner", "ValidationRunner"),
        ]

        for module_name, class_name in import_targets:
            self.assertIn(
                f"from orchrl.trainer.{module_name} import {class_name}",
                trainer_text,
                f"MultiAgentsPPOTrainer must import {class_name} from orchrl.trainer.{module_name}",
            )


class AdapterSupportRemovalSafetyTests(unittest.TestCase):
    def test_key_training_paths_do_not_reference_removed_adapters(self):
        repo_root = Path(__file__).resolve().parents[1]
        guarded_paths = [
            "orchrl/trainer/train.py",
            "orchrl/trainer/policy_trainer_registry.py",
            "orchrl/trainer/multi_agents_ppo_trainer.py",
            "orchrl/trainer/training_step_executor.py",
            "orchrl/trainer/validation_runner.py",
            "orchrl/config/ppo_trainer/base.yaml",
            "experiments/search_mas/train.yaml",
        ]
        forbidden_tokens = (
            "lo" + "ra",
            "q" + "lo" + "ra",
            "pe" + "ft",
        )

        for rel_path in guarded_paths:
            file_text = (repo_root / rel_path).read_text(encoding="utf-8").lower()
            for token in forbidden_tokens:
                self.assertNotIn(
                    token,
                    file_text,
                    f"{rel_path} should not reference removed token '{token}'",
                )


class PolicyTrainerRegistryBehaviorTests(unittest.TestCase):
    class _FakeRayPPOTrainer:
        def __init__(
            self,
            *,
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
        ):
            self.config = config
            self.tokenizer = tokenizer
            self.role_worker_mapping = role_worker_mapping
            self.resource_pool_manager = resource_pool_manager
            self.ray_worker_group_cls = ray_worker_group_cls
            self.async_rollout_manager = SimpleNamespace(server_handles=[f"{tokenizer}-handle"])
            self.checkpoint_manager = f"{tokenizer}-ckpt"
            self.global_steps = None
            self.init_workers_calls = 0

        def init_workers(self):
            self.init_workers_calls += 1

    def _build_config(self, specialization: str):
        return OmegaConf.create(
            {
                "specialization": specialization,
                "training": {
                    "train_batch_size": 8,
                    "experiment_name": "exp-registry",
                    "model_checkpoints_dir": "outputs/training_runs/exp-registry/run-001/checkpoints",
                },
                "models": {
                    "m1": {
                        "name": "policy_a",
                        "ppo_trainer_config": {
                            "data": {},
                            "trainer": {
                                "experiment_name": "seed-a",
                                "default_local_dir": "seed-checkpoints-a",
                            },
                            "actor_rollout_ref": {
                                "rollout": {
                                    "prometheus": {
                                        "served_model_name": "served-policy-a"
                                    }
                                },
                                "model": {"path": "model-a"},
                            },
                        },
                    },
                    "m2": {
                        "name": "policy_b",
                        "ppo_trainer_config": {
                            "data": {},
                            "trainer": {
                                "experiment_name": "seed-b",
                                "default_local_dir": "seed-checkpoints-b",
                            },
                            "actor_rollout_ref": {
                                "rollout": {},
                                "model": {"path": "model-b"},
                            },
                        },
                    },
                },
            }
        )

    def test_registry_owns_single_trainer_creation_logic(self):
        from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry

        config = self._build_config("role_sharing")
        registry = PolicyTrainerRegistry(
            config=config,
            tokenizer_dict={"policy_a": "tok-a"},
            role_worker_mapping={"actor": "worker"},
            resource_pool_manager=["pool-0"],
            ray_worker_group_cls=object,
            ppo_trainer_cls=self._FakeRayPPOTrainer,
        )

        registry.initialize_ppo_trainers()

        self.assertEqual(list(registry.ppo_trainer_dict.keys()), ["policy_a"])
        self.assertEqual(
            registry.ppo_trainer_config_dict["policy_a"].data["train_batch_size"], 8
        )
        self.assertEqual(registry.ppo_trainer_dict["policy_a"].global_steps, 0)

    def test_registry_owns_multi_trainer_creation_and_worker_initialization_logic(self):
        from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry

        config = self._build_config("role_specific")
        registry = PolicyTrainerRegistry(
            config=config,
            tokenizer_dict={"policy_a": "tok-a", "policy_b": "tok-b"},
            role_worker_mapping={"actor": "worker"},
            resource_pool_manager=["pool-0", "pool-1"],
            ray_worker_group_cls=object,
            ppo_trainer_cls=self._FakeRayPPOTrainer,
        )

        registry.initialize_ppo_trainers()
        registry.init_workers()

        self.assertEqual(set(registry.ppo_trainer_dict.keys()), {"policy_a", "policy_b"})
        for model_name, trainer in registry.ppo_trainer_dict.items():
            self.assertEqual(trainer.init_workers_calls, 1, f"{model_name} should init workers once")
            self.assertEqual(trainer.config.trainer.experiment_name, "exp-registry")

    def test_registry_assigns_per_policy_checkpoint_dirs_for_multi_policy_training(self):
        from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry

        config = self._build_config("role_specific")
        registry = PolicyTrainerRegistry(
            config=config,
            tokenizer_dict={"policy_a": "tok-a", "policy_b": "tok-b"},
            role_worker_mapping={"actor": "worker"},
            resource_pool_manager=["pool-0", "pool-1"],
            ray_worker_group_cls=object,
            ppo_trainer_cls=self._FakeRayPPOTrainer,
        )

        registry.initialize_ppo_trainers()

        self.assertEqual(
            registry.ppo_trainer_dict["policy_a"].config.trainer.default_local_dir,
            "outputs/training_runs/exp-registry/run-001/checkpoints/policy_a",
        )
        self.assertEqual(
            registry.ppo_trainer_dict["policy_b"].config.trainer.default_local_dir,
            "outputs/training_runs/exp-registry/run-001/checkpoints/policy_b",
        )

    def test_registry_provides_runtime_handle_accessors(self):
        from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry

        config = self._build_config("role_specific")
        registry = PolicyTrainerRegistry(
            config=config,
            tokenizer_dict={},
            role_worker_mapping={},
            resource_pool_manager=[],
            ray_worker_group_cls=object,
            ppo_trainer_cls=self._FakeRayPPOTrainer,
        )
        registry.ppo_trainer_dict = {
            "policy_a": self._FakeRayPPOTrainer(
                config=config.models.m1.ppo_trainer_config,
                tokenizer="tok-a",
                role_worker_mapping={},
                resource_pool_manager="pool-0",
                ray_worker_group_cls=object,
            ),
            "policy_b": self._FakeRayPPOTrainer(
                config=config.models.m2.ppo_trainer_config,
                tokenizer="tok-b",
                role_worker_mapping={},
                resource_pool_manager="pool-1",
                ray_worker_group_cls=object,
            ),
        }
        registry.ppo_trainer_config_dict = {
            "policy_a": config.models.m1.ppo_trainer_config,
            "policy_b": config.models.m2.ppo_trainer_config,
        }

        registry.collect_runtime_handles()

        self.assertEqual(
            registry.get_checkpoint_managers(),
            {"policy_a": "tok-a-ckpt", "policy_b": "tok-b-ckpt"},
        )
        self.assertEqual(
            registry.get_tokenizers(),
            {"policy_a": "tok-a", "policy_b": "tok-b"},
        )
        self.assertEqual(
            registry.get_server_handles(),
            {"policy_a": ["tok-a-handle"], "policy_b": ["tok-b-handle"]},
        )
        self.assertEqual(
            registry.get_policy_server_names(),
            {"policy_a": "served-policy-a", "policy_b": "model-b"},
        )

    def test_registry_rejects_legacy_specialization_values(self):
        from orchrl.trainer.policy_trainer_registry import PolicyTrainerRegistry

        config = self._build_config("prompt")
        registry = PolicyTrainerRegistry(
            config=config,
            tokenizer_dict={"policy_a": "tok-a"},
            role_worker_mapping={"actor": "worker"},
            resource_pool_manager=["pool-0"],
            ray_worker_group_cls=object,
            ppo_trainer_cls=self._FakeRayPPOTrainer,
        )

        with self.assertRaisesRegex(ValueError, "role_sharing"):
            registry.initialize_ppo_trainers()


class TrainConfigNormalizationTests(unittest.TestCase):
    def test_role_specific_topology_rejects_duplicate_resolved_served_model_names(self):
        from orchrl.trainer.train import _validate_unique_role_specific_served_model_names

        config = OmegaConf.create(
            {
                "specialization": "role_specific",
                "models": {
                    "model_0": {
                        "name": "policy_a",
                        "ppo_trainer_config": {
                            "actor_rollout_ref": {
                                "rollout": {},
                                "model": {"path": "/models/shared"},
                            }
                        },
                    },
                    "model_1": {
                        "name": "policy_b",
                        "ppo_trainer_config": {
                            "actor_rollout_ref": {
                                "rollout": {},
                                "model": {"path": "/models/shared"},
                            }
                        },
                    },
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "Duplicate served model names"):
            _validate_unique_role_specific_served_model_names(config)

    def test_search_mas_config_assigns_unique_explicit_served_model_names(self):
        from hydra import compose, initialize_config_dir
        from orchrl.utils.served_model_name import resolve_policy_server_name

        config_dir = str(
            Path(__file__).resolve().parents[1] / "experiments" / "search_mas"
        )
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="train")
        OmegaConf.resolve(config)

        resolved_names = []
        for model_cfg in config.models.values():
            resolved_names.append(
                resolve_policy_server_name(
                    model_cfg.name,
                    model_cfg.ppo_trainer_config,
                )
            )

        self.assertEqual(
            resolved_names,
            ["verifier_model", "searcher_model", "answerer_model"],
        )
        self.assertEqual(len(set(resolved_names)), 3)
        self.assertFalse(
            hasattr(
                config.models.model_0.ppo_trainer_config.actor_rollout_ref.rollout,
                "served_model_name",
            )
        )
        self.assertEqual(
            config.models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.prometheus.served_model_name,
            "verifier_model",
        )

    def test_single_base_model_role_specific_expands_unique_policy_names(self):
        from orchrl.trainer.train import _expand_single_base_model_role_specific

        config = OmegaConf.create(
            {
                "specialization": "role_specific",
                "base_models": {
                    "policy_0": {
                        "path": "/models/base",
                        "name": "shared_model",
                    }
                },
                "models": {
                    "model_0": {
                        "path": "${base_models.policy_0.path}",
                        "name": "${base_models.policy_0.name}",
                        "ppo_trainer_config": {
                            "actor_rollout_ref": {
                                "model": {
                                    "path": "${base_models.policy_0.path}",
                                    "override_config": {
                                        "model_id": "${base_models.policy_0.name}"
                                    },
                                }
                            }
                        },
                    }
                },
            }
        )

        expanded = _expand_single_base_model_role_specific(
            config,
            {
                "verifier": "verifier_model",
                "searcher": "searcher_model",
            },
        )
        OmegaConf.resolve(expanded)

        self.assertEqual(
            [expanded.models.model_0.name, expanded.models.model_1.name],
            ["verifier_model", "searcher_model"],
        )
        self.assertEqual(
            [expanded.base_models.policy_0.name, expanded.base_models.policy_1.name],
            ["verifier_model", "searcher_model"],
        )
        self.assertEqual(expanded.models.model_0.path, "/models/base")
        self.assertEqual(expanded.models.model_1.path, "/models/base")
        self.assertEqual(
            expanded.models.model_0.ppo_trainer_config.actor_rollout_ref.model.override_config.model_id,
            "verifier_model",
        )
        self.assertEqual(
            expanded.models.model_1.ppo_trainer_config.actor_rollout_ref.model.override_config.model_id,
            "searcher_model",
        )

    def test_single_base_model_expansion_is_noop_when_not_applicable(self):
        from orchrl.trainer.train import _expand_single_base_model_role_specific

        config = OmegaConf.create(
            {
                "specialization": "role_sharing",
                "base_models": {"policy_0": {"path": "/models/base", "name": "shared_model"}},
                "models": {"model_0": {"path": "/models/base", "name": "shared_model"}},
            }
        )

        expanded = _expand_single_base_model_role_specific(
            config,
            {"verifier": "verifier_model"},
        )

        self.assertIs(expanded, config)

    def test_role_specific_expansion_rejects_legacy_specialization_values(self):
        from orchrl.trainer.train import validate_specialization_mode

        with self.assertRaisesRegex(ValueError, "Unsupported specialization 'full'") as full_ctx:
            validate_specialization_mode("full")
        self.assertNotIn("Use 'role_specific' instead", str(full_ctx.exception))

        with self.assertRaisesRegex(ValueError, "Unsupported specialization 'prompt'") as prompt_ctx:
            validate_specialization_mode("prompt")
        self.assertNotIn("Use 'role_sharing' instead", str(prompt_ctx.exception))

    def test_train_multi_agents_only_expands_in_special_case(self):
        from orchrl.trainer import train as train_module

        config = OmegaConf.create(
            {
                "specialization": "role_sharing",
                "resource": {"n_gpus_per_node": 1, "nnodes": 1},
                "agent_policy_configs": {
                    "agent_configs": {
                        "agent_0": {"name": "verifier", "policy_name": "policy_a"}
                    }
                },
                "base_models": {
                    "policy_0": {"path": "/models/base", "name": "policy_a"}
                },
                "models": {
                    "model_0": {
                        "path": "/models/base",
                        "name": "policy_a",
                    }
                },
            }
        )

        call_count = {"expand": 0}

        def fake_expand(cfg, mapping):
            call_count["expand"] += 1
            return cfg

        fake_remote_worker = object()

        class _FakeResourcePoolManager:
            def __init__(self, resource_pool_spec, mapping):
                self.resource_pool_spec = resource_pool_spec
                self.mapping = mapping

            def create_resource_pool(self):
                return None

        class _FakeTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.init_workers_calls = 0
                self.init_runtime_calls = 0
                self.fit_calls = 0

            def init_workers(self):
                self.init_workers_calls += 1

            def init_mate_rollout_runtime(self):
                self.init_runtime_calls += 1

            def fit(self):
                self.fit_calls += 1

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    train_module,
                    "_expand_single_base_model_role_specific",
                    side_effect=fake_expand,
                )
            )
            stack.enter_context(
                mock.patch(
                    "orchrl.trainer.train.copy_local_path_from_hdfs",
                    side_effect=lambda path: path,
                    create=True,
                )
            )
            stack.enter_context(
                mock.patch(
                    "verl.utils.fs.copy_local_path_from_hdfs",
                    side_effect=lambda path: path,
                )
            )
            stack.enter_context(
                mock.patch(
                    "verl.utils.hf_tokenizer",
                    side_effect=lambda path, trust_remote_code=False: f"tok:{path}",
                )
            )
            stack.enter_context(
                mock.patch(
                    "orchrl.trainer.train.hf_tokenizer",
                    side_effect=lambda path, trust_remote_code=False: f"tok:{path}",
                    create=True,
                )
            )
            stack.enter_context(
                mock.patch(
                    "verl.trainer.ppo.ray_trainer.ResourcePoolManager",
                    _FakeResourcePoolManager,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    train_module,
                    "MultiAgentsPPOTrainer",
                    _FakeTrainer,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    train_module.ray,
                    "remote",
                    side_effect=lambda *args, **kwargs: (lambda cls: fake_remote_worker),
                )
            )

            train_module.train_multi_agents(config)

        self.assertEqual(call_count["expand"], 0)

    def test_multi_agents_trainer_init_workers_delegates_to_registry(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        calls = []

        trainer.policy_trainer_registry = SimpleNamespace(
            init_workers=lambda: calls.append("called")
        )

        trainer.init_workers()
        self.assertEqual(calls, ["called"])

    def test_multi_agents_runtime_handle_collection_delegates_to_registry(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        call_order = []

        trainer.policy_trainer_registry = SimpleNamespace(
            collect_runtime_handles=lambda: call_order.append("collect"),
            get_async_rollout_managers=lambda: {"policy_a": "async-a"},
            get_checkpoint_managers=lambda: {"policy_a": "ckpt-a"},
            get_tokenizers=lambda: {"policy_a": "tok-a"},
            get_server_handles=lambda: {"policy_a": ["h-a"]},
            get_policy_server_names=lambda: {"policy_a": "served-a"},
        )
        trainer.mate_runtime = SimpleNamespace(
            initialize=lambda **kwargs: call_order.append(("runtime", kwargs)),
            mate_config={"role_policy_mapping": {"searcher": "policy_a"}},
            monitor_pool_manager="monitor-pool",
            mate_train_prompt_loader="train-loader",
            mate_val_prompt_loader="val-loader",
            mate_reward_provider="reward-provider",
            mate_rollout_adapter="train-adapter",
            mate_val_rollout_adapter="val-adapter",
        )

        trainer.init_mate_rollout_runtime()

        self.assertEqual(call_order[0], "collect")
        self.assertEqual(
            call_order[1],
            (
                "runtime",
                {
                    "tokenizer_dict": {"policy_a": "tok-a"},
                    "async_rollout_manager_dict": {"policy_a": "async-a"},
                    "server_handle_dict": {"policy_a": ["h-a"]},
                    "policy_server_name_mapping": {"policy_a": "served-a"},
                },
            ),
        )
        self.assertEqual(trainer.async_rollout_manager_dict, {"policy_a": "async-a"})
        self.assertEqual(trainer.checkpoint_manager_dict, {"policy_a": "ckpt-a"})
        self.assertEqual(trainer.tokenizer_dict, {"policy_a": "tok-a"})
        self.assertEqual(trainer.server_handle_dict, {"policy_a": ["h-a"]})
        self.assertEqual(trainer.policy_server_name_mapping, {"policy_a": "served-a"})


class MateRuntimeBehaviorTests(unittest.TestCase):
    def _build_runtime_config(self):
        return OmegaConf.create(
            {
                "training": {
                    "mate": {"raw": "cfg"},
                    "train_data_path": "/tmp/train.jsonl",
                    "val_data_path": "/tmp/val.jsonl",
                }
            }
        )

    def test_mate_runtime_owns_validated_config_and_rollout_dependencies(self):
        from orchrl.trainer.mate.runtime import MateRuntime

        validated_mate_config = {
            "role_policy_mapping": {"searcher": "policy_a"},
            "prompt_loader": {
                "source_type": "jsonl",
                "prompt_keys": ["prompt"],
                "expected_keys": ["expected"],
            },
            "reward": {"type": "plugin"},
        }
        config = self._build_runtime_config()

        with (
            mock.patch(
                "orchrl.trainer.mate.runtime.validate_mate_config",
                return_value=validated_mate_config,
            ),
            mock.patch(
                "orchrl.trainer.mate.runtime.build_reward_provider",
                return_value="reward-provider",
            ) as reward_provider_mock,
            mock.patch(
                "orchrl.trainer.mate.runtime.MatePromptLoader",
                side_effect=["train-loader", "val-loader"],
            ) as prompt_loader_mock,
            mock.patch(
                "orchrl.trainer.mate.runtime.MateRolloutAdapter",
                side_effect=["train-adapter", "val-adapter"],
            ) as rollout_adapter_mock,
        ):
            runtime = MateRuntime(
                config=config,
                agent_policy_mapping={"searcher": "policy_a"},
            )
            monitor_pool_manager = mock.Mock()
            runtime._build_mate_monitor_pool_manager = mock.Mock(
                return_value=monitor_pool_manager
            )

            runtime.initialize(
                tokenizer_dict={"policy_a": "tok-a"},
                async_rollout_manager_dict={
                    "policy_a": mock.Mock(
                        server_addresses=["addr"],
                        server_handles=["handle-a"],
                        global_load_balancer="lb",
                    )
                },
                server_handle_dict={"policy_a": ["handle-a"]},
                policy_server_name_mapping={"policy_a": "served-a"},
            )

        self.assertEqual(runtime.mate_config, validated_mate_config)
        runtime._build_mate_monitor_pool_manager.assert_called_once_with()
        monitor_pool_manager.start.assert_called_once_with()
        reward_provider_mock.assert_called_once_with({"type": "plugin"})
        self.assertEqual(prompt_loader_mock.call_count, 2)
        self.assertEqual(
            prompt_loader_mock.call_args_list[0].kwargs["path"],
            "/tmp/train.jsonl",
        )
        self.assertEqual(
            prompt_loader_mock.call_args_list[1].kwargs["path"],
            "/tmp/val.jsonl",
        )
        self.assertEqual(
            rollout_adapter_mock.call_args_list[0].kwargs["prompt_loader"],
            "train-loader",
        )
        self.assertEqual(
            rollout_adapter_mock.call_args_list[1].kwargs["prompt_loader"],
            "val-loader",
        )
        validation_mate_config = rollout_adapter_mock.call_args_list[1].kwargs["config"]
        self.assertEqual(validation_mate_config["rollout_mode"], "parallel")
        self.assertEqual(validation_mate_config["n_samples_per_prompt"], 1)
        self.assertEqual(runtime.mate_rollout_adapter, "train-adapter")
        self.assertEqual(runtime.mate_val_rollout_adapter, "val-adapter")

    def test_multi_agents_trainer_init_mate_runtime_delegates_to_mate_runtime(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        call_order = []

        trainer.policy_trainer_registry = SimpleNamespace(
            collect_runtime_handles=lambda: call_order.append("collect"),
            get_async_rollout_managers=lambda: {"policy_a": "async-a"},
            get_checkpoint_managers=lambda: {"policy_a": "ckpt-a"},
            get_tokenizers=lambda: {"policy_a": "tok-a"},
            get_server_handles=lambda: {"policy_a": ["h-a"]},
            get_policy_server_names=lambda: {"policy_a": "served-a"},
        )
        trainer.mate_runtime = SimpleNamespace(
            initialize=lambda **kwargs: call_order.append(("runtime", kwargs)),
            mate_config={"role_policy_mapping": {"searcher": "policy_a"}},
            monitor_pool_manager="monitor-pool",
            mate_train_prompt_loader="train-loader",
            mate_val_prompt_loader="val-loader",
            mate_reward_provider="reward-provider",
            mate_rollout_adapter="train-adapter",
            mate_val_rollout_adapter="val-adapter",
        )

        trainer.init_mate_rollout_runtime()

        self.assertEqual(call_order[0], "collect")
        self.assertEqual(
            call_order[1],
            (
                "runtime",
                {
                    "tokenizer_dict": {"policy_a": "tok-a"},
                    "async_rollout_manager_dict": {"policy_a": "async-a"},
                    "server_handle_dict": {"policy_a": ["h-a"]},
                    "policy_server_name_mapping": {"policy_a": "served-a"},
                },
            ),
        )
        self.assertEqual(trainer.mate_config, {"role_policy_mapping": {"searcher": "policy_a"}})
        self.assertEqual(trainer.monitor_pool_manager, "monitor-pool")
        self.assertEqual(trainer.mate_train_prompt_loader, "train-loader")
        self.assertEqual(trainer.mate_val_prompt_loader, "val-loader")
        self.assertEqual(trainer.mate_reward_provider, "reward-provider")
        self.assertEqual(trainer.mate_rollout_adapter, "train-adapter")
        self.assertEqual(trainer.mate_val_rollout_adapter, "val-adapter")


class ValidationRunnerBehaviorTests(unittest.TestCase):
    def _build_episode(self, reward):
        return SimpleNamespace(final_reward=reward)

    def _build_mas_episode(self, *, episode_id, reward, include_search=True):
        search_turns = [
            TurnData(
                agent_role="searcher",
                turn_index=0,
                messages=[],
                response_text="<search>query</search>",
                token_ids=[3],
                logprobs=None,
                finish_reason="stop",
                timestamp=1.0,
            )
        ] if include_search else []
        return EpisodeResult(
            trajectory=EpisodeTrajectory(
                episode_id=episode_id,
                agent_trajectories={
                    "verifier": [
                        TurnData(
                            agent_role="verifier",
                            turn_index=0,
                            messages=[],
                            response_text="<verify>yes</verify>",
                            token_ids=[1],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=0.0,
                        )
                    ],
                    "searcher": search_turns,
                    "answerer": [
                        TurnData(
                            agent_role="answerer",
                            turn_index=0,
                            messages=[],
                            response_text="<answer>answer</answer>",
                            token_ids=[4],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=2.0,
                        )
                    ],
                },
            ),
            rewards={"verifier": reward, "answerer": reward},
            final_reward=reward,
        )

    def test_validation_runner_iterates_batches_computes_sample_average_reward_and_saves_best(self):
        from orchrl.trainer.validation_runner import ValidationRunner

        prompt_batches = [
            [{"prompt": "q0"}, {"prompt": "q1"}],
            [{"prompt": "q2"}],
        ]
        episode_batches = [
            [self._build_episode(0.5), self._build_episode(1.0)],
            [self._build_episode(0.0)],
        ]

        async def collect_prompt_batch_rollouts(*, prompts, n_samples_per_prompt):
            self.assertEqual(n_samples_per_prompt, 1)
            return episode_batches.pop(0)

        checkpoint_manager = mock.Mock()
        trainer = mock.Mock()
        registry = SimpleNamespace(
            get_checkpoint_managers=lambda: {"policy_a": checkpoint_manager},
            ppo_trainer_dict={"policy_a": trainer},
        )
        mate_runtime = SimpleNamespace(
            mate_val_prompt_loader=SimpleNamespace(
                iter_batches=lambda batch_size: iter(prompt_batches)
            ),
            mate_val_rollout_adapter=SimpleNamespace(
                collect_prompt_batch_rollouts=collect_prompt_batch_rollouts
            ),
        )
        runner = ValidationRunner(
            config=OmegaConf.create(
                {
                    "training": {
                        "validate_batch_size": 2,
                        "train_batch_size": 4,
                        "if_save": True,
                    },
                    "specialization": "role_sharing",
                }
            ),
            policy_trainer_registry=registry,
            mate_runtime=mate_runtime,
            agent_policy_mapping={},
        )

        metrics = runner.validate(global_steps=1)

        self.assertEqual(metrics["validation/sample_avg_reward"], 0.5)
        self.assertAlmostEqual(metrics["validation/accuracy"], 1 / 3)
        self.assertEqual(metrics["validation/failed_sample_count"], 0)
        self.assertEqual(metrics["validation/failed_sample_rate"], 0.0)
        self.assertEqual(metrics["mas/validation/sample_avg_reward"], 0.5)
        self.assertAlmostEqual(metrics["mas/validation/accuracy"], 1 / 3)
        self.assertEqual(metrics["mas/validation/success_rate"], 1.0)
        self.assertEqual(metrics["mas/validation/failed_rate"], 0.0)
        checkpoint_manager.update_weights.assert_called_once_with()
        checkpoint_manager.sleep_replicas.assert_called_once_with()
        trainer._save_checkpoint.assert_called_once_with()
        self.assertEqual(runner.best_validation_reward, 0.5)

    def test_validation_runner_reports_mas_level_behavior_metrics(self):
        from orchrl.agent_trajectory_engine.datatypes import MateCollectedRollouts
        from orchrl.trainer.validation_runner import ValidationRunner

        rollout_batches = [
            MateCollectedRollouts(
                episodes=[
                    self._build_mas_episode(
                        episode_id="ep-0",
                        reward=1.0,
                        include_search=True,
                    )
                ],
                expected_job_count=2,
                success_count=1,
                failed_count=1,
                failures=[],
            )
        ]

        async def collect_prompt_batch_rollouts(*, prompts, n_samples_per_prompt):
            return rollout_batches.pop(0)

        registry = SimpleNamespace(
            get_checkpoint_managers=lambda: {},
            ppo_trainer_dict={},
        )
        mate_runtime = SimpleNamespace(
            mate_val_prompt_loader=SimpleNamespace(
                iter_batches=lambda batch_size: iter([[{"prompt": "q0"}, {"prompt": "q1"}]])
            ),
            mate_val_rollout_adapter=SimpleNamespace(
                collect_prompt_batch_rollouts=collect_prompt_batch_rollouts
            ),
        )
        runner = ValidationRunner(
            config=OmegaConf.create(
                {
                    "training": {
                        "validate_batch_size": 2,
                        "train_batch_size": 4,
                        "if_save": False,
                    },
                    "specialization": "role_sharing",
                }
            ),
            policy_trainer_registry=registry,
            mate_runtime=mate_runtime,
            agent_policy_mapping={},
        )

        metrics = runner.validate(global_steps=1)

        self.assertEqual(metrics["validation/sample_avg_reward"], 0.5)
        self.assertEqual(metrics["validation/accuracy"], 0.5)
        self.assertEqual(metrics["validation/failed_sample_rate"], 0.5)
        self.assertEqual(metrics["mas/validation/sample_avg_reward"], 0.5)
        self.assertEqual(metrics["mas/validation/accuracy"], 0.5)
        self.assertEqual(metrics["mas/validation/success_rate"], 0.5)
        self.assertEqual(metrics["mas/validation/failed_rate"], 0.5)
        self.assertEqual(metrics["mas/validation/avg_turns"], 3.0)
        self.assertEqual(metrics["mas/validation/avg_search_calls"], 1.0)
        self.assertEqual(metrics["mas/validation/search_call_rate"], 1.0)
        self.assertEqual(metrics["mas/validation/answer_rate"], 1.0)
        self.assertEqual(metrics["mas/validation/verifier_yes_rate"], 1.0)

    def test_multi_agents_trainer_validate_delegates_to_validation_runner(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        trainer.best_validation_reward = float("-inf")
        trainer.validation_runner = SimpleNamespace(
            validate=lambda global_steps: {"validation/sample_avg_reward": 0.75},
            best_validation_reward=0.75,
        )

        metrics = trainer._validate(global_steps=3)

        self.assertEqual(metrics, {"validation/sample_avg_reward": 0.75})
        self.assertEqual(trainer.best_validation_reward, 0.75)


class RolloutAccountingTests(unittest.TestCase):
    def test_parallel_rollout_returns_expected_success_and_failure_counts(self):
        from orchrl.agent_trajectory_engine.parallel import _summarize_parallel_results

        results = _summarize_parallel_results(
            prompts=["q0", "q1"],
            gathered=[
                "ep-0",
                RuntimeError("boom"),
            ],
            n_samples_per_prompt=1,
        )

        self.assertEqual(results.expected_job_count, 2)
        self.assertEqual(results.success_count, 1)
        self.assertEqual(results.failed_count, 1)

    def test_mate_rollout_adapter_surfaces_job_accounting(self):
        from orchrl.trainer.mate.rollout_adapter import MateRolloutAdapter

        adapter = MateRolloutAdapter(
            config={
                "roles": ["searcher"],
                "role_policy_mapping": {"searcher": "policy_a"},
                "batch_size": 2,
                "n_samples_per_prompt": 1,
                "rollout_mode": "parallel",
                "mas_command_template": "python run.py --config {config_path}",
                "config_template": {"llm": {}, "agents": {}},
            },
            prompt_loader=mock.Mock(),
            reward_provider=mock.Mock(),
            role_policy_mapping={"searcher": "policy_a"},
            policy_server_name_mapping={"policy_a": "served-a"},
            monitor_pool_manager=mock.Mock(),
        )

        async def fake_parallel_rollout(**kwargs):
            prompts = list(kwargs["prompts"])
            if prompts == ["q0"]:
                return SimpleNamespace(
                    episodes=[
                        SimpleNamespace(
                            metadata={},
                            trajectory=SimpleNamespace(metadata={}),
                        )
                    ],
                    expected_job_count=1,
                    success_count=1,
                    failed_count=0,
                    failures=[],
                )
            return SimpleNamespace(
                episodes=[],
                expected_job_count=1,
                success_count=0,
                failed_count=1,
                failures=[{"error_type": "RuntimeError", "message": "boom"}],
            )

        with mock.patch(
            "orchrl.trainer.mate.rollout_adapter.parallel_rollout",
            side_effect=fake_parallel_rollout,
        ):
            result = asyncio.run(
                adapter.collect_prompt_batch_rollouts(
                    prompts=[{"prompt": "q0"}, {"prompt": "q1"}],
                    n_samples_per_prompt=1,
                )
            )

        self.assertEqual(result.expected_job_count, 2)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(len(result.episodes), 1)


class OrchRLCodeHygieneTests(unittest.TestCase):
    def test_runtime_uses_prompt_loader_only(self):
        repo_root = Path(__file__).resolve().parents[1]
        runtime_text = (
            repo_root / "orchrl/trainer/mate/runtime.py"
        ).read_text(encoding="utf-8")

        self.assertIn('self.mate_config.get("prompt_loader", {})', runtime_text)
        self.assertNotIn("prompt_source", runtime_text)

    def test_training_step_executor_restores_uid_assignment_summary_logging(self):
        repo_root = Path(__file__).resolve().parents[1]
        executor_text = (
            repo_root / "orchrl/trainer/training_step_executor.py"
        ).read_text(encoding="utf-8")

        self.assertIn("[DEBUG UID]", executor_text)
        self.assertIn("UID assignment summary", executor_text)

    def test_performance_utils_remove_unused_debug_helpers(self):
        repo_root = Path(__file__).resolve().parents[1]
        performance_text = (
            repo_root / "orchrl/utils/performance.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("def log_print", performance_text)
        self.assertNotIn("class SimplerTimer", performance_text)
        self.assertNotIn("def create_timer", performance_text)
        self.assertNotIn("def marked_timer", performance_text)
        self.assertNotIn("def reduce_timing", performance_text)

    def test_dataproto_adapter_removes_dead_tree_policy_batch_helper(self):
        repo_root = Path(__file__).resolve().parents[1]
        adapter_text = (
            repo_root / "orchrl/trainer/mate/dataproto_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("def tree_episodes_to_policy_batches", adapter_text)

    def test_served_model_name_removes_legacy_passthrough_helper(self):
        repo_root = Path(__file__).resolve().parents[1]
        served_model_text = (
            repo_root / "orchrl/utils/served_model_name.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("def _legacy_model_name", served_model_text)

    def test_unused_display_helper_is_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse(
            (repo_root / "orchrl/agent_trajectory_engine/_support/display.py").exists()
        )


class MultiAgentsTrainerResumeBehaviorTests(unittest.TestCase):
    class _FakeProgressBar:
        def update(self, value):
            self.last_update = value

        def set_description(self, value):
            self.last_description = value

        def close(self):
            self.closed = True

    class _ResumeTrainer:
        def __init__(self, loaded_step):
            self.loaded_step = loaded_step
            self.global_steps = 0
            self.load_calls = 0

        def _load_checkpoint(self):
            self.load_calls += 1
            self.global_steps = self.loaded_step
            return None

    def _build_trainer(self, ppo_trainer_dict, *, total_training_steps):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        trainer.config = OmegaConf.create(
            {
                "training": {
                    "project_name": "orchrl",
                    "experiment_name": "resume-test",
                    "logger": ["console"],
                    "total_training_steps": total_training_steps,
                }
            }
        )
        trainer.ppo_trainer_dict = ppo_trainer_dict
        trainer._initialize_logger_safely = lambda: SimpleNamespace(
            log=lambda **kwargs: None
        )
        return trainer

    def test_fit_resumes_all_policy_trainers_when_load_only_sets_global_steps(self):
        trainer = self._build_trainer(
            {
                "policy_a": self._ResumeTrainer(loaded_step=7),
                "policy_b": self._ResumeTrainer(loaded_step=7),
            },
            total_training_steps=7,
        )

        with mock.patch(
            "orchrl.trainer.multi_agents_ppo_trainer.tqdm",
            return_value=self._FakeProgressBar(),
        ):
            trainer.fit()

        self.assertEqual(trainer.global_steps, 7)
        self.assertEqual(trainer.ppo_trainer_dict["policy_a"].load_calls, 1)
        self.assertEqual(trainer.ppo_trainer_dict["policy_b"].load_calls, 1)

    def test_fit_raises_when_policy_resume_steps_are_inconsistent(self):
        trainer = self._build_trainer(
            {
                "policy_a": self._ResumeTrainer(loaded_step=7),
                "policy_b": self._ResumeTrainer(loaded_step=8),
            },
            total_training_steps=8,
        )

        with mock.patch(
            "orchrl.trainer.multi_agents_ppo_trainer.tqdm",
            return_value=self._FakeProgressBar(),
        ):
            with self.assertRaisesRegex(RuntimeError, "Inconsistent checkpoint steps"):
                trainer.fit()


class TrainingStepExecutorBehaviorTests(unittest.TestCase):
    def _build_mas_episode(self, *, final_reward=1.0):
        return EpisodeResult(
            trajectory=EpisodeTrajectory(
                episode_id="ep-train",
                agent_trajectories={
                    "verifier": [
                        TurnData(
                            agent_role="verifier",
                            turn_index=0,
                            messages=[],
                            response_text="<verify>no</verify>",
                            token_ids=[1],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=0.0,
                        ),
                        TurnData(
                            agent_role="verifier",
                            turn_index=1,
                            messages=[],
                            response_text="<verify>yes</verify>",
                            token_ids=[2],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=3.0,
                        ),
                    ],
                    "searcher": [
                        TurnData(
                            agent_role="searcher",
                            turn_index=0,
                            messages=[],
                            response_text="<search>query</search>",
                            token_ids=[3],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=1.0,
                        )
                    ],
                    "answerer": [
                        TurnData(
                            agent_role="answerer",
                            turn_index=0,
                            messages=[],
                            response_text="<answer>answer</answer>",
                            token_ids=[4],
                            logprobs=None,
                            finish_reason="stop",
                            timestamp=4.0,
                        )
                    ],
                },
            ),
            rewards={"verifier": final_reward, "searcher": final_reward, "answerer": final_reward},
            final_reward=final_reward,
        )

    def test_step_executor_collects_policy_batches_and_exports_trajectories(self):
        from orchrl.trainer.training_step_executor import TrainingStepExecutor

        async def collect_step_rollouts(*, step_idx):
            self.assertEqual(step_idx, 5)
            return ["episode-a"]

        checkpoint_manager = mock.Mock()
        registry = SimpleNamespace(
            get_checkpoint_managers=lambda: {"policy_a": checkpoint_manager},
            get_tokenizers=lambda: {"policy_a": "tok-a"},
            ppo_trainer_dict={
                "policy_a": SimpleNamespace(
                    config=OmegaConf.create(
                        {"data": {"max_prompt_length": 32, "max_response_length": 16}}
                    )
                )
            },
        )
        mate_runtime = SimpleNamespace(
            mate_rollout_adapter=SimpleNamespace(collect_step_rollouts=collect_step_rollouts),
            mate_config={
                "role_policy_mapping": {"searcher": "policy_a"},
                "rollout_mode": "parallel",
            },
        )

        with (
            mock.patch(
                "orchrl.trainer.training_step_executor.maybe_export_prompt_trajectories"
            ) as export_mock,
            mock.patch(
                "orchrl.trainer.training_step_executor.episodes_to_policy_batches",
                return_value={"policy_a": "batch-a"},
            ) as adapter_mock,
        ):
            executor = TrainingStepExecutor(
                config=OmegaConf.create(
                    {"training": {"max_prompt_length": None, "max_response_length": None}}
                ),
                policy_trainer_registry=registry,
                mate_runtime=mate_runtime,
                agent_policy_mapping={"searcher": "policy_a"},
                agent_untrained=[],
            )
            output = executor.collect_mate_step_batches(step_idx=5)

        self.assertEqual(output, {"policy_a": "batch-a"})
        checkpoint_manager.update_weights.assert_called_once_with()
        checkpoint_manager.sleep_replicas.assert_called_once_with()
        export_mock.assert_called_once_with(
            episodes=["episode-a"],
            step_idx=5,
            rollout_mode="parallel",
            mate_config=mate_runtime.mate_config,
        )
        adapter_mock.assert_called_once()

    def test_step_executor_execute_step_updates_present_policies_and_tracks_missing(self):
        from verl import DataProto
        import torch

        from orchrl.trainer.training_step_executor import TrainingStepExecutor

        fake_batch = DataProto.from_dict(
            tensors={"responses": torch.tensor([[1, 2]])},
            non_tensors={"uid": ["a0"]},
        )
        trainer_a = SimpleNamespace(
            actor_rollout_wg=SimpleNamespace(world_size=1),
            config=SimpleNamespace(filter_ratio=0.0, filter_method="uid"),
        )
        trainer_b = SimpleNamespace(
            actor_rollout_wg=SimpleNamespace(world_size=1),
            config=SimpleNamespace(filter_ratio=0.0, filter_method="uid"),
        )
        registry = SimpleNamespace(
            ppo_trainer_dict={"policy_a": trainer_a, "policy_b": trainer_b},
        )
        executor = TrainingStepExecutor(
            config=OmegaConf.create({"training": {}}),
            policy_trainer_registry=registry,
            mate_runtime=SimpleNamespace(mate_config={"role_policy_mapping": {}}),
            agent_policy_mapping={},
            agent_untrained=[],
        )
        def collect_mate_step_batches(step_idx):
            executor._last_mate_episodes = [self._build_mas_episode(final_reward=1.0)]
            executor._last_mate_rollout_accounting = {
                "expected_job_count": 2,
                "success_count": 1,
                "failed_count": 1,
                "failures": [],
            }
            return {"policy_a": fake_batch}

        executor.collect_mate_step_batches = mock.Mock(
            side_effect=collect_mate_step_batches
        )
        executor.filter_batch_by_existing_uid_groups = mock.Mock(
            side_effect=lambda data_proto, filter_ratio, mode: data_proto
        )
        executor.update_parameters = mock.Mock(return_value=fake_batch)

        with mock.patch(
            "orchrl.trainer.training_step_executor.pad_dataproto_to_divisor",
            side_effect=lambda batch, world_size: (batch, None),
        ):
            result = executor.execute_training_step(step_idx=2)

        self.assertEqual(result.present_policy_names, ["policy_a"])
        self.assertEqual(result.missing_policy_names, ["policy_b"])
        self.assertEqual(result.batch_per_trainer["policy_a"], fake_batch)
        self.assertEqual(result.metrics["training/present_policy_count"], 1)
        self.assertEqual(result.metrics["training/skipped_policy_count"], 1)
        self.assertNotIn("training/skipped_policies", result.metrics)
        for metric_name, metric_value in result.metrics.items():
            self.assertIsInstance(
                metric_value,
                (int, float),
                f"{metric_name} should be a numeric scalar metric",
            )
        self.assertEqual(result.metrics["mas/train/sample_avg_reward"], 0.5)
        self.assertEqual(result.metrics["mas/train/success_rate"], 0.5)
        self.assertEqual(result.metrics["mas/train/failed_rate"], 0.5)
        executor.update_parameters.assert_called_once_with(
            fake_batch,
            trainer_a,
            mock.ANY,
        )
        self.assertIn("collect_trajectory", result.timing_raw)
        self.assertIn("update_parameters", result.timing_raw)

    def test_step_executor_reports_mas_level_train_metrics(self):
        from orchrl.agent_trajectory_engine.datatypes import MateCollectedRollouts
        from orchrl.trainer.training_step_executor import TrainingStepExecutor

        episode = self._build_mas_episode(final_reward=1.0)
        rollout_result = MateCollectedRollouts(
            episodes=[episode],
            expected_job_count=2,
            success_count=1,
            failed_count=1,
            failures=[],
        )
        registry = SimpleNamespace(
            get_checkpoint_managers=lambda: {},
            get_tokenizers=lambda: {"policy_a": "tok-a"},
            ppo_trainer_dict={
                "policy_a": SimpleNamespace(
                    config=OmegaConf.create(
                        {"data": {"max_prompt_length": 32, "max_response_length": 16}}
                    )
                )
            },
        )
        async def collect_step_rollouts(*, step_idx):
            return rollout_result

        mate_runtime = SimpleNamespace(
            mate_rollout_adapter=SimpleNamespace(
                collect_step_rollouts=collect_step_rollouts
            ),
            mate_config={
                "role_policy_mapping": {"verifier": "policy_a"},
                "rollout_mode": "parallel",
                "failure_policy": {"mode": "threshold", "max_failed_rate": 0.8},
            },
        )

        with (
            mock.patch("orchrl.trainer.training_step_executor.maybe_export_prompt_trajectories"),
            mock.patch(
                "orchrl.trainer.training_step_executor.episodes_to_policy_batches",
                return_value={"policy_a": "batch-a"},
            ),
        ):
            executor = TrainingStepExecutor(
                config=OmegaConf.create(
                    {"training": {"max_prompt_length": None, "max_response_length": None}}
                ),
                policy_trainer_registry=registry,
                mate_runtime=mate_runtime,
                agent_policy_mapping={"verifier": "policy_a"},
                agent_untrained=[],
            )
            executor.collect_mate_step_batches(step_idx=3)

        accounting = executor._last_mate_rollout_accounting
        self.assertEqual(accounting["failed_count"], 1)
        metrics = executor.build_mas_train_metrics(
            episodes=[episode],
            expected_job_count=accounting["expected_job_count"],
            failed_count=accounting["failed_count"],
        )

        self.assertEqual(metrics["mas/train/sample_avg_reward"], 0.5)
        self.assertEqual(metrics["mas/train/success_rate"], 0.5)
        self.assertEqual(metrics["mas/train/failed_rate"], 0.5)
        self.assertEqual(metrics["mas/train/avg_turns"], 4.0)
        self.assertEqual(metrics["mas/train/avg_search_calls"], 1.0)
        self.assertEqual(metrics["mas/train/search_call_rate"], 1.0)
        self.assertEqual(metrics["mas/train/answer_rate"], 1.0)
        self.assertEqual(metrics["mas/train/verifier_yes_rate"], 0.5)
        self.assertEqual(metrics["mas/train/verifier_no_rate"], 0.5)

    def test_multi_agents_trainer_collect_phase_delegates_to_training_step_executor(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        trainer.global_steps = 4
        trainer.training_step_executor = SimpleNamespace(
            collect_mate_step_batches=lambda step_idx: {"policy_a": f"batch-{step_idx}"}
        )

        collected = trainer.fit_one_collect_phase_for_test()

        self.assertEqual(collected, {"policy_a": "batch-4"})


class TrainingFailurePolicyTests(unittest.TestCase):
    def _build_executor(self, *, failure_policy):
        from orchrl.trainer.training_step_executor import TrainingStepExecutor

        return TrainingStepExecutor(
            config=OmegaConf.create(
                {
                    "training": {
                        "mate": {
                            "failure_policy": failure_policy,
                        }
                    }
                }
            ),
            policy_trainer_registry=mock.Mock(),
            mate_runtime=SimpleNamespace(
                mate_config={
                    "failure_policy": failure_policy,
                }
            ),
            agent_policy_mapping={},
            agent_untrained=[],
        )

    def test_training_step_allows_small_failure_below_threshold(self):
        executor = self._build_executor(
            failure_policy={
                "mode": "threshold",
                "max_failed_jobs": 1,
                "max_failed_rate": 0.5,
            }
        )

        executor._enforce_rollout_failure_policy(
            expected_job_count=4,
            failed_count=1,
            failures=[{"message": "boom"}],
        )

    def test_training_step_threshold_is_controlled_only_by_failure_rate(self):
        executor = self._build_executor(
            failure_policy={
                "mode": "threshold",
                "max_failed_jobs": 1,
                "max_failed_rate": 0.5,
            }
        )

        executor._enforce_rollout_failure_policy(
            expected_job_count=10,
            failed_count=2,
            failures=[{"message": "boom-0"}, {"message": "boom-1"}],
        )

    def test_training_step_raises_when_failure_threshold_exceeded(self):
        executor = self._build_executor(
            failure_policy={
                "mode": "threshold",
                "max_failed_jobs": 1,
                "max_failed_rate": 0.2,
            }
        )

        with self.assertRaisesRegex(RuntimeError, "rollout job failures"):
            executor._enforce_rollout_failure_policy(
                expected_job_count=4,
                failed_count=2,
                failures=[{"message": "boom-0"}, {"message": "boom-1"}],
        )

    def test_training_step_default_threshold_allows_failures_below_eighty_percent(self):
        executor = self._build_executor(failure_policy={})

        executor._enforce_rollout_failure_policy(
            expected_job_count=10,
            failed_count=5,
            failures=[{"message": "boom"}],
        )

    def test_training_step_default_threshold_raises_above_eighty_percent(self):
        executor = self._build_executor(failure_policy={})

        with self.assertRaisesRegex(RuntimeError, "rollout job failures"):
            executor._enforce_rollout_failure_policy(
                expected_job_count=10,
                failed_count=9,
                failures=[{"message": "boom"}],
            )


class MateFailurePolicyConfigTests(unittest.TestCase):
    def test_failure_policy_normalization_defaults_rate_threshold_to_eighty_percent(self):
        from orchrl.trainer.mate.config import validate_mate_config

        config = validate_mate_config(
            {
                "roles": ["searcher"],
                "role_policy_mapping": {"searcher": "policy_a"},
                "rollout_mode": "parallel",
            },
            {"searcher": "policy_a"},
        )

        self.assertEqual(
            config["failure_policy"],
            {
                "mode": "threshold",
                "max_failed_rate": 0.8,
            },
        )

    def test_failure_policy_normalization_drops_legacy_job_count_threshold(self):
        from orchrl.trainer.mate.config import validate_mate_config

        config = validate_mate_config(
            {
                "roles": ["searcher"],
                "role_policy_mapping": {"searcher": "policy_a"},
                "rollout_mode": "parallel",
                "failure_policy": {
                    "mode": "threshold",
                    "max_failed_jobs": 1,
                    "max_failed_rate": 0.5,
                },
            },
            {"searcher": "policy_a"},
        )

        self.assertEqual(
            config["failure_policy"],
            {
                "mode": "threshold",
                "max_failed_rate": 0.5,
            },
        )


class MASLauncherLoggingTests(unittest.TestCase):
    def test_launcher_accepts_explicit_log_paths(self):
        from orchrl.agent_trajectory_engine._support.launcher import MASLauncher

        with tempfile.TemporaryDirectory() as tmp_dir:
            launcher = MASLauncher(work_dir=tmp_dir)
            stdout_path, stderr_path = launcher.prepare_log_paths(
                root_dir=Path(tmp_dir) / "logs",
                episode_id="ep-1",
            )

        self.assertTrue(str(stdout_path).endswith("ep-1.stdout.log"))
        self.assertTrue(str(stderr_path).endswith("ep-1.stderr.log"))


class MultiAgentsTrainerLoggingCompatibilityTests(unittest.TestCase):
    def test_logger_initialization_provides_top_level_trainer_config_for_tracking(self):
        trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
        trainer.config = OmegaConf.create(
            {
                "training": {
                    "project_name": "orchrl",
                    "experiment_name": "wandb-compat-test",
                    "logger": ["console", "wandb"],
                }
            }
        )

        with mock.patch("verl.utils.tracking.Tracking") as tracking_cls:
            trainer._initialize_logger_safely()

        tracking_kwargs = tracking_cls.call_args.kwargs
        self.assertEqual(tracking_kwargs["project_name"], "orchrl")
        self.assertEqual(tracking_kwargs["experiment_name"], "wandb-compat-test")
        self.assertEqual(tracking_kwargs["default_backend"], ["console", "wandb"])
        self.assertIn("training", tracking_kwargs["config"])
        self.assertIn("trainer", tracking_kwargs["config"])
        self.assertEqual(tracking_kwargs["config"]["trainer"], {})


if __name__ == "__main__":
    unittest.main()
