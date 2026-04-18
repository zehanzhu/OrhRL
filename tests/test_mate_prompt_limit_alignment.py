from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self, prompt_ids):
        self._prompt_ids = list(prompt_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return list(self._prompt_ids)


class ChatRendererPromptLimitTests(unittest.TestCase):
    def test_renderer_left_truncates_prompt_ids_at_runtime(self):
        from orchrl.agent_trajectory_engine import ChatRenderer

        renderer = ChatRenderer.from_tokenizer(
            _FakeTokenizer(range(10)),
            model_name="served-model",
            max_prompt_length=4,
            truncation="left",
        )

        prompt_ids, _ = renderer.render(
            [{"role": "user", "content": "prompt"}],
            add_generation_prompt=True,
        )

        self.assertEqual(prompt_ids, [6, 7, 8, 9])


class MateRuntimePromptLimitTests(unittest.TestCase):
    def test_runtime_builds_renderers_with_prompt_limit_settings(self):
        from orchrl.trainer.mate.runtime import MateRuntime

        config = OmegaConf.create(
            {
                "training": {
                    "mate": {
                        "roles": ["searcher"],
                        "role_policy_mapping": {"searcher": "policy_a"},
                        "prompt_loader": {
                            "source_type": "jsonl",
                            "prompt_keys": ["prompt"],
                            "expected_keys": [],
                        },
                        "reward": {},
                        "monitor_pool": {"size": 1},
                        "max_prompt_length": 32,
                        "prompt_truncation": "left",
                        "max_response_length": 48,
                    },
                    "train_data_path": "/tmp/train.jsonl",
                    "val_data_path": "/tmp/val.jsonl",
                }
            }
        )
        runtime = MateRuntime(
            config=config,
            agent_policy_mapping={"searcher": "policy_a"},
        )
        runtime.tokenizer_dict = {"policy_a": "tok-a"}
        runtime.server_handle_dict = {"policy_a": ["handle-a"]}
        runtime.policy_server_name_mapping = {"policy_a": "served-a"}

        with (
            mock.patch(
                "orchrl.agent_trajectory_engine.ChatRenderer.from_tokenizer",
                return_value="renderer",
            ) as renderer_mock,
            mock.patch(
                "orchrl.agent_trajectory_engine.MonitorPoolManager",
                return_value="pool-manager",
            ),
            mock.patch(
                "orchrl.agent_trajectory_engine.RolloutBackend",
                return_value="backend",
            ),
            mock.patch(
                "orchrl.agent_trajectory_engine.ModelMappingEntry",
                side_effect=lambda actual_model: SimpleNamespace(actual_model=actual_model),
            ),
            mock.patch(
                "verl.experimental.agent_loop.AsyncLLMServerManager",
                return_value="async-manager",
            ),
        ):
            runtime._build_mate_monitor_pool_manager()

        renderer_mock.assert_called_once_with(
            "tok-a",
            model_name="served-a",
            max_prompt_length=32,
            truncation="left",
        )


class MateRolloutAdapterTemplateLimitTests(unittest.TestCase):
    def test_load_config_template_aligns_agent_max_tokens_with_response_limit(self):
        from orchrl.trainer.mate.rollout_adapter import MateRolloutAdapter

        adapter = MateRolloutAdapter(
            config={
                "roles": ["verifier", "searcher", "answerer"],
                "role_policy_mapping": {
                    "verifier": "policy_v",
                    "searcher": "policy_s",
                    "answerer": "policy_a",
                },
                "batch_size": 1,
                "n_samples_per_prompt": 1,
                "mas_command_template": "python run.py --config {config_path}",
                "config_template": {
                    "llm": {"max_tokens": 1536},
                    "agents": {
                        "verifier": {"max_tokens": 4096},
                        "searcher": {"max_tokens": 4096},
                        "answerer": {"max_tokens": 4096},
                    },
                },
                "max_response_length": 128,
            },
            prompt_loader=mock.Mock(),
            reward_provider=mock.Mock(),
            role_policy_mapping={
                "verifier": "policy_v",
                "searcher": "policy_s",
                "answerer": "policy_a",
            },
            policy_server_name_mapping={
                "policy_v": "served-v",
                "policy_s": "served-s",
                "policy_a": "served-a",
            },
            monitor_pool_manager=mock.Mock(),
        )

        template = adapter._load_config_template()

        self.assertEqual(template["llm"]["max_tokens"], 128)
        self.assertEqual(template["agents"]["verifier"]["max_tokens"], 128)
        self.assertEqual(template["agents"]["searcher"]["max_tokens"], 128)
        self.assertEqual(template["agents"]["answerer"]["max_tokens"], 128)


class MateDataProtoAdapterLengthValidationTests(unittest.TestCase):
    def test_prompt_ids_longer_than_runtime_limit_raise_instead_of_retruncating(self):
        from orchrl.trainer.mate.dataproto_adapter import _resolve_prompt_ids

        turn = SimpleNamespace(
            prompt_ids=[0, 1, 2, 3, 4],
            agent_role="searcher",
            turn_index=0,
        )

        with self.assertRaisesRegex(ValueError, "longer than configured max_prompt_length"):
            _resolve_prompt_ids(turn=turn, max_prompt_length=4)

    def test_response_ids_longer_than_runtime_limit_raise_instead_of_retruncating(self):
        from orchrl.trainer.mate.dataproto_adapter import _normalize_response_ids

        with self.assertRaisesRegex(ValueError, "longer than configured max_response_length"):
            _normalize_response_ids([0, 1, 2, 3, 4], 4)
