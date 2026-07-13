from types import SimpleNamespace
import unittest
from unittest import mock

from omegaconf import OmegaConf
import torch

from verl import DataProto

from orchrl.trainer.v1_tq_adapter import (
    ensure_v1_ppo_config,
    prepare_dataproto_for_v1_transfer_queue,
)


class V1TransferQueueAdapterTests(unittest.TestCase):
    def test_writes_prompt_tags_and_trajectory_fields_grouped_for_replay_buffer(self):
        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[0, 11, 12], [0, 11, 13]], dtype=torch.long),
                "responses": torch.tensor([[21, 0], [22, 23]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
            },
            non_tensors={
                "uid": ["same_prompt", "same_prompt"],
                "reward": [0.5, 1.0],
                "agent_name": ["verifier", "verifier"],
                "turn_idx": [0, 1],
            },
        )
        trainer = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

        with mock.patch("orchrl.trainer.v1_tq_adapter.tq.kv_batch_put") as put:
            sample_batch_size = prepare_dataproto_for_v1_transfer_queue(
                data_proto=batch,
                trainer=trainer,
                global_steps=3,
            )

        self.assertEqual(sample_batch_size, 1)
        self.assertEqual(put.call_count, 2)

        prompt_call = put.call_args_list[0].kwargs
        self.assertEqual(prompt_call["partition_id"], "train")
        self.assertEqual(len(prompt_call["keys"]), 1)
        self.assertEqual(
            prompt_call["tags"],
            [{"is_prompt": True, "status": "finished", "global_steps": 3}],
        )

        trajectory_call = put.call_args_list[1].kwargs
        trajectory_keys = trajectory_call["keys"]
        self.assertEqual(len(trajectory_keys), 2)
        self.assertTrue(all(key.startswith(prompt_call["keys"][0] + "_") for key in trajectory_keys))
        self.assertEqual([tag["response_len"] for tag in trajectory_call["tags"]], [1, 2])
        self.assertEqual([tag["global_steps"] for tag in trajectory_call["tags"]], [3, 3])

        fields = trajectory_call["fields"]
        self.assertEqual(fields["prompts"].unbind()[0].tolist(), [11, 12])
        self.assertEqual(fields["responses"].unbind()[1].tolist(), [22, 23])
        self.assertEqual(fields["rm_scores"].unbind()[0].tolist(), [0.5])
        self.assertEqual(fields["rm_scores"].unbind()[1].tolist(), [0.0, 1.0])

    def test_v1_backend_rejects_async_trainer_modes_until_supported(self):
        config = OmegaConf.create(
            {
                "trainer": {
                    "v1": {
                        "trainer_mode": "colocate_async",
                    }
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "sync only"):
            ensure_v1_ppo_config(config)


if __name__ == "__main__":
    unittest.main()
