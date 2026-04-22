from __future__ import annotations

from pathlib import Path
import unittest

import yaml


class SearchTemperatureAlignmentTests(unittest.TestCase):
    def test_search_configs_use_uniform_point_six_temperature(self):
        repo_root = Path(__file__).resolve().parents[1]

        train_cfg = yaml.safe_load(
            (repo_root / "experiments/search_mas/train.yaml").read_text(
                encoding="utf-8"
            )
        )
        inference_cfg = yaml.safe_load(
            (repo_root / "mas_apps/search/configs/inference.yaml").read_text(
                encoding="utf-8"
            )
        )
        example_cfg = yaml.safe_load(
            (repo_root / "mas_apps/search/configs/search_mas_example.yaml").read_text(
                encoding="utf-8"
            )
        )
        generated_cfgs = [
            yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in sorted((repo_root / "mas_apps/search").glob("mate_mas_*.yaml"))
        ]
        ppo_base_cfg = yaml.safe_load(
            (repo_root / "orchrl/config/ppo_trainer/base.yaml").read_text(
                encoding="utf-8"
            )
        )
        search_readme_text = (repo_root / "mas_apps/search/README.md").read_text(
            encoding="utf-8"
        )

        self.assertEqual(train_cfg["training"]["sample_temperature"], 0.6)

        for agent_cfg in train_cfg["agent_policy_configs"]["agent_configs"].values():
            self.assertEqual(agent_cfg["train_llm_config"]["temperature"], 0.6)
            self.assertEqual(agent_cfg["val_llm_config"]["temperature"], 0.6)

        self.assertEqual(inference_cfg["llm"]["temperature"], 0.6)
        self.assertEqual(example_cfg["llm"]["temperature"], 0.6)

        for role in ("verifier", "searcher", "answerer"):
            self.assertEqual(inference_cfg["agents"][role]["temperature"], 0.6)
            self.assertEqual(example_cfg["agents"][role]["temperature"], 0.6)

        for generated_cfg in generated_cfgs:
            self.assertEqual(generated_cfg["llm"]["temperature"], 0.6)
            for role in ("verifier", "searcher", "answerer"):
                self.assertEqual(generated_cfg["agents"][role]["temperature"], 0.6)

        self.assertEqual(
            ppo_base_cfg["actor_rollout_ref"]["rollout"]["temperature"],
            0.6,
        )
        self.assertEqual(
            ppo_base_cfg["actor_rollout_ref"]["rollout"]["val_kwargs"][
                "temperature"
            ],
            0.6,
        )
        self.assertNotIn("temperature: 0.0", search_readme_text)
        self.assertNotIn("temperature: 0.2", search_readme_text)
        self.assertNotIn("temperature: 0.4", search_readme_text)


if __name__ == "__main__":
    unittest.main()
