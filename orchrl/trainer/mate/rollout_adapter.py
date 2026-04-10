from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from orchrl.agent_trajectory_engine import (
    AgentPipeConfig,
    ModelMappingEntry,
    MonitorPoolManager,
    TreeEpisodeResult,
    parallel_rollout,
    tree_rollout,
)
from orchrl.trainer.mate.config import to_plain_dict


class _JobAwareRewardProvider:
    def __init__(self, reward_provider, metadata: dict[str, Any]):
        self._reward_provider = reward_provider
        self._metadata = metadata

    def compute(self, trajectory):
        trajectory.metadata.update(self._metadata)
        return self._reward_provider.compute(trajectory)


class MateRolloutAdapter:
    def __init__(
        self,
        *,
        config,
        prompt_loader,
        reward_provider,
        role_policy_mapping,
        policy_server_name_mapping,
        monitor_pool_manager: MonitorPoolManager,
    ):
        self._config = to_plain_dict(config)
        self._prompt_loader = prompt_loader
        self._reward_provider = reward_provider
        self._role_policy_mapping = dict(role_policy_mapping)
        self._policy_server_name_mapping = dict(policy_server_name_mapping)
        self._roles = list(self._config.get("roles", self._role_policy_mapping.keys()))
        sampling_cfg = self._config.get("sampling", {})
        self._batch_size = int(self._config.get("batch_size", sampling_cfg.get("n_prompts_per_step", 1)))
        self._n_samples_per_prompt = int(self._config.get("n_samples_per_prompt", sampling_cfg.get("n_samples_per_prompt", 1)))
        max_concurrent = self._config.get("max_concurrent_episodes", sampling_cfg.get("max_concurrent_episodes"))
        self._max_concurrent_episodes = int(max_concurrent) if max_concurrent is not None else None
        self._rollout_mode = str(self._config.get("rollout_mode", "parallel"))
        tree_cfg = self._config.get("tree", {})
        k_branches = tree_cfg.get("k_branches", self._config.get("k_branches", 1))
        max_concurrent_branches = tree_cfg.get(
            "max_concurrent_branches",
            self._config.get("max_concurrent_branches"),
        )
        self._k_branches = int(k_branches)
        self._max_concurrent_branches = (
            int(max_concurrent_branches) if max_concurrent_branches is not None else None
        )
        self._monitor_pool_manager = monitor_pool_manager

    async def collect_step_rollouts(self, step_idx: int):
        prompts = self._prompt_loader.get_step_batch(step_idx=step_idx, batch_size=self._batch_size)
        return await self.collect_prompt_batch_rollouts(prompts)

    async def collect_prompt_batch_rollouts(self, prompts, *, n_samples_per_prompt: int | None = None):
        if not prompts:
            return []

        sample_count = self._n_samples_per_prompt if n_samples_per_prompt is None else int(n_samples_per_prompt)
        pipe_config = self._build_pipe_config()
        jobs = [
            {
                "prompt_item": prompt_item,
                "prompt_group_id": f"prompt-{prompt_idx}",
                "sample_idx": sample_idx,
            }
            for prompt_idx, prompt_item in enumerate(prompts)
            for sample_idx in range(sample_count)
        ]
        semaphore = asyncio.Semaphore(self._max_concurrent_episodes) if self._max_concurrent_episodes is not None else None

        async def run_job(job):
            if semaphore is None:
                return await self._collect_single_job(job=job, pipe_config=pipe_config)
            async with semaphore:
                return await self._collect_single_job(job=job, pipe_config=pipe_config)

        gathered = await asyncio.gather(*(run_job(job) for job in jobs))
        episodes = []
        for results in gathered:
            episodes.extend(results)
        return episodes

    async def _collect_single_job(self, *, job, pipe_config: AgentPipeConfig):
        job_metadata = {
            "prompt": job["prompt_item"]["prompt"],
            "expected": job["prompt_item"].get("expected"),
            "prompt_row": job["prompt_item"].get("raw"),
            "prompt_group_id": job["prompt_group_id"],
            "sample_idx": job["sample_idx"],
        }
        reward_provider = _JobAwareRewardProvider(self._reward_provider, job_metadata)
        if self._rollout_mode == "tree":
            result = await tree_rollout(
                prompt=job["prompt_item"]["prompt"],
                reward_provider=reward_provider,
                config=pipe_config,
                k_branches=self._k_branches,
                max_concurrent_branches=self._max_concurrent_branches,
                monitor_pool_manager=self._monitor_pool_manager,
            )
            self._annotate_tree_result(result, job_metadata)
            return [result]

        results = await parallel_rollout(
            prompts=[job["prompt_item"]["prompt"]],
            reward_provider=reward_provider,
            config=pipe_config,
            n_samples_per_prompt=1,
            max_concurrent=None,
            monitor_pool_manager=self._monitor_pool_manager,
        )
        for result in results:
            result.metadata.update(job_metadata)
            result.trajectory.metadata.update(job_metadata)
        return results

    def _build_pipe_config(self) -> AgentPipeConfig:
        model_mapping: dict[str, ModelMappingEntry] = {}
        for role in self._roles:
            policy_name = self._role_policy_mapping[role]
            actual_model = self._policy_server_name_mapping.get(policy_name, policy_name)
            model_mapping[role] = ModelMappingEntry(
                actual_model=actual_model,
            )

        return AgentPipeConfig(
            mas_command_template=self._config["mas_command_template"],
            config_template=self._load_config_template(),
            model_mapping=model_mapping,
            timeout=float(self._config.get("timeout", 300.0)),
            mas_work_dir=Path(self._config["mas_work_dir"]) if self._config.get("mas_work_dir") else None,
        )

    def _load_config_template(self) -> dict[str, Any]:
        inline_template = self._config.get("config_template")
        if isinstance(inline_template, dict):
            return dict(inline_template)

        template_path = self._config.get("config_template_path")
        if not isinstance(template_path, str) or not template_path:
            raise ValueError("mate config requires either config_template or config_template_path")

        with open(template_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj)
        if not isinstance(loaded, dict):
            raise ValueError("mate config template must load as a dict")
        return loaded

    @staticmethod
    def _annotate_tree_result(result: TreeEpisodeResult, metadata: dict[str, Any]) -> None:
        def annotate_episode(episode_result) -> None:
            episode_result.metadata.update(metadata)
            episode_result.trajectory.metadata.update(metadata)

        annotate_episode(result.pilot_result)
        for branch in result.branch_results:
            annotate_episode(branch.episode_result)
