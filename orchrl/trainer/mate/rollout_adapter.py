from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import yaml

from orchrl.agent_trajectory_engine import (
    AgentPipeConfig,
    MateCollectedRollouts,
    ModelMappingEntry,
    MonitorPoolManager,
    RolloutFailure,
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
            return MateCollectedRollouts(
                episodes=[],
                expected_job_count=0,
                success_count=0,
                failed_count=0,
            )

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

        gathered = await asyncio.gather(
            *(run_job(job) for job in jobs),
            return_exceptions=True,
        )
        episodes = []
        expected_job_count = 0
        success_count = 0
        failed_count = 0
        failures = []
        for item in gathered:
            if isinstance(item, Exception):
                failures.append(
                    RolloutFailure(
                        error_type=type(item).__name__,
                        message=str(item),
                    )
                )
                expected_job_count += 1
                failed_count += 1
                continue
            if isinstance(item, BaseException):
                raise item
            episodes.extend(list(getattr(item, "episodes", []) or []))
            expected_job_count += int(getattr(item, "expected_job_count", 0))
            success_count += int(getattr(item, "success_count", 0))
            failed_count += int(getattr(item, "failed_count", 0))
            failures.extend(list(getattr(item, "failures", []) or []))
        return MateCollectedRollouts(
            episodes=episodes,
            expected_job_count=expected_job_count,
            success_count=success_count,
            failed_count=failed_count,
            failures=failures,
        )

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
            try:
                result = await tree_rollout(
                    prompt=job["prompt_item"]["prompt"],
                    reward_provider=reward_provider,
                    config=pipe_config,
                    k_branches=self._k_branches,
                    max_concurrent_branches=self._max_concurrent_branches,
                    monitor_pool_manager=self._monitor_pool_manager,
                )
            except Exception as exc:
                return MateCollectedRollouts(
                    episodes=[],
                    expected_job_count=1,
                    success_count=0,
                    failed_count=1,
                    failures=[
                        RolloutFailure(
                            error_type=type(exc).__name__,
                            message=str(exc),
                            prompt=job_metadata["prompt"],
                            sample_idx=job_metadata["sample_idx"],
                            metadata={"prompt_group_id": job_metadata["prompt_group_id"]},
                        )
                    ],
                )
            self._annotate_tree_result(result, job_metadata)
            if result.pilot_result.status != "success":
                return MateCollectedRollouts(
                    episodes=[],
                    expected_job_count=1,
                    success_count=0,
                    failed_count=1,
                    failures=[
                        RolloutFailure(
                            error_type="TreePilotFailed",
                            message=str(result.pilot_result.failure_info or "tree pilot failed"),
                            prompt=job_metadata["prompt"],
                            sample_idx=job_metadata["sample_idx"],
                            metadata={"prompt_group_id": job_metadata["prompt_group_id"]},
                        )
                    ],
                )
            expected_branch_count = int(result.tree_metadata.get("expected_branch_count", 0))
            failed_branch_count = int(result.tree_metadata.get("failed_branch_count", 0))
            return MateCollectedRollouts(
                episodes=[result],
                expected_job_count=1 + expected_branch_count,
                success_count=1 + len(result.branch_results),
                failed_count=failed_branch_count,
                failures=[
                    RolloutFailure(
                        error_type="TreeBranchFailed",
                        message="tree branch rollout failed",
                        prompt=job_metadata["prompt"],
                        sample_idx=job_metadata["sample_idx"],
                        metadata={"prompt_group_id": job_metadata["prompt_group_id"]},
                    )
                    for _ in range(failed_branch_count)
                ],
            )

        results = await parallel_rollout(
            prompts=[job["prompt_item"]["prompt"]],
            reward_provider=reward_provider,
            config=pipe_config,
            n_samples_per_prompt=1,
            max_concurrent=None,
            monitor_pool_manager=self._monitor_pool_manager,
        )
        for result in results.episodes:
            result.metadata.update(job_metadata)
            result.trajectory.metadata.update(job_metadata)
        return MateCollectedRollouts(
            episodes=list(results.episodes),
            expected_job_count=results.expected_job_count,
            success_count=results.success_count,
            failed_count=results.failed_count,
            failures=[
                self._annotate_failure(
                    failure,
                    metadata=job_metadata,
                )
                for failure in results.failures
            ],
        )

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
            mas_log_dir=Path(self._config["mas_log_dir"]) if self._config.get("mas_log_dir") else None,
        )

    def _load_config_template(self) -> dict[str, Any]:
        inline_template = self._config.get("config_template")
        if isinstance(inline_template, dict):
            loaded = copy.deepcopy(inline_template)
            return self._apply_generation_limits(loaded)

        template_path = self._config.get("config_template_path")
        if not isinstance(template_path, str) or not template_path:
            raise ValueError("mate config requires either config_template or config_template_path")

        with open(template_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj)
        if not isinstance(loaded, dict):
            raise ValueError("mate config template must load as a dict")
        return self._apply_generation_limits(copy.deepcopy(loaded))

    def _apply_generation_limits(self, config_template: dict[str, Any]) -> dict[str, Any]:
        max_response_length = self._config.get("max_response_length")
        if max_response_length is None:
            return config_template

        resolved_limit = int(max_response_length)
        llm_cfg = config_template.setdefault("llm", {})
        if isinstance(llm_cfg, dict):
            llm_cfg["max_tokens"] = resolved_limit

        agents_cfg = config_template.setdefault("agents", {})
        if not isinstance(agents_cfg, dict):
            agents_cfg = {}
            config_template["agents"] = agents_cfg

        for role in self._roles:
            role_cfg = agents_cfg.setdefault(role, {})
            if not isinstance(role_cfg, dict):
                role_cfg = {}
                agents_cfg[role] = role_cfg
            role_cfg["max_tokens"] = resolved_limit
            role_llm_cfg = role_cfg.get("llm")
            if isinstance(role_llm_cfg, dict):
                role_llm_cfg["max_tokens"] = resolved_limit

        return config_template

    @staticmethod
    def _annotate_tree_result(result: TreeEpisodeResult, metadata: dict[str, Any]) -> None:
        def annotate_episode(episode_result) -> None:
            episode_result.metadata.update(metadata)
            episode_result.trajectory.metadata.update(metadata)

        annotate_episode(result.pilot_result)
        for branch in result.branch_results:
            annotate_episode(branch.episode_result)

    @staticmethod
    def _annotate_failure(failure, *, metadata: dict[str, Any]):
        if isinstance(failure, RolloutFailure):
            failure.prompt = metadata["prompt"]
            failure.sample_idx = metadata["sample_idx"]
            failure.metadata.update({"prompt_group_id": metadata["prompt_group_id"]})
            return failure
        if isinstance(failure, dict):
            merged_metadata = dict(failure.get("metadata") or {})
            merged_metadata.update({"prompt_group_id": metadata["prompt_group_id"]})
            return RolloutFailure(
                error_type=str(failure.get("error_type", "RuntimeError")),
                message=str(failure.get("message", "")),
                prompt=metadata["prompt"],
                sample_idx=metadata["sample_idx"],
                metadata=merged_metadata,
            )
        return RolloutFailure(
            error_type=type(failure).__name__,
            message=str(failure),
            prompt=metadata["prompt"],
            sample_idx=metadata["sample_idx"],
            metadata={"prompt_group_id": metadata["prompt_group_id"]},
        )
