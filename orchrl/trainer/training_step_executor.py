from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import torch
import transfer_queue as tq

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
)
from verl.utils.torch_functional import pad_sequence_to_length

from orchrl.trainer.mate.dataproto_adapter import (
    episodes_to_policy_batches,
    tree_episodes_to_decision_point_batches,
)
from orchrl.trainer.mate.mas_metrics import build_mas_metrics
from orchrl.trainer.mate.trajectory_export import maybe_export_prompt_trajectories
from orchrl.trainer.v1_tq_adapter import prepare_dataproto_for_v1_transfer_queue
from orchrl.utils.performance import colorful_print, simple_timer

@dataclass
class TrainingStepResult:
    batch_per_trainer: dict[str, DataProto]
    present_policy_names: list[str]
    missing_policy_names: list[str]
    metrics: dict[str, object]
    timing_raw: dict[str, float]


class TrainingStepExecutor:
    def __init__(
        self,
        *,
        config,
        policy_trainer_registry,
        mate_runtime,
        agent_policy_mapping,
        agent_untrained,
    ):
        self.config = config
        self.policy_trainer_registry = policy_trainer_registry
        self.mate_runtime = mate_runtime
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.agent_untrained = agent_untrained or []
        self._last_mate_rollout_accounting = {
            "expected_job_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "failures": [],
        }
        self._last_mate_episodes = []

    def collect_mate_episodes(self, step_idx: int):
        checkpoint_managers = self.policy_trainer_registry.get_checkpoint_managers()
        for checkpoint_manager in checkpoint_managers.values():
            checkpoint_manager.update_weights()
        try:
            return asyncio.run(
                self.mate_runtime.mate_rollout_adapter.collect_step_rollouts(
                    step_idx=step_idx
                )
            )
        finally:
            for checkpoint_manager in checkpoint_managers.values():
                checkpoint_manager.sleep_replicas()

    def collect_mate_step_batches(self, step_idx: int):
        rollout_result = self.collect_mate_episodes(step_idx=step_idx)
        episodes = self._extract_rollout_episodes(rollout_result)
        self._last_mate_episodes = episodes
        self._last_mate_rollout_accounting = self._build_rollout_accounting(
            rollout_result,
            episodes,
        )
        self._enforce_rollout_failure_policy(
            expected_job_count=self._last_mate_rollout_accounting["expected_job_count"],
            failed_count=self._last_mate_rollout_accounting["failed_count"],
            failures=self._last_mate_rollout_accounting["failures"],
        )
        maybe_export_prompt_trajectories(
            episodes=episodes,
            step_idx=step_idx,
            rollout_mode=self.mate_rollout_mode(),
            mate_config=self.mate_runtime.mate_config or {},
        )

        max_prompt_length = getattr(
            self.config.training,
            "max_prompt_length",
            None,
        )
        max_response_length = getattr(
            self.config.training,
            "max_response_length",
            None,
        )
        ppo_trainer_dict = self.policy_trainer_registry.ppo_trainer_dict
        if max_prompt_length is None:
            max_prompt_length = next(iter(ppo_trainer_dict.values())).config.data.max_prompt_length
        if max_response_length is None:
            max_response_length = next(iter(ppo_trainer_dict.values())).config.data.max_response_length

        role_names = (
            list(self.agent_policy_mapping.keys())
            if self.agent_policy_mapping
            else list(self.mate_runtime.mate_config["role_policy_mapping"].keys())
        )
        adapter_fn = (
            tree_episodes_to_decision_point_batches
            if self.mate_rollout_mode() == "tree"
            else episodes_to_policy_batches
        )
        return adapter_fn(
            episodes=episodes,
            tokenizer_dict=self.policy_trainer_registry.get_tokenizers(),
            role_policy_mapping=self.mate_runtime.mate_config["role_policy_mapping"],
            role_index_mapping={role: idx for idx, role in enumerate(role_names)},
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            credit_assignment=self.mate_runtime.mate_config.get(
                "credit_assignment",
                self.mate_runtime.mate_config.get("reward", {}).get(
                    "credit_assignment",
                    "all_turns",
                ),
            ),
        )

    @staticmethod
    def build_mas_train_metrics(*, episodes, expected_job_count: int, failed_count: int):
        return build_mas_metrics(
            episodes=episodes,
            expected_sample_count=expected_job_count,
            failed_count=failed_count,
            prefix="mas/train",
        )

    @staticmethod
    def _extract_rollout_episodes(rollout_result):
        if hasattr(rollout_result, "episodes"):
            return list(getattr(rollout_result, "episodes") or [])
        return list(rollout_result or [])

    @staticmethod
    def _build_rollout_accounting(rollout_result, episodes):
        if hasattr(rollout_result, "expected_job_count"):
            return {
                "expected_job_count": int(getattr(rollout_result, "expected_job_count", 0)),
                "success_count": int(getattr(rollout_result, "success_count", len(episodes))),
                "failed_count": int(getattr(rollout_result, "failed_count", 0)),
                "failures": list(getattr(rollout_result, "failures", []) or []),
            }
        return {
            "expected_job_count": len(episodes),
            "success_count": len(episodes),
            "failed_count": 0,
            "failures": [],
        }

    def _failure_policy_config(self) -> dict[str, object]:
        mate_config = getattr(self.mate_runtime, "mate_config", None) or {}
        failure_policy = dict(mate_config.get("failure_policy") or {})
        if not failure_policy:
            raw_training = getattr(self.config, "training", None)
            raw_mate = getattr(raw_training, "mate", None) if raw_training is not None else None
            raw_failure_policy = getattr(raw_mate, "failure_policy", None) if raw_mate is not None else None
            if raw_failure_policy is not None:
                failure_policy = dict(raw_failure_policy)
        resolved = {
            "mode": "threshold",
            "max_failed_rate": 0.8,
        }
        resolved.update(failure_policy)
        return resolved

    def _enforce_rollout_failure_policy(self, *, expected_job_count, failed_count, failures):
        if failed_count <= 0:
            return

        failure_policy = self._failure_policy_config()
        mode = str(failure_policy.get("mode", "threshold"))
        failed_rate = failed_count / expected_job_count if expected_job_count > 0 else 0.0
        sample_failure = failures[:1] if isinstance(failures, list) else []

        if mode == "warn":
            colorful_print(
                (
                    "Warning: rollout job failures detected: "
                    f"failed={failed_count}/{expected_job_count}"
                ),
                "yellow",
            )
            return

        if mode == "error":
            raise RuntimeError(
                "MATE rollout job failures detected: "
                f"failed={failed_count}/{expected_job_count}; sample={sample_failure}"
            )

        if mode != "threshold":
            raise ValueError("mate.failure_policy.mode must be one of warn/threshold/error")

        max_failed_rate = float(failure_policy.get("max_failed_rate", 0.8))
        if failed_rate > max_failed_rate:
            raise RuntimeError(
                "MATE rollout job failures exceeded threshold: "
                f"failed={failed_count}/{expected_job_count} "
                f"(rate={failed_rate:.4f}, max_failed_rate={max_failed_rate}); "
                f"sample={sample_failure}"
            )

    @staticmethod
    def build_rollout_failure_metrics(accounting):
        expected_job_count = int(accounting.get("expected_job_count", 0))
        failed_count = int(accounting.get("failed_count", 0))
        success_count = int(accounting.get("success_count", 0))
        failed_rate = failed_count / expected_job_count if expected_job_count > 0 else 0.0
        return {
            "training/rollout_expected_job_count": expected_job_count,
            "training/rollout_success_job_count": success_count,
            "training/rollout_failed_job_count": failed_count,
            "training/rollout_failed_job_rate": float(failed_rate),
        }

    def mate_rollout_mode(self) -> str:
        if not self.mate_runtime.mate_config:
            return "parallel"
        return str(self.mate_runtime.mate_config.get("rollout_mode", "parallel"))

    @staticmethod
    def resolve_batch_group_keys(batch):
        uids = [str(uid) for uid in batch.non_tensor_batch["uid"]]
        group_ids = batch.non_tensor_batch.get("group_id")
        if group_ids is None:
            return uids, uids, False
        group_keys = [
            str(group_id) if group_id is not None else uid
            for uid, group_id in zip(uids, group_ids)
        ]
        uses_group_id = any(group_id is not None for group_id in group_ids)
        if uses_group_id:
            return uids, group_keys, True
        return uids, uids, False

    def resolve_mate_policy_batches(self, gen_batch_output_per_policy):
        expected_policy_names = list(self.policy_trainer_registry.ppo_trainer_dict.keys())
        actual_policy_names = list(gen_batch_output_per_policy.keys())
        if not actual_policy_names:
            raise RuntimeError(
                "MATE rollout produced no policy batches; all rollout episodes likely failed before returning trajectories."
            )

        present_policy_names = [
            model_name
            for model_name in expected_policy_names
            if model_name in gen_batch_output_per_policy
        ]
        missing_policy_names = [
            model_name
            for model_name in expected_policy_names
            if model_name not in gen_batch_output_per_policy
        ]
        return present_policy_names, missing_policy_names

    @staticmethod
    def build_mate_policy_presence_metrics(present_policy_names, missing_policy_names):
        return {
            "training/present_policy_count": len(present_policy_names),
            "training/skipped_policy_count": len(missing_policy_names),
        }

    @staticmethod
    def has_real_batch(batch) -> bool:
        return batch is not None and getattr(batch, "batch", None) is not None

    def execute_training_step(self, step_idx: int):
        batch_per_trainer: dict[str, DataProto] = {}
        present_policy_names = []
        missing_policy_names = []
        metrics = {}
        timing_raw = {}

        with simple_timer("collect_trajectory", timing_raw):
            gen_batch_output_per_policy = self.collect_mate_step_batches(step_idx=step_idx)
            metrics.update(
                self.build_rollout_failure_metrics(
                    self._last_mate_rollout_accounting,
                )
            )
            metrics.update(
                self.build_mas_train_metrics(
                    episodes=getattr(self, "_last_mate_episodes", []),
                    expected_job_count=self._last_mate_rollout_accounting[
                        "expected_job_count"
                    ],
                    failed_count=self._last_mate_rollout_accounting["failed_count"],
                )
            )
            present_policy_names, missing_policy_names = self.resolve_mate_policy_batches(
                gen_batch_output_per_policy
            )
            metrics.update(
                self.build_mate_policy_presence_metrics(
                    present_policy_names=present_policy_names,
                    missing_policy_names=missing_policy_names,
                )
            )
            if missing_policy_names:
                colorful_print(
                    (
                        "Warning: MATE rollout missing policy batches for "
                        f"{missing_policy_names}; skipping updates for these policies this step. "
                        f"Available policies: {present_policy_names}"
                    ),
                    "yellow",
                )

            for model_name in present_policy_names:
                trainer = self.policy_trainer_registry.ppo_trainer_dict[model_name]
                dp_world_size = trainer.actor_rollout_wg.world_size
                batch_per_trainer_temp, _ = pad_dataproto_to_divisor(
                    gen_batch_output_per_policy[model_name],
                    dp_world_size,
                )
                existing_batch = batch_per_trainer.get(model_name)
                if not self.has_real_batch(existing_batch):
                    batch_per_trainer[model_name] = batch_per_trainer_temp
                else:
                    batch_per_trainer[model_name] = DataProto.concat(
                        [existing_batch, batch_per_trainer_temp]
                    )

        update_timing_raw = {}
        with simple_timer("update_parameters", update_timing_raw):
            for model_name in present_policy_names:
                trainer = self.policy_trainer_registry.ppo_trainer_dict[model_name]
                if model_name in batch_per_trainer and self.has_real_batch(
                    batch_per_trainer[model_name]
                ):
                    filter_ratio = getattr(trainer.config, "filter_ratio", 0.0)
                    filter_method = getattr(trainer.config, "filter_method", "uid")
                    batch_per_trainer[model_name] = self.filter_batch_by_existing_uid_groups(
                        batch_per_trainer[model_name],
                        filter_ratio=filter_ratio,
                        mode=filter_method,
                    )

            for model_name in present_policy_names:
                trainer = self.policy_trainer_registry.ppo_trainer_dict[model_name]
                local_timing_raw = {}
                updated_batch = self.update_parameters(
                    batch_per_trainer[model_name],
                    trainer,
                    local_timing_raw,
                )

                for key, value in local_timing_raw.items():
                    update_timing_raw[key] = max(update_timing_raw.get(key, 0), value)

                if updated_batch is not None:
                    batch_per_trainer[model_name] = updated_batch

        timing_raw.update(update_timing_raw)

        return TrainingStepResult(
            batch_per_trainer=batch_per_trainer,
            present_policy_names=present_policy_names,
            missing_policy_names=missing_policy_names,
            metrics=metrics,
            timing_raw=timing_raw,
        )

    def execute_v1_tq_training_step(self, step_idx: int):
        batch_per_trainer: dict[str, DataProto] = {}
        present_policy_names = []
        missing_policy_names = []
        metrics = {}
        timing_raw = {}

        with simple_timer("collect_trajectory", timing_raw):
            gen_batch_output_per_policy = self.collect_mate_step_batches(step_idx=step_idx)
            metrics.update(
                self.build_rollout_failure_metrics(
                    self._last_mate_rollout_accounting,
                )
            )
            metrics.update(
                self.build_mas_train_metrics(
                    episodes=getattr(self, "_last_mate_episodes", []),
                    expected_job_count=self._last_mate_rollout_accounting[
                        "expected_job_count"
                    ],
                    failed_count=self._last_mate_rollout_accounting["failed_count"],
                )
            )
            present_policy_names, missing_policy_names = self.resolve_mate_policy_batches(
                gen_batch_output_per_policy
            )
            metrics.update(
                self.build_mate_policy_presence_metrics(
                    present_policy_names=present_policy_names,
                    missing_policy_names=missing_policy_names,
                )
            )
            if missing_policy_names:
                colorful_print(
                    (
                        "Warning: MATE rollout missing policy batches for "
                        f"{missing_policy_names}; skipping updates for these policies this step. "
                        f"Available policies: {present_policy_names}"
                    ),
                    "yellow",
                )

            for model_name in present_policy_names:
                batch_per_trainer[model_name] = gen_batch_output_per_policy[model_name]

        update_timing_raw = {}
        with simple_timer("update_parameters", update_timing_raw):
            updated_policy_names = []
            for model_name in present_policy_names:
                trainer = self.policy_trainer_registry.ppo_trainer_dict[model_name]
                batch = batch_per_trainer.get(model_name)
                if not self.has_real_batch(batch):
                    continue

                batch = self.filter_untrained_agents(batch)
                if not self.has_real_batch(batch) or len(batch) == 0:
                    continue

                filter_ratio = getattr(trainer.config, "filter_ratio", 0.0)
                filter_method = getattr(trainer.config, "filter_method", "uid")
                batch = self.filter_batch_by_existing_uid_groups(
                    batch,
                    filter_ratio=filter_ratio,
                    mode=filter_method,
                )
                batch_per_trainer[model_name] = batch

                trainer.global_steps = int(step_idx) + 1
                local_metrics = {}
                local_timing_raw = {}
                batch_meta = None

                try:
                    if hasattr(trainer, "on_step_begin"):
                        trainer.on_step_begin()

                    sample_batch_size = prepare_dataproto_for_v1_transfer_queue(
                        data_proto=batch,
                        trainer=trainer,
                        global_steps=trainer.global_steps,
                    )
                    if sample_batch_size <= 0:
                        continue

                    batch_meta = trainer._step_once(
                        local_metrics,
                        local_timing_raw,
                        sample_batch_size,
                    )

                    if hasattr(trainer, "_compute_metrics"):
                        trainer._compute_metrics(
                            batch_meta,
                            local_metrics,
                            local_timing_raw,
                            global_steps=trainer.global_steps,
                            epoch=int(step_idx),
                        )
                    updated_policy_names.append(model_name)
                finally:
                    if batch_meta is not None:
                        tq.kv_clear(keys=batch_meta.keys, partition_id=batch_meta.partition_id)
                    if hasattr(trainer, "on_step_end"):
                        trainer.on_step_end()

                for key, value in local_timing_raw.items():
                    update_timing_raw[key] = max(update_timing_raw.get(key, 0), value)
                for key, value in local_metrics.items():
                    metrics[f"{model_name}_{key}"] = value

            present_policy_names = updated_policy_names

        timing_raw.update(update_timing_raw)

        return TrainingStepResult(
            batch_per_trainer=batch_per_trainer,
            present_policy_names=present_policy_names,
            missing_policy_names=missing_policy_names,
            metrics=metrics,
            timing_raw=timing_raw,
        )

    def filter_untrained_agents(self, batch):
        if not self.agent_untrained or len(self.agent_untrained) == 0:
            return batch
        if "agent_name" not in batch.non_tensor_batch:
            return batch

        agent_names = batch.non_tensor_batch["agent_name"]
        keep_indices = [
            i for i, name in enumerate(agent_names) if name not in self.agent_untrained
        ]

        if len(keep_indices) == len(agent_names):
            return batch

        colorful_print(
            (
                "Filtering training data: keeping "
                f"{len(keep_indices)}/{len(agent_names)} samples "
                f"(excluding agents: {self.agent_untrained})"
            ),
            "yellow",
        )
        if not keep_indices:
            colorful_print(
                "Warning: All samples filtered out, skipping parameter update",
                "red",
            )
            return batch.select_idxs([])
        return batch.select_idxs(keep_indices)

    def update_parameters(self, batch, ppo_trainer, timing_raw):
        if not hasattr(batch, "meta_info"):
            batch.meta_info = {}
        if "metrics" not in batch.meta_info:
            batch.meta_info["metrics"] = {}

        batch = self.filter_untrained_agents(batch)
        if len(batch) == 0:
            return batch

        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in batch.batch["prompts"]],
            batch_first=True,
            padding_value=ppo_trainer.tokenizer.pad_token_id,
        ).flip(dims=[1])
        responses_batch = torch.nn.utils.rnn.pad_sequence(
            [i for i in batch.batch["responses"]],
            batch_first=True,
            padding_value=ppo_trainer.tokenizer.pad_token_id,
        )
        if "response_mask" in batch.batch.keys():
            response_mask_batch = torch.nn.utils.rnn.pad_sequence(
                [i for i in batch.batch["response_mask"]],
                batch_first=True,
                padding_value=0,
            )
        else:
            response_mask_batch = None

        prompts_batch = pad_sequence_to_length(
            prompts_batch,
            ppo_trainer.config.data.max_prompt_length,
            ppo_trainer.tokenizer.pad_token_id,
            left_pad=True,
        )
        responses_batch = pad_sequence_to_length(
            responses_batch,
            ppo_trainer.config.data.max_response_length,
            ppo_trainer.tokenizer.pad_token_id,
            left_pad=False,
        )
        if response_mask_batch is not None:
            response_mask_batch = pad_sequence_to_length(
                response_mask_batch,
                ppo_trainer.config.data.max_response_length,
                0,
                left_pad=False,
            )
        input_ids_batch = torch.cat([prompts_batch, responses_batch], dim=1)
        attention_mask_batch = torch.where(
            input_ids_batch != ppo_trainer.tokenizer.pad_token_id,
            1,
            0,
        )
        position_ids = (
            torch.cumsum(attention_mask_batch, dim=1) - 1
        ) * attention_mask_batch

        batch.batch["prompts"] = prompts_batch
        batch.batch["responses"] = responses_batch
        batch.batch["input_ids"] = input_ids_batch
        batch.batch["attention_mask"] = attention_mask_batch
        batch.batch["position_ids"] = position_ids
        if response_mask_batch is None:
            response_mask_batch = (
                responses_batch != ppo_trainer.tokenizer.pad_token_id
            ).to(attention_mask_batch.dtype)
        batch.batch["response_mask"] = response_mask_batch
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"],
            dim=-1,
        ).tolist()

        reward_tensor = torch.zeros_like(
            batch.batch["responses"],
            dtype=torch.float32,
        )
        response_attention_mask = (
            responses_batch != ppo_trainer.tokenizer.pad_token_id
        )
        valid_token_counts = response_attention_mask.sum(dim=-1)
        valid_sequences_mask = valid_token_counts > 0

        if valid_sequences_mask.any():
            valid_batch_indices = torch.where(valid_sequences_mask)[0]
            last_valid_positions = valid_token_counts[valid_batch_indices] - 1
            rewards_tensor = torch.tensor(
                [batch.non_tensor_batch["reward"][i] for i in valid_batch_indices.tolist()],
                dtype=torch.float32,
                device=reward_tensor.device,
            )
            reward_tensor[valid_batch_indices, last_valid_positions] = rewards_tensor

        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        with simple_timer("old_log_prob", timing_raw):
            try:
                dp_world_size = ppo_trainer.actor_rollout_wg.world_size
            except Exception:
                dp_world_size = 1
            if dp_world_size > 1:
                batch, _ = pad_dataproto_to_divisor(batch, dp_world_size)
            old_log_prob = ppo_trainer.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

        need_ref_log_prob = (
            ppo_trainer.use_reference_policy
            or ppo_trainer.config.algorithm.use_kl_in_reward
        )
        if need_ref_log_prob:
            with simple_timer("ref", timing_raw):
                if not ppo_trainer.ref_in_actor:
                    ref_log_prob = ppo_trainer.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = ppo_trainer.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if ppo_trainer.use_critic:
            with simple_timer("values", timing_raw):
                values = ppo_trainer.critic_wg.compute_values(batch)
                batch = batch.union(values)

        if ppo_trainer.config.algorithm.use_kl_in_reward:
            with simple_timer("kl_penalty", timing_raw):
                if not hasattr(ppo_trainer, "kl_ctrl_in_reward"):
                    ppo_trainer.kl_ctrl_in_reward = core_algos.get_kl_controller(
                        ppo_trainer.config.algorithm.kl_ctrl
                    )
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=ppo_trainer.kl_ctrl_in_reward,
                    kl_penalty=ppo_trainer.config.algorithm.kl_penalty,
                )
                batch.meta_info["metrics"].update(kl_metrics)
                colorful_print(f"Applied KL penalty: {kl_metrics}", "cyan")

        with simple_timer("adv", timing_raw):
            norm_adv_by_std_in_grpo = ppo_trainer.config.algorithm.get(
                "norm_adv_by_std_in_grpo",
                True,
            )
            original_uids, group_keys, _ = self.resolve_batch_group_keys(batch)
            batch.non_tensor_batch["uid"] = np.array(group_keys, dtype=object)

            try:
                batch = compute_advantage(
                    batch,
                    adv_estimator=ppo_trainer.config.algorithm.adv_estimator,
                    gamma=ppo_trainer.config.algorithm.gamma,
                    lam=ppo_trainer.config.algorithm.lam,
                    num_repeat=ppo_trainer.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=ppo_trainer.config.algorithm,
                )
            finally:
                batch.non_tensor_batch["uid"] = np.array(original_uids, dtype=object)

        if ppo_trainer.use_critic:
            with simple_timer("update_critic", timing_raw):
                critic_output = ppo_trainer.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            batch.meta_info["metrics"].update(critic_output_metrics)

        with simple_timer("update_actor", timing_raw):
            batch.meta_info["multi_turn"] = (
                ppo_trainer.config.actor_rollout_ref.rollout.multi_turn.enable
            )
            actor_output = ppo_trainer.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])

            batch.meta_info["metrics"].update(actor_output_metrics)

        rollout_data_dir = ppo_trainer.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with simple_timer("dump_rollout_generations", timing_raw):
                reward_extra_infos_dict: dict[str, list] = {}
                inputs = ppo_trainer.tokenizer.batch_decode(
                    batch.batch["prompts"],
                    skip_special_tokens=True,
                )
                outputs = ppo_trainer.tokenizer.batch_decode(
                    batch.batch["responses"],
                    skip_special_tokens=True,
                )
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )
                ppo_trainer._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

            return batch

    def filter_batch_by_existing_uid_groups(
        self,
        data_proto,
        filter_ratio=0.0,
        mode="uid",
    ):
        from collections import defaultdict

        required_keys = ("uid", "prompt_group_id", "agent_idx")
        missing_keys = [
            key for key in required_keys if key not in data_proto.non_tensor_batch
        ]
        if missing_keys:
            raise ValueError(
                "MATE batch missing required metadata: " + ", ".join(missing_keys)
            )

        uids, group_keys, _ = self.resolve_batch_group_keys(data_proto)
        rewards = data_proto.non_tensor_batch.get("reward", [])

        uid_reward_groups = defaultdict(list)
        all_rewards = []

        data_proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        if group_keys is not None:
            data_proto.non_tensor_batch["group_id"] = np.array(group_keys, dtype=object)

        for i, group_key in enumerate(group_keys):
            if len(rewards) > 0:
                reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                uid_reward_groups[str(group_key)].append((i, reward_val))
                all_rewards.append(reward_val)

        def range_normalized_variance(rewards_in_group):
            rewards_in_group = np.asarray(rewards_in_group, dtype=float)
            rng = np.max(rewards_in_group) - np.min(rewards_in_group)
            if rng == 0:
                return 0.0
            return np.var(rewards_in_group, ddof=0) / (rng ** 2)

        sample_to_remove = set()
        if mode == "dapo":
            uids_to_remove = []
            for uid, samples in uid_reward_groups.items():
                rewards_in_group = [sample[1] for sample in samples]
                variance = range_normalized_variance(rewards_in_group)
                if variance == 0:
                    uids_to_remove.append(uid)
            for uid in uids_to_remove:
                for sample_idx, _ in uid_reward_groups.get(uid, []):
                    sample_to_remove.add(sample_idx)
        elif filter_ratio > 0:
            if mode == "std":
                uid_variances = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [sample[1] for sample in samples]
                        uid_variances[uid] = range_normalized_variance(rewards_in_group)
                    else:
                        uid_variances[uid] = 0.0

                total_uids = len(uid_variances)
                num_to_remove = int(total_uids * filter_ratio)
                if num_to_remove > 0:
                    sorted_uids = sorted(uid_variances.items(), key=lambda item: item[1])
                    for uid, _ in sorted_uids[:num_to_remove]:
                        for sample_idx, _ in uid_reward_groups.get(uid, []):
                            sample_to_remove.add(sample_idx)
            elif mode == "mean":
                uid_means = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [sample[1] for sample in samples]
                        uid_means[uid] = np.mean(rewards_in_group)
                    else:
                        uid_means[uid] = 0.0

                total_uids = len(uid_means)
                num_to_remove = int(total_uids * filter_ratio)
                if num_to_remove > 0:
                    sorted_uids = sorted(uid_means.items(), key=lambda item: item[1])
                    for uid, _ in sorted_uids[:num_to_remove]:
                        for sample_idx, _ in uid_reward_groups.get(uid, []):
                            sample_to_remove.add(sample_idx)
            elif mode == "uid":
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [sample[1] for sample in samples]
                        group_mean = np.mean(rewards_in_group)
                        samples_with_deviation = [
                            (sample[0], abs(sample[1] - group_mean))
                            for sample in samples
                        ]
                        samples_with_deviation.sort(
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        num_to_remove = int(
                            len(samples_with_deviation) * filter_ratio
                        )
                        for i in range(num_to_remove):
                            sample_idx, _ = samples_with_deviation[i]
                            sample_to_remove.add(sample_idx)

        if sample_to_remove:
            keep_indices = [
                i for i in range(len(data_proto)) if i not in sample_to_remove
            ]
            if len(keep_indices) < len(data_proto):
                data_proto = data_proto.select_idxs(keep_indices)

        if all_rewards:
            summary = {
                "total_samples": len(all_rewards),
                "mean_reward": float(np.mean(all_rewards)),
                "std_reward": float(np.std(all_rewards)),
                "filtered_samples": len(sample_to_remove) if filter_ratio > 0 else 0,
                "remain_samples": len(data_proto),
            }
            print(
                f"[DEBUG UID] Output: total_samples={len(all_rewards)}, mean_reward={np.mean(all_rewards):.4f}, remain_samples={len(data_proto)}, removed={len(sample_to_remove)}"
            )
            colorful_print(f"UID assignment summary: {summary}", "green")

        return data_proto
