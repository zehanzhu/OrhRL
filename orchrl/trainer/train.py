# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import sys
import os
import logging

# Configure unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

import hydra
import ray
from omegaconf import OmegaConf, DictConfig, open_dict
from verl.single_controller.ray import RayWorkerGroup

from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
)
from orchrl.utils.clean_up import cleanup_ray_runtime, install_cleanup_hooks
from orchrl.utils.output_paths import prepare_training_output_dirs
from orchrl.utils.ray_utils import init_ray_with_temp_dirs
from orchrl.utils.served_model_name import resolve_policy_server_name

install_cleanup_hooks()


def _ray_remote_actor_worker(worker_cls):
    return ray.remote(max_concurrency=2048)(worker_cls)


def _ray_remote_critic_worker(worker_cls):
    return ray.remote(worker_cls)


def _get_policy_strategy_and_reference_requirement(ppo_config: DictConfig) -> tuple[str, bool]:
    actor_cfg = getattr(getattr(ppo_config, "actor_rollout_ref", None), "actor", None)
    algorithm_cfg = getattr(ppo_config, "algorithm", None)
    strategy = str(getattr(actor_cfg, "strategy", "fsdp"))
    needs_reference_policy = bool(
        getattr(actor_cfg, "use_kl_loss", False)
        or getattr(algorithm_cfg, "use_kl_in_reward", False)
    )
    return strategy, needs_reference_policy


def _get_critic_strategy_and_enablement(ppo_config: DictConfig) -> tuple[str, bool]:
    critic_cfg = getattr(ppo_config, "critic", None)
    strategy = str(getattr(critic_cfg, "strategy", "fsdp"))
    critic_enabled = bool(getattr(critic_cfg, "enable", False))
    return strategy, critic_enabled


def _get_representative_policy_ppo_config(config: DictConfig) -> DictConfig:
    models_cfg = getattr(config, "models", None)
    if models_cfg:
        ppo_configs = []
        for model_key, model_cfg in models_cfg.items():
            ppo_config = getattr(model_cfg, "ppo_trainer_config", None)
            if ppo_config is None:
                continue
            strategy, needs_reference_policy = _get_policy_strategy_and_reference_requirement(
                ppo_config
            )
            critic_strategy, critic_enabled = _get_critic_strategy_and_enablement(ppo_config)
            ppo_configs.append(
                (
                    model_key,
                    ppo_config,
                    strategy,
                    needs_reference_policy,
                    critic_strategy,
                    critic_enabled,
                )
            )

        if ppo_configs:
            (
                representative_key,
                representative_ppo_config,
                representative_strategy,
                representative_needs_ref,
                representative_critic_strategy,
                representative_critic_enabled,
            ) = (
                ppo_configs[0]
            )
            for (
                model_key,
                _,
                strategy,
                needs_reference_policy,
                critic_strategy,
                critic_enabled,
            ) in ppo_configs[1:]:
                if strategy != representative_strategy:
                    raise ValueError(
                        "OrchRL policy training does not support mixed backend strategies "
                        f"across models: {representative_key}={representative_strategy}, "
                        f"{model_key}={strategy}"
                    )
                if needs_reference_policy != representative_needs_ref:
                    raise ValueError(
                        "OrchRL policy training does not support mixed reference-policy "
                        "requirements across models: "
                        f"{representative_key}={representative_needs_ref}, "
                        f"{model_key}={needs_reference_policy}"
                    )
                if critic_strategy != representative_critic_strategy:
                    raise ValueError(
                        "OrchRL policy training does not support mixed critic backend "
                        f"strategies across models: {representative_key}="
                        f"{representative_critic_strategy}, {model_key}={critic_strategy}"
                    )
                if critic_enabled != representative_critic_enabled:
                    raise ValueError(
                        "OrchRL policy training does not support mixed critic enablement "
                        f"across models: {representative_key}={representative_critic_enabled}, "
                        f"{model_key}={critic_enabled}"
                    )
            return representative_ppo_config
    return config


def _need_reference_policy(config: DictConfig) -> bool:
    ppo_config = _get_representative_policy_ppo_config(config)
    _, needs_reference_policy = _get_policy_strategy_and_reference_requirement(ppo_config)
    return needs_reference_policy


def _critic_enabled(config: DictConfig) -> bool:
    ppo_config = _get_representative_policy_ppo_config(config)
    _, critic_enabled = _get_critic_strategy_and_enablement(ppo_config)
    return critic_enabled


def _get_model_ppo_config(model_cfg: DictConfig) -> DictConfig:
    ppo_config = getattr(model_cfg, "ppo_trainer_config", None)
    if ppo_config is not None:
        return ppo_config
    return model_cfg


def _normalize_megatron_router_replay_config(config: DictConfig) -> None:
    models_cfg = getattr(config, "models", None)
    if not models_cfg:
        return

    for model_cfg in models_cfg.values():
        ppo_config = _get_model_ppo_config(model_cfg)
        actor_rollout_ref = getattr(ppo_config, "actor_rollout_ref", None)
        actor_cfg = getattr(actor_rollout_ref, "actor", None) if actor_rollout_ref is not None else None
        if actor_cfg is None:
            continue
        if str(getattr(actor_cfg, "strategy", "fsdp")) != "megatron":
            continue
        if getattr(actor_cfg, "router_replay", None) is not None:
            continue

        megatron_cfg = getattr(actor_cfg, "megatron", None)
        if megatron_cfg is None:
            continue

        router_replay_cfg = getattr(megatron_cfg, "router_replay", None)
        if router_replay_cfg is None:
            continue

        OmegaConf.set_struct(actor_cfg, False)
        actor_cfg.router_replay = OmegaConf.create(
            OmegaConf.to_container(router_replay_cfg, resolve=True)
        )
        OmegaConf.set_struct(actor_cfg, True)


def _normalize_megatron_train_configs(config: DictConfig) -> None:
    models_cfg = getattr(config, "models", None)
    if not models_cfg:
        return

    actor_forbidden_keys = {"fsdp_config", "grad_clip", "ulysses_sequence_parallel_size"}
    actor_optim_forbidden_keys = {"optimizer_impl", "min_lr_ratio", "num_cycles", "warmup_style"}
    ref_forbidden_keys = {"fsdp_config", "ulysses_sequence_parallel_size"}

    for model_cfg in models_cfg.values():
        ppo_config = _get_model_ppo_config(model_cfg)
        actor_rollout_ref = getattr(ppo_config, "actor_rollout_ref", None)
        if actor_rollout_ref is None:
            continue

        actor_cfg = getattr(actor_rollout_ref, "actor", None)
        if actor_cfg is None or str(getattr(actor_cfg, "strategy", "fsdp")) != "megatron":
            continue

        model_cfg = getattr(actor_rollout_ref, "model", None)
        if model_cfg is not None:
            with open_dict(model_cfg):
                if "mtp" not in model_cfg:
                    model_cfg.mtp = OmegaConf.create({"enable": False})

        with open_dict(actor_cfg):
            for key in actor_forbidden_keys:
                actor_cfg.pop(key, None)

            actor_optim_cfg = getattr(actor_cfg, "optim", None)
            if actor_optim_cfg is not None:
                with open_dict(actor_optim_cfg):
                    for key in actor_optim_forbidden_keys:
                        actor_optim_cfg.pop(key, None)

        ref_cfg = getattr(actor_rollout_ref, "ref", None)
        if ref_cfg is None:
            continue

        with open_dict(ref_cfg):
            for key in ref_forbidden_keys:
                ref_cfg.pop(key, None)

        rollout_cfg = getattr(actor_rollout_ref, "rollout", None)
        if rollout_cfg is not None:
            with open_dict(rollout_cfg):
                if getattr(rollout_cfg, "log_prob_micro_batch_size", None) is None:
                    rollout_cfg.log_prob_micro_batch_size = getattr(
                        ref_cfg, "log_prob_micro_batch_size", None
                    )
                if getattr(rollout_cfg, "log_prob_micro_batch_size_per_gpu", None) is None:
                    rollout_cfg.log_prob_micro_batch_size_per_gpu = getattr(
                        ref_cfg, "log_prob_micro_batch_size_per_gpu", None
                    )
                if getattr(rollout_cfg, "log_prob_use_dynamic_bsz", None) is None:
                    rollout_cfg.log_prob_use_dynamic_bsz = getattr(
                        ref_cfg, "log_prob_use_dynamic_bsz", False
                    )
                if getattr(rollout_cfg, "log_prob_max_token_len_per_gpu", None) is None:
                    rollout_cfg.log_prob_max_token_len_per_gpu = getattr(
                        ref_cfg, "log_prob_max_token_len_per_gpu", None
                    )


def _is_external_mas_reward_flow(config: DictConfig) -> bool:
    if str(getattr(config, "workflow_type", "")) == "external_mas":
        return True

    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        return False

    if str(getattr(training_cfg, "rollout_source", "")) == "mate":
        return True

    mate_cfg = getattr(training_cfg, "mate", None)
    reward_cfg = getattr(mate_cfg, "reward", None) if mate_cfg is not None else None
    return bool(getattr(reward_cfg, "provider", None))


def _patch_verl_reward_loop_for_external_mas(config: DictConfig) -> None:
    if not _is_external_mas_reward_flow(config):
        return

    from verl.experimental.reward_loop import reward_loop as reward_loop_module

    if getattr(reward_loop_module, "_orchrl_external_mas_reward_loop_patched", False):
        return

    original_init_reward_loop_workers = (
        reward_loop_module.RewardLoopManager._init_reward_loop_workers
    )

    def _init_reward_loop_workers(self):
        # OrchRL external MAS computes rewards via the MATE reward provider,
        # so VERL generic reward-loop workers are redundant here and clash
        # across multiple policy trainers because they use fixed Ray actor names.
        self.reward_loop_workers = None
        return

    reward_loop_module.RewardLoopManager._orchrl_original_init_reward_loop_workers = (
        original_init_reward_loop_workers
    )
    reward_loop_module.RewardLoopManager._init_reward_loop_workers = _init_reward_loop_workers
    reward_loop_module._orchrl_external_mas_reward_loop_patched = True


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):   
    output_paths = prepare_training_output_dirs(config)
    if output_paths:
        logging.info(
            "Prepared training output directories: %s",
            ", ".join(f"{key}={value}" for key, value in sorted(output_paths.items())),
        )
    OmegaConf.to_yaml(config)
    run_ppo(config)


def run_ppo(config):
    try:
        # Initialize Ray with temporary directories
        init_ray_with_temp_dirs(config)

        if _run_training_in_driver(config):
            train_multi_agents(config)
        else:
            multiagent_training_engine = _make_trainer_remote(config)
            ray.get(multiagent_training_engine.remote(config))
    finally:
        cleanup_ray_runtime()


def _run_training_in_driver(config) -> bool:
    resource_cfg = getattr(config, "resource", None)
    if resource_cfg is None:
        return False
    return bool(getattr(resource_cfg, "run_training_in_driver", False))


def _make_trainer_remote(config):
    resource_cfg = getattr(config, "resource", None)
    configured_num_cpus = getattr(resource_cfg, "trainer_remote_num_cpus", None)
    if configured_num_cpus is None:
        configured_num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1))

    trainer_remote_options = {"num_cpus": int(configured_num_cpus)}
    configured_resources = getattr(resource_cfg, "trainer_remote_resources", None)
    if configured_resources:
        trainer_remote_options["resources"] = OmegaConf.to_container(
            configured_resources, resolve=True
        )

    return ray.remote(**trainer_remote_options)(train_multi_agents)

def _build_role_worker_mapping(config: DictConfig):
    from verl.trainer.ppo.ray_trainer import Role

    ppo_config = _get_representative_policy_ppo_config(config)
    strategy, _ = _get_policy_strategy_and_reference_requirement(ppo_config)

    if strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

        actor_role = Role.ActorRolloutRef
        worker_cls = AsyncActorRolloutRefWorker
    elif strategy == "megatron":
        from orchrl.workers.megatron_workers import AsyncActorRolloutRefWorker

        actor_role = Role.ActorRollout
        worker_cls = AsyncActorRolloutRefWorker
    else:
        raise ValueError(f"Unsupported actor strategy for OrchRL policy training: {strategy}")

    worker = _ray_remote_actor_worker(worker_cls)
    role_worker_mapping = {actor_role: worker}

    if _critic_enabled(config):
        critic_strategy, _ = _get_critic_strategy_and_enablement(ppo_config)
        if critic_strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif critic_strategy == "megatron":
            from orchrl.workers.megatron_workers import CriticWorker
        else:
            raise ValueError(
                f"Unsupported critic strategy for OrchRL policy training: {critic_strategy}"
            )
        role_worker_mapping[Role.Critic] = _ray_remote_critic_worker(CriticWorker)

    if _need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = worker

    return role_worker_mapping, actor_role


def train_multi_agents(config):
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    _patch_verl_reward_loop_for_external_mas(config)
    _normalize_megatron_router_replay_config(config)
    _normalize_megatron_train_configs(config)

    agent_policy_mapping = {}
    for agent_config in config.agent_policy_configs.agent_configs.values():
        agent_policy_mapping[agent_config.name] = agent_config.policy_name
        print(f"Agent mapping: {agent_config.name} -> {agent_config.policy_name}")
    num_base_models = len(config.base_models) if hasattr(config, 'base_models') else 0
    num_models = len(config.models) if hasattr(config, 'models') else 0
    specialization = validate_specialization_mode(config.specialization)

    if num_base_models != num_models:
        error_msg = (
            f"Configuration error: Number of base_models ({num_base_models}) does not match "
        )
        print("="*80)
        print(f"ERROR: {error_msg}")
        print("="*80)
        raise ValueError(error_msg)

    if specialization == ROLE_SHARING:
        if num_models != 1:
            raise ValueError(
                f"For specialization={specialization}', expected exactly 1 model, but got {num_models}"
            )

    if (
        specialization == ROLE_SPECIFIC
        and num_base_models == 1
        and num_models == 1
        and len(agent_policy_mapping) > 1
    ):
        config = _expand_single_base_model_role_specific(
            config,
            agent_policy_mapping,
        )

    _validate_unique_role_specific_served_model_names(config)

    OmegaConf.to_container(config, resolve=True)
    OmegaConf.resolve(config)
    
    tokenizer_dict = {}

    for model_key, model_config in config.models.items():
        model_path = model_config.path
        model_name = model_config.name
        
        print(f"Processing model: {model_name} at path: {model_path}")
        
        local_path = copy_local_path_from_hdfs(model_path)
        
        trust_remote_code = getattr(model_config, 'trust_remote_code', False)
        if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
            trust_remote_code = config.resource.trust_remote_code
        
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_dict[model_name] = tokenizer

    role_worker_mapping, actor_role = _build_role_worker_mapping(config)
    managers = _build_policy_resource_pool_managers(config, actor_role=actor_role)
    
    trainer = MultiAgentsPPOTrainer(
        config=config,
        tokenizer_dict=tokenizer_dict,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=managers,
        ray_worker_group_cls=RayWorkerGroup,
        agent_policy_mapping=agent_policy_mapping,
    )
    trainer.init_workers()
    
    trainer.init_mate_rollout_runtime()
    
    trainer.fit()


def _build_policy_resource_pool_managers(config, actor_role=None):
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    resource_cfg = getattr(config, "resource", None)
    n_gpus_per_node = int(getattr(resource_cfg, "n_gpus_per_node", 1))
    configured_gpus_per_policy = getattr(resource_cfg, "gpus_per_policy", None)
    gpus_per_policy = int(configured_gpus_per_policy or n_gpus_per_node)
    policy_nnodes = int(getattr(resource_cfg, "policy_nnodes", 1))
    bundle_resources_cfg = getattr(resource_cfg, "policy_bundle_resources", None)
    bundle_resources = (
        OmegaConf.to_container(bundle_resources_cfg, resolve=True)
        if bundle_resources_cfg
        else None
    )

    managers = []
    for model_key, model_cfg in config.models.items():
        ppo_config = _get_model_ppo_config(model_cfg)
        _, model_needs_reference_policy = _get_policy_strategy_and_reference_requirement(
            ppo_config
        )
        _, model_critic_enabled = _get_critic_strategy_and_enablement(ppo_config)
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {global_pool_id: [gpus_per_policy] * policy_nnodes}
        mapping = {(actor_role or Role.ActorRolloutRef): global_pool_id}
        if model_critic_enabled:
            mapping[Role.Critic] = global_pool_id
        if model_needs_reference_policy:
            mapping[Role.RefPolicy] = global_pool_id
        manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
            bundle_resources={global_pool_id: bundle_resources} if bundle_resources else {},
            n_gpus_per_node=n_gpus_per_node,
        )
        managers.append(manager)

    return managers


def _expand_single_base_model_role_specific(
    config: DictConfig,
    agent_policy_mapping: dict[str, str],
) -> DictConfig:
    num_base_models = len(config.base_models) if hasattr(config, "base_models") else 0
    num_models = len(config.models) if hasattr(config, "models") else 0
    if (
        validate_specialization_mode(config.specialization) != ROLE_SPECIFIC
        or num_base_models != 1
        or num_models != 1
        or len(agent_policy_mapping) <= 1
    ):
        return config

    from copy import deepcopy

    print("=" * 80)
    print("SPECIAL MODE: specialization='role_specific' with single base_model detected")
    print(f"Replicating configurations to match {len(agent_policy_mapping)} agents...")

    base_model_config = config.base_models[list(config.base_models.keys())[0]]
    original_model_config = config.models[list(config.models.keys())[0]]
    base_model_path = str(base_model_config.path)

    new_base_models_dict = {}
    new_models_dict = {}
    for index, policy_name in enumerate(agent_policy_mapping.values()):
        base_model_copy = deepcopy(base_model_config)
        model_copy = deepcopy(original_model_config)

        base_model_copy.name = policy_name
        model_copy.path = base_model_path
        model_copy.name = policy_name

        ppo_cfg = getattr(model_copy, "ppo_trainer_config", None)
        if ppo_cfg is not None:
            actor_rollout_ref = getattr(ppo_cfg, "actor_rollout_ref", None)
            if actor_rollout_ref is not None:
                actor_model_cfg = getattr(actor_rollout_ref, "model", None)
                if actor_model_cfg is not None:
                    actor_model_cfg.path = base_model_path

                    override_config = getattr(actor_model_cfg, "override_config", None)
                    if override_config is None:
                        actor_model_cfg.override_config = OmegaConf.create(
                            {"model_id": policy_name}
                        )
                    else:
                        OmegaConf.set_struct(actor_model_cfg, False)
                        actor_model_cfg.override_config["model_id"] = policy_name
                        OmegaConf.set_struct(actor_model_cfg, True)

        new_base_models_dict[f"policy_{index}"] = base_model_copy
        new_models_dict[f"model_{index}"] = model_copy

    OmegaConf.set_struct(config, False)
    config.base_models = OmegaConf.create(new_base_models_dict)
    config.models = OmegaConf.create(new_models_dict)
    OmegaConf.set_struct(config, True)
    return config


def _validate_unique_role_specific_served_model_names(config: DictConfig) -> None:
    if validate_specialization_mode(config.specialization) != ROLE_SPECIFIC:
        return

    served_name_to_policies: dict[str, list[str]] = {}
    for model_config in config.models.values():
        policy_name = str(model_config.name)
        served_name = resolve_policy_server_name(
            policy_name,
            getattr(model_config, "ppo_trainer_config", None),
        )
        served_name_to_policies.setdefault(str(served_name), []).append(policy_name)

    collisions = {
        served_name: policies
        for served_name, policies in served_name_to_policies.items()
        if len(policies) > 1
    }
    if not collisions:
        return

    collision_details = ", ".join(
        f"{served_name} <- {policies}"
        for served_name, policies in sorted(collisions.items())
    )
    raise ValueError(
        "Duplicate served model names detected for role_specific training: "
        f"{collision_details}. Configure unique "
        "actor_rollout_ref.rollout.prometheus.served_model_name values."
    )


if __name__ == "__main__":
    main()
