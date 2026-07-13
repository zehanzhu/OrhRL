# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import sys
import os
import logging
import inspect

# Configure unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

import hydra
import ray
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.engine_workers import ActorRolloutRefWorker

from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from orchrl.trainer.specialization_mode import (
    ROLE_SHARING,
    ROLE_SPECIFIC,
    validate_specialization_mode,
)
from orchrl.trainer.v1_tq_adapter import (
    close_transfer_queue_for_v1,
    ensure_top_level_transfer_queue_config,
    initialize_transfer_queue_for_v1,
    is_v1_tq_backend,
)
from orchrl.utils.clean_up import cleanup_ray_runtime, install_cleanup_hooks
from orchrl.utils.output_paths import prepare_training_output_dirs
from orchrl.utils.ray_utils import init_ray_with_temp_dirs
from orchrl.utils.served_model_name import resolve_policy_server_name

install_cleanup_hooks()


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
        # V1 treats None as "compute colocated rewards now"; [] means external
        # rewards have already been supplied through TransferQueue rm_scores.
        self.reward_loop_workers = []
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
        if is_v1_tq_backend(config):
            ensure_top_level_transfer_queue_config(config)
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


def train_multi_agents(config):
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    _patch_verl_reward_loop_for_external_mas(config)
    v1_tq_enabled = is_v1_tq_backend(config)
    if v1_tq_enabled:
        initialize_transfer_queue_for_v1(config)

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

    role_worker_mapping = {
        Role.ActorRolloutRef: ray.remote(max_concurrency=2048)(ActorRolloutRefWorker),
    }

    managers = _build_policy_resource_pool_managers(config)
    
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

    try:
        trainer.fit()
    finally:
        if v1_tq_enabled:
            close_transfer_queue_for_v1()


def _build_policy_resource_pool_managers(config):
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
    for model_key in config.models.keys():
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {global_pool_id: [gpus_per_policy] * policy_nnodes}
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        manager_kwargs = {
            "resource_pool_spec": resource_pool_spec,
            "mapping": mapping,
        }
        if _supports_keyword(ResourcePoolManager, "bundle_resources"):
            manager_kwargs["bundle_resources"] = (
                {global_pool_id: bundle_resources} if bundle_resources else {}
            )
        if _supports_keyword(ResourcePoolManager, "n_gpus_per_node"):
            manager_kwargs["n_gpus_per_node"] = n_gpus_per_node

        manager = ResourcePoolManager(**manager_kwargs)
        managers.append(manager)

    return managers


def _supports_keyword(callable_obj, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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
