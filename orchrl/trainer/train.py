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
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

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
        
        # Create and execute remote trainer
        def make_trainer_remote():
            num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1)) 
            return ray.remote(num_cpus=num_cpus)(train_multi_agents)

        multiagent_training_engine = make_trainer_remote()
        ray.get(multiagent_training_engine.remote(config))
    finally:
        cleanup_ray_runtime()

def train_multi_agents(config):
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

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

    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1)
    nnodes = getattr(config.resource, 'nnodes', 1)
    OmegaConf.to_container(config, resolve=True)
    #pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    tokenizer_dict = {}
    model_num = 0

    for model_key, model_config in config.models.items():
        model_num += 1
        model_path = model_config.path
        model_name = model_config.name
        
        print(f"Processing model: {model_name} at path: {model_path}")
        
        local_path = copy_local_path_from_hdfs(model_path)
        
        trust_remote_code = getattr(model_config, 'trust_remote_code', False)
        if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
            trust_remote_code = config.resource.trust_remote_code
        
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer_dict[model_name] = tokenizer

    n_gpus_per_model = n_gpus_per_node // model_num
    print(f"n_gpus_per_model: {n_gpus_per_model}")
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }
    
    managers = []
    for model_key, model_config in config.models.items():
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {global_pool_id: [n_gpus_per_model] * nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        #print(f"Creating resource pool for {model_key}: {resource_pool_spec}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        resource_pool_manager.create_resource_pool()
        managers.append(resource_pool_manager)
    
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
