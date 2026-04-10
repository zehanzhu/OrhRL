from __future__ import annotations

from omegaconf import OmegaConf

from orchrl.trainer.mate.config import validate_mate_config
from orchrl.trainer.mate.prompt_loader import MatePromptLoader
from orchrl.trainer.mate.reward_bridge import build_reward_provider
from orchrl.trainer.mate.rollout_adapter import MateRolloutAdapter


class MateRuntime:
    def __init__(self, *, config, agent_policy_mapping):
        self.config = config
        self.mate_config = validate_mate_config(
            config.training.mate,
            agent_policy_mapping,
        )
        self.monitor_pool_manager = None
        self.mate_train_prompt_loader = None
        self.mate_val_prompt_loader = None
        self.mate_reward_provider = None
        self.mate_rollout_adapter = None
        self.mate_val_rollout_adapter = None
        self.tokenizer_dict = {}
        self.server_handle_dict = {}
        self.policy_server_name_mapping = {}

    def initialize(
        self,
        *,
        tokenizer_dict,
        server_handle_dict,
        policy_server_name_mapping,
    ):
        self.tokenizer_dict = dict(tokenizer_dict)
        self.server_handle_dict = dict(server_handle_dict)
        self.policy_server_name_mapping = dict(policy_server_name_mapping)

        self.monitor_pool_manager = self._build_mate_monitor_pool_manager()
        self.monitor_pool_manager.start()
        self._init_mate_rollout_adapter()

    def _build_mate_monitor_pool_manager(self):
        if not self.mate_config:
            raise ValueError("mate_config is required to build MonitorPoolManager")

        monitor_pool_cfg = self.mate_config.get("monitor_pool", {})

        from orchrl.agent_trajectory_engine import (
            ChatRenderer,
            ModelMappingEntry,
            MonitorPoolManager,
            RolloutBackend,
        )
        from verl.experimental.agent_loop import AsyncLLMServerManager

        role_policy_mapping = self.mate_config["role_policy_mapping"]
        roles = list(self.mate_config.get("roles", role_policy_mapping.keys()))

        model_mapping = {}
        renderer_mapping = {}
        for role in roles:
            policy_name = role_policy_mapping[role]
            actual_model = self.policy_server_name_mapping.get(policy_name, policy_name)
            model_mapping[role] = ModelMappingEntry(
                actual_model=actual_model,
            )

            tokenizer = self.tokenizer_dict.get(policy_name)
            if tokenizer is None:
                raise ValueError(
                    f"No tokenizer configured for prompt-id MATE policy '{policy_name}'"
                )
            renderer_mapping[role] = ChatRenderer.from_tokenizer(
                tokenizer,
                model_name=actual_model,
            )

        async_server_manager_dict = {}
        for policy_name in sorted(set(role_policy_mapping.values())):
            raw_server_handles = self.server_handle_dict.get(policy_name)
            if isinstance(raw_server_handles, (list, tuple)):
                server_handles = list(raw_server_handles)
            elif raw_server_handles is None:
                server_handles = []
            else:
                server_handles = [raw_server_handles]

            if not server_handles:
                raise ValueError(
                    f"No server handles configured for prompt-id policy '{policy_name}'"
                )

            manager_config = OmegaConf.create({"policy_name": policy_name})
            async_server_manager_dict[policy_name] = AsyncLLMServerManager(
                manager_config,
                server_handles,
            )

        backend = RolloutBackend(
            policy_to_manager=async_server_manager_dict,
            policy_to_tokenizer=self.tokenizer_dict,
            policy_to_actual_model=self.policy_server_name_mapping,
        )

        return MonitorPoolManager(
            backend=backend,
            model_mapping=model_mapping,
            size=int(monitor_pool_cfg["size"]),
            host=str(monitor_pool_cfg["host"]),
            base_port=int(monitor_pool_cfg["base_port"]),
            acquire_timeout_sec=float(monitor_pool_cfg["acquire_timeout_sec"]),
            renderer=renderer_mapping or None,
        )

    def _init_mate_rollout_adapter(self):
        prompt_loader_cfg = self.mate_config.get("prompt_loader", {}) if self.mate_config else {}
        reward_cfg = self.mate_config.get("reward", {}) if self.mate_config else {}
        self.mate_train_prompt_loader = self._build_mate_prompt_loader(
            prompt_loader_cfg=prompt_loader_cfg,
            data_path=self.config.training.train_data_path,
        )
        self.mate_val_prompt_loader = self._build_mate_prompt_loader(
            prompt_loader_cfg=prompt_loader_cfg,
            data_path=self.config.training.val_data_path,
        )
        self.mate_reward_provider = build_reward_provider(reward_cfg)
        self.mate_rollout_adapter = MateRolloutAdapter(
            config=self.mate_config,
            prompt_loader=self.mate_train_prompt_loader,
            reward_provider=self.mate_reward_provider,
            role_policy_mapping=self.mate_config["role_policy_mapping"],
            policy_server_name_mapping=self.policy_server_name_mapping,
            monitor_pool_manager=self.monitor_pool_manager,
        )
        validation_mate_config = dict(self.mate_config)
        validation_mate_config["rollout_mode"] = "parallel"
        validation_mate_config["n_samples_per_prompt"] = 1
        self.mate_val_rollout_adapter = MateRolloutAdapter(
            config=validation_mate_config,
            prompt_loader=self.mate_val_prompt_loader,
            reward_provider=self.mate_reward_provider,
            role_policy_mapping=validation_mate_config["role_policy_mapping"],
            policy_server_name_mapping=self.policy_server_name_mapping,
            monitor_pool_manager=self.monitor_pool_manager,
        )

    def _build_mate_prompt_loader(self, *, prompt_loader_cfg, data_path):
        return MatePromptLoader(
            source_type=prompt_loader_cfg.get(
                "source_type",
                prompt_loader_cfg.get("type"),
            ),
            path=str(data_path),
            prompt_keys=list(prompt_loader_cfg["prompt_keys"]),
            expected_keys=list(prompt_loader_cfg.get("expected_keys", [])),
        )
