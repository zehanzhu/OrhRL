from __future__ import annotations

import importlib.util
import inspect
import os
import sys


def _prepend_env_path(env_var: str) -> None:
    path = os.getenv(env_var)
    if not path or not os.path.isdir(path):
        return
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def patch_mbridge_autobridge_api() -> bool:
    try:
        auto_bridge_module = importlib.import_module("mbridge.core.auto_bridge")
        bridge_module = importlib.import_module("mbridge.core.bridge")
        from transformers import AutoConfig
    except Exception:
        return False
    AutoBridge = auto_bridge_module.AutoBridge
    _MODEL_REGISTRY = bridge_module._MODEL_REGISTRY

    if getattr(AutoBridge, "_orchrl_dtype_compat_patched", False):
        return False

    from_config_sig = inspect.signature(AutoBridge.from_config)
    from_pretrained_sig = inspect.signature(AutoBridge.from_pretrained)

    from_config_supports_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in from_config_sig.parameters.values()
    )
    from_pretrained_supports_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in from_pretrained_sig.parameters.values()
    )

    if from_config_supports_kwargs and from_pretrained_supports_kwargs:
        return False

    def _from_config(cls, hf_config: AutoConfig, **kwargs):
        model_type = hf_config.model_type
        if model_type in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[model_type](hf_config, **kwargs)
        raise ValueError(
            f"Unregistered model type: {model_type}, now only support {_MODEL_REGISTRY.keys()}"
        )

    def _from_pretrained(cls, hf_model_path, trust_remote_code=False, **kwargs):
        config = AutoConfig.from_pretrained(
            hf_model_path, trust_remote_code=trust_remote_code
        )
        return cls.from_config(config, **kwargs)

    AutoBridge.from_config = classmethod(_from_config)
    AutoBridge.from_pretrained = classmethod(_from_pretrained)
    AutoBridge._orchrl_dtype_compat_patched = True
    return True


def patch_mbridge_transformer_engine_fallback() -> bool:
    if importlib.util.find_spec("transformer_engine") is not None:
        return False

    try:
        llm_bridge_module = importlib.import_module("mbridge.core.llm_bridge")
    except Exception:
        return False

    LLMBridge = llm_bridge_module.LLMBridge
    if getattr(LLMBridge, "_orchrl_no_te_patched", False):
        return False

    def _get_transformer_layer_spec(self, vp_stage=None):
        assert self.config.normalization == "RMSNorm", "only RMSNorm is supported for now"
        sig = inspect.signature(llm_bridge_module.get_gpt_decoder_block_spec)
        self.has_vp_stage = "vp_stage" in sig.parameters
        extra_args = {}
        if self.has_vp_stage:
            extra_args["vp_stage"] = vp_stage
        if "normalization" in sig.parameters:
            extra_args["normalization"] = self.config.normalization
        return llm_bridge_module.get_gpt_decoder_block_spec(
            self.config,
            use_transformer_engine=False,
            **extra_args,
        )

    LLMBridge._get_transformer_layer_spec = _get_transformer_layer_spec
    LLMBridge._orchrl_no_te_patched = True
    return True


def bootstrap_mcore_runtime_compat() -> None:
    _prepend_env_path("ORCHRL_TRANSFORMER_ENGINE_HOME")
    _prepend_env_path("ORCHRL_MCORE_PYDEPS_HOME")
    _prepend_env_path("ORCHRL_MEGATRON_LM_HOME")
    patch_mbridge_autobridge_api()
    patch_mbridge_transformer_engine_fallback()
