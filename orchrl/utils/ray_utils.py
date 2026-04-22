"""Ray initialization utilities"""
import json
import os
from pathlib import Path

import ray


def init_ray_with_temp_dirs(config=None, n_gpus_per_node=None):
    """
    Initialize Ray with temporary directories and spilling configuration
    
    Args:
        config: Optional config object with resource settings
        n_gpus_per_node: Number of GPUs per node (overrides config if provided)
    
    Returns:
        Tuple of (ray_tmp_dir, ray_spill_dir)
    """
    from .clean_up import register_ray_process_matchers, register_temp_dirs
    
    if ray.is_initialized():
        print("Ray is already initialized")
        return None, None
    
    # Create experiment-specific temporary directories using process ID
    pid = os.getpid()
    ray_tmp_dir = f"/tmp/verl_ray_{pid}"
    ray_spill_dir = f"/tmp/verl_spill_{pid}"
    os.makedirs(ray_tmp_dir, exist_ok=True)
    os.makedirs(ray_spill_dir, exist_ok=True)
    _ensure_writable_ray_log_dir(config, ray_tmp_dir)
    
    # Register directories for cleanup
    register_temp_dirs(ray_tmp_dir, ray_spill_dir)
    
    # Configure spilling
    spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
    system_config = {"object_spilling_config": json.dumps(spilling_conf)}
    
    # Determine GPU count
    if n_gpus_per_node is None:
        n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if config and hasattr(config, 'resource') else 1
    
    # Validate GPU availability
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible_devices:
        available_gpu_count = len(cuda_visible_devices.split(','))
        n_gpus_per_node = min(n_gpus_per_node, available_gpu_count)
    
    print(f"Initializing Ray with {n_gpus_per_node} GPUs")
    runtime_env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        # Needed for multiple colocated NPU processes that each initialize HCCL.
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    }
    runtime_env_vars = {key: value for key, value in runtime_env_vars.items() if os.environ.get(key) is None}

    ray_context = ray.init(
        num_gpus=n_gpus_per_node,
        runtime_env={"env_vars": runtime_env_vars},
        _temp_dir=ray_tmp_dir,
        _system_config=system_config
    )

    session_dir = None
    address_info = getattr(ray_context, "address_info", None)
    if isinstance(address_info, dict):
        session_dir = address_info.get("session_dir")
    register_ray_process_matchers(ray_tmp_dir, session_dir)
    
    return ray_tmp_dir, ray_spill_dir


def _ensure_writable_ray_log_dir(config, ray_tmp_dir: str) -> str:
    """Redirect Ray logs to a writable directory when platform defaults are read-only."""
    redirect_log_dir = os.environ.get("BYTED_RAY_REDIRECT_LOG", "").strip()
    if redirect_log_dir and _can_write_directory(redirect_log_dir):
        return redirect_log_dir

    fallback_dir = _resolve_fallback_ray_log_dir(config, ray_tmp_dir)
    os.makedirs(fallback_dir, exist_ok=True)
    os.environ["BYTED_RAY_REDIRECT_LOG"] = fallback_dir
    print(f"Using writable Ray log directory: {fallback_dir}")
    return fallback_dir


def _resolve_fallback_ray_log_dir(config, ray_tmp_dir: str) -> str:
    training_cfg = getattr(config, "training", None) if config is not None else None
    run_dir = getattr(training_cfg, "run_dir", None) if training_cfg is not None else None
    if run_dir:
        candidate = Path(str(run_dir)).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return str((candidate / "ray_logs").resolve())
    return str((Path(ray_tmp_dir) / "logs").resolve())


def _can_write_directory(path: str) -> bool:
    candidate = Path(path).expanduser()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(candidate, os.W_OK)
