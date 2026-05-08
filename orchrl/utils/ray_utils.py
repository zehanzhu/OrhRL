"""Ray initialization utilities"""
import json
import os
from pathlib import Path
import socket
import sys
from typing import Callable

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
    
    ray_address = _resolve_ray_address(config)
    is_external_cluster = bool(ray_address)

    # Create experiment-specific temporary directories using process ID for local Ray.
    pid = os.getpid()
    ray_tmp_dir = None
    ray_spill_dir = None
    if not is_external_cluster:
        ray_tmp_dir = f"/tmp/verl_ray_{pid}"
        ray_spill_dir = f"/tmp/verl_spill_{pid}"
        os.makedirs(ray_tmp_dir, exist_ok=True)
        os.makedirs(ray_spill_dir, exist_ok=True)
        _ensure_writable_ray_log_dir(config, ray_tmp_dir, force_override=True)

        # Register directories for cleanup only for locally created Ray runtime.
        register_temp_dirs(ray_tmp_dir, ray_spill_dir)
    else:
        _ensure_writable_ray_log_dir(config, f"/tmp/verl_ray_driver_{pid}")

    # Configure spilling for local Ray only.
    system_config = None
    if ray_spill_dir is not None:
        spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
        system_config = {"object_spilling_config": json.dumps(spilling_conf)}

    # Determine GPU count only when this process starts a local Ray runtime.
    if n_gpus_per_node is None:
        n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if config and hasattr(config, 'resource') else 1

    if not is_external_cluster:
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            available_gpu_count = len(cuda_visible_devices.split(','))
            n_gpus_per_node = min(n_gpus_per_node, available_gpu_count)

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
    runtime_env_vars["PYTHONPATH"] = _build_runtime_pythonpath()
    filtered_runtime_env_vars = {}
    for key, value in runtime_env_vars.items():
        if key == "PYTHONPATH":
            filtered_runtime_env_vars[key] = value
            continue
        if os.environ.get(key) is None:
            filtered_runtime_env_vars[key] = value
    runtime_env_vars = filtered_runtime_env_vars

    ray_init_kwargs = {
        "runtime_env": {"env_vars": runtime_env_vars},
    }
    if is_external_cluster:
        ray_init_kwargs["address"] = ray_address
        print(f"Connecting to existing Ray cluster at {ray_address}")
    else:
        local_node_ip = _ensure_local_ray_ipv4_env()
        local_num_cpus = _resolve_local_ray_num_cpus(config, n_gpus_per_node)
        ray_init_kwargs["num_gpus"] = n_gpus_per_node
        ray_init_kwargs["num_cpus"] = local_num_cpus
        ray_init_kwargs["include_dashboard"] = False
        ray_init_kwargs["_node_ip_address"] = local_node_ip
        ray_init_kwargs["_temp_dir"] = ray_tmp_dir
        ray_init_kwargs["_system_config"] = system_config
        print(
            f"Initializing local Ray with {n_gpus_per_node} GPUs, "
            f"{local_num_cpus} CPUs on {local_node_ip}"
        )

    if not is_external_cluster:
        ray_context = _init_local_ray_with_process_env_overrides(
            ray_init_kwargs, local_node_ip
        )
    else:
        ray_context = ray.init(**ray_init_kwargs)

    session_dir = None
    address_info = getattr(ray_context, "address_info", None)
    if isinstance(address_info, dict):
        session_dir = address_info.get("session_dir")
    if not is_external_cluster:
        register_ray_process_matchers(ray_tmp_dir, session_dir)
    
    return ray_tmp_dir, ray_spill_dir


def _resolve_ray_address(config=None):
    env_ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if env_ray_address:
        return env_ray_address

    resource_cfg = getattr(config, "resource", None) if config is not None else None
    cfg_ray_address = getattr(resource_cfg, "ray_address", None) if resource_cfg is not None else None
    if cfg_ray_address is None:
        return None

    resolved = str(cfg_ray_address).strip()
    if not resolved or resolved.lower() in {"none", "null"}:
        return None
    return resolved


def _ensure_writable_ray_log_dir(
    config, ray_tmp_dir: str, *, force_override: bool = False
) -> str:
    """Redirect Ray logs to a writable directory when platform defaults are read-only."""
    redirect_log_dir = os.environ.get("BYTED_RAY_REDIRECT_LOG", "").strip()
    if not force_override and redirect_log_dir and _can_write_directory(redirect_log_dir):
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


def _build_runtime_pythonpath() -> str:
    pythonpath_entries = []

    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        pythonpath_entries.extend(existing.split(":"))

    cwd = str(Path.cwd().resolve())
    repo_parent = str(Path(cwd).parent.resolve())

    pythonpath_entries.extend([cwd, repo_parent])
    pythonpath_entries.extend(path for path in sys.path if path)

    deduped_entries = []
    seen = set()
    for entry in pythonpath_entries:
        normalized = str(Path(entry).expanduser()) if entry else entry
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_entries.append(normalized)
    return ":".join(deduped_entries)


def _coerce_positive_int(value) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(1, coerced)


def _resolve_local_ray_num_cpus(config, n_gpus_per_node: int) -> int:
    ray_kwargs = getattr(config, "ray_kwargs", None) if config is not None else None
    ray_init_cfg = getattr(ray_kwargs, "ray_init", None) if ray_kwargs is not None else None
    configured_num_cpus = (
        _coerce_positive_int(getattr(ray_init_cfg, "num_cpus", None))
        if ray_init_cfg is not None
        else None
    )
    if configured_num_cpus is not None:
        return configured_num_cpus

    resource_cfg = getattr(config, "resource", None) if config is not None else None
    trainer_remote_num_cpus = _coerce_positive_int(
        getattr(resource_cfg, "trainer_remote_num_cpus", None)
        if resource_cfg is not None
        else None,
    )
    if trainer_remote_num_cpus is not None:
        return trainer_remote_num_cpus

    detected_cpu_count = _coerce_positive_int(os.cpu_count())
    if detected_cpu_count is not None:
        return detected_cpu_count

    return max(1, int(n_gpus_per_node))


def _ensure_local_ray_ipv4_env() -> str:
    """Force local Ray to use IPv4 when platform defaults inject IPv6 pod IPs."""
    local_node_ip = _resolve_local_ray_node_ip()
    _apply_local_ray_process_env_overrides(os.environ, local_node_ip)
    return local_node_ip


def _resolve_local_ray_node_ip() -> str:
    for env_var in ("RAY_NODE_IP_ADDRESS", "MY_HOST_IP", "MY_POD_IP"):
        candidate = _normalize_ipv4_env_value(os.environ.get(env_var))
        if candidate:
            return candidate

    host_name = socket.gethostname()
    fqdn = socket.getfqdn(host_name)
    for candidate in (
        _normalize_ipv4_env_value(socket.gethostbyname(host_name)),
        _normalize_ipv4_env_value(socket.gethostbyname(fqdn)),
    ):
        if candidate:
            return candidate

    raise RuntimeError(
        "Failed to resolve an IPv4 address for local Ray startup. "
        "Set RAY_NODE_IP_ADDRESS or MY_HOST_IP explicitly."
    )


def _normalize_ipv4_env_value(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().strip("[]")
    if not candidate or ":" in candidate:
        return None
    return candidate


def _build_local_ray_process_env_overrides(local_node_ip: str) -> dict[str, str]:
    return {
        "BYTED_RAY_POD_IP": local_node_ip,
        "RAY_IP": local_node_ip,
        "MY_HOST_IP": local_node_ip,
        "MY_POD_IP": local_node_ip,
        "MY_HOST_IPV6": "",
        "MY_POD_IPV6": "",
        "RAY_enable_worker_prestart": "false",
        "RAY_prestart_worker_first_driver": "false",
    }


def _apply_local_ray_process_env_overrides(
    target_env: os._Environ[str], local_node_ip: str
) -> None:
    target_env.update(_build_local_ray_process_env_overrides(local_node_ip))


def _init_local_ray_with_process_env_overrides(
    ray_init_kwargs: dict, local_node_ip: str
):
    import ray._private.services as ray_services

    original_start_ray_process: Callable = ray_services.start_ray_process
    env_overrides = _build_local_ray_process_env_overrides(local_node_ip)

    def _start_ray_process_with_ipv4_env(*args, **kwargs):
        env_updates = dict(kwargs.get("env_updates") or {})
        env_updates.update(env_overrides)
        kwargs["env_updates"] = env_updates
        return original_start_ray_process(*args, **kwargs)

    ray_services.start_ray_process = _start_ray_process_with_ipv4_env
    try:
        return ray.init(**ray_init_kwargs)
    finally:
        ray_services.start_ray_process = original_start_ray_process
