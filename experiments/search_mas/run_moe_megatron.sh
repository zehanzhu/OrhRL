#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$REPO_ROOT/experiments/search_mas"

DEFAULT_CONFIG_NAME="train_megatron_moe_role_specific"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DEFAULT_MEGATRON_LM_HOME="/mnt/bn/chenghao1026/resouces/libs/Megatron-LM"
DEFAULT_MCORE_PYDEPS_HOME="/mnt/bn/chenghao1026/resouces/libs/verl-mcore-pydeps-0131"
DEFAULT_TRANSFORMER_ENGINE_HOME="/mnt/bn/chenghao1026/resouces/libs/transformer-engine-cu128-torch290"
DEFAULT_TORCH_LIB_HOME="/usr/local/lib/python3.11/dist-packages/torch/lib"
DEFAULT_CUBLAS_LIB_HOME="/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib"
DEFAULT_CUDNN_LIB_HOME="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
DEFAULT_NCCL_LIB_HOME="/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_PATH="$REPO_ROOT/outputs/logs/search_mas_megatron_moe_${TIMESTAMP}.log"

CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG_NAME}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"
ORCHRL_MEGATRON_LM_HOME="${ORCHRL_MEGATRON_LM_HOME:-$DEFAULT_MEGATRON_LM_HOME}"
ORCHRL_MCORE_PYDEPS_HOME="${ORCHRL_MCORE_PYDEPS_HOME:-$DEFAULT_MCORE_PYDEPS_HOME}"
ORCHRL_TRANSFORMER_ENGINE_HOME="${ORCHRL_TRANSFORMER_ENGINE_HOME:-$DEFAULT_TRANSFORMER_ENGINE_HOME}"

mkdir -p "$REPO_ROOT/outputs/logs"
mkdir -p "$(dirname "$LOG_PATH")"
cd "$REPO_ROOT"

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

if ! RUNTIME_ENV="$(
  CONFIG_DIR="$CONFIG_DIR" CONFIG_NAME="$CONFIG_NAME" python3 - <<'PY'
import os
import shlex
import sys

from hydra import compose, initialize_config_dir


def shell_assign(key, value):
    print(f"{key}={shlex.quote(str(value))}")


def die(message):
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


config_dir = os.environ["CONFIG_DIR"]
config_name = os.environ["CONFIG_NAME"]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name)

resource_cfg = getattr(cfg, "resource", None)
ray_address = getattr(resource_cfg, "ray_address", None) if resource_cfg is not None else None
env_ray_address = os.environ.get("RAY_ADDRESS", "").strip()
use_external_ray = bool(env_ray_address)
if not use_external_ray and ray_address is not None:
    resolved = str(ray_address).strip().lower()
    use_external_ray = resolved not in {"", "none", "null"}

models_cfg = getattr(cfg, "models", None)
if not models_cfg:
    die(f"Config '{config_name}' has no models section")

errors = []
megatron_summaries = []
for model_key, model_cfg in models_cfg.items():
    ppo_cfg = getattr(model_cfg, "ppo_trainer_config", model_cfg)
    actor_rollout_ref = getattr(ppo_cfg, "actor_rollout_ref", None)
    actor_cfg = getattr(actor_rollout_ref, "actor", None) if actor_rollout_ref is not None else None
    ref_cfg = getattr(actor_rollout_ref, "ref", None) if actor_rollout_ref is not None else None
    rollout_cfg = getattr(actor_rollout_ref, "rollout", None) if actor_rollout_ref is not None else None

    actor_strategy = str(getattr(actor_cfg, "strategy", "")).strip().lower()
    if actor_strategy != "megatron":
        errors.append(f"{model_key} actor.strategy={actor_strategy or '<missing>'}")
        continue

    megatron_cfg = getattr(actor_cfg, "megatron", None)
    if megatron_cfg is None:
        errors.append(f"{model_key} actor.megatron=<missing>")
        continue

    if ref_cfg is not None:
        ref_strategy = str(getattr(ref_cfg, "strategy", "")).strip().lower()
        if ref_strategy and ref_strategy != "megatron":
            errors.append(f"{model_key} ref.strategy={ref_strategy}")

    summary = (
        f"{model_key}:{getattr(model_cfg, 'name', model_key)} "
        f"actor_tp={getattr(megatron_cfg, 'tensor_model_parallel_size', '<unset>')} "
        f"actor_pp={getattr(megatron_cfg, 'pipeline_model_parallel_size', '<unset>')} "
        f"actor_cp={getattr(megatron_cfg, 'context_parallel_size', '<unset>')} "
        f"actor_ep={getattr(megatron_cfg, 'expert_model_parallel_size', '<unset>')} "
        f"rollout_tp={getattr(rollout_cfg, 'tensor_model_parallel_size', '<unset>')}"
    )
    megatron_summaries.append(summary)

if errors:
    die(
        f"Config '{config_name}' is not Megatron-only: "
        + "; ".join(errors)
        + ". Use CONFIG_NAME=train_megatron_moe_smoke or another Megatron config."
    )

shell_assign("USE_EXTERNAL_RAY", "1" if use_external_ray else "0")
shell_assign("MAS_WORK_DIR", cfg.training.mate.mas_work_dir)
shell_assign("CONFIG_TEMPLATE_PATH", cfg.training.mate.config_template_path)
shell_assign("TRAIN_PROMPT_DATA_PATH", cfg.training.train_data_path)
shell_assign("VAL_PROMPT_DATA_PATH", cfg.training.val_data_path)
shell_assign("MEGATRON_CONFIG_SUMMARY", " | ".join(megatron_summaries))

model_paths = []
for idx, policy_cfg in enumerate(cfg.base_models.values()):
    shell_assign(f"MODEL_PATH_{idx}", policy_cfg.path)
    model_paths.append(str(policy_cfg.path))

shell_assign("MODEL_PATH_COUNT", len(model_paths))
shell_assign("MODEL_PATHS_JOINED", " | ".join(model_paths))
PY
)"; then
  echo "[ERROR] Failed to resolve Megatron runtime config: $CONFIG_NAME" >&2
  exit 1
fi
eval "$RUNTIME_ENV"

if [[ "$USE_EXTERNAL_RAY" == "1" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[INFO] External Ray mode detected; preserving existing CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  else
    echo "[INFO] External Ray mode detected; not forcing CUDA_VISIBLE_DEVICES on the driver"
  fi
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
fi

for required_dir in \
  "$MAS_WORK_DIR" \
  "$ORCHRL_MEGATRON_LM_HOME" \
  "$ORCHRL_MCORE_PYDEPS_HOME" \
  "$ORCHRL_TRANSFORMER_ENGINE_HOME"
do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory not found: $required_dir" >&2
    exit 1
  fi
done

for required_file in "$CONFIG_TEMPLATE_PATH" "$TRAIN_PROMPT_DATA_PATH" "$VAL_PROMPT_DATA_PATH"; do
  if [[ ! -e "$required_file" ]]; then
    echo "[ERROR] Required path not found: $required_file" >&2
    exit 1
  fi
done

for ((i=0; i<MODEL_PATH_COUNT; i++)); do
  model_path_var="MODEL_PATH_${i}"
  model_path="${!model_path_var}"
  if [[ ! -e "$model_path" ]]; then
    echo "[ERROR] Required model path not found: $model_path" >&2
    exit 1
  fi
done

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

USER_SITE_PACKAGES="$(
  python3 - <<'PY'
import site
print(site.getusersitepackages())
PY
)"
if [[ -d "$USER_SITE_PACKAGES" ]]; then
  export PYTHONPATH="$USER_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

for lib_dir in \
  "$DEFAULT_TORCH_LIB_HOME" \
  "$DEFAULT_CUBLAS_LIB_HOME" \
  "$DEFAULT_CUDNN_LIB_HOME" \
  "$DEFAULT_NCCL_LIB_HOME"
do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done


export PYTHONPATH="$ORCHRL_MEGATRON_LM_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ORCHRL_MCORE_PYDEPS_HOME${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ORCHRL_TRANSFORMER_ENGINE_HOME${PYTHONPATH:+:$PYTHONPATH}"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Experiment dir: $CONFIG_DIR"
echo "[INFO] Megatron config: $CONFIG_NAME"
echo "[INFO] Megatron parallelism: $MEGATRON_CONFIG_SUMMARY"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] Log path: $LOG_PATH"
echo "[INFO] Transformer Engine home: $ORCHRL_TRANSFORMER_ENGINE_HOME"
echo "[INFO] MCore pydeps home: $ORCHRL_MCORE_PYDEPS_HOME"
echo "[INFO] Megatron-LM home: $ORCHRL_MEGATRON_LM_HOME"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "[INFO] PYTHONPATH: ${PYTHONPATH:-<unset>}"
echo "[INFO] MAS work dir: $MAS_WORK_DIR"
echo "[INFO] Train prompt data: $TRAIN_PROMPT_DATA_PATH"
echo "[INFO] Val prompt data: $VAL_PROMPT_DATA_PATH"
echo "[INFO] Model paths: $MODEL_PATHS_JOINED"

python3 -m orchrl.trainer.train \
  --config-path "$CONFIG_DIR" \
  --config-name "$CONFIG_NAME" 2>&1 | tee "$LOG_PATH"
