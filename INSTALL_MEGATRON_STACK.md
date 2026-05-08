# Megatron / TE / Apex / mbridge 安装说明

本文档提供一套可重复执行的安装命令，目标是适配当前 OrchRL Megatron 训练环境：

- Megatron-LM（源码路径）
- mbridge（隔离 pydeps 目录）
- Transformer Engine（隔离目录，避免污染系统包）
- Apex（用户 site-packages）

已存在的目录会自动跳过，不会重复下载。  
特别是 **Megatron-LM 已存在时，不需要重新下载**，只需做导入校验。

## 1. 一键安装脚本

在仓库根目录执行：

```bash
cat > setup_megatron_stack.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

# ===== 0) 路径与编译环境 =====
export LIB_ROOT=/mnt/bn/chenghao1026/resouces/libs
export ORCHRL_MEGATRON_LM_HOME="$LIB_ROOT/Megatron-LM"
export ORCHRL_MCORE_PYDEPS_HOME="$LIB_ROOT/verl-mcore-pydeps-0131"
export ORCHRL_TRANSFORMER_ENGINE_HOME="$LIB_ROOT/transformer-engine-cu128-torch290"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/torch/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

# FORCE_REBUILD=1 时会重建 Transformer Engine
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "$LIB_ROOT" "$ORCHRL_MCORE_PYDEPS_HOME"

python3 -m pip install -U pip setuptools wheel packaging cmake ninja

# ===== 1) Megatron-LM（源码路径，可跳过）=====
if [[ ! -d "$ORCHRL_MEGATRON_LM_HOME/.git" ]]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git "$ORCHRL_MEGATRON_LM_HOME"
else
  echo "[SKIP] Megatron-LM already exists: $ORCHRL_MEGATRON_LM_HOME"
fi

# ===== 2) mbridge（安装到隔离 pydeps 目录）=====
python3 -m pip install -U -t "$ORCHRL_MCORE_PYDEPS_HOME" mbridge

# ===== 3) Transformer Engine（安装到隔离目录）=====
if [[ "$FORCE_REBUILD" == "1" ]]; then
  rm -rf "$ORCHRL_TRANSFORMER_ENGINE_HOME"
fi

if [[ -f "$ORCHRL_TRANSFORMER_ENGINE_HOME/transformer_engine/__init__.py" ]]; then
  echo "[SKIP] Transformer Engine already exists: $ORCHRL_TRANSFORMER_ENGINE_HOME"
else
  mkdir -p "$ORCHRL_TRANSFORMER_ENGINE_HOME"
  TMP_TE="$(mktemp -d)"
  git clone --depth 1 https://github.com/NVIDIA/TransformerEngine.git "$TMP_TE/TransformerEngine"
  pushd "$TMP_TE/TransformerEngine"
  git submodule update --init --recursive
  NVTE_FRAMEWORK=pytorch python3 -m pip install -v --no-build-isolation . -t "$ORCHRL_TRANSFORMER_ENGINE_HOME"
  popd
  rm -rf "$TMP_TE"
fi

# ===== 4) Apex（安装到用户 site-packages）=====
if python3 - <<'PY'
import apex
print(apex.__file__)
PY
then
  echo "[SKIP] apex already importable"
else
  TMP_APEX="$(mktemp -d)"
  git clone --depth 1 https://github.com/NVIDIA/apex.git "$TMP_APEX/apex"
  pushd "$TMP_APEX/apex"
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python3 -m pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir .
  popd
  rm -rf "$TMP_APEX"
fi

# ===== 5) 验证导入 =====
PYTHONPATH="$ORCHRL_MEGATRON_LM_HOME:$ORCHRL_MCORE_PYDEPS_HOME:$ORCHRL_TRANSFORMER_ENGINE_HOME:${PYTHONPATH:-}" \
python3 - <<'PY'
import transformer_engine
import apex
import megatron
import mbridge
print("OK transformer_engine:", transformer_engine.__file__)
print("OK apex:", apex.__file__)
print("OK megatron:", megatron.__file__)
print("OK mbridge:", mbridge.__file__)
PY
BASH

chmod +x setup_megatron_stack.sh
./setup_megatron_stack.sh
```

## 2. Megatron-LM 已存在时的最小校验（可单独执行）

```bash
PYTHONPATH="/mnt/bn/chenghao1026/resouces/libs/Megatron-LM:$PYTHONPATH" \
python3 -c "import megatron; print(megatron.__file__)"
```

能打印路径就说明 Megatron-LM 可用，可跳过它的安装步骤。

## 3. 强制重编 Transformer Engine

```bash
FORCE_REBUILD=1 ./setup_megatron_stack.sh
```

## 4. 与训练脚本联动

`experiments/search_mas/run_train_e2e.sh` 默认会使用以下路径并 prepend 到 `PYTHONPATH`：

- `ORCHRL_MEGATRON_LM_HOME=/mnt/bn/chenghao1026/resouces/libs/Megatron-LM`
- `ORCHRL_MCORE_PYDEPS_HOME=/mnt/bn/chenghao1026/resouces/libs/verl-mcore-pydeps-0131`
- `ORCHRL_TRANSFORMER_ENGINE_HOME=/mnt/bn/chenghao1026/resouces/libs/transformer-engine-cu128-torch290`

如果你安装到了别的路径，启动前覆盖这些环境变量即可。

## 5. Apex 与 `PYTHONNOUSERSITE=1` 注意事项

脚本中 `apex` 默认安装到 `~/.local/lib/python3.11/site-packages`。  
当设置 `PYTHONNOUSERSITE=1` 时，Python 默认不会加载用户 site-packages，因此可能出现 `ModuleNotFoundError: apex`。

建议在启动前显式补回用户 site-packages：

```bash
USER_SITE_PACKAGES="$(python3 - <<'PY'
import site
print(site.getusersitepackages())
PY
)"
export PYTHONPATH="$USER_SITE_PACKAGES:${PYTHONPATH:-}"
```

说明：`experiments/search_mas/run_train_e2e.sh` 已包含这段逻辑，通常不需要你手动再加。
