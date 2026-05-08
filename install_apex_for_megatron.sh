#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Apex install used for this OrchRL Megatron stack.
# Defaults match the current machine: torch 2.9.0+cu128 with /usr/local/cuda
# pointing at CUDA 12.9. Apex upstream rejects this minor-version mismatch, so
# this script adds a small opt-in bypass in the temporary Apex checkout.

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
APEX_TMP_ROOT="${APEX_TMP_ROOT:-$(mktemp -d /tmp/apex-install-XXXXXX)}"
APEX_SRC="${APEX_SRC:-$APEX_TMP_ROOT/apex}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST

echo "=== Environment Check ==="
python3 - <<'PY'
import os
import site
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch: {type(exc).__name__}: {exc}") from exc

print("python_executable=", sys.executable)
print("python_version=", sys.version.replace("\n", " "))
print("torch_version=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("user_site=", site.getusersitepackages())
print("CUDA_HOME=", os.environ.get("CUDA_HOME"))
print("TORCH_CUDA_ARCH_LIST=", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "[ERROR] nvcc not found: $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

"$CUDA_HOME/bin/nvcc" --version

echo
echo "=== Fetch Apex ==="
if [[ -d "$APEX_SRC/.git" ]]; then
  echo "[SKIP] Apex source already exists: $APEX_SRC"
else
  mkdir -p "$(dirname "$APEX_SRC")"
  git clone --depth 1 https://github.com/NVIDIA/apex.git "$APEX_SRC"
fi

cd "$APEX_SRC"

echo
echo "=== Patch Apex CUDA Minor-Version Check ==="
python3 - <<'PY'
from pathlib import Path

path = Path("setup.py")
text = path.read_text()
old = "if bare_metal_version != torch_binary_version:"
new = (
    'if bare_metal_version != torch_binary_version '
    'and os.environ.get("APEX_IGNORE_CUDA_MISMATCH", "0") != "1":'
)

if new in text:
    print("[SKIP] setup.py already patched")
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print("[OK] patched setup.py")
else:
    raise SystemExit("[ERROR] expected CUDA version check not found in setup.py")
PY

echo
echo "=== Build And Install Apex ==="
APEX_IGNORE_CUDA_MISMATCH=1 \
APEX_CPP_EXT=1 \
APEX_CUDA_EXT=1 \
python3 -m pip install \
  -v \
  --user \
  --no-build-isolation \
  --disable-pip-version-check \
  --no-cache-dir \
  .

echo
echo "=== Verify Apex ==="
python3 - <<'PY'
import apex
import apex_C
import amp_C
import fused_layer_norm_cuda

print("apex", apex.__file__)
print("apex_C", apex_C.__file__)
print("amp_C", amp_C.__file__)
print("fused_layer_norm_cuda", fused_layer_norm_cuda.__file__)
PY

echo
echo "Apex install complete. Source kept at: $APEX_SRC"
