#!/usr/bin/env bash
set -euo pipefail

export ORCHRL_MEGATRON_LM_HOME="${ORCHRL_MEGATRON_LM_HOME:-/mnt/bn/chenghao1026/resouces/libs/Megatron-LM}"
export ORCHRL_MCORE_PYDEPS_HOME="${ORCHRL_MCORE_PYDEPS_HOME:-/mnt/bn/chenghao1026/resouces/libs/verl-mcore-pydeps-0131}"
export ORCHRL_TRANSFORMER_ENGINE_HOME="${ORCHRL_TRANSFORMER_ENGINE_HOME:-/mnt/bn/chenghao1026/resouces/libs/transformer-engine-cu128-torch290}"

USER_SITE_PACKAGES="$(
python3 - <<'PY'
import site
print(site.getusersitepackages())
PY
)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ORCHRL_MEGATRON_LM_HOME:$ORCHRL_MCORE_PYDEPS_HOME:$ORCHRL_TRANSFORMER_ENGINE_HOME:$USER_SITE_PACKAGES:${PYTHONPATH:-}"

echo "=== Path Check ==="
echo "ORCHRL_MEGATRON_LM_HOME=$ORCHRL_MEGATRON_LM_HOME"
echo "ORCHRL_MCORE_PYDEPS_HOME=$ORCHRL_MCORE_PYDEPS_HOME"
echo "ORCHRL_TRANSFORMER_ENGINE_HOME=$ORCHRL_TRANSFORMER_ENGINE_HOME"
echo "USER_SITE_PACKAGES=$USER_SITE_PACKAGES"

python3 - <<'PY'
import importlib

mods = [
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "mbridge",
    "mbridge.core.auto_bridge",
    "transformer_engine",
    "apex",
]

print("\n=== Import Check ===")
failed = False
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"[OK]   {m}: {getattr(mod, '__file__', '<builtin>')}")
    except Exception as e:
        failed = True
        print(f"[FAIL] {m}: {type(e).__name__}: {e}")

print("\n=== Megatron Minimal Runtime Check ===")
try:
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=8)
    print(f"[OK]   TransformerConfig instantiate: hidden_size={cfg.hidden_size}, num_attention_heads={cfg.num_attention_heads}")
except Exception as e:
    failed = True
    print(f"[FAIL] TransformerConfig instantiate: {type(e).__name__}: {e}")

if failed:
    raise SystemExit(1)

print("\nAll checks passed.")
PY
