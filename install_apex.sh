TMP_APEX="$(mktemp -d /tmp/apex-install-XXXXXX)"
git clone --depth 1 https://github.com/NVIDIA/apex.git "$TMP_APEX/apex"
cd "$TMP_APEX/apex"
sed -i 's/if bare_metal_version != torch_binary_version:/if bare_metal_version != torch_binary_version and os.environ.get("APEX_IGNORE_CUDA_MISMATCH", "0") != "1":/' setup.py

CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
APEX_IGNORE_CUDA_MISMATCH=1 \
APEX_CPP_EXT=1 \
APEX_CUDA_EXT=1 \
TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' \
python3 -m pip install -v --user --no-build-isolation --disable-pip-version-check --no-cache-dir .

# check_apex_installation_status
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
