#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scratch_root=${VLX_ML_DENSITY_SCRATCH:-/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density}
env_dir="$scratch_root/env/pet-density-gh200-py311"
export PIP_CACHE_DIR="$scratch_root/env/pip-cache-gh200"
export TMPDIR="$scratch_root/env/tmp-gh200"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
ml cray-python/3.11.7

if [[ ! -x "$env_dir/bin/python" ]]; then
    python3 -m venv "$env_dir"
fi

source "$env_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
    'torch==2.10.0' \
    --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
    -r "$repo_root/tools/ml_initial_density/requirements-gh200.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu130

python -m pip freeze > "$scratch_root/env/pet-density-gh200-py311.freeze.txt"
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA PyTorch is installed but no CUDA device is available")

print("GH200 environment import check: OK")
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
probe = torch.ones(16, device="cuda")
if float(probe.sum()) != 16.0:
    raise RuntimeError("CUDA execution probe returned an incorrect result")
print("CUDA execution probe: OK")
PY
