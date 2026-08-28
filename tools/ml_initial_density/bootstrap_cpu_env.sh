#!/bin/bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scratch_root=${VLX_ML_DENSITY_SCRATCH:-/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density}
env_dir="$scratch_root/env/pet-density-cpu-py311"
export PIP_CACHE_DIR="$scratch_root/env/pip-cache"
export TMPDIR="$scratch_root/env/tmp"

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

ml cray-python/3.11.7

if [[ ! -x "$env_dir/bin/python" ]]; then
    python3 -m venv "$env_dir"
fi

source "$env_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$repo_root/tools/ml_initial_density/requirements-reproduce.txt"

python -m pip freeze > "$scratch_root/env/pet-density-cpu-py311.freeze.txt"
python - <<'PY'
import ase
import metatensor.torch
import numpy
import pyscf
import scipy
import torch

print("environment import check: OK")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"pyscf={pyscf.__version__}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__} ase={ase.__version__}")
PY
