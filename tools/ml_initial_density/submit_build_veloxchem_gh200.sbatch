#!/bin/bash -l
#SBATCH -A pdc-software-test-gh
#SBATCH -J vlx-pet-build
#SBATCH -p gpugh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus-per-task=1
#SBATCH -t 00:20:00
#SBATCH -o /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/build_logs/veloxchem_pet_build_%j.out
#SBATCH -e /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/build_logs/veloxchem_pet_build_%j.err

set -euo pipefail

repo_root=/cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
dependency_env=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3/vlxenv

if ml -t list 2>&1 | grep -q '^PrgEnv-cray'; then
    ml swap PrgEnv-cray PrgEnv-nvidia
elif ! ml -t list 2>&1 | grep -q '^PrgEnv-nvidia'; then
    ml load PrgEnv-nvidia
fi
ml nvidia/24.11
ml gcc-native/13.2
ml craype-network-ofi
ml cray-mpich/8.1.32
ml cray-python/3.11.7

export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/math_libs/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/compilers/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/cfs/klemming/projects/snic/panor/lxin/gitlab/VeloxChemMP.cuda/dependencies/libxc/lib:$LD_LIBRARY_PATH"
export PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/compilers/bin:$PATH"

source "$dependency_env/bin/activate"
# Remove objects left by an interrupted or differently configured build.
make -C "$repo_root/src" clean
make -C "$repo_root/src" release -j "$SLURM_CPUS_PER_TASK"

export PYTHONPATH="$repo_root/build/python"
export VLXBASISPATH="$repo_root/basis"
python -c "import veloxchem; print(veloxchem.__file__)"
