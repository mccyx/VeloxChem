#!/bin/bash -l
#SBATCH -A pdc-software-test
#SBATCH -J pet-density-cpu
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_cpu_%j.out
#SBATCH -e /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_cpu_%j.err

set -euo pipefail

repo_root=/cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
scratch_root=/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density

export VLX_ML_DENSITY_SCRATCH="$scratch_root"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$repo_root/tools/ml_initial_density/bootstrap_cpu_env.sh"
source "$scratch_root/env/pet-density-cpu-py311/bin/activate"

srun python "$repo_root/tools/ml_initial_density/reproduce_pet_pyscf.py" \
    --device cpu \
    --threads "$SLURM_CPUS_PER_TASK"
