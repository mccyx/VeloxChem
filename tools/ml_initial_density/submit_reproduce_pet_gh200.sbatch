#!/bin/bash -l
#SBATCH -A pdc-software-test-gh
#SBATCH -J pet-reproduce-gh200
#SBATCH -p gpugh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_reproduce_gh200_%j.out
#SBATCH -e /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_reproduce_gh200_%j.err

set -euo pipefail

repo_root=/cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
scratch_root=/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density

export VLX_ML_DENSITY_SCRATCH="$scratch_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

source "$scratch_root/env/pet-density-gh200-py311/bin/activate"
export LD_LIBRARY_PATH="$scratch_root/env/pet-density-gh200-py311/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
srun python "$repo_root/tools/ml_initial_density/reproduce_pet_pyscf.py" \
    --device cuda \
    --threads "$SLURM_CPUS_PER_TASK" \
    --output-dir "$scratch_root/benchmark_results/pet_pyscf_reproduction_gh200_${SLURM_JOB_ID}"
