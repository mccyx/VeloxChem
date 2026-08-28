#!/bin/bash -l
#SBATCH -A pdc-software-test-gh
#SBATCH -J pet-gh200-smoke
#SBATCH -p gpugh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH --gpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_gh200_smoke_%j.out
#SBATCH -e /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_gh200_smoke_%j.err

set -euo pipefail

repo_root=/cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
scratch_root=/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density

export VLX_ML_DENSITY_SCRATCH="$scratch_root"
export CUDA_VISIBLE_DEVICES=0

source "$scratch_root/env/pet-density-gh200-py311/bin/activate"
srun python "$repo_root/tools/ml_initial_density/smoke_pet_inference.py" \
    --device cuda \
    --model "$scratch_root/checkpoints/pet-density.pt" \
    --output "$scratch_root/benchmark_results/pet_gh200_smoke_${SLURM_JOB_ID}.json"
