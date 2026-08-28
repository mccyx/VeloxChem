#!/bin/bash -l
#SBATCH -A pdc-software-test-gh
#SBATCH -J pet-repeat-timing
#SBATCH -p gpugh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_repeated_timing_%j.out
#SBATCH -e /cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/benchmark_results/pet_repeated_timing_%j.err

set -euo pipefail

repo_root=/cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
scratch_root=/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density
benchmark_threads="${BENCH_THREADS:-8}"
benchmark_repeats="${BENCH_REPEATS:-5}"

export VLX_ML_DENSITY_SCRATCH="$scratch_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="$benchmark_threads"
export OPENBLAS_NUM_THREADS="$benchmark_threads"

source "$scratch_root/env/pet-density-gh200-py311/bin/activate"
export LD_LIBRARY_PATH="$scratch_root/env/pet-density-gh200-py311/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
srun python "$repo_root/tools/ml_initial_density/benchmark_initial_density_end_to_end.py" \
    --device cuda \
    --threads "$benchmark_threads" \
    --repeats "$benchmark_repeats" \
    --output-dir "$scratch_root/benchmark_results/pet_repeated_timing_${SLURM_JOB_ID}"
