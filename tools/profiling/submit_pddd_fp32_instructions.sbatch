#!/usr/bin/env bash
#SBATCH --job-name=pddd-rs-ncu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=288
#SBATCH --partition=gpugh
#SBATCH --time=00:55:00
#SBATCH --account=pdc-software-test
#SBATCH --exclude=nid002890,nid002891
#SBATCH --output=tools/profiling/slurm_logs/pddd-rs-ncu_%j.out
#SBATCH --error=tools/profiling/slurm_logs/pddd-rs-ncu_%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR}"
cd "${repo_root}"
source env.sh

export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

output_dir="ncu_results/pddd_fp32_instructions_${SLURM_JOB_ID}"
tools/profiling/ncu/run_pddd_fp32_instructions.sh \
    guanine-8-hf.inp "${output_dir}"
