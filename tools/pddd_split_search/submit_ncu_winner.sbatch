#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J pddd-k4-ncu
#SBATCH -o ncu_results/pddd_k4_%j.out
#SBATCH -e ncu_results/pddd_k4_%j.err

set -euo pipefail

repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/ncu_results/pddd_k4_winner_${SLURM_JOB_ID}"
profiler="$repo/tools/profiling/ncu/profile_exact_exchange_instructions_ncu.sh"
mkdir -p "$result_dir"
cd "$repo"
source env.sh

export VLX_EXCHANGE_PDDD_SPLIT=rs5
for index in 0 1 2 3 4; do
    "$profiler" "computeExchangeFockPDDD${index}_RS_FP32" guanine-8-hf.inp "$result_dir"
done

export VLX_EXCHANGE_PDDD_SPLIT=k4_m23
for index in 0 1 2 3; do
    "$profiler" "computeExchangeFockPDDD${index}_K4_M23_FP32" guanine-8-hf.inp "$result_dir"
done
