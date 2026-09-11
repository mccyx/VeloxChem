#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dddd-k16-ncu
#SBATCH -o ncu_results/dddd_k16_%j.out
#SBATCH -e ncu_results/dddd_k16_%j.err

set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/ncu_results/dddd_old_k16_${SLURM_JOB_ID}"
profiler="$repo/tools/profiling/ncu/profile_exact_exchange_instructions_ncu.sh"
mkdir -p "$result_dir"
cd "$repo"
source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=rs5
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime

for index in $(seq 0 18); do
    "$profiler" "computeExchangeFockDDDD${index}_FP32" guanine-8-hf.inp "$result_dir"
done
for index in $(seq 0 15); do
    "$profiler" "computeExchangeFockDDDD${index}_K16_OLD_RUNTIME_FP32" guanine-8-hf.inp "$result_dir"
done
