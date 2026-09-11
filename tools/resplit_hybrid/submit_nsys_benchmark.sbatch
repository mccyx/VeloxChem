#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J exchange-winners-nsys
#SBATCH -o nsys_results/exchange_winners_%j.out
#SBATCH -e nsys_results/exchange_winners_%j.err

set -euo pipefail

repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/nsys_results/exchange_winners_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cd "$repo"
source env.sh

export VLX_EXCHANGE_PDDD_SPLIT=k4_m23
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
export VLX_EXCHANGE_DDDP_SPLIT=old_k5
export VLX_EXCHANGE_DPDD_SPLIT=rs_k4
export VLX_ABLATION_LOG="$result_dir/ablation.log"

nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --stats=false \
    -o "$result_dir/all_winners" \
    vlx guanine-8-hf.inp > "$result_dir/run.log" 2>&1

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$result_dir/all_winners_cuda_gpu_kern_sum" \
    "$result_dir/all_winners.nsys-rep" > "$result_dir/stats.log" 2>&1
