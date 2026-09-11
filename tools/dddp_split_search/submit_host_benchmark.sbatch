#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dddp-k5-host
#SBATCH -o build_logs/dddp_k5_host_%j.out
#SBATCH -e build_logs/dddp_k5_host_%j.err

set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/build_logs/dddp_k5_host_${SLURM_JOB_ID}"
mkdir -p "$result_dir"; cd "$repo"; source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=rs5
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
for variant in rs6 rs_k5 old_k5; do
    for repeat in 1 2 3; do
        export VLX_EXCHANGE_DDDP_SPLIT="$variant"
        export VLX_ABLATION_LOG="$result_dir/${variant}_run${repeat}_ablation.log"
        vlx guanine-8-hf.inp > "$result_dir/${variant}_run${repeat}.log" 2>&1
    done
done
