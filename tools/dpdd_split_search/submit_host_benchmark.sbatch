#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dpdd-split-host
#SBATCH -o build_logs/dpdd_split_host_%j.out
#SBATCH -e build_logs/dpdd_split_host_%j.err
set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3; dir="$repo/build_logs/dpdd_split_host_${SLURM_JOB_ID}"; mkdir -p "$dir"; cd "$repo"; source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=k4_m23 VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime VLX_EXCHANGE_DDDP_SPLIT=old_k5
for variant in rs5 old_k5 rs_k4 old_k4; do
 for repeat in 1 2 3; do export VLX_EXCHANGE_DPDD_SPLIT="$variant" VLX_ABLATION_LOG="$dir/${variant}_run${repeat}_ablation.log"; vlx guanine-8-hf.inp > "$dir/${variant}_run${repeat}.log" 2>&1; done
done
