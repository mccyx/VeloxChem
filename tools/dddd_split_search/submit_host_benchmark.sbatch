#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dddd-k19-host
#SBATCH -o build_logs/dddd_k19_host_%j.out
#SBATCH -e build_logs/dddd_k19_host_%j.err

set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/build_logs/dddd_k19_host_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cd "$repo"
source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=rs5

for variant in rs26 k19_static k19_runtime k19_light; do
    for repeat in 1 2 3; do
        export VLX_EXCHANGE_DDDD_SPLIT="$variant"
        export VLX_ABLATION_LOG="$result_dir/${variant}_run${repeat}_ablation.log"
        vlx guanine-8-hf.inp > "$result_dir/${variant}_run${repeat}.log" 2>&1
    done
done
