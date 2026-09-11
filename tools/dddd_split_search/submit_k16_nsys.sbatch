#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dddd-k16-nsys
#SBATCH -o nsys_results/dddd_k16_%j.out
#SBATCH -e nsys_results/dddd_k16_%j.err

set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/nsys_results/dddd_k16_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cd "$repo"
source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=rs5

for variant in rs_k16_runtime old_k16_runtime; do
    export VLX_EXCHANGE_DDDD_SPLIT="$variant"
    export VLX_ABLATION_LOG="$result_dir/${variant}_ablation.log"
    nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --stats=false \
        -o "$result_dir/${variant}" vlx guanine-8-hf.inp > "$result_dir/${variant}.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output "$result_dir/${variant}_cuda_gpu_kern_sum" "$result_dir/${variant}.nsys-rep" \
        > "$result_dir/${variant}_stats.log" 2>&1
done
