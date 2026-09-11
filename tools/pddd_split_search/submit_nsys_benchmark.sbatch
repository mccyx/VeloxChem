#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J pddd-k4-nsys
#SBATCH -o nsys_results/pddd_k4_%j.out
#SBATCH -e nsys_results/pddd_k4_%j.err

set -euo pipefail

repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/nsys_results/pddd_k4_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cd "$repo"
source env.sh

for variant in rs5 k4_m01 k4_m12 k4_m23 k4_m34; do
    export VLX_EXCHANGE_PDDD_SPLIT="$variant"
    export VLX_ABLATION_LOG="$result_dir/${variant}_ablation.log"
    nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --stats=false \
        -o "$result_dir/${variant}" \
        vlx guanine-8-hf.inp > "$result_dir/${variant}.log" 2>&1
    nsys stats \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output "$result_dir/${variant}_cuda_gpu_kern_sum" \
        "$result_dir/${variant}.nsys-rep" > "$result_dir/${variant}_stats.log" 2>&1
done
