#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dddp-k5-nsys
#SBATCH -o nsys_results/dddp_k5_%j.out
#SBATCH -e nsys_results/dddp_k5_%j.err
set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
dir="$repo/nsys_results/dddp_k5_${SLURM_JOB_ID}"; mkdir -p "$dir"; cd "$repo"; source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=rs5 VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
for variant in rs6 rs_k5 old_k5; do
 export VLX_EXCHANGE_DDDP_SPLIT="$variant" VLX_ABLATION_LOG="$dir/${variant}_ablation.log"
 nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --stats=false -o "$dir/$variant" vlx guanine-8-hf.inp > "$dir/$variant.log" 2>&1
 nsys stats --report cuda_gpu_kern_sum --format csv --output "$dir/${variant}_cuda_gpu_kern_sum" "$dir/$variant.nsys-rep" > "$dir/${variant}_stats.log" 2>&1
done
