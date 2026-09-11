#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dpdd-split-nsys
#SBATCH -o nsys_results/dpdd_split_%j.out
#SBATCH -e nsys_results/dpdd_split_%j.err
set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3;dir="$repo/nsys_results/dpdd_split_${SLURM_JOB_ID}";mkdir -p "$dir";cd "$repo";source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=k4_m23 VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime VLX_EXCHANGE_DDDP_SPLIT=old_k5
for v in rs5 old_k5 rs_k4 old_k4;do export VLX_EXCHANGE_DPDD_SPLIT="$v" VLX_ABLATION_LOG="$dir/${v}_ablation.log";nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --stats=false -o "$dir/$v" vlx guanine-8-hf.inp >"$dir/$v.log" 2>&1;nsys stats --report cuda_gpu_kern_sum --format csv --output "$dir/${v}_cuda_gpu_kern_sum" "$dir/$v.nsys-rep" >"$dir/${v}_stats.log" 2>&1;done
