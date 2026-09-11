#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J exchange-production-nsys
#SBATCH -o build_logs/exchange_production_nsys_%j.out
#SBATCH -e build_logs/exchange_production_nsys_%j.err

set -euo pipefail

source_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.mixed-precision-exchange-test
result_dir="$source_repo/build_logs/exchange_production_nsys_${SLURM_JOB_ID}"
mkdir -p "$result_dir"

cd "$source_repo"
source env.sh
export VLXHOME="$repo"
export PYTHONPATH="$repo/build/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$repo/build/bin:$PATH"

for threshold in 1e-6 1e-5; do
    run_dir="$result_dir/k${threshold}"
    mkdir -p "$run_dir"
    cp "$source_repo/guanine-8.xyz" "$run_dir/"
    sed "s/mixed_precision_threshold_k: 1e-6/mixed_precision_threshold_k: $threshold/" \
        "$source_repo/guanine-8-hf.inp" > "$run_dir/guanine-8-hf.inp"
    cd "$run_dir"
    nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none \
        --stats=false -o profile vlx guanine-8-hf.inp > run.log 2>&1
    nsys stats --report cuda_gpu_kern_sum --format csv \
        --output cuda_gpu_kern_sum profile.nsys-rep > nsys-stats.log 2>&1
done
