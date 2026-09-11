#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J exchange-original-bench
#SBATCH -o build_logs/exchange_original_benchmark_%j.out
#SBATCH -e build_logs/exchange_original_benchmark_%j.err

set -euo pipefail

source_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-4-original
result_dir="$source_repo/build_logs/exchange_original_benchmark_${SLURM_JOB_ID}"
mkdir -p "$result_dir"

cd "$source_repo"
source env.sh
export VLXHOME="$repo"
export PYTHONPATH="$repo/build/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$repo/build/bin:$PATH"

prepare_run() {
    run_dir=$1
    mkdir -p "$run_dir"
    cp "$source_repo/guanine-8.xyz" "$run_dir/"
    sed '/mixed_precision_threshold_[jk]:/d' \
        "$source_repo/guanine-8-hf.inp" > "$run_dir/guanine-8-hf.inp"
}

for run in 1 2 3; do
    run_dir="$result_dir/host_run${run}"
    prepare_run "$run_dir"
    cd "$run_dir"
    vlx guanine-8-hf.inp > run.log 2>&1
done

run_dir="$result_dir/nsys"
prepare_run "$run_dir"
cd "$run_dir"
nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none \
    --stats=false -o profile vlx guanine-8-hf.inp > run.log 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv \
    --output cuda_gpu_kern_sum profile.nsys-rep > nsys-stats.log 2>&1
