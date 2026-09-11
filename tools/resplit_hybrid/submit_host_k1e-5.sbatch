#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J exchange-k1e-5-host
#SBATCH -o build_logs/exchange_winners_k1e-5_%j.out
#SBATCH -e build_logs/exchange_winners_k1e-5_%j.err

set -euo pipefail

repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
result_dir="$repo/build_logs/exchange_winners_k1e-5_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
cd "$repo"
source env.sh

export VLX_EXCHANGE_PDDD_SPLIT=k4_m23
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
export VLX_EXCHANGE_DDDP_SPLIT=old_k5
export VLX_EXCHANGE_DPDD_SPLIT=rs_k4

for repeat in 1 2 3; do
    export VLX_ABLATION_LOG="$result_dir/run${repeat}_ablation.log"
    vlx guanine-8-hf-k1e-5.inp > "$result_dir/run${repeat}.log" 2>&1
done

python3 tools/resplit_hybrid/summarize_host.py "$result_dir"
