#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J dpdd-k4-ncu
#SBATCH -o ncu_results/dpdd_k4_%j.out
#SBATCH -e ncu_results/dpdd_k4_%j.err
set -euo pipefail
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3;dir="$repo/ncu_results/dpdd_rs_k4_${SLURM_JOB_ID}";prof="$repo/tools/profiling/ncu/profile_exact_exchange_instructions_ncu.sh";mkdir -p "$dir";cd "$repo";source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=k4_m23 VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime VLX_EXCHANGE_DDDP_SPLIT=old_k5 VLX_EXCHANGE_DPDD_SPLIT=rs_k4
for i in $(seq 0 6);do "$prof" "computeExchangeFockDPDD${i}_FP32" guanine-8-hf.inp "$dir";done
for i in $(seq 0 3);do "$prof" "computeExchangeFockDPDD${i}_K4_RS_FP32" guanine-8-hf.inp "$dir";done
