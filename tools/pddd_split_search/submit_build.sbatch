#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J pddd-k4-build
#SBATCH -o build_logs/pddd_k4_build_%j.out
#SBATCH -e build_logs/pddd_k4_build_%j.err

set -euo pipefail

repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
log_dir="$repo/build_logs/pddd_k4_build_${SLURM_JOB_ID}"
mkdir -p "$log_dir"
cd "$repo"
source env.sh
make -C src -j 144 2>&1 | tee "$log_dir/build.log"
