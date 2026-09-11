#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:20:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH -J exchange-production-rebuild
#SBATCH -o build_logs/exchange_production_rebuild_%j.out
#SBATCH -e build_logs/exchange_production_rebuild_%j.err

set -euo pipefail

source_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.mixed-precision-exchange-test
log_dir="$source_repo/build_logs/exchange_production_rebuild_${SLURM_JOB_ID}"
mkdir -p "$log_dir"
cd "$source_repo"
source env.sh
export VLXHOME="$repo"
export PYTHONPATH="$repo/build/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$repo/build/bin:$PATH"
cd "$repo"
make -C src -j 144 2>&1 | tee "$log_dir/build.log"
