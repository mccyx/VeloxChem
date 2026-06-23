#!/usr/bin/env bash
#SBATCH -A pdc-software-test
#SBATCH -p gpugh
#SBATCH -J gsp-1em6-4gpu
#SBATCH -t 00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --output=/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/gsp_1em6_4gpu_%j.out
#SBATCH --error=/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/gsp_1em6_4gpu_%j.err
#SBATCH -x nid002890

set -euo pipefail

export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"

/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/run_gsp_1em6_4gpu_nsys.sh
