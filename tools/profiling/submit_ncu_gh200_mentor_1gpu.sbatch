#!/usr/bin/env bash
#SBATCH -J ncu-mentor-1gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 04:00:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890
#SBATCH --exclusive

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../.." && pwd)"
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

job="${1:-guanine-8}"
input_file="${2:-${job}.inp}"
run_timestamp="${VLX_PROFILING_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)_mentor_1gpu}"

cd "${repo_root}"
source env.sh

export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"
export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0
export VLX_PROFILING_TIMESTAMP="${run_timestamp}"
export VLX_NCU_OUT_DIR="${repo_root}/ncu_results/${run_timestamp}"

echo "Job ID:            ${SLURM_JOB_ID:-unknown}"
echo "Node list:         ${SLURM_JOB_NODELIST:-unknown}"
echo "Submit dir:        ${SLURM_SUBMIT_DIR:-unknown}"
echo "Repository:        ${repo_root}"
echo "Job prefix:        ${job}"
echo "Input file:        ${input_file}"
echo "NCU output dir:    ${VLX_NCU_OUT_DIR}"
echo

"${repo_root}/tools/profiling/ncu/profile_mentor_dddd_v2_fp32_scalar_compare_ncu.sh" "${job}" "${input_file}"
