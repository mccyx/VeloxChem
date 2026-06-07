#!/usr/bin/env bash
#SBATCH -J ncu-gh200
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 02:00:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890

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

mode="${1:-all-dddd-v2-fp32}"
job="${2:-guanine-8}"
input_file="${3:-${job}.inp}"

mkdir -p "${repo_root}/tools/profiling/slurm_logs"

echo "Job ID:            ${SLURM_JOB_ID:-unknown}"
echo "Node list:         ${SLURM_JOB_NODELIST:-unknown}"
echo "Submit dir:        ${SLURM_SUBMIT_DIR:-unknown}"
echo "Repository:        ${repo_root}"
echo "Mode:              ${mode}"
echo "Job prefix:        ${job}"
echo "Input file:        ${input_file}"
echo

cd "${repo_root}"

"${repo_root}/tools/profiling/run_ncu_gh200_one_gpu.sh" "${mode}" "${job}" "${input_file}"
