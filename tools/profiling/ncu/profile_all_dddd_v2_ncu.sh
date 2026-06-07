#!/usr/bin/env bash
set -euo pipefail

job="${1:-guanine-8}"
input_file="${2:-${job}.inp}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-\{0\}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Job:               ${job}"
echo "Input file:        ${input_file}"
echo "Kernel family:     split26_v2_fp64+fp32"
echo "OMP_NUM_THREADS:   ${OMP_NUM_THREADS}"
echo "OMP_PLACES:        ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

"${script_dir}/profile_all_dddd_v2_fp64_ncu.sh" "${job}" "${input_file}"
"${script_dir}/profile_all_dddd_v2_fp32_ncu.sh" "${job}" "${input_file}"
