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
echo "Kernel family:     split26_v2_fp32 baseline+auto_s_ast"
echo "OMP_NUM_THREADS:   ${OMP_NUM_THREADS}"
echo "OMP_PLACES:        ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

for i in $(seq 0 25); do
    echo "===== DDDDv2_${i} FP32 baseline ====="
    "${script_dir}/profile_and_export_ncu_fp32.sh" "computeCoulombFockDDDDv2_${i}_FP32" "${input_file}"
    echo

    echo "===== DDDDv2_${i} FP32 auto_s_ast ====="
    "${script_dir}/profile_and_export_ncu_fp32.sh" "computeCoulombFockDDDDv2_${i}_FP32_auto_s_ast" "${input_file}"
    echo
done
