#!/usr/bin/env bash
set -euo pipefail

job="${1:-guanine-8}"
input_file="${2:-${job}.inp}"

out_dir="${VLX_NCU_OUT_DIR:-ncu_results/$(date +%Y%m%d_%H%M%S)_mentor}"
mkdir -p "${out_dir}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-\{0\}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Job:               ${job}"
echo "Input file:        ${input_file}"
echo "Kernel family:     split26_v2_fp32 baseline+auto_s_ast"
echo "NCU command style: mentor, no launch-skip/launch-count"
echo "NCU output dir:    ${out_dir}"
echo "OMP_NUM_THREADS:   ${OMP_NUM_THREADS}"
echo "OMP_PLACES:        ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

profile_one() {
    local kernel_name="$1"
    local stem="$2"
    local report="${out_dir}/${job}.${stem}.ncu-rep"

    echo "ncu --set full --kernel-name \"${kernel_name}\" -f -o \"${report}\" vlx \"${input_file}\""
    ncu --set full --kernel-name "${kernel_name}" -f -o "${report}" vlx "${input_file}"
    ncu --import "${report}" --page source --print-source sass --csv \
        > "${out_dir}/${job}.${stem}.source.csv"
    ncu --import "${report}" --section WarpStateStats --page details --print-details all --csv \
        > "${out_dir}/${job}.${stem}.warpstate.csv"
    ncu --import "${report}" --page details \
        > "${out_dir}/${job}.${stem}.details.txt"
    ncu --import "${report}" \
        > "${out_dir}/${job}.${stem}.summary.txt"
}

for i in $(seq 0 25); do
    echo "===== DDDDv2_${i} FP32 baseline ====="
    profile_one "computeCoulombFockDDDDv2_${i}_FP32" "ddddv2_${i}_fp32"
    echo

    echo "===== DDDDv2_${i} FP32 auto_s_ast ====="
    profile_one "computeCoulombFockDDDDv2_${i}_FP32_auto_s_ast" "ddddv2_${i}_fp32_auto_s_ast"
    echo
done
