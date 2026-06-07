#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <kernel> [input_file]"
    echo
    echo "Examples:"
    echo "  $0 DDDD3"
    echo "  $0 computeCoulombFockDDDD3_FP64 guanine-8.inp"
    exit 1
fi

kernel_arg="$1"
input_file="${2:-guanine-8.inp}"
out_dir="${VLX_NCU_OUT_DIR:-ncu_results}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-\{0\}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${kernel_arg}" == computeCoulombFock* ]]; then
    kernel_func="${kernel_arg}"
else
    kernel_func="computeCoulombFock${kernel_arg}_FP64"
fi

mkdir -p "${out_dir}"

stem="$(echo "${kernel_func#computeCoulombFock}" | tr '[:upper:]' '[:lower:]')"
job="$(basename "${input_file}")"
job="${job%.*}"
report="${out_dir}/${job}.${stem}.ncu-rep"
details="${out_dir}/${stem}.txt"

echo "Profiling kernel: ${kernel_func}"
echo "Input file:       ${input_file}"
echo "OMP_NUM_THREADS:  ${OMP_NUM_THREADS}"
echo "OMP_PLACES:       ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output directory: ${out_dir}"
echo "Report:           ${report}"
echo "Details:          ${details}"
ncu --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --kernel-name "regex:^${kernel_func}$" \
    --import-source yes \
    -o "${report}" \
    vlx "${input_file}"

ncu --import "${report}" --page details > "${details}"

echo "Done."
