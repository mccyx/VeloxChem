#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <exact-kernel-name> <input-file> [output-directory]" >&2
    exit 1
fi

kernel_name="$1"
input_file="$2"
output_dir="${3:-ncu_results/exchange_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "${kernel_name}" =~ ^computeExchangeFock[A-Za-z0-9_]+$ ]]; then
    echo "Expected an exact computeExchangeFock kernel name, got: ${kernel_name}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-\{0\}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${output_dir}"

job="$(basename "${input_file}")"
job="${job%.*}"
stem="$(printf '%s' "${kernel_name#computeExchangeFock}" | tr '[:upper:]' '[:lower:]')"
report="${output_dir}/${job}.${stem}.ncu-rep"
summary="${output_dir}/${job}.${stem}.summary.txt"
source_csv="${output_dir}/${job}.${stem}.source.csv"
warpstate_csv="${output_dir}/${job}.${stem}.warpstate.csv"

echo "Kernel:        ${kernel_name}"
echo "Input:         ${input_file}"
echo "Output report: ${report}"

ncu --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --kernel-name "regex:^${kernel_name}$" \
    --import-source yes \
    --force-overwrite \
    -o "${report}" \
    vlx "${input_file}"

ncu --import "${report}" > "${summary}"
ncu --import "${report}" --page source --print-source sass --csv > "${source_csv}"
ncu --import "${report}" --section WarpStateStats --page details \
    --print-details all --csv > "${warpstate_csv}"

printf '%s\n' "${report}" "${summary}" "${source_csv}" "${warpstate_csv}"
