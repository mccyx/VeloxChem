#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <kernel-function> <input-file> <output-dir>" >&2
    exit 1
fi

kernel_func="$1"
input_file="$2"
out_dir="$3"
job="$(basename "${input_file}")"
job="${job%.*}"
stem="$(echo "${kernel_func#computeExchangeFock}" | tr '[:upper:]' '[:lower:]')"
report="${out_dir}/${job}.${stem}.ncu-rep"

mkdir -p "${out_dir}"
export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

ncu --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --kernel-name "regex:^${kernel_func}$" \
    --import-source yes \
    -o "${report}" \
    vlx "${input_file}"

echo "${report}"
