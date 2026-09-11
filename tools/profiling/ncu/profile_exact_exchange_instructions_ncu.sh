#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <exact-kernel-name> <input-file> [output-directory]" >&2
    exit 1
fi

kernel_name="$1"
input_file="$2"
output_dir="${3:-ncu_results/exchange_instructions_$(date +%Y%m%d_%H%M%S)}"

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
report="${output_dir}/${job}.${stem}.instructions.ncu-rep"
summary="${output_dir}/${job}.${stem}.instructions.txt"
csv_output="${output_dir}/${job}.${stem}.instructions.csv"

metrics=(
    smsp__inst_executed.sum
    smsp__inst_issued.sum
    smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
)
metric_list="$(IFS=,; echo "${metrics[*]}")"

echo "Kernel:        ${kernel_name}"
echo "Input:         ${input_file}"
echo "Output report: ${report}"

ncu --metrics "${metric_list}" \
    --launch-skip 0 \
    --launch-count 1 \
    --kernel-name "regex:^${kernel_name}$" \
    --force-overwrite \
    -o "${report}" \
    vlx "${input_file}"

ncu --import "${report}" > "${summary}"
ncu --import "${report}" --csv > "${csv_output}"

printf '%s\n' "${report}" "${summary}" "${csv_output}"
