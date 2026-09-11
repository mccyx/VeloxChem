#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
output_dir="${2:-ncu_results/pddd_fp32_instructions_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kernels=()
for index in $(seq 0 7); do
    kernels+=("computeExchangeFockPDDD${index}_FP32")
done
for index in $(seq 0 4); do
    kernels+=("computeExchangeFockPDDD${index}_RS_FP32")
done

mkdir -p "${output_dir}"
for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exact_exchange_instructions_ncu.sh" \
        "${kernel}" "${input_file}" "${output_dir}"
done

echo "PDDD FP32 instruction reports: ${output_dir}"
