#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
output_dir="${2:-ncu_results/ppdd_old_rs_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kernels=(
    computeExchangeFockPPDD_FP32
    computeExchangeFockPPDD0_RS_FP32
    computeExchangeFockPPDD1_RS_FP32
    computeExchangeFockPPDD_FP64
    computeExchangeFockPPDD0_RS_FP64
    computeExchangeFockPPDD1_RS_FP64
)

mkdir -p "${output_dir}"
for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exact_exchange_ncu.sh" \
        "${kernel}" "${input_file}" "${output_dir}"
done

echo "PPDD old-versus-RS reports: ${output_dir}"
