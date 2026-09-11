#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
out_dir="${2:-ncu_results/exchange_typical_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

# PDDD winner group 2 is the merge of RS groups 2 and 3.
kernels=(
    computeExchangeFockPDDD2_RS_FP32
    computeExchangeFockPDDD3_RS_FP32
    computeExchangeFockPDDD2_K4_M23_FP32

    # DPDD is the strongest complete-layout instruction reduction.
    computeExchangeFockDPDD1_FP32
    computeExchangeFockDPDD2_FP32
    computeExchangeFockDPDD1_K4_RS_FP32

    # DDDD is the marginal-win control: old groups 0 and 1 merge into K16 group 0.
    computeExchangeFockDDDD0_FP32
    computeExchangeFockDDDD1_FP32
    computeExchangeFockDDDD0_K16_OLD_RUNTIME_FP32
)

mkdir -p "${out_dir}"
printf '%s\n' "${kernels[@]}" > "${out_dir}/kernel_manifest.txt"

for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exchange_kernel.sh" "${kernel}" "${input_file}" "${out_dir}"
done

echo "Reports written to ${out_dir}"
