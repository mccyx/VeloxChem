#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
out_dir="${2:-ncu_results/exchange_typical_dpdd_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0
export VLX_EXCHANGE_DPDD_SPLIT=rs5

# The selected K4 RS group 1 is the direct merge of supplied RS5 groups 1+2.
kernels=(
    computeExchangeFockDPDD1_RS_FP32
    computeExchangeFockDPDD2_RS_FP32
)

mkdir -p "${out_dir}"
printf '%s\n' "${kernels[@]}" > "${out_dir}/kernel_manifest.txt"

for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exchange_kernel.sh" "${kernel}" "${input_file}" "${out_dir}"
done

echo "Reports written to ${out_dir}"
