#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
out_dir="${2:-ncu_results/exchange_typical_dddp_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

# Selected old-derived K5 mapping: [0] [1+2] [3] [4+5] [6].
kernels=(
    computeExchangeFockDDDP1_FP32
    computeExchangeFockDDDP2_FP32
    computeExchangeFockDDDP1_K5_OLD_FP32
    computeExchangeFockDDDP4_FP32
    computeExchangeFockDDDP5_FP32
    computeExchangeFockDDDP3_K5_OLD_FP32
)

mkdir -p "${out_dir}"
printf '%s\n' "${kernels[@]}" > "${out_dir}/kernel_manifest.txt"

for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exchange_kernel.sh" "${kernel}" "${input_file}" "${out_dir}"
done

echo "Reports written to ${out_dir}"
