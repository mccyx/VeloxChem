#!/usr/bin/env bash
set -euo pipefail

input_file="${1:-guanine-8-hf.inp}"
out_dir="${2:-ncu_results/exchange_old_typical_$(date +%Y%m%d_%H%M%S)}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

# Highest old-layout FP32 times in the all-exchange Nsys baseline, excluding
# families already studied through detailed resplitting experiments.
kernels=(
    computeExchangeFockPPDP_FP32
    computeExchangeFockPPPP_FP32
    computeExchangeFockSPPP_FP32
    computeExchangeFockPDDP_FP32
    computeExchangeFockPPDD_FP32
    computeExchangeFockPDPP_FP32
)

mkdir -p "${out_dir}"
printf '%s\n' "${kernels[@]}" > "${out_dir}/kernel_manifest.txt"

for kernel in "${kernels[@]}"; do
    "${script_dir}/profile_exchange_kernel.sh" "${kernel}" "${input_file}" "${out_dir}"
done

echo "Reports written to ${out_dir}"
