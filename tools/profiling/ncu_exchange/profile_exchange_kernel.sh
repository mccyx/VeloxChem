#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <exact-kernel-name> <input-file> [output-directory]"
    exit 1
fi

kernel_func="$1"
input_file="$2"
out_dir="${3:-ncu_results/exchange_typical_$(date +%Y%m%d_%H%M%S)}"
helper_dir="/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-experiments/tools/profiling/ncu"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-\{0\}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${kernel_func}" in
    computeExchangeFockPDDD*_K4_M23_*) export VLX_EXCHANGE_PDDD_SPLIT=k4_m23 ;;
    computeExchangeFockDPDD*_K4_RS_*) export VLX_EXCHANGE_DPDD_SPLIT=rs_k4 ;;
    computeExchangeFockDDDP*_K5_OLD_*) export VLX_EXCHANGE_DDDP_SPLIT=old_k5 ;;
    computeExchangeFockDDDD*_K16_OLD_RUNTIME_*) export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime ;;
esac

mkdir -p "${out_dir}"
stem="$(printf '%s' "${kernel_func#computeExchangeFock}" | tr '[:upper:]' '[:lower:]')"
job="$(basename "${input_file}")"
job="${job%.*}"
report="${out_dir}/${job}.${stem}.ncu-rep"
details="${out_dir}/${job}.${stem}.details.txt"

echo "Profiling kernel: ${kernel_func}"
echo "Output report:    ${report}"
ncu --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --kernel-name "regex:^${kernel_func}$" \
    --import-source yes \
    --force-overwrite \
    -o "${report}" \
    vlx "${input_file}"

ncu --import "${report}" --page details > "${details}"
"${helper_dir}/export_ncu_artifacts.sh" "${report}" "${out_dir}"
