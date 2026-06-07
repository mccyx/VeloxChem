#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <kernel> [input_file]"
    echo
    echo "Examples:"
    echo "  $0 DDDD26"
    echo "  $0 computeCoulombFockDDDD26_FP32 guanine-8.inp"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel_arg="$1"
input_file="${2:-guanine-8.inp}"

"${script_dir}/profile_ncu_fp32.sh" "${kernel_arg}" "${input_file}"

if [[ "${kernel_arg}" == computeCoulombFock* ]]; then
    kernel_func="${kernel_arg}"
else
    kernel_func="computeCoulombFock${kernel_arg}_FP32"
fi

stem="$(echo "${kernel_func#computeCoulombFock}" | tr '[:upper:]' '[:lower:]')"
job="$(basename "${input_file}")"
job="${job%.*}"
out_dir="${VLX_NCU_OUT_DIR:-ncu_results}"
report_path="${out_dir}/${job}.${stem}.ncu-rep"

"${script_dir}/export_ncu_artifacts.sh" "${report_path}" "${out_dir}"

echo
echo "Suggested next steps:"
echo "  python3 ${script_dir}/extract_summary_metrics.py ${out_dir}/${job}.${stem}.summary.txt"
echo "  python3 ${script_dir}/top_kernel_metrics.py ${out_dir}/${job}.${stem}.source.csv"
echo "  python3 ${script_dir}/top_warpstate_metrics.py ${out_dir}/${job}.${stem}.warpstate.csv"
