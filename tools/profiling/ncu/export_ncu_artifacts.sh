#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <report.ncu-rep> [output_dir]"
    echo
    echo "Exports three companion files used by the helper Python scripts:"
    echo "  - *.summary.txt"
    echo "  - *.source.csv"
    echo "  - *.warpstate.csv"
    exit 1
fi

report_path="$1"
output_dir="${2:-$(dirname "${report_path}")}"

report_base="$(basename "${report_path}")"
stem="${report_base%.ncu-rep}"

summary_path="${output_dir}/${stem}.summary.txt"
source_path="${output_dir}/${stem}.source.csv"
warpstate_path="${output_dir}/${stem}.warpstate.csv"

mkdir -p "${output_dir}"

echo "Import report:     ${report_path}"
echo "Summary output:    ${summary_path}"
echo "Source output:     ${source_path}"
echo "Warpstate output:  ${warpstate_path}"

# Summary text used by extract_summary_metrics.py
ncu --import "${report_path}" > "${summary_path}"

# Source CSV with SASS annotations used by top_kernel_metrics.py and assess_kernels.py
ncu --import "${report_path}" --page source --print-source sass --csv > "${source_path}"

# Warp-state CSV used by top_warpstate_metrics.py and assess_kernels.py
ncu --import "${report_path}" --section WarpStateStats --page details --print-details all --csv > "${warpstate_path}"

echo "Done."
