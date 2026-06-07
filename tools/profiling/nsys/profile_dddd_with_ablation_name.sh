#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_file> [output_suffix]"
    echo
    echo "Environment:"
    echo "  VLX_ABLATION_LOG   Optional benchmark log path used to derive the nsys output name."
    echo
    echo "Examples:"
    echo "  VLX_ABLATION_LOG=benchmarks/ablation_results_gh200_guanine8_split26_2026-04-17.log \\"
    echo "    $0 guanine-8.inp"
    echo
    echo "  VLX_ABLATION_LOG=benchmarks/ablation_results_gh200_guanine8_split26_2026-04-17.log \\"
    echo "    $0 guanine-8.inp split_compare"
    exit 1
fi

input_file="$1"
output_suffix="${2:-}"

ablation_log="${VLX_ABLATION_LOG:-ablation_results.log}"
ablation_base="$(basename "${ablation_log}")"
ablation_stem="${ablation_base%.log}"
run_timestamp="${VLX_PROFILING_TIMESTAMP:-}"

input_base="$(basename "${input_file}")"
input_stem="${input_base%.*}"

if [[ -n "${output_suffix}" ]]; then
    profile_stem="${ablation_stem}_${output_suffix}"
else
    profile_stem="${ablation_stem}_${input_stem}"
fi

if [[ -n "${run_timestamp}" ]]; then
    out_dir="${VLX_NSYS_OUT_DIR:-nsys_results/${ablation_stem}/${run_timestamp}}"
else
    out_dir="${VLX_NSYS_OUT_DIR:-nsys_results/${ablation_stem}}"
fi
profile_path="${out_dir}/${profile_stem}"

mkdir -p "${out_dir}"

echo "Ablation log:      ${ablation_log}"
echo "Input file:        ${input_file}"
if [[ -n "${run_timestamp}" ]]; then
    echo "Run timestamp:     ${run_timestamp}"
fi
echo "Output directory:  ${out_dir}"
echo "Profile stem:      ${profile_stem}"

nsys profile \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --sample=none \
    --stats=false \
    -o "${profile_path}" \
    vlx "${input_file}"

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "${profile_path}_cuda_gpu_kern_sum" \
    "${profile_path}.nsys-rep"

echo
echo "Generated:"
echo "  ${profile_path}.nsys-rep"
echo "  ${profile_path}_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv"
