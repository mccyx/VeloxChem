#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage:"
    echo "  $0 profile <input_file> [output_suffix]"
    echo "  $0 summarize <ablation_log>"
    echo "  $0 full <ablation_log> <input_file> [output_suffix]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
mode="$1"
shift
run_timestamp="${VLX_PROFILING_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "${repo_root}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
source env.sh

# Force the advisor-recommended single-GH200-GPU profiling setup after env.sh.
export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0
export VLX_PROFILING_TIMESTAMP="${run_timestamp}"

echo "Repository:        ${repo_root}"
echo "Mode:              ${mode}"
echo "Run timestamp:     ${VLX_PROFILING_TIMESTAMP}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS:   ${OMP_NUM_THREADS}"
echo "OMP_PLACES:        ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

case "${mode}" in
    profile)
        exec "${repo_root}/tools/profiling/nsys/profile_dddd_with_ablation_name.sh" "$@"
        ;;
    summarize)
        exec "${repo_root}/tools/profiling/nsys/summarize_from_ablation_name.sh" "$@"
        ;;
    full)
        if [[ $# -lt 2 ]]; then
            echo "Usage: $0 full <ablation_log> <input_file> [output_suffix]" >&2
            exit 1
        fi
        export VLX_ABLATION_LOG="$1"
        shift
        "${repo_root}/tools/profiling/nsys/profile_dddd_with_ablation_name.sh" "$@"
        "${repo_root}/tools/profiling/nsys/summarize_from_ablation_name.sh" "${VLX_ABLATION_LOG}"
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        echo "Expected one of: profile, summarize, full" >&2
        exit 1
        ;;
esac
