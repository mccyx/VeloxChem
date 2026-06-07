#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  $0 single-fp32 <kernel> [input_file]"
    echo "  $0 single-fp64 <kernel> [input_file]"
    echo "  $0 all-dddd [job] [input_file]"
    echo "  $0 all-dddd-v2 [job] [input_file]"
    echo "  $0 all-dddd-v2-fp64 [job] [input_file]"
    echo "  $0 all-dddd-v2-fp32 [job] [input_file]"
    echo "  $0 all-dddd-v2-fp32-scalar-compare [job] [input_file]"
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
export VLX_NCU_OUT_DIR="${VLX_NCU_OUT_DIR:-${repo_root}/ncu_results/${run_timestamp}}"

echo "Repository:        ${repo_root}"
echo "Mode:              ${mode}"
echo "Run timestamp:     ${VLX_PROFILING_TIMESTAMP}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "NCU output dir:    ${VLX_NCU_OUT_DIR}"
echo "OMP_NUM_THREADS:   ${OMP_NUM_THREADS}"
echo "OMP_PLACES:        ${OMP_PLACES}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

case "${mode}" in
    single-fp32)
        exec "${repo_root}/tools/profiling/ncu/profile_and_export_ncu_fp32.sh" "$@"
        ;;
    single-fp64)
        exec "${repo_root}/tools/profiling/ncu/profile_and_export_ncu_fp64.sh" "$@"
        ;;
    all-dddd)
        exec "${repo_root}/tools/profiling/ncu/profile_all_dddd_ncu.sh" "$@"
        ;;
    all-dddd-v2)
        exec "${repo_root}/tools/profiling/ncu/profile_all_dddd_v2_ncu.sh" "$@"
        ;;
    all-dddd-v2-fp64)
        exec "${repo_root}/tools/profiling/ncu/profile_all_dddd_v2_fp64_ncu.sh" "$@"
        ;;
    all-dddd-v2-fp32)
        exec "${repo_root}/tools/profiling/ncu/profile_all_dddd_v2_fp32_ncu.sh" "$@"
        ;;
    all-dddd-v2-fp32-scalar-compare)
        exec "${repo_root}/tools/profiling/ncu/profile_all_dddd_v2_fp32_scalar_compare_ncu.sh" "$@"
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        echo "Expected one of: single-fp32, single-fp64, all-dddd, all-dddd-v2, all-dddd-v2-fp64, all-dddd-v2-fp32, all-dddd-v2-fp32-scalar-compare" >&2
        exit 1
        ;;
esac
