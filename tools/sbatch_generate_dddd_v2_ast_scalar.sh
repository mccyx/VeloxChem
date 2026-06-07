#!/usr/bin/env bash
#SBATCH -A pdc-software-test
#SBATCH -p gpugh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -t 00:55:00
#SBATCH -x nid002890
#SBATCH -J dddd-v2-ast-scalar

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

mkdir -p "${repo_root}/tools/profiling/slurm_logs"

exec > >(tee -a "${repo_root}/tools/profiling/slurm_logs/${SLURM_JOB_NAME:-dddd-v2-ast-scalar}_${SLURM_JOB_ID:-manual}.out")
exec 2> >(tee -a "${repo_root}/tools/profiling/slurm_logs/${SLURM_JOB_NAME:-dddd-v2-ast-scalar}_${SLURM_JOB_ID:-manual}.err" >&2)

cd "${repo_root}"

echo "Job ID:              ${SLURM_JOB_ID:-unknown}"
echo "Node list:           ${SLURM_JOB_NODELIST:-unknown}"
echo "Submit dir:          ${SLURM_SUBMIT_DIR:-unknown}"
echo "Repository:          ${repo_root}"
echo "DDDD_V2_INDICES:     ${DDDD_V2_INDICES:-0-25}"
echo "FP32 tag:            ${DDDD_AST_SCALAR_TAG_FP32:-auto_s_ast}"
echo "FP64 tag:            ${DDDD_AST_SCALAR_TAG_FP64:-auto_s_ast_fp64}"
echo

export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"

source env.sh

if [[ -z "${VLX_AST_COMPILER:-}" ]]; then
    for candidate in \
        /opt/cray/pe/cce/19.0.0/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/19.0.0/cce-clang/x86_64/bin/clang++ \
        /opt/cray/pe/cce/18.0.1/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/18.0.1/cce-clang/x86_64/bin/clang++ \
        /opt/cray/pe/cce/18.0.0/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/18.0.0/cce-clang/x86_64/bin/clang++; do
        if [[ -x "${candidate}" ]]; then
            export VLX_AST_COMPILER="${candidate}"
            break
        fi
    done
fi

echo "After env.sh:"
echo "  SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "  OMP_NUM_THREADS:     ${OMP_NUM_THREADS:-unset}"
echo "  VLXHOME:             ${VLXHOME:-unset}"
echo "  python3:             $(command -v python3)"
echo "  cc:                  $(command -v cc)"
echo "  clang++:             $(command -v clang++ || true)"
echo "  VLX_AST_COMPILER:    ${VLX_AST_COMPILER:-auto-detect-in-wrapper}"
echo

python3 tools/generate_dddd_variant_bundle.py \
    --function-set dddd-v2-fp32 \
    --indices "${DDDD_V2_INDICES:-0-25}" \
    --passes scalar \
    --tag "${DDDD_AST_SCALAR_TAG_FP32:-auto_s_ast}" \
    --matcher ast \
    --combine-output \
    --combined-prefix "${DDDD_AST_SCALAR_PREFIX_FP32:-dddd_v2_fp32_auto_s_ast}"

python3 tools/generate_dddd_variant_bundle.py \
    --function-set dddd-v2-fp64 \
    --indices "${DDDD_V2_INDICES:-0-25}" \
    --passes scalar \
    --tag "${DDDD_AST_SCALAR_TAG_FP64:-auto_s_ast_fp64}" \
    --matcher ast \
    --combine-output \
    --combined-prefix "${DDDD_AST_SCALAR_PREFIX_FP64:-dddd_v2_fp64_auto_s_ast}"
