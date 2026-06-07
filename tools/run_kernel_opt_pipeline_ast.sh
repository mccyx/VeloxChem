#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

find_ast_compiler() {
    if [[ -n "${VLX_AST_COMPILER:-}" && -x "${VLX_AST_COMPILER}" ]]; then
        printf '%s\n' "${VLX_AST_COMPILER}"
        return 0
    fi

    if command -v clang++ >/dev/null 2>&1; then
        command -v clang++
        return 0
    fi

    local candidate
    for candidate in \
        /opt/cray/pe/cce/19.0.0/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/19.0.0/cce-clang/x86_64/bin/clang++ \
        /opt/cray/pe/cce/18.0.1/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/18.0.1/cce-clang/x86_64/bin/clang++ \
        /opt/cray/pe/cce/18.0.0/cce-clang/aarch64/bin/clang++ \
        /opt/cray/pe/cce/18.0.0/cce-clang/x86_64/bin/clang++; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "Could not find a clang++ AST compiler. Set VLX_AST_COMPILER=/path/to/clang++." >&2
    return 1
}

ast_compiler="$(find_ast_compiler)"

exec python3 "${repo_root}/tools/kernel_opt_pipeline.py" --matcher ast --ast-compiler "${ast_compiler}" "$@"
