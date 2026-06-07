#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] use tools/profiling/run_ncu_gh200_one_gpu.sh instead" >&2
exec "${script_dir}/run_ncu_gh200_one_gpu.sh" "$@"
