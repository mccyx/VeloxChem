#!/usr/bin/env bash
set -euo pipefail

ablation_log="${1:-${VLX_ABLATION_LOG:-ablation_results.log}}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/summarize_dddd_split_benchmark.py" "${ablation_log}"
