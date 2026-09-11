#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH --array=0-3
#SBATCH -J exchange-series
#SBATCH -o build_logs/exchange_series_%A_%a.out
#SBATCH -e build_logs/exchange_series_%A_%a.err

set -euo pipefail

source_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
original_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-4-original
mp_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.mixed-precision-exchange-test
systems=(guanine-4 guanine-8 guanine-12 guanine-sugar)
system=${systems[$SLURM_ARRAY_TASK_ID]}
result_dir="$source_repo/build_logs/exchange_series_${SLURM_ARRAY_JOB_ID}/${system}"
mkdir -p "$result_dir"

cd "$source_repo"
source env.sh
base_path=$PATH
base_pythonpath=${PYTHONPATH-}

prepare_input() {
    run_dir=$1
    threshold=${2-}
    mkdir -p "$run_dir"
    cp "$source_repo/exchange_series_inputs/${system}.xyz" "$run_dir/"
    {
        printf '%s\n' \
            '@jobs' \
            'task: scf' \
            '@end' \
            '' \
            '@method_settings' \
            'basis: def2-svp' \
            '@end' \
            '' \
            '@scf' \
            'timing: yes' \
            'timing_gpu: yes' \
            'acc_type: diis' \
            'conv_thresh: 2e-4'
        if [[ -n "$threshold" ]]; then
            printf 'mixed_precision_threshold_j: 1e-6\n'
            printf 'mixed_precision_threshold_k: %s\n' "$threshold"
        fi
        printf '%s\n' \
            '@end' \
            '' \
            '@molecule' \
            'charge: 0' \
            'multiplicity: 1' \
            "xyzfile: ${system}.xyz" \
            '@end'
    } > "$run_dir/${system}.inp"
}

run_case() {
    label=$1
    repo=$2
    threshold=${3-}
    run_dir="$result_dir/$label"
    prepare_input "$run_dir" "$threshold"
    export VLXHOME="$repo"
    export PYTHONPATH="$repo/build/python${base_pythonpath:+:$base_pythonpath}"
    export PATH="$repo/build/bin:$base_path"
    cd "$run_dir"
    vlx "${system}.inp" > run.log 2>&1
    rm -f ./*.h5
}

run_case original_fp64 "$original_repo"
run_case k1e-6 "$mp_repo" 1e-6
run_case k1e-5 "$mp_repo" 1e-5
run_case k1e-4 "$mp_repo" 1e-4
