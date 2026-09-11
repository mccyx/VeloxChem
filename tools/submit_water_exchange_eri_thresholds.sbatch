#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -c 288
#SBATCH -p gpugh
#SBATCH -t 00:55:00
#SBATCH -A pdc-software-test
#SBATCH -x nid002890,nid002891
#SBATCH --array=0-11%3
#SBATCH -J water-exchange-eri
#SBATCH -o build_logs/water_exchange_eri_%A_%a.out
#SBATCH -e build_logs/water_exchange_eri_%A_%a.err

set -euo pipefail

source_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-mixed-precision-3
original_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.gpu-dev-4-original
mp_repo=/cfs/klemming/home/y/yuch4126/work/VeloxChem.mixed-precision-exchange-test
scratch_root=/cfs/klemming/scratch/y/yuch4126/veloxchem_exchange_water
systems=(006 010 014)
labels=(original_fp64 k1e-6 k1e-5 k1e-4)
thresholds=('' 1e-6 1e-5 1e-4)

system_index=$((SLURM_ARRAY_TASK_ID / 4))
case_index=$((SLURM_ARRAY_TASK_ID % 4))
system=${systems[$system_index]}
label=${labels[$case_index]}
threshold=${thresholds[$case_index]}
run_dir="$scratch_root/${SLURM_ARRAY_JOB_ID}/${system}/${label}"
result_dir="$source_repo/build_logs/water_exchange_eri_${SLURM_ARRAY_JOB_ID}/${system}/${label}"
mkdir -p "$run_dir" "$result_dir"

cd "$source_repo"
source env.sh
base_path=$PATH
base_pythonpath=${PYTHONPATH-}
cp "$source_repo/water_exchange_inputs/${system}.xyz" "$run_dir/"

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

if [[ "$label" == original_fp64 ]]; then
    repo=$original_repo
else
    repo=$mp_repo
fi
export VLXHOME="$repo"
export PYTHONPATH="$repo/build/python${base_pythonpath:+:$base_pythonpath}"
export PATH="$repo/build/bin:$base_path"

cd "$run_dir"
set +e
vlx "${system}.inp" > run.log 2>&1
status=$?
set -e
cp "${system}.inp" run.log "$result_dir/"
rm -f ./*.h5
exit "$status"
