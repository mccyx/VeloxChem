#!/usr/bin/env bash
set -euo pipefail

repo_root="/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2"
job_id="${SLURM_JOB_ID:-manual}"
job_dir="${repo_root}/nsys_results/gsp_1em6_4gpu_${job_id}"

mkdir -p "${job_dir}"
cd "${repo_root}"

source env.sh
unset CUDA_VISIBLE_DEVICES
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"
export VLX_ABLATION_LOG="${job_dir}/ablation_results_gsp_1em6_4gpu.log"

echo "Job dir: ${job_dir}"
echo "Input: guanine-sugar-phosphate_1em6.inp"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS-}"
echo "OMP_PLACES=${OMP_PLACES-}"
echo

make -C src -j144 2>&1 | tee "${job_dir}/build.log"

vlx guanine-sugar-phosphate_1em6.inp 2>&1 | tee "${job_dir}/gsp_1em6_correctness_4gpu.log"

nsys profile \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --sample=none \
    --stats=false \
    -o "${job_dir}/gsp_1em6_4gpu" \
    vlx guanine-sugar-phosphate_1em6.inp 2>&1 | tee "${job_dir}/gsp_1em6_nsys_4gpu_run.log"

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "${job_dir}/gsp_1em6_4gpu_cuda_gpu_kern_sum" \
    "${job_dir}/gsp_1em6_4gpu.nsys-rep" 2>&1 | tee "${job_dir}/nsys_stats.log"

echo
echo "Done."
echo "${job_dir}"
