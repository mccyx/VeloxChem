#!/usr/bin/env bash
#SBATCH -A pdc-software-test
#SBATCH -p gpugh
#SBATCH -J resplit-scalar
#SBATCH -t 00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --output=/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/resplit_scalar_%j.out
#SBATCH --error=/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/resplit_scalar_%j.err
#SBATCH -x nid002890

set -euo pipefail

repo_root="/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2"
job_dir="${repo_root}/nsys_results/resplit_scalar_${SLURM_JOB_ID}"
mkdir -p "${job_dir}"

cd "${repo_root}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"

echo "Job: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_JOB_NODELIST}"
echo "Repo: ${repo_root}"
echo "Output: ${job_dir}"
echo

source env.sh

echo "Environment after env.sh:"
echo "  VLXHOME=${VLXHOME:-}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
echo "  OMP_PLACES=${OMP_PLACES:-}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

echo "Building..."
make -C src -j144 2>&1 | tee "${job_dir}/build.log"

export VLX_ABLATION_LOG="${job_dir}/ablation_results_resplit_scalar.log"

echo
echo "Running correctness check..."
vlx guanine-8.inp 2>&1 | tee "${job_dir}/guanine8_correctness.log"

echo
echo "Running nsys profile..."
export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0

profile_path="${job_dir}/guanine8_resplit_scalar"
nsys profile \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --sample=none \
    --stats=false \
    -o "${profile_path}" \
    vlx guanine-8.inp 2>&1 | tee "${job_dir}/guanine8_nsys_run.log"

nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "${profile_path}_cuda_gpu_kern_sum" \
    "${profile_path}.nsys-rep" 2>&1 | tee "${job_dir}/nsys_stats.log"

kernel_csv="${profile_path}_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv"
filtered_csv="${job_dir}/coulomb_target_kernel_times.csv"
filtered_md="${job_dir}/coulomb_target_kernel_times.md"

python3 - "${kernel_csv}" "${filtered_csv}" "${filtered_md}" <<'PY'
import csv
import re
import sys
from pathlib import Path

kernel_csv = Path(sys.argv[1])
filtered_csv = Path(sys.argv[2])
filtered_md = Path(sys.argv[3])

patterns = [
    "computeCoulombFockPPDD",
    "computeCoulombFockPDPD",
    "computeCoulombFockDDSD",
    "computeCoulombFockDDPP",
    "computeCoulombFockDDPD",
]

rows = []
with kernel_csv.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        name = row.get("Name", "")
        if not any(pattern in name for pattern in patterns):
            continue
        short = name
        short = re.sub(r"^gpu::", "", short)
        short = short.split("(", 1)[0]
        total_ms = float(row["Total Time (ns)"]) / 1.0e6
        avg_ms = float(row["Avg (ns)"]) / 1.0e6
        rows.append(
            {
                "kernel": short,
                "instances": int(row["Instances"]),
                "total_ms": total_ms,
                "avg_ms": avg_ms,
                "time_percent": float(row["Time (%)"]),
            }
        )

rows.sort(key=lambda item: (item["kernel"], item["avg_ms"]))

with filtered_csv.open("w", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["kernel", "instances", "total_ms", "avg_ms", "time_percent"],
    )
    writer.writeheader()
    writer.writerows(rows)

with filtered_md.open("w") as fh:
    fh.write("# Coulomb Target Kernel Times\\n\\n")
    fh.write(f"Source: `{kernel_csv}`\\n\\n")
    fh.write("| Kernel | Instances | Total ms | Avg ms | Time % |\\n")
    fh.write("|---|---:|---:|---:|---:|\\n")
    for row in rows:
        fh.write(
            "| {kernel} | {instances} | {total_ms:.6f} | {avg_ms:.6f} | {time_percent:.3f} |\\n".format(
                **row
            )
        )

print(f"Wrote {filtered_csv}")
print(f"Wrote {filtered_md}")
PY

echo
echo "Done."
echo "Outputs:"
echo "  ${job_dir}/build.log"
echo "  ${job_dir}/guanine8_correctness.log"
echo "  ${job_dir}/ablation_results_resplit_scalar.log"
echo "  ${profile_path}.nsys-rep"
echo "  ${kernel_csv}"
echo "  ${filtered_csv}"
echo "  ${filtered_md}"
