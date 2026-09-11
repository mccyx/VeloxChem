#!/usr/bin/env python3

"""Summarize normalized DDDD K16 kernel durations from Nsys."""

import csv
import re
import sys
from pathlib import Path


VARIANTS = {"rs_k16_runtime": "K16_RS_RUNTIME", "old_k16_runtime": "K16_OLD_RUNTIME"}
KERNEL_RE = re.compile(r"computeExchangeFockDDDD(?P<index>\d+)(?P<suffix>_[A-Z0-9_]+)?\(")


def load(path):
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = KERNEL_RE.search(row["Name"])
            if match:
                rows[(int(match.group("index")), match.group("suffix") or "")] = (int(row["Total Time (ns)"]), int(row["Instances"]))
    return rows


def total(rows, indexes, suffix, instances):
    selected = [rows[(index, suffix)] for index in indexes]
    if {item[1] for item in selected} != {instances}:
        raise ValueError(f"Unexpected instances for {suffix}")
    return sum(item[0] / item[1] for item in selected) / 1.0e6


def main():
    result_dir = Path(sys.argv[1])
    output = []
    for variant, tag in VARIANTS.items():
        path = next(result_dir.glob(f"{variant}_cuda_gpu_kern_sum*.csv"))
        rows = load(path)
        old_ref = total(rows, range(19), "", 12)
        old_mp = total(rows, range(19), "_FP64", 8) + total(rows, range(19), "_FP32", 8)
        selected_ref = total(rows, range(16), f"_{tag}", 4)
        selected_mp = total(rows, range(16), f"_{tag}_FP64", 4) + total(rows, range(16), f"_{tag}_FP32", 4)
        output.append((variant, old_ref, selected_ref, old_ref / selected_ref, old_mp, selected_mp, old_mp / selected_mp))
    with (result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("variant", "old_original_ms", "selected_original_ms", "original_speedup", "old_mp_ms", "selected_mp_ms", "mp_speedup")); writer.writerows(output)
    lines = ["# DDDD K16 Nsys Kernel Timing", "", "Times are CUDA kernel execution only, normalized to one SCF interaction.", "", "| variant | selected original (ms) | original speedup | selected MP (ms) | MP speedup |", "|---|---:|---:|---:|---:|"]
    for variant, _, selected_ref, ref_speedup, _, selected_mp, mp_speedup in output:
        lines.append(f"| {variant} | {selected_ref:.3f} | {ref_speedup:.4f}x | {selected_mp:.3f} | {mp_speedup:.4f}x |")
    (result_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(result_dir / "README.md")


if __name__ == "__main__":
    main()
