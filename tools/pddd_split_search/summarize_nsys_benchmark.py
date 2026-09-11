#!/usr/bin/env python3

"""Summarize normalized PDDD kernel-only durations from Nsys CSV files."""

import argparse
import csv
import re
from pathlib import Path


VARIANTS = {
    "rs5": "RS",
    "k4_m01": "K4_M01",
    "k4_m12": "K4_M12",
    "k4_m23": "K4_M23",
    "k4_m34": "K4_M34",
}
NAME_RE = re.compile(r"computeExchangeFockPDDD(?P<index>\d+)(?P<suffix>_[A-Z0-9_]+)?\(")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def load(path):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = NAME_RE.search(row["Name"])
            if match:
                key = (int(match.group("index")), match.group("suffix") or "")
                result[key] = {
                    "total_ns": int(row["Total Time (ns)"]),
                    "instances": int(row["Instances"]),
                }
    return result


def sum_group(data, indexes, suffix, expected_instances):
    rows = [data[(index, suffix)] for index in indexes]
    instances = {row["instances"] for row in rows}
    if instances != {expected_instances}:
        raise ValueError(f"Expected {expected_instances} instances for {suffix}, got {instances}")
    # Four SCF interactions; repeated baseline validation paths are removed by
    # taking the per-instance duration before summing the subkernels.
    return sum(row["total_ns"] / row["instances"] for row in rows) / 1.0e6


def main():
    args = parse_args()
    rows = []
    for variant, selected_tag in VARIANTS.items():
        paths = list(args.result_dir.glob(f"{variant}_cuda_gpu_kern_sum*.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one Nsys CSV for {variant}, got {paths}")
        data = load(paths[0])
        old_original = sum_group(data, range(8), "", 12)
        old_mp = sum_group(data, range(8), "_FP64", 8) + sum_group(data, range(8), "_FP32", 8)
        selected_indexes = range(5) if variant == "rs5" else range(4)
        selected_original = sum_group(data, selected_indexes, f"_{selected_tag}", 4)
        selected_mp = sum_group(data, selected_indexes, f"_{selected_tag}_FP64", 4) + sum_group(
            data, selected_indexes, f"_{selected_tag}_FP32", 4
        )
        rows.append(
            {
                "variant": variant,
                "old_original_ms": old_original,
                "selected_original_ms": selected_original,
                "original_speedup": old_original / selected_original,
                "old_mp_ms": old_mp,
                "selected_mp_ms": selected_mp,
                "mp_speedup": old_mp / selected_mp,
            }
        )

    output_csv = args.result_dir / "summary.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: row["selected_mp_ms"])
    lines = [
        "# PDDD K4 Nsys Kernel Timing",
        "",
        "Guanine-8, one GH200 GPU, one profiled program run with four SCF interactions per variant.",
        "",
        "Durations contain only CUDA execution time for the named PDDD kernel groups. They exclude host launch overhead, stream synchronization, allocation, zeroing, host cut-layout construction, GPU cut construction, copies, and validation. Existing duplicate old-kernel validation paths were normalized using the per-instance duration reported by Nsys: old original has 12 instances per subkernel, old MP has 8, and each selected variant has 4.",
        "",
        "| variant | selected original (ms/interaction) | original speedup | selected MP (ms/interaction) | MP speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['selected_original_ms']:.3f} | {row['original_speedup']:.4f}x | "
            f"{row['selected_mp_ms']:.3f} | {row['mp_speedup']:.4f}x |"
        )
    lines.extend(
        [
            "",
            f"Fastest MP split by Nsys kernel duration: `{best['variant']}` at "
            f"`{best['selected_mp_ms']:.3f} ms/interaction`, or `{best['mp_speedup']:.4f}x` versus old MP.",
            "",
        ]
    )
    report = args.result_dir / "README.md"
    report.write_text("\n".join(lines))
    print(report)
    print(output_csv)


if __name__ == "__main__":
    main()
