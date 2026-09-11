#!/usr/bin/env python3

"""Compare NCU instruction counts for PDDD RS5 and K4 M23 FP32 bundles."""

import argparse
import csv
from pathlib import Path


METRICS = (
    "smsp__inst_executed.sum",
    "smsp__inst_issued.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
)
GROUPS = ((0,), (1,), (2, 3), (4,))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def load(path):
    with path.open(newline="") as handle:
        return {row["Metric Name"]: int(row["Metric Value"].replace(",", "")) for row in csv.DictReader(handle)}


def main():
    args = parse_args()
    rs = [load(args.result_dir / f"guanine-8-hf.pddd{i}_rs_fp32.instructions.csv") for i in range(5)]
    k4 = [load(args.result_dir / f"guanine-8-hf.pddd{i}_k4_m23_fp32.instructions.csv") for i in range(4)]

    output_csv = args.result_dir / "summary.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "rs5", "k4_m23", "rs5_over_k4_m23", "reduction_percent"))
        for metric in METRICS:
            old = sum(row[metric] for row in rs)
            new = sum(row[metric] for row in k4)
            writer.writerow((metric, old, new, old / new, 100.0 * (old - new) / old))

    group_rows = []
    for k4_index, sources in enumerate(GROUPS):
        old = sum(rs[index]["smsp__inst_executed.sum"] for index in sources)
        new = k4[k4_index]["smsp__inst_executed.sum"]
        group_rows.append((k4_index, "+".join(str(index) for index in sources), old, new, old / new))

    executed_old = sum(row["smsp__inst_executed.sum"] for row in rs)
    executed_new = sum(row["smsp__inst_executed.sum"] for row in k4)
    lines = [
        "# PDDD K4 M23 NCU Instruction Comparison",
        "",
        "One profiled launch per FP32 subkernel on one GH200 GPU. K4 M23 maps RS groups as `[0] [1] [2+3] [4]`.",
        "",
        "| K4 kernel | RS source kernel(s) | RS executed instructions | K4 executed instructions | RS/K4 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, sources, old, new, ratio in group_rows:
        lines.append(f"| {index} | {sources} | {old} | {new} | {ratio:.4f}x |")
    lines.extend(
        [
            "",
            "| metric | RS5 total | K4 M23 total | RS5/K4 | reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        old = sum(row[metric] for row in rs)
        new = sum(row[metric] for row in k4)
        lines.append(f"| `{metric}` | {old} | {new} | {old / new:.4f}x | {100.0 * (old - new) / old:.2f}% |")
    lines.extend(
        [
            "",
            f"The complete FP32 bundle executes `{100.0 * (executed_old - executed_new) / executed_old:.2f}%` fewer SM instructions after merging RS2 and RS3. This supports reduced duplicated per-kernel setup as the main reason K4 M23 is faster; the ERI arithmetic assigned to those groups is retained in the merged kernel.",
            "",
        ]
    )
    report = args.result_dir / "README.md"
    report.write_text("\n".join(lines))
    print(report)
    print(output_csv)


if __name__ == "__main__":
    main()
