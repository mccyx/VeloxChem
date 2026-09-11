#!/usr/bin/env python3

"""Compare DDDD old19 and old-derived K16 FP32 instruction bundles."""

import csv
import sys
from pathlib import Path


METRICS = ("smsp__inst_executed.sum", "smsp__inst_issued.sum", "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
GROUPS = ((0, 1), (2,), (3,), (4,), (5,), (6,), (7,), (8, 9), (10,), (11,), (12,), (13, 14), (15,), (16,), (17,), (18,))


def load(path):
    with path.open(newline="") as handle:
        return {row["Metric Name"]: int(row["Metric Value"].replace(",", "")) for row in csv.DictReader(handle)}


def main():
    directory = Path(sys.argv[1])
    old = [load(directory / f"guanine-8-hf.dddd{i}_fp32.instructions.csv") for i in range(19)]
    new = [load(directory / f"guanine-8-hf.dddd{i}_k16_old_runtime_fp32.instructions.csv") for i in range(16)]
    lines = ["# DDDD Old19 vs Old-derived K16 NCU", "", "One profiled FP32 launch per subkernel.", "", "| K16 | old source(s) | old executed | K16 executed | old/K16 |", "|---:|---:|---:|---:|---:|"]
    for index, sources in enumerate(GROUPS):
        old_value = sum(old[i][METRICS[0]] for i in sources); new_value = new[index][METRICS[0]]
        lines.append(f"| {index} | {'+'.join(map(str, sources))} | {old_value} | {new_value} | {old_value / new_value:.4f}x |")
    lines.extend(["", "| metric | old19 | K16 | old/K16 | reduction |", "|---|---:|---:|---:|---:|"])
    rows = []
    for metric in METRICS:
        old_value = sum(item[metric] for item in old); new_value = sum(item[metric] for item in new)
        reduction = 100.0 * (old_value - new_value) / old_value
        rows.append((metric, old_value, new_value, old_value / new_value, reduction))
        lines.append(f"| `{metric}` | {old_value} | {new_value} | {old_value / new_value:.4f}x | {reduction:.2f}% |")
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("metric", "old19", "old_k16", "old_over_k16", "reduction_percent")); writer.writerows(rows)
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(directory / "README.md")


if __name__ == "__main__":
    main()
