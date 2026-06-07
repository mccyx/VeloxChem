#!/usr/bin/env python3

import csv
import sys
from pathlib import Path


TOP_N = 10


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <csv-file>")

    csv_path = Path(sys.argv[1])

    kernel_names: set[str] = set()
    warp_states: list[tuple[str, float]] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel_name = row["Kernel Name"].strip()
            if kernel_name:
                kernel_names.add(kernel_name)

            if row["Section Name"] != "Warp State Statistics":
                continue

            if row["Body Item Label"] != "Warp State (All Cycles)":
                continue

            metric_name = row["Metric Name"].strip()
            metric_value = float(row["Metric Value"])
            warp_states.append((metric_name, metric_value))

    if len(kernel_names) != 1:
        raise ValueError(
            f"Expected exactly one kernel name in the CSV, found {len(kernel_names)}."
        )

    kernel_name = next(iter(kernel_names)).split("(", 1)[0].strip()

    print(f"Kernel Name: {kernel_name}")
    print()
    print("Top 10 Warp State Metrics (All Cycles)")
    print(f"{'Rank':>4}  {'Warp State Metric':<36} {'Value':>12}")
    for rank, (metric_name, metric_value) in enumerate(
        sorted(warp_states, key=lambda item: item[1], reverse=True)[:TOP_N],
        start=1,
    ):
        print(f"{rank:>4}  {metric_name:<36} {metric_value:>12.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
