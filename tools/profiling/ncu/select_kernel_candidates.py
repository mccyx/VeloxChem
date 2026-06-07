#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def parse_percent(text):
    return float(text.strip().rstrip("%"))


def parse_duration_us(text):
    value, unit = text.strip().split()
    value = float(value)
    if unit == "ms":
        return value * 1000.0
    if unit == "us":
        return value
    raise ValueError("Unsupported duration unit: %s" % unit)


def load_rows(csv_path, precision):
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row["File"]
            if precision and precision not in file_name.lower():
                continue
            row["_file"] = file_name
            row["_roofline"] = parse_percent(row["Roofline"])
            row["_compute"] = parse_percent(row["Compute"])
            row["_memory"] = parse_percent(row["Memory"])
            row["_duration_us"] = parse_duration_us(row["Duration"])
            row["_regs"] = int(row["Regs"])
            rows.append(row)
    return rows


def representative_by_bucket(sorted_rows, bucket_idx, bucket_count):
    n = len(sorted_rows)
    start = int(math.floor(bucket_idx * n / bucket_count))
    end = int(math.floor((bucket_idx + 1) * n / bucket_count))
    end = max(end, start + 1)
    bucket = sorted_rows[start:end]
    return bucket[len(bucket) // 2]


def unique_by_file(rows):
    seen = set()
    out = []
    for row in rows:
        key = row["_file"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def emit_table(title, rows):
    print(title)
    for row in rows:
        print(
            "  {file}: roofline={roof}, duration={dur}, regs={regs}, top_warp={warp}, top_stall={stall}".format(
                file=row["_file"],
                roof=row["Roofline"],
                dur=row["Duration"],
                regs=row["Regs"],
                warp=row["Top Warp State"],
                stall=row["Top Source Stall Not Issued"],
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Select representative and metrics-driven kernel candidates from assess_overview.csv"
    )
    parser.add_argument("csv_path", help="Path to assess_overview.csv")
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp64", "all"],
        help="Filter rows by precision suffix in File column",
    )
    args = parser.parse_args()

    precision = None if args.precision == "all" else args.precision
    rows = load_rows(args.csv_path, precision)
    if not rows:
        raise SystemExit("No rows matched the requested precision filter")

    by_roofline = sorted(rows, key=lambda row: row["_roofline"])
    low = representative_by_bucket(by_roofline, 0, 3)
    mid = representative_by_bucket(by_roofline, 1, 3)
    high = representative_by_bucket(by_roofline, 2, 3)

    top_math = max(
        (row for row in rows if "stall_math" in row["Top Source Stall Not Issued"]),
        key=lambda row: (row["_duration_us"], row["_roofline"]),
        default=None,
    )
    top_wait = max(
        (row for row in rows if "stall_wait" in row["Top Source Stall Not Issued"]),
        key=lambda row: (row["_duration_us"], row["_roofline"]),
        default=None,
    )
    top_regs = max(rows, key=lambda row: (row["_regs"], row["_duration_us"]))
    top_duration = max(rows, key=lambda row: row["_duration_us"])

    quantile_rows = [low, mid, high]
    metrics_rows = unique_by_file(
        [row for row in [top_math, top_wait, top_regs, top_duration] if row is not None]
    )

    emit_table("Quantile representatives", quantile_rows)
    print("")
    emit_table("Metrics-driven extras", metrics_rows)
    print("")
    print("Suggested combined set")
    for row in unique_by_file(quantile_rows + metrics_rows):
        print("  %s" % row["_file"])


if __name__ == "__main__":
    main()
