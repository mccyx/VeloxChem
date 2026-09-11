#!/usr/bin/env python3

"""Rank old and resplit Exchange subkernels from an Nsys kernel-summary CSV."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


KERNEL_RE = re.compile(
    r"(?:\w+::)*computeExchangeFock"
    r"(?P<family>[A-Z]{4})(?P<index>\d*)"
    r"(?P<rs>_RS)?(?P<precision>_FP32|_FP64)?\("
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report per-launch time and runtime share for old and resplit "
            "Exchange subkernels."
        )
    )
    parser.add_argument("csv", type=Path, help="Nsys cuda_gpu_kern_sum CSV")
    parser.add_argument(
        "--families",
        default="DDDD,DDDP,DDDS,DPDD,DSDD,PDDD,PPDD,SDDD",
        help="Comma-separated family filter",
    )
    parser.add_argument(
        "--precision",
        choices=("all", "original", "fp64", "fp32"),
        default="all",
    )
    return parser.parse_args()


def load_rows(path, families, precision_filter):
    rows = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            match = KERNEL_RE.search(record["Name"])
            if not match or match.group("family") not in families:
                continue

            precision = {
                None: "original",
                "_FP64": "fp64",
                "_FP32": "fp32",
            }[match.group("precision")]
            if precision_filter != "all" and precision != precision_filter:
                continue

            index = match.group("index") or "-"
            layout = "rs" if match.group("rs") else "old"
            rows.append(
                {
                    "family": match.group("family"),
                    "layout": layout,
                    "precision": precision,
                    "index": index,
                    "kernel": match.group(0)[:-1].split("::")[-1],
                    "instances": int(record["Instances"]),
                    "avg_ms": float(record["Avg (ns)"]) / 1.0e6,
                }
            )
    return rows


def main():
    args = parse_args()
    families = {item.strip().upper() for item in args.families.split(",") if item.strip()}
    rows = load_rows(args.csv, families, args.precision)

    totals = defaultdict(float)
    for row in rows:
        key = (row["family"], row["layout"], row["precision"])
        totals[key] += row["avg_ms"]

    rows.sort(
        key=lambda row: (
            row["family"],
            row["precision"],
            row["layout"],
            -row["avg_ms"],
        )
    )

    print(
        "family,layout,precision,index,avg_ms,family_share_pct,instances,kernel"
    )
    for row in rows:
        key = (row["family"], row["layout"], row["precision"])
        share = 100.0 * row["avg_ms"] / totals[key]
        print(
            f'{row["family"]},{row["layout"]},{row["precision"]},'
            f'{row["index"]},{row["avg_ms"]:.6f},{share:.3f},'
            f'{row["instances"]},{row["kernel"]}'
        )


if __name__ == "__main__":
    main()
