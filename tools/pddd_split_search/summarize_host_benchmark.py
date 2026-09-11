#!/usr/bin/env python3

"""Summarize PDDD split host timings and validation errors."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


VARIANTS = ("rs5", "k4_m01", "k4_m12", "k4_m23", "k4_m34")
TIMING_RE = re.compile(
    r"=== PDDD (?P<variant>\S+) exchange resplit timing ===\s+"
    r"old original = (?P<old_ref>[0-9.]+) ms\s+"
    r"RS original  = (?P<selected_ref>[0-9.]+) ms\s+"
    r"old MP       = (?P<old_mp>[0-9.]+) ms\s+"
    r"RS MP        = (?P<selected_mp>[0-9.]+) ms"
)
ERROR_RE = re.compile(
    r"=== PDDD (?P<label>RS original vs old original|RS MP vs RS original) ===\s+"
    r"max \|dJ\|\s+= (?P<absolute>[0-9.eE+-]+).*?"
    r"rel error\s+= (?P<relative>[0-9.eE+-]+)",
    re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for variant in VARIANTS:
        timings = []
        errors = {"RS original vs old original": [], "RS MP vs RS original": []}
        for path in sorted(args.result_dir.glob(f"{variant}_run*_ablation.log")):
            text = path.read_text()
            matches = [m for m in TIMING_RE.finditer(text) if m.group("variant") == variant]
            if len(matches) != 4:
                raise ValueError(f"Expected 4 PDDD interactions in {path}, got {len(matches)}")
            timings.extend(
                {key: float(match.group(key)) for key in ("old_ref", "selected_ref", "old_mp", "selected_mp")}
                for match in matches
            )
            parsed_errors = list(ERROR_RE.finditer(text))
            for label in errors:
                selected = [m for m in parsed_errors if m.group("label") == label]
                if len(selected) != 4:
                    raise ValueError(f"Expected 4 '{label}' blocks in {path}, got {len(selected)}")
                errors[label].extend(
                    (float(match.group("absolute")), float(match.group("relative")))
                    for match in selected
                )
        if len(timings) != 12:
            raise ValueError(f"Expected 12 samples for {variant}, got {len(timings)}")
        old_ref = mean(item["old_ref"] for item in timings)
        selected_ref = mean(item["selected_ref"] for item in timings)
        old_mp = mean(item["old_mp"] for item in timings)
        selected_mp_values = [item["selected_mp"] for item in timings]
        selected_mp = mean(selected_mp_values)
        rows.append(
            {
                "variant": variant,
                "samples": len(timings),
                "old_original_ms": old_ref,
                "selected_original_ms": selected_ref,
                "original_speedup": old_ref / selected_ref,
                "old_mp_ms": old_mp,
                "selected_mp_ms": selected_mp,
                "selected_mp_std_ms": stdev(selected_mp_values),
                "mp_speedup": old_mp / selected_mp,
                "original_max_abs_error": max(x[0] for x in errors["RS original vs old original"]),
                "original_max_rel_error": max(x[1] for x in errors["RS original vs old original"]),
                "mp_max_abs_error": max(x[0] for x in errors["RS MP vs RS original"]),
                "mp_max_rel_error": max(x[1] for x in errors["RS MP vs RS original"]),
            }
        )

    csv_path = args.result_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: row["selected_mp_ms"])
    lines = [
        "# PDDD K4 Host Benchmark",
        "",
        "Guanine-8, one GH200 GPU, three program runs and four SCF interactions per run (12 timing samples per variant).",
        "",
        "The host timer covers the selected kernel launch group, GPU execution, and the final stream synchronization. It excludes buffer allocation, buffer zeroing, host cut-layout construction, GPU cut construction, and validation copies/checks. All variants use the same executable; `VLX_EXCHANGE_PDDD_SPLIT` selects exactly one split per run.",
        "",
        "| variant | selected original (ms) | original speedup | selected MP (ms) | MP speedup | MP std (ms) | original max abs error | MP max abs error | MP max rel error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['selected_original_ms']:.3f} | {row['original_speedup']:.4f}x | "
            f"{row['selected_mp_ms']:.3f} | {row['mp_speedup']:.4f}x | {row['selected_mp_std_ms']:.3f} | "
            f"{row['original_max_abs_error']:.6e} | {row['mp_max_abs_error']:.6e} | {row['mp_max_rel_error']:.6e} |"
        )
    lines.extend(
        [
            "",
            f"Fastest MP split by host timing: `{best['variant']}` at `{best['selected_mp_ms']:.3f} ms`, "
            f"or `{best['mp_speedup']:.4f}x` versus old MP.",
            "",
        ]
    )
    report_path = args.result_dir / "README.md"
    report_path.write_text("\n".join(lines))
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
