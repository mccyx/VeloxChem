#!/usr/bin/env python3

"""Summarize DDDD split host timings and validation errors."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


DEFAULT_VARIANTS = "rs26,k19_static,k19_runtime,k19_light"
TIMING_RE = re.compile(
    r"=== DDDD (?P<variant>\S+) exchange resplit timing ===\s+"
    r"old original = (?P<old_ref>[0-9.]+) ms\s+"
    r"RS original  = (?P<selected_ref>[0-9.]+) ms\s+"
    r"old MP       = (?P<old_mp>[0-9.]+) ms\s+"
    r"RS MP        = (?P<selected_mp>[0-9.]+) ms"
)
ERROR_RE = re.compile(
    r"=== DDDD (?P<label>RS original vs old original|RS MP vs RS original) ===\s+"
    r"max \|dJ\|\s+= (?P<absolute>[0-9.eE+-]+).*?rel error\s+= (?P<relative>[0-9.eE+-]+)",
    re.DOTALL,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    rows = []
    for variant in variants:
        timings = []
        errors = {"RS original vs old original": [], "RS MP vs RS original": []}
        paths = sorted(args.result_dir.glob(f"{variant}_run*_ablation.log"))
        for path in paths:
            text = path.read_text()
            matches = [m for m in TIMING_RE.finditer(text) if m.group("variant") == variant]
            if len(matches) != 4:
                raise ValueError(f"Expected 4 interactions in {path}, got {len(matches)}")
            timings.extend({k: float(m.group(k)) for k in ("old_ref", "selected_ref", "old_mp", "selected_mp")} for m in matches)
            parsed = list(ERROR_RE.finditer(text))
            for label in errors:
                selected = [m for m in parsed if m.group("label") == label]
                if len(selected) != 4:
                    raise ValueError(f"Expected 4 {label} blocks in {path}")
                errors[label].extend((float(m.group("absolute")), float(m.group("relative"))) for m in selected)
        if len(timings) != 12:
            raise ValueError(f"Expected 12 samples for {variant}")
        old_ref = mean(x["old_ref"] for x in timings)
        selected_ref = mean(x["selected_ref"] for x in timings)
        old_mp = mean(x["old_mp"] for x in timings)
        mp_values = [x["selected_mp"] for x in timings]
        selected_mp = mean(mp_values)
        rows.append({
            "variant": variant,
            "samples": 12,
            "old_original_ms": old_ref,
            "selected_original_ms": selected_ref,
            "original_speedup": old_ref / selected_ref,
            "old_mp_ms": old_mp,
            "selected_mp_ms": selected_mp,
            "selected_mp_std_ms": stdev(mp_values),
            "mp_speedup": old_mp / selected_mp,
            "original_max_abs_error": max(x[0] for x in errors["RS original vs old original"]),
            "mp_max_abs_error": max(x[0] for x in errors["RS MP vs RS original"]),
            "mp_max_rel_error": max(x[1] for x in errors["RS MP vs RS original"]),
        })
    with (args.result_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# DDDD Split Host Benchmark", "",
        "Guanine-8, one GH200 GPU, three runs and four SCF interactions per run. The host timer covers selected launches, GPU execution, and final synchronization; it excludes allocation, zeroing, cut construction, copies, and validation.", "",
        "| variant | selected original (ms) | original speedup | selected MP (ms) | MP speedup | MP std (ms) | original max abs | MP max abs | MP max rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['selected_original_ms']:.3f} | {row['original_speedup']:.4f}x | {row['selected_mp_ms']:.3f} | {row['mp_speedup']:.4f}x | {row['selected_mp_std_ms']:.3f} | {row['original_max_abs_error']:.6e} | {row['mp_max_abs_error']:.6e} | {row['mp_max_rel_error']:.6e} |")
    (args.result_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(args.result_dir / "README.md")


if __name__ == "__main__":
    main()
