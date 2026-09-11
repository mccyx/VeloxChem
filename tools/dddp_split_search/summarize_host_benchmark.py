#!/usr/bin/env python3

"""Summarize DDDP K5 host timing and validation."""

import csv
import re
import sys
from pathlib import Path
from statistics import mean, stdev


VARIANTS = ("rs6", "rs_k5", "old_k5")
TIMING = re.compile(r"=== DDDP (?P<variant>\S+) exchange resplit timing ===\s+old original = (?P<old_ref>[0-9.]+) ms\s+RS original  = (?P<selected_ref>[0-9.]+) ms\s+old MP       = (?P<old_mp>[0-9.]+) ms\s+RS MP        = (?P<selected_mp>[0-9.]+) ms")
ERROR = re.compile(r"=== DDDP (?P<label>RS original vs old original|RS MP vs RS original) ===\s+max \|dJ\|\s+= (?P<absolute>[0-9.eE+-]+).*?rel error\s+= (?P<relative>[0-9.eE+-]+)", re.DOTALL)


def main():
    directory = Path(sys.argv[1]); rows = []
    for variant in VARIANTS:
        timings = []; errors = {"RS original vs old original": [], "RS MP vs RS original": []}
        for path in sorted(directory.glob(f"{variant}_run*_ablation.log")):
            text = path.read_text(); matches = [m for m in TIMING.finditer(text) if m.group("variant") == variant]
            if len(matches) != 4: raise ValueError(f"Expected 4 interactions in {path}, got {len(matches)}")
            timings.extend({key: float(m.group(key)) for key in ("old_ref", "selected_ref", "old_mp", "selected_mp")} for m in matches)
            parsed = list(ERROR.finditer(text))
            for label in errors:
                selected = [m for m in parsed if m.group("label") == label]
                if len(selected) != 4: raise ValueError(f"Expected 4 {label} in {path}")
                errors[label].extend((float(m.group("absolute")), float(m.group("relative"))) for m in selected)
        old_ref = mean(x["old_ref"] for x in timings); selected_ref = mean(x["selected_ref"] for x in timings)
        old_mp = mean(x["old_mp"] for x in timings); mp_values = [x["selected_mp"] for x in timings]; selected_mp = mean(mp_values)
        rows.append({"variant": variant, "old_original_ms": old_ref, "selected_original_ms": selected_ref, "original_speedup": old_ref / selected_ref, "old_mp_ms": old_mp, "selected_mp_ms": selected_mp, "mp_speedup": old_mp / selected_mp, "mp_std_ms": stdev(mp_values), "original_max_abs": max(x[0] for x in errors["RS original vs old original"]), "mp_max_abs": max(x[0] for x in errors["RS MP vs RS original"]), "mp_max_rel": max(x[1] for x in errors["RS MP vs RS original"])})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    lines = ["# DDDP K5 Host Benchmark", "", "Three runs and four SCF interactions per run on one GH200. Timer includes selected launches, GPU execution, and final synchronization; it excludes allocation, zeroing, cuts, copies, and validation.", "", "| variant | original (ms) | original speedup | MP (ms) | MP speedup | MP std | original max abs | MP max abs | MP max rel |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows: lines.append(f"| {row['variant']} | {row['selected_original_ms']:.3f} | {row['original_speedup']:.4f}x | {row['selected_mp_ms']:.3f} | {row['mp_speedup']:.4f}x | {row['mp_std_ms']:.3f} | {row['original_max_abs']:.6e} | {row['mp_max_abs']:.6e} | {row['mp_max_rel']:.6e} |")
    (directory / "README.md").write_text("\n".join(lines) + "\n"); print(directory / "README.md")


if __name__ == "__main__": main()
