#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

WINNER_TAGS = {
    "PDDD": "K4_M23",
    "DDDD": "K16_OLD_RUNTIME",
    "DDDP": "K5_OLD",
    "DPDD": "K4_RS",
}
KERNEL = re.compile(
    r"computeExchangeFock(?P<family>[SPDF]{4})(?P<index>\d*)"
    r"(?P<suffix>_[A-Z0-9_]+)?\("
)


def main():
    csv_path = Path(sys.argv[1])
    kernels = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = KERNEL.search(row["Name"])
            if match:
                kernels.append((
                    match.group("family"),
                    match.group("suffix") or "",
                    float(row["Avg (ns)"]) / 1.0e6,
                ))

    families = sorted({family for family, suffix, _ in kernels if suffix == ""})
    if len(families) != 54:
        raise ValueError("expected 54 old-layout families, found {}".format(len(families)))

    reference = sum(ms for _, suffix, ms in kernels if suffix == "")
    old_mp = sum(ms for _, suffix, ms in kernels if suffix in ("_FP64", "_FP32"))
    winner_mp = old_mp
    replacements = []
    for family, tag in WINNER_TAGS.items():
        old = sum(ms for f, suffix, ms in kernels
                  if f == family and suffix in ("_FP64", "_FP32"))
        selected = sum(ms for f, suffix, ms in kernels
                       if f == family and suffix in ("_" + tag + "_FP64", "_" + tag + "_FP32"))
        if not old or not selected:
            raise ValueError("missing old/winner kernels for {}".format(family))
        winner_mp += selected - old
        replacements.append((family, old, selected))

    lines = [
        "# All Exchange Winners: Nsys Aggregate",
        "",
        "CUDA kernel execution time normalized to one SCF interaction. The sum",
        "contains all 54 exchange families; four old MP layouts are replaced by",
        "their selected winners.",
        "",
        "| metric | Nsys kernel time (ms) | speedup vs old original FP64 |",
        "|---|---:|---:|",
        "| old original FP64 | {:.3f} | 1.0000x |".format(reference),
        "| old-split MP | {:.3f} | {:.4f}x |".format(old_mp, reference / old_mp),
        "| all-winner MP | {:.3f} | {:.4f}x |".format(winner_mp, reference / winner_mp),
        "",
        "| replaced family | old MP (ms) | winner MP (ms) | reduction (ms) |",
        "|---|---:|---:|---:|",
    ]
    lines.extend("| {} | {:.3f} | {:.3f} | {:.3f} |".format(f, old, new, old - new)
                 for f, old, new in replacements)
    output = csv_path.parent / "SUMMARY.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
