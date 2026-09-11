#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path
from statistics import mean, stdev

WINNERS = {
    "PDDD": "k4_m23",
    "DDDD": "old_k16_runtime",
    "DDDP": "old_k5",
    "DPDD": "rs_k4",
}

VALIDATION = re.compile(
    r"=== (?P<family>[SPDF]{4}) exchange validation timing ===\s+"
    r"cuts\s+= (?P<cuts>[0-9.]+) ms\s+"
    r"FP64\+FP32\s+= (?P<mp>[0-9.]+) ms\s+"
    r"reference\s+= (?P<reference>[0-9.]+) ms",
)
RESPLIT = re.compile(
    r"=== (?P<family>[SPDF]{4}) (?P<variant>\S+) exchange resplit timing ===\s+"
    r"old original = (?P<old_ref>[0-9.]+) ms\s+"
    r"RS original\s+= (?P<selected_ref>[0-9.]+) ms\s+"
    r"old MP\s+= (?P<old_mp>[0-9.]+) ms\s+"
    r"RS MP\s+= (?P<selected_mp>[0-9.]+) ms",
)
CUT_STATS = re.compile(
    r"=== (?P<family>[SPDF]{4}) exchange cut stats ===.*?"
    r"FP64 fraction\s+= (?P<fp64>[0-9.]+) %\s+"
    r"FP32 fraction\s+= (?P<fp32>[0-9.]+) %",
    re.S,
)


def parse_interactions(path):
    text = path.read_text()
    validations = list(VALIDATION.finditer(text))
    if not validations:
        raise ValueError("no validation timings in {}".format(path))

    validation_by_family = {}
    for match in validations:
        validation_by_family.setdefault(match.group("family"), []).append(match)
    family_count = len(validation_by_family)
    counts = {family: len(matches) for family, matches in validation_by_family.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("unequal family sample counts in {}: {}".format(path, counts))
    interaction_count = next(iter(counts.values()))

    resplits = list(RESPLIT.finditer(text))
    selected = {}
    for family, variant in WINNERS.items():
        matches = [m for m in resplits if m.group("family") == family and m.group("variant") == variant]
        if len(matches) != interaction_count:
            raise ValueError("{} {}: expected {} timings, found {} in {}".format(
                family, variant, interaction_count, len(matches), path))
        selected[family] = matches

    rows = []
    for interaction in range(interaction_count):
        group = [matches[interaction] for matches in validation_by_family.values()]

        reference = sum(float(m.group("reference")) for m in group)
        old_mp = sum(float(m.group("mp")) for m in group)
        cuts = sum(float(m.group("cuts")) for m in group)
        winner_mp = old_mp
        for family in WINNERS:
            m = selected[family][interaction]
            winner_mp += float(m.group("selected_mp")) - float(m.group("old_mp"))

        rows.append({
            "run": path.stem.replace("_ablation", ""),
            "interaction": interaction + 1,
            "families": family_count,
            "reference_ms": reference,
            "old_mp_ms": old_mp,
            "winner_mp_ms": winner_mp,
            "cuts_ms": cuts,
            "old_mp_plus_cuts_ms": old_mp + cuts,
            "winner_mp_plus_cuts_ms": winner_mp + cuts,
        })
    return rows


def main():
    result_dir = Path(sys.argv[1])
    rows = []
    paths = sorted(result_dir.glob("run*_ablation.log"))
    for path in paths:
        rows.extend(parse_interactions(path))
    if not rows:
        raise ValueError("no run logs in {}".format(result_dir))

    fields = list(rows[0])
    with (result_dir / "interaction_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    validation_samples = {}
    fraction_samples = {}
    selected_samples = {}
    for path in paths:
        text = path.read_text()
        for match in VALIDATION.finditer(text):
            validation_samples.setdefault(match.group("family"), []).append(match)
        for match in CUT_STATS.finditer(text):
            fraction_samples.setdefault(match.group("family"), []).append(match)
        for match in RESPLIT.finditer(text):
            family = match.group("family")
            if WINNERS.get(family) == match.group("variant"):
                selected_samples.setdefault(family, []).append(match)

    family_rows = []
    for family in sorted(validation_samples):
        timings = validation_samples[family]
        fractions = fraction_samples[family]
        reference = mean(float(m.group("reference")) for m in timings)
        old_mp = mean(float(m.group("mp")) for m in timings)
        cuts = mean(float(m.group("cuts")) for m in timings)
        fp32 = mean(float(m.group("fp32")) for m in fractions) / 100.0
        fp64 = mean(float(m.group("fp64")) for m in fractions) / 100.0
        layout = WINNERS.get(family, "old")
        selected_mp = old_mp
        if family in WINNERS:
            selected_mp = mean(float(m.group("selected_mp")) for m in selected_samples[family])
        theory = 1.0 / (fp64 + fp32 / 2.0)
        family_rows.append({
            "family": family,
            "layout": layout,
            "fp32_fraction_pct": fp32 * 100.0,
            "original_fp64_ms": reference,
            "selected_mp_ms": selected_mp,
            "cuts_ms": cuts,
            "theoretical_kernel_speedup": theory,
            "achieved_kernel_speedup": reference / selected_mp,
            "achieved_speedup_with_cuts": reference / (selected_mp + cuts),
        })

    family_fields = list(family_rows[0])
    with (result_dir / "per_combination_winners.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=family_fields)
        writer.writeheader()
        writer.writerows(family_rows)

    family_lines = [
        "| family | layout | FP32 | old original FP64 (ms) | selected MP (ms) | cuts (ms) | theory kernel | achieved kernel | achieved + cuts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    family_lines.extend(
        "| {family} | `{layout}` | {fp32_fraction_pct:.3f}% | {original_fp64_ms:.3f} | {selected_mp_ms:.3f} | {cuts_ms:.3f} | {theoretical_kernel_speedup:.4f}x | {achieved_kernel_speedup:.4f}x | {achieved_speedup_with_cuts:.4f}x |".format(**row)
        for row in family_rows
    )
    (result_dir / "PER_COMBINATION.md").write_text("\n".join(family_lines) + "\n")

    def avg(key):
        return mean(row[key] for row in rows)

    ref = avg("reference_ms")
    old_mp = avg("old_mp_ms")
    winner_mp = avg("winner_mp_ms")
    cuts = avg("cuts_ms")
    old_total = avg("old_mp_plus_cuts_ms")
    winner_total = avg("winner_mp_plus_cuts_ms")
    winner_samples = [row["winner_mp_ms"] for row in rows]

    lines = [
        "# All Exchange Winners: Host-Timer Aggregate",
        "",
        "Input: `guanine-8-hf.inp`; three runs, four SCF interactions per run.",
        "All 54 exchange families are included. PDDD, DDDD, DDDP, and DPDD use",
        "their selected split winners; the other families retain the old split.",
        "",
        "| metric | average (ms) | speedup vs original FP64 |",
        "|---|---:|---:|",
        "| old original FP64 kernels | {:.3f} | 1.0000x |".format(ref),
        "| old-split MP kernels | {:.3f} | {:.4f}x |".format(old_mp, ref / old_mp),
        "| all-winner MP kernels | {:.3f} | {:.4f}x |".format(winner_mp, ref / winner_mp),
        "| old-split MP kernels + cut building | {:.3f} | {:.4f}x |".format(old_total, ref / old_total),
        "| all-winner MP kernels + cut building | {:.3f} | {:.4f}x |".format(winner_total, ref / winner_total),
        "",
        "Mean cut-building time: {:.3f} ms. All-winner MP kernel standard deviation".format(cuts),
        "over {} interaction samples: {:.3f} ms.".format(len(rows), stdev(winner_samples)),
        "",
        "Host kernel timings include launches, GPU execution, and the final stream",
        "synchronization. They exclude buffer allocation/zeroing, cut building, data",
        "copies, and accuracy validation. The `+ cut building` row adds the measured",
        "GPU cut-building time; allocation and zeroing remain outside the timer.",
    ]
    output = result_dir / "SUMMARY.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
