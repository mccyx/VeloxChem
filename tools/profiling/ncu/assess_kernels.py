#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

from extract_summary_metrics import parse_summary_file


SECTION_RE = re.compile(r"^\s+Section: (.+?)\n(.*?)(?=^\s+Section: |\Z)", re.M | re.S)


def infer_related_files(summary_path: Path) -> Tuple[Path, Path]:
    if not summary_path.name.endswith(".summary.txt"):
        raise ValueError(f"Expected a .summary.txt file, got: {summary_path.name}")

    base = summary_path.name[:-len(".summary.txt")]
    source_path = summary_path.with_name(f"{base}.source.csv")
    warpstate_path = summary_path.with_name(f"{base}.warpstate.csv")

    if not source_path.exists():
        raise ValueError(f"Missing matching source CSV: {source_path.name}")
    if not warpstate_path.exists():
        raise ValueError(f"Missing matching warpstate CSV: {warpstate_path.name}")

    return source_path, warpstate_path


def normalize_instruction(source: str) -> str:
    text = source.strip()
    if not text:
        return "UNKNOWN"

    if text.startswith("@"):
        parts = text.split(None, 1)
        text = parts[1].strip() if len(parts) > 1 else ""

    if not text:
        return "UNKNOWN"

    opcode = text.split(None, 1)[0]
    base = opcode.split(".", 1)[0]
    base = re.sub(r"^[^A-Z]*", "", base.upper())

    if len(base) > 1 and base.startswith("U") and base[1:].isalpha():
        base = base[1:]

    return base or "UNKNOWN"


def parse_source_metrics(path: Path) -> Tuple[str, str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        metadata = next(reader, None)
        if not metadata or metadata[0] != "Kernel Name":
            raise ValueError(f"Unexpected source CSV format: {path.name}")

        header = next(reader, None)
        if not header:
            raise ValueError(f"Missing header in source CSV: {path.name}")

        source_idx = header.index("Source")
        executed_idx = header.index("Instructions Executed")
        stall_columns = [
            (idx, name)
            for idx, name in enumerate(header)
            if name.startswith("stall_") and "Not Issued" not in name
        ]
        stall_not_issued_columns = [
            (idx, name)
            for idx, name in enumerate(header)
            if name.startswith("stall_") and "Not Issued" in name
        ]

        instruction_totals = defaultdict(int)  # type: Dict[str, int]
        stall_totals = {name: 0 for _, name in stall_columns}
        stall_not_issued_totals = {name: 0 for _, name in stall_not_issued_columns}

        for row in reader:
            if not row or row[0] in ("Kernel Name", "Address"):
                continue
            instruction = normalize_instruction(row[source_idx])
            instruction_totals[instruction] += int(row[executed_idx])
            for idx, name in stall_columns:
                stall_totals[name] += int(row[idx])
            for idx, name in stall_not_issued_columns:
                stall_not_issued_totals[name] += int(row[idx])

    top_inst = max(instruction_totals.items(), key=lambda item: item[1])[0]
    top_stall = max(stall_totals.items(), key=lambda item: item[1])
    top_not_issued = max(stall_not_issued_totals.items(), key=lambda item: item[1])
    return top_inst, top_not_issued[0]


def parse_warpstate_metrics(path: Path) -> Tuple[Tuple[str, float], Dict[str, float]]:
    kernel_names = set()  # type: Set[str]
    metrics = {}  # type: Dict[str, float]

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel_name = row["Kernel Name"].strip()
            if kernel_name:
                kernel_names.add(kernel_name)

            if row["Section Name"] != "Warp State Statistics":
                continue

            if row["Body Item Label"] != "Warp State (All Cycles)":
                continue

            metrics[row["Metric Name"].strip()] = float(row["Metric Value"])

    if len(kernel_names) != 1:
        raise ValueError(
            f"Expected exactly one kernel name in the warpstate CSV, found {len(kernel_names)}."
        )

    top_metric = max(metrics.items(), key=lambda item: item[1])
    return top_metric, metrics


def parse_instruction_summary(path: Path) -> Dict[str, str]:
    text = path.read_text()
    instruction_section = None
    for match in SECTION_RE.finditer(text):
        if match.group(1).strip() == "Instruction Statistics":
            instruction_section = match.group(2)
            break

    if instruction_section is None:
        raise ValueError(f"Missing Instruction Statistics section: {path.name}")

    def metric_value(metric_name: str) -> str:
        pattern = re.compile(
            r"^\s*" + re.escape(metric_name) + r"\s+\S+\s+([0-9][0-9,\.]*)\s*$",
            re.M,
        )
        match = pattern.search(instruction_section)
        if not match:
            raise ValueError(f"Missing metric {metric_name}: {path.name}")
        return match.group(1)

    fp32_fused = ""
    fp32_non_fused = ""
    fp32_match = re.search(
        r"This kernel executes\s+([0-9,]+)\s+fused and\s+([0-9,]+)\s+non-fused FP32 instructions",
        instruction_section,
    )
    if fp32_match:
        fp32_fused = fp32_match.group(1)
        fp32_non_fused = fp32_match.group(2)

    return {
        "executed_inst": metric_value("Executed Instructions"),
        "issued_inst": metric_value("Issued Instructions"),
        "fp32_fused_inst": fp32_fused,
        "fp32_non_fused_inst": fp32_non_fused,
    }


def classify_precision(path: Path) -> str:
    match = re.search(r"_(fp32|fp64)\.summary\.txt$", path.name)
    if not match:
        raise ValueError(f"Cannot determine precision from filename: {path.name}")
    return match.group(1)


def assess_file(summary_path: Path) -> Dict[str, str]:
    source_path, warpstate_path = infer_related_files(summary_path)
    summary = parse_summary_file(summary_path)
    instruction_summary = parse_instruction_summary(summary_path)

    top_inst, top_source_not_issued = parse_source_metrics(source_path)
    top_warp_metric, _ = parse_warpstate_metrics(warpstate_path)

    return {
        "file": summary_path.name[:-len(".summary.txt")],
        "duration": summary["duration"],
        "roofline": summary["roofline_peak_percent"],
        "compute": summary["compute_throughput"],
        "memory": summary["memory_throughput"],
        "registers": summary["register_count"],
        "occ": f"{summary['achieved_occupancy']}/{summary['theoretical_occupancy']}",
        "executed_inst": instruction_summary["executed_inst"],
        "issued_inst": instruction_summary["issued_inst"],
        "fp32_fused_inst": instruction_summary["fp32_fused_inst"],
        "fp32_non_fused_inst": instruction_summary["fp32_non_fused_inst"],
        "top_inst": top_inst,
        "top_warp": top_warp_metric[0],
        "top_source_not_issued": top_source_not_issued,
    }


def write_csv(rows, columns, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header for _, header in columns])
        for row in rows:
            writer.writerow([row[key] for key, _ in columns])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize NCU summary/source/warpstate artifacts into a compact table."
    )
    parser.add_argument(
        "--csv-out",
        dest="csv_out",
        help="Optional path to write the same assessed table as CSV.",
    )
    parser.add_argument(
        "summary_files",
        nargs="+",
        help="One or more .summary.txt files. Matching .source.csv and .warpstate.csv are inferred automatically.",
    )
    args = parser.parse_args()

    rows = [assess_file(Path(arg)) for arg in args.summary_files]

    columns = [
        ("file", "File"),
        ("duration", "Duration"),
        ("roofline", "Roofline"),
        ("compute", "Compute"),
        ("memory", "Memory"),
        ("registers", "Regs"),
        ("occ", "Ach/Theor Occ"),
        ("executed_inst", "Executed Inst."),
        ("issued_inst", "Issued Inst."),
        ("fp32_fused_inst", "FP32 Fused Inst."),
        ("fp32_non_fused_inst", "FP32 Non-Fused Inst."),
        ("top_inst", "Top Inst."),
        ("top_warp", "Top Warp State"),
        ("top_source_not_issued", "Top Source Stall Not Issued"),
    ]

    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header in columns
    }

    print("  ".join(header.ljust(widths[key]) for key, header in columns))
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key, _ in columns))

    if args.csv_out:
        output_path = Path(args.csv_out)
        if output_path.parent != Path(""):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, columns, output_path)
        print()
        print(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
