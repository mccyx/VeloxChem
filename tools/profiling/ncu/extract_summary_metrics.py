#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from typing import Dict, Tuple


SECTION_RE = re.compile(r"^\s+Section: (.+?)\n(.*?)(?=^\s+Section: |\Z)", re.M | re.S)


def get_section(text: str, name: str) -> str:
    for match in SECTION_RE.finditer(text):
        if match.group(1).strip() == name:
            return match.group(2)
    raise ValueError(f"Missing section: {name}")


def extract_metric(section: str, metric_name: str) -> Tuple[str, str]:
    pattern = re.compile(
        rf"^\s*{re.escape(metric_name)}\s+(\S*)\s+([0-9][0-9,\.]*)\s*$",
        re.M,
    )
    match = pattern.search(section)
    if not match:
        raise ValueError(f"Missing metric: {metric_name}")
    return match.group(1), match.group(2)


def extract_roofline_percent(section: str, precision: str) -> str:
    # Nsight Compute may print either "achieved 36% ..." or
    # "achieved close to 1% ...", and it may use "this device's"
    # for FP32 but "its" for FP64. Extract the two percentages separately.
    fp32_match = re.search(
        r"achieved\s+(?:close\s+to\s+)?([0-9]+(?:\.[0-9]+)?)%\s+of\s+(?:this\s+device's\s+)?fp32\s+peak\s+performance",
        section,
        re.I | re.S,
    )
    fp64_match = re.search(
        r"and\s+(?:close\s+to\s+)?([0-9]+(?:\.[0-9]+)?)%\s+of\s+(?:its|this\s+device's)\s+fp64\s+peak\s+performance",
        section,
        re.I | re.S,
    )
    if not fp32_match or not fp64_match:
        raise ValueError("Missing roofline peak-performance percentages")
    return fp32_match.group(1) if precision == "fp32" else fp64_match.group(1)


def parse_summary_file(path: Path) -> Dict[str, str]:
    text = path.read_text()
    throughput = get_section(text, "GPU Speed Of Light Throughput")
    roofline = get_section(text, "GPU Speed Of Light Roofline Chart")
    launch = get_section(text, "Launch Statistics")
    occupancy = get_section(text, "Occupancy")

    precision_match = re.search(r"_(fp32|fp64)(?:_[^.]+)*\.summary\.txt$", path.name)
    if not precision_match:
        raise ValueError(
            f"Filename does not encode fp32/fp64 precision as expected: {path.name}"
        )
    precision = precision_match.group(1)

    duration_unit, duration_value = extract_metric(throughput, "Duration")
    _, compute_throughput = extract_metric(throughput, "Compute (SM) Throughput")
    _, memory_throughput = extract_metric(throughput, "Memory Throughput")
    _, register_count = extract_metric(launch, "Registers Per Thread")
    _, theoretical_occupancy = extract_metric(occupancy, "Theoretical Occupancy")
    _, achieved_occupancy = extract_metric(occupancy, "Achieved Occupancy")

    return {
        "file": path.name,
        "duration": f"{duration_value} {duration_unit}",
        "compute_throughput": f"{compute_throughput}%",
        "memory_throughput": f"{memory_throughput}%",
        "roofline_peak_percent": f"{extract_roofline_percent(roofline, precision)}%",
        "register_count": register_count,
        "theoretical_occupancy": f"{theoretical_occupancy}%",
        "achieved_occupancy": f"{achieved_occupancy}%",
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <summary.txt> [<summary.txt> ...]")

    rows = [parse_summary_file(Path(arg)) for arg in sys.argv[1:]]

    columns = [
        ("file", "File"),
        ("duration", "Duration"),
        ("compute_throughput", "Compute"),
        ("memory_throughput", "Memory"),
        ("roofline_peak_percent", "Roofline"),
        ("register_count", "Registers"),
        ("theoretical_occupancy", "Theoretical Occ"),
        ("achieved_occupancy", "Achieved Occ"),
    ]

    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header in columns
    }

    print("  ".join(header.ljust(widths[key]) for key, header in columns))
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key, _ in columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
