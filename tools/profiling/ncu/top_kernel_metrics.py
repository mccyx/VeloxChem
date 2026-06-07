#!/usr/bin/env python3

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


TOP_N = 10


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


def print_ranked_list(title: str, label: str, values: dict[str, int]) -> None:
    print(title)
    print(f"{'Rank':>4}  {label:<36} {'Value':>12}")
    for rank, (name, value) in enumerate(
        sorted(values.items(), key=lambda item: item[1], reverse=True)[:TOP_N],
        start=1,
    ):
        print(f"{rank:>4}  {name:<36} {value:>12}")
    print()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <csv-file>")

    csv_path = Path(sys.argv[1])
    instructions_executed: dict[str, int] = defaultdict(int)
    stall_totals: dict[str, int] = {}
    stall_not_issued_totals: dict[str, int] = {}
    kernel_name_rows = 0
    kernel_name = None

    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        metadata = next(reader, None)
        if not metadata:
            raise ValueError("CSV file is empty.")

        if metadata[0] == "Kernel Name":
            kernel_name_rows += 1
            kernel_name = metadata[1].strip() if len(metadata) > 1 else ""
        else:
            raise ValueError("CSV file does not start with a 'Kernel Name' row.")

        header = next(reader, None)
        if not header:
            raise ValueError("CSV file is missing the header row.")

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
        if not stall_columns:
            raise ValueError("No warp stall reason columns were found.")
        if not stall_not_issued_columns:
            raise ValueError("No not-issued warp stall reason columns were found.")
        stall_totals = {name: 0 for _, name in stall_columns}
        stall_not_issued_totals = {
            name: 0 for _, name in stall_not_issued_columns
        }

        for row in reader:
            if not row:
                continue

            if row[0] == "Kernel Name":
                kernel_name_rows += 1
                continue

            if row[0] == "Address" and len(row) > 1 and row[1] == "Source":
                continue

            instruction = normalize_instruction(row[source_idx])
            instructions_executed[instruction] += int(row[executed_idx])

            for idx, name in stall_columns:
                stall_totals[name] += int(row[idx])
            for idx, name in stall_not_issued_columns:
                stall_not_issued_totals[name] += int(row[idx])

    if kernel_name_rows != 1:
        raise ValueError(
            f"Expected exactly one 'Kernel Name' row, found {kernel_name_rows}."
        )

    display_kernel_name = kernel_name.split("(", 1)[0].strip()
    print(f"Kernel Name: {display_kernel_name}")
    print()
    print_ranked_list(
        "Top 10 Instructions by Instructions Executed",
        "Instruction",
        instructions_executed,
    )
    print_ranked_list(
        "Top 10 Stall Reasons by Summed Source Samples",
        "Stall Reason",
        stall_totals,
    )
    print_ranked_list(
        "Top 10 Stall Reasons by Summed Source Samples (Not Issued)",
        "Stall Reason (Not Issued)",
        stall_not_issued_totals,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
