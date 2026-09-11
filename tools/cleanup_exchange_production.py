#!/usr/bin/env python3
"""Remove validation-only exchange helpers from a production host source."""

import argparse
import re
from pathlib import Path


DEAD_HELPERS = (
    "check_J_against_ref",
    "print_exchange_validation_timing",
    "print_exchange_k_reuse_status",
    "print_exchange_resplit_timing",
)


def remove_function(text: str, name: str) -> str:
    match = re.search(r"^void\s+" + re.escape(name) + r"\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing function {name}")

    brace = text.find("{", match.end())
    if brace < 0:
        raise ValueError(f"missing body for {name}")

    depth = 0
    end = None
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError(f"unclosed body for {name}")

    while end < len(text) and text[end] == "\n":
        end += 1
    return text[:match.start()] + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    text = args.source.read_text()
    for helper in DEAD_HELPERS:
        text = remove_function(text, helper)
    args.source.write_text(text)


if __name__ == "__main__":
    main()
