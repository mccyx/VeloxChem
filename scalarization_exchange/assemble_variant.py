#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
from typing import Tuple


def function_span(text: str, function: str) -> Tuple[int, int]:
    start = text.find(function + "(")
    if start < 0:
        raise ValueError(f"function not found: {function}")

    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"function has no body: {function}")

    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return start, pos + 1

    raise ValueError(f"unbalanced function body: {function}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("fragments", nargs="*")
    args = parser.parse_args()

    if args.source.startswith("git:"):
        revision_path = args.source[len("git:"):]
        text = subprocess.check_output(
            ["git", "show", revision_path], universal_newlines=True
        )
    else:
        text = Path(args.source).read_text()
    for fragment_path in args.fragments:
        fragment = Path(fragment_path).read_text()
        function = fragment.split("(", 1)[0].strip()
        start, end = function_span(text, function)
        text = text[:start] + fragment + text[end:]

    Path(args.output).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
