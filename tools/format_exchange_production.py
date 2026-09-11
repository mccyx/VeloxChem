#!/usr/bin/env python3
"""Tidy mechanically promoted exchange production blocks."""

import argparse
import re
from pathlib import Path


def format_block(match):
    body = match.group("body")
    lines = []
    for line in body.splitlines():
        if line.startswith("gpu::computeExchangeFock"):
            line = "            " + line
        elif line.startswith("        ") and not line.startswith("            "):
            line = "    " + line
        lines.append(line)
    return "        {\n" + "\n".join(lines) + "\n        }"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    text = args.source.read_text()
    block = re.compile(
        r"^        \{\n(?P<body>(?:(?!^        \}$).)*?"
        r"build_exchange_cuts_device(?:(?!^        \}$).)*)\n"
        r"^        \}$",
        re.MULTILINE | re.DOTALL,
    )
    text, count = block.subn(format_block, text)
    if count != 54:
        raise ValueError(f"expected 54 production blocks, found {count}")

    stale_comments = (
        "            // omptimers[thread_id].stop(\"    K PPPP MP cuts layout\");\n",
        "            // Sync first to drain the stream, then time H2D alone.\n",
        "            // gpuSafe(gpuStreamSynchronize(stream));\n",
        "            // omptimers[thread_id].start(\"    K PPPP MP H2D displ+fp32\");\n",
        "            // omptimers[thread_id].stop(\"    K PPPP MP H2D displ+fp32\");\n",
        "            // omptimers[thread_id].start(\"    K PPPP MP cuts compute\");\n",
    )
    for comment in stale_comments:
        text = text.replace(comment, "")
    text = re.sub(r"^        \}\n\n\n+", "        }\n\n", text, flags=re.MULTILINE)
    args.source.write_text(text)


if __name__ == "__main__":
    main()
