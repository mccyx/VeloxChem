#!/usr/bin/env python3

"""Compare static Fock-expression work in old and resplit Exchange CUDA kernels."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


FUNCTION_RE = re.compile(
    r"^computeExchangeFock(?P<family>[A-Z]{4})(?P<index>\d*)"
    r"(?P<rs>_RS)?\(",
    re.MULTILINE,
)
ERI_RE = re.compile(r"const\s+(?:double|float)\s+eri_ijkl(?:_f)?\s*=")
BOYS_RE = re.compile(
    r"gpu::computeBoysFunction(?:_f)?\([^,]+,[^,]+,\s*(\d+)\s*,"
)
TOKEN_RE = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[fF]?"
    r"|[A-Za-z_]\w*|\+|-|\*|/|\(|\)|\[|\]|,"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count static arithmetic in old and RS Exchange kernels."
    )
    parser.add_argument("cuda", type=Path, help="Generated EriExchange CUDA file")
    parser.add_argument(
        "--families",
        default="DDDD,DDDP,DDDS,DPDD,DSDD,PDDD,PPDD,SDDD",
        help="Comma-separated family filter",
    )
    return parser.parse_args()


def extract_braced_block(text, search_start):
    start = text.find("{", search_start)
    if start < 0:
        raise ValueError("Function body opening brace not found")

    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ValueError("Unterminated function body")


def extract_eri_expression(body):
    match = ERI_RE.search(body)
    if not match:
        raise ValueError("eri_ijkl expression not found")
    end = body.find(";", match.end())
    if end < 0:
        raise ValueError("eri_ijkl expression terminator not found")
    return body[match.end() : end]


def count_arithmetic(expression):
    counts = {"add": 0, "sub": 0, "mul": 0, "div": 0}
    previous = None
    for token in TOKEN_RE.findall(expression):
        if token in ("+", "-"):
            # A sign after another operator or opening delimiter is unary.
            if previous is not None and previous not in ("+", "-", "*", "/", "(", "[", ","):
                counts["add" if token == "+" else "sub"] += 1
        elif token == "*":
            counts["mul"] += 1
        elif token == "/":
            counts["div"] += 1
        previous = token
    counts["total"] = sum(counts.values())
    return counts


def count_boys_groups(expression):
    # Count visible, top-level Boys-function groups after factoring. This is
    # not the generator's pre-factoring mathematical term count.
    return len(
        re.findall(
            r"(?:^|\n)\s*[+-]?\s*F\d+_t(?:_f)?\[\d+\].*?\*\s*\(",
            expression,
        )
    )


def load_kernels(path, families):
    text = path.read_text()
    rows = []
    for match in FUNCTION_RE.finditer(text):
        family = match.group("family")
        if family not in families:
            continue

        body = extract_braced_block(text, match.end())
        expression = extract_eri_expression(body)
        counts = count_arithmetic(expression)
        boys = BOYS_RE.search(body)
        index = match.group("index") or "-"
        layout = "rs" if match.group("rs") else "old"
        rows.append(
            {
                "family": family,
                "layout": layout,
                "index": index,
                "kernel": match.group(0)[:-1],
                "boys_groups": count_boys_groups(expression),
                "boys_order": int(boys.group(1)) if boys else -1,
                **counts,
            }
        )
    return rows


def main():
    args = parse_args()
    families = {item.strip().upper() for item in args.families.split(",") if item.strip()}
    rows = load_kernels(args.cuda, families)
    rows.sort(key=lambda row: (row["family"], row["layout"], str(row["index"])))

    print("family,layout,index,boys_groups,boys_order,add,sub,mul,div,total,kernel")
    for row in rows:
        print(
            f'{row["family"]},{row["layout"]},{row["index"]},'
            f'{row["boys_groups"]},{row["boys_order"]},{row["add"]},{row["sub"]},'
            f'{row["mul"]},{row["div"]},{row["total"]},{row["kernel"]}'
        )

    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        aggregate = totals[(row["family"], row["layout"])]
        aggregate["kernels"] += 1
        for key in ("boys_groups", "add", "sub", "mul", "div", "total"):
            aggregate[key] += row[key]

    print()
    print("family,layout,kernels,boys_groups,add,sub,mul,div,total")
    for (family, layout), aggregate in sorted(totals.items()):
        print(
            f'{family},{layout},{aggregate["kernels"]},{aggregate["boys_groups"]},'
            f'{aggregate["add"]},{aggregate["sub"]},{aggregate["mul"]},'
            f'{aggregate["div"]},{aggregate["total"]}'
        )


if __name__ == "__main__":
    main()
