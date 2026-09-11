#!/usr/bin/env python3

"""Design DDDD RS26-to-K19 candidates from static work and Nsys runtime."""

import argparse
import csv
import json
import re
from pathlib import Path


SEGMENTS = ((0, 4), (5, 12), (13, 20), (21, 24), (25, 25))
TARGET_COUNTS = (4, 5, 6, 3, 1)  # 19 kernels, with merges kept within Boys order.
KERNEL_RE = re.compile(r"computeExchangeFockDDDD(?P<index>\d+)_RS_FP32\(")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=Path, default=Path("src/gpu/EriExchange.cu"))
    parser.add_argument(
        "--nsys-csv",
        type=Path,
        default=Path("nsys_results/resplit_comparison_job22840458/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("tools/dddd_split_search/candidates.json"))
    return parser.parse_args()


def function_body(text, name):
    pos = text.find(f"\n{name}(")
    start = text.find("{", pos)
    depth = 0
    for end in range(start, len(text)):
        depth += text[end] == "{"
        depth -= text[end] == "}"
        if depth == 0:
            return text[start : end + 1]
    raise ValueError(name)


def static_weights(cuda):
    text = cuda.read_text()
    weights = []
    for index in range(26):
        body = function_body(text, f"computeExchangeFockDDDD{index}_RS")
        start = re.search(r"const\s+double\s+eri_ijkl\s*=", body).end()
        expression = body[start : body.find(";", start)]
        weights.append(len(re.findall(r"(?<![eE])[+*/-]", expression)))
    return weights


def runtime_weights(path):
    weights = [None] * 26
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = KERNEL_RE.search(row["Name"])
            if match:
                weights[int(match.group("index"))] = float(row["Avg (ns)"]) / 1.0e6
    if any(value is None for value in weights):
        raise ValueError("Missing DDDD RS FP32 Nsys rows")
    return weights


def linear_partition(weights, start, end, groups):
    values = weights[start : end + 1]
    count = len(values)
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    dp = [[float("inf")] * (groups + 1) for _ in range(count + 1)]
    cut = [[None] * (groups + 1) for _ in range(count + 1)]
    dp[0][0] = 0.0
    for items in range(1, count + 1):
        for group_count in range(1, min(groups, items) + 1):
            for previous in range(group_count - 1, items):
                cost = max(dp[previous][group_count - 1], prefix[items] - prefix[previous])
                if cost < dp[items][group_count]:
                    dp[items][group_count] = cost
                    cut[items][group_count] = previous
    result = []
    items = count
    group_count = groups
    while group_count:
        previous = cut[items][group_count]
        result.append(list(range(start + previous, start + items)))
        items = previous
        group_count -= 1
    return list(reversed(result))


def balanced(weights):
    groups = []
    for (start, end), count in zip(SEGMENTS, TARGET_COUNTS):
        groups.extend(linear_partition(weights, start, end, count))
    return groups


def light_setup(runtime):
    groups = [[index] for index in range(26)]
    while len(groups) > 19:
        choices = []
        for pos in range(len(groups) - 1):
            left, right = groups[pos], groups[pos + 1]
            same_segment = any(left[0] >= start and right[-1] <= end for start, end in SEGMENTS)
            if same_segment:
                choices.append((sum(runtime[i] for i in left + right), pos))
        _, pos = min(choices)
        groups[pos : pos + 2] = [groups[pos] + groups[pos + 1]]
    return groups


def main():
    args = parse_args()
    static = static_weights(args.cuda)
    runtime = runtime_weights(args.nsys_csv)
    candidates = {
        "static_balanced": balanced(static),
        "runtime_balanced": balanced(runtime),
        "light_setup": light_setup(runtime),
    }
    for name, groups in candidates.items():
        flat = [item for group in groups for item in group]
        if len(groups) != 19 or flat != list(range(26)):
            raise ValueError(f"Invalid candidate {name}")
    payload = {"segments": SEGMENTS, "target_counts": TARGET_COUNTS, "static": static, "fp32_ms": runtime, "candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
