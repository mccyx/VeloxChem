#!/usr/bin/env python3

"""Design runtime-balanced DDDD K16 candidates from RS26 and old19."""

import csv
import json
import re
from pathlib import Path


NSYS = Path("nsys_results/resplit_comparison_job22840458/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv")
OUTPUT = Path("tools/dddd_split_search/k16_candidates.json")
LAYOUTS = {
    "rs_k16_runtime": {"suffix": "_RS", "count": 26, "segments": ((0, 4), (5, 12), (13, 20), (21, 24), (25, 25))},
    "old_k16_runtime": {"suffix": "", "count": 19, "segments": ((0, 1), (2, 3), (4, 9), (10, 14), (15, 17), (18, 18))},
}
KERNEL_RE = re.compile(r"computeExchangeFockDDDD(?P<index>\d+)(?P<layout>_RS)?_FP32\(")


def load_runtime():
    result = {name: [None] * cfg["count"] for name, cfg in LAYOUTS.items()}
    with NSYS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = KERNEL_RE.search(row["Name"])
            if not match:
                continue
            for name, cfg in LAYOUTS.items():
                if (match.group("layout") or "") == cfg["suffix"]:
                    index = int(match.group("index"))
                    if index < cfg["count"]:
                        result[name][index] = float(row["Avg (ns)"]) / 1.0e6
    for name, weights in result.items():
        if any(value is None for value in weights):
            raise ValueError(f"Missing Nsys weights for {name}")
    return result


def partition_segment(weights, start, end, groups):
    values = weights[start : end + 1]
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    size = len(values)
    dp = [[float("inf")] * (groups + 1) for _ in range(size + 1)]
    cut = [[None] * (groups + 1) for _ in range(size + 1)]
    dp[0][0] = 0.0
    for items in range(1, size + 1):
        for count in range(1, min(groups, items) + 1):
            for previous in range(count - 1, items):
                cost = max(dp[previous][count - 1], prefix[items] - prefix[previous])
                if cost < dp[items][count]:
                    dp[items][count] = cost
                    cut[items][count] = previous
    result = []
    items, count = size, groups
    while count:
        previous = cut[items][count]
        result.append(list(range(start + previous, start + items)))
        items, count = previous, count - 1
    return list(reversed(result)), dp[size][groups]


def allocate(weights, segments, total_groups):
    options = {}
    for segment_index, (start, end) in enumerate(segments):
        options[segment_index] = {}
        for groups in range(1, end - start + 2):
            options[segment_index][groups] = partition_segment(weights, start, end, groups)
    states = {(0, 0): (0.0, [])}
    for segment_index in range(len(segments)):
        next_states = {}
        for (_, used), (cost, allocations) in states.items():
            for groups, (_, segment_cost) in options[segment_index].items():
                key = (segment_index + 1, used + groups)
                candidate = (max(cost, segment_cost), allocations + [groups])
                if key not in next_states or candidate[0] < next_states[key][0]:
                    next_states[key] = candidate
        states = next_states
    _, allocations = states[(len(segments), total_groups)]
    result = []
    for segment_index, groups in enumerate(allocations):
        result.extend(options[segment_index][groups][0])
    return result, allocations


def main():
    runtime = load_runtime()
    payload = {}
    for name, cfg in LAYOUTS.items():
        groups, allocations = allocate(runtime[name], cfg["segments"], 16)
        flat = [index for group in groups for index in group]
        if len(groups) != 16 or flat != list(range(cfg["count"])):
            raise ValueError(f"Invalid {name}")
        payload[name] = {**cfg, "segment_group_counts": allocations, "fp32_ms": runtime[name], "groups": groups}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: data["groups"] for name, data in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
