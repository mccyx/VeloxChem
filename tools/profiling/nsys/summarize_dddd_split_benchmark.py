#!/usr/bin/env python3
import csv
import os
import re
import statistics
import sys
from pathlib import Path


def extract_last_n_samples(text, tag, n):
    pattern = rf"=== {re.escape(tag)} ===\n  elapsed ms = ([0-9.]+)"
    values = [float(x) for x in re.findall(pattern, text)]
    return values[-n:]


def extract_last_block(text, tag):
    pattern = rf"=== {re.escape(tag)} ===\n(.*?)(?=\n============================)"
    matches = list(re.finditer(pattern, text, re.S))
    if not matches:
        raise ValueError(f"Tag not found: {tag}")
    block = matches[-1].group(1)
    result = {}
    for line in block.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def fmt_samples(values):
    return ", ".join(f"{v:.6f}" for v in values)


def load_kernel_rows(csv_path):
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            try:
                total_ns = float(row[1])
                calls = int(float(row[2]))
                avg_ns = float(row[3])
            except ValueError:
                continue
            rows.append((row[-1], total_ns, calls, avg_ns))
    return rows


def reconstruct_nsys_totals(rows):
    full_ref_ms = 0.0
    old_split30_ms = 0.0
    v2_ms = 0.0
    v2_top = []

    for name, total_ns, calls, avg_ns in rows:
        mv2 = re.search(r"computeCoulombFockDDDDv2_(\d+)_FP(32|64)\(", name)
        mold = re.search(r"computeCoulombFockDDDD(\d+)_FP(32|64)\(", name)
        mfull = re.search(r"computeCoulombFockDDDD(\d+)\(", name)

        if mv2:
            v2_ms += total_ns / 1e6
            v2_top.append((name, calls, total_ns / 1e6))
            continue

        if mold:
            idx = int(mold.group(1))
            prec = mold.group(2)
            # Reconstruct one benchmark mixed-path pass from per-call averages.
            # Old FP32 kernels 3/4/21/26 are also re-launched in separate ablation blocks,
            # so raw totals overcount them.
            if prec == "32" and idx in {3, 4, 21, 26}:
                old_split30_ms += avg_ns * 4 / 1e6
            else:
                old_split30_ms += total_ns / 1e6
            continue

        if mfull and "_FP" not in name:
            # The full-double benchmark reference and the production path both launch the same kernels.
            # One benchmark reference pass corresponds to 4 calls in this profile.
            full_ref_ms += avg_ns * 4 / 1e6

    v2_top.sort(key=lambda x: x[2], reverse=True)
    return {
        "full_ref_ms": full_ref_ms,
        "old_split30_ms": old_split30_ms,
        "v2_ms": v2_ms,
        "v2_top": v2_top[:5],
    }


def main() -> int:
    ablation_log = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("VLX_ABLATION_LOG", "ablation_results.log")
    )
    if not ablation_log.exists():
        raise SystemExit(f"Ablation log not found: {ablation_log}")

    stem = ablation_log.stem
    nsys_csv = (
        Path("nsys_results")
        / stem
        / f"{stem}_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv"
    )
    if len(sys.argv) > 2:
        nsys_csv = Path(sys.argv[2])
    if not nsys_csv.exists():
        candidates = sorted(
            Path("nsys_results").glob("**/*split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            nsys_csv = candidates[0]
        else:
            raise SystemExit(f"NSYS CSV not found: {nsys_csv}")

    text = ablation_log.read_text()
    split30_samples = extract_last_n_samples(text, "DDDD split30 mixed path timing", 4)
    split26_samples = extract_last_n_samples(text, "DDDD split26 v2 mixed path timing", 4)
    if not split30_samples or not split26_samples:
        raise SystemExit("Missing split30 or split26 host timing samples in ablation log.")

    split30_avg = statistics.mean(split30_samples)
    split26_avg = statistics.mean(split26_samples)
    host_speedup = split30_avg / split26_avg
    host_improvement_pct = (split30_avg - split26_avg) / split30_avg * 100.0

    split30_ref = extract_last_block(text, "DDDD Two Separate Kernels (J2_2kernels vs ref)")
    split26_ref = extract_last_block(text, "DDDD split26 v2 mixed result (Jv2 vs ref)")
    split26_vs_old = extract_last_block(text, "DDDD split26 v2 mixed result (Jv2 vs J2_2kernels)")
    split26_delta = extract_last_block(text, "DDDD split26 v2 delta contribution (Jv2-J2 vs 0)")

    rows = load_kernel_rows(nsys_csv)
    totals = reconstruct_nsys_totals(rows)
    full_ref_ms = float(totals["full_ref_ms"])
    old_split30_ms = float(totals["old_split30_ms"])
    v2_ms = float(totals["v2_ms"])
    v2_top = totals["v2_top"]

    nsys_speedup = old_split30_ms / v2_ms
    full_old_speedup = full_ref_ms / old_split30_ms
    full_v2_speedup = full_ref_ms / v2_ms

    summary_suffix = os.environ.get("VLX_PROFILING_TIMESTAMP", "").strip()
    if summary_suffix:
        md_path = Path("benchmarks") / f"{stem}_{summary_suffix}.md"
    else:
        md_path = Path("benchmarks") / f"{stem}.md"
    md = f"""# DDDD Split Benchmark Summary

## Data sources
- `{ablation_log}`
- `{nsys_csv}`

## Host Timing Summary
From the latest 4 timing samples in `{ablation_log}`:

| Variant | Samples (ms) | Avg (ms) | Speedup vs `split30` |
|---|---|---:|---:|
| `split30 mixed` | {fmt_samples(split30_samples)} | {split30_avg:.6f} | 1.0000x |
| `split26 v2 mixed` | {fmt_samples(split26_samples)} | {split26_avg:.6f} | {host_speedup:.4f}x |

`split26 v2 mixed` is about **{host_improvement_pct:.2f}% faster** than `split30 mixed` on this host-side end-to-end DDDD benchmark segment.

## Accuracy Summary

### `split30 mixed` vs reference
- max `|ΔJ|` = `{split30_ref.get('max |ΔJ|', 'N/A')}`
- rms `|ΔJ|` = `{split30_ref.get('rms |ΔJ|', 'N/A')}`
- rel error = `{split30_ref.get('rel error', 'N/A')}`

### `split26 v2 mixed` vs reference
- max `|ΔJ|` = `{split26_ref.get('max |ΔJ|', 'N/A')}`
- rms `|ΔJ|` = `{split26_ref.get('rms |ΔJ|', 'N/A')}`
- rel error = `{split26_ref.get('rel error', 'N/A')}`

### `split26 v2 mixed` vs `split30 mixed`
- max `|ΔJ|` = `{split26_vs_old.get('max |ΔJ|', 'N/A')}`
- rms `|ΔJ|` = `{split26_vs_old.get('rms |ΔJ|', 'N/A')}`
- rel error = `{split26_vs_old.get('rel error', 'N/A')}`

### `split26 v2 - split30` delta contribution vs zero
- max `|ΔJ|` = `{split26_delta.get('max |ΔJ|', 'N/A')}`
- rms `|ΔJ|` = `{split26_delta.get('rms |ΔJ|', 'N/A')}`

## NSYS Reconstructed Totals
These totals are reconstructed from per-call averages so repeated old-kernel ablation launches do not distort the benchmark-path comparison.

| Variant | DDDD total GPU time (ms) | Speedup |
|---|---:|---:|
| `split30 mixed` reconstructed | {old_split30_ms:.6f} | 1.0000x |
| `split26 v2 mixed` | {v2_ms:.6f} | {nsys_speedup:.4f}x |
| benchmark full-double reference | {full_ref_ms:.6f} | - |

### Speedup vs benchmark full-double reference
- `split30 mixed`: `{full_old_speedup:.4f}x`
- `split26 v2 mixed`: `{full_v2_speedup:.4f}x`

## Heaviest `split26 v2` kernels from NSYS

| Kernel | Calls | Total time (ms) |
|---|---:|---:|
"""

    for name, calls, total_ms in v2_top:
        kernel = name.split("gpu::", 1)[-1]
        kernel = kernel.split("(", 1)[0]
        md += f"| `{kernel}` | {calls} | {total_ms:.3f} |\n"

    md += f"""

## Conclusion
- `split26 v2` improves the mixed-precision DDDD path by about **{host_speedup:.4f}x** on host end-to-end timing.
- The `nsys` reconstructed totals give a consistent result of about **{nsys_speedup:.4f}x**.
- Relative to the benchmark full-double reference path, `split26 v2` reaches about **{full_v2_speedup:.4f}x** speedup.
"""

    md_path.write_text(md)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
