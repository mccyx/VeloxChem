# Nsight Systems Helpers

This directory supports a lightweight `profile -> export csv -> summarize` workflow for the DDDD split benchmark.

## Files
- [`profile_dddd_with_ablation_name.sh`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/nsys/profile_dddd_with_ablation_name.sh)
  Runs `nsys profile` and exports `cuda_gpu_kern_sum` as CSV.
  Output naming is derived from `VLX_ABLATION_LOG`, and the top-level wrapper adds a timestamped run directory by default.

- [`summarize_from_ablation_name.sh`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/nsys/summarize_from_ablation_name.sh)
  Reads the benchmark log and matching `nsys` CSV, then generates a markdown summary in `benchmarks/`.

- [`summarize_dddd_split_benchmark.py`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/nsys/summarize_dddd_split_benchmark.py)
  Python backend used by the wrapper above.

## Quick start

### 1. Run the benchmark with a named ablation log
```bash
cd /cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2
source env.sh

export VLX_ABLATION_LOG=benchmarks/ablation_results_gh200_guanine8_split26_2026-04-17.log
vlx guanine-8.inp
```

### 2. Run NSYS with the same experiment name
```bash
tools/profiling/nsys/profile_dddd_with_ablation_name.sh guanine-8.inp split_compare
```

This produces files like:

```text
nsys_results/ablation_results_gh200_guanine8_split26_2026-04-17/20260421_143122/
  ablation_results_gh200_guanine8_split26_2026-04-17_split_compare.nsys-rep
  ablation_results_gh200_guanine8_split26_2026-04-17_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
```

### 3. Generate a markdown summary
```bash
tools/profiling/nsys/summarize_from_ablation_name.sh
```

This generates:

```text
benchmarks/
  ablation_results_gh200_guanine8_split26_2026-04-17_20260421_143122.md
```

## What the summary includes
- host timing summary for:
  - `DDDD split30 mixed path`
  - `DDDD split26 v2 mixed path`
- accuracy summary from `ablation_results.log`
- reconstructed `nsys` DDDD totals based on per-call averages
- speedup vs benchmark full-double reference
- top `split26 v2` kernels by total GPU time

## Recommended workflow for split30 vs split26
1. Build and run `vlx` with `VLX_ABLATION_LOG` set.
2. Run [`profile_dddd_with_ablation_name.sh`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/nsys/profile_dddd_with_ablation_name.sh) on the same input.
3. Run [`summarize_from_ablation_name.sh`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/nsys/summarize_from_ablation_name.sh).
4. Use host timing as the main speedup result.
5. Use `nsys` reconstructed totals as supporting evidence and for hotspot inspection.

## Notes
- If `VLX_ABLATION_LOG` is not set, the scripts fall back to `ablation_results.log`.
- If `VLX_PROFILING_TIMESTAMP` is set, the summary markdown gets the same timestamp suffix so repeated summarize runs do not overwrite each other.
- The `profile_dddd_with_ablation_name.sh` wrapper exports only `cuda_gpu_kern_sum` because that is enough for the current DDDD total-time analysis.
- The summary script prefers a same-stem `split_compare` CSV under `nsys_results/<ablation_stem>/...`.
- If it cannot find an exact same-stem CSV, it falls back to the most recent matching `*split_compare*_cuda_gpu_kern_sum*.csv` under `nsys_results/`.
- The raw `.nsys-rep` file remains the source of truth if you later want deeper analysis with `nsys stats` or the GUI.
