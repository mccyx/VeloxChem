# DDDD Split Benchmark Summary

## Data sources
- `ablation_results.log`
- `nsys_results/split26_compare/guanine8_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`

## Host Timing Summary
From the latest 4 timing samples in `ablation_results.log`:

| Variant | Samples (ms) | Avg (ms) | Speedup vs `split30` |
|---|---|---:|---:|
| `split30 mixed` | 45.209447, 45.206791, 45.185512, 45.371043 | 45.243198 | 1.0000x |
| `split26 v2 mixed` | 40.324522, 40.354216, 40.252363, 40.442565 | 40.343417 | 1.1215x |

`split26 v2 mixed` is about **10.83% faster** than `split30 mixed` on this host-side end-to-end DDDD benchmark segment.

## Accuracy Summary

### `split30 mixed` vs reference
- max `|ΔJ|` = `9.367355e-11`
- rms `|ΔJ|` = `8.990751e-12`
- rel error = `6.987230e-09`

### `split26 v2 mixed` vs reference
- max `|ΔJ|` = `9.352699e-11`
- rms `|ΔJ|` = `9.007277e-12`
- rel error = `6.976298e-09`

### `split26 v2 mixed` vs `split30 mixed`
- max `|ΔJ|` = `2.218671e-12`
- rms `|ΔJ|` = `2.415167e-13`
- rel error = `1.654935e-10`

### `split26 v2 - split30` delta contribution vs zero
- max `|ΔJ|` = `2.218671e-12`
- rms `|ΔJ|` = `2.415167e-13`

## NSYS Reconstructed Totals
These totals are reconstructed from per-call averages so repeated old-kernel ablation launches do not distort the benchmark-path comparison.

| Variant | DDDD total GPU time (ms) | Speedup |
|---|---:|---:|
| `split30 mixed` reconstructed | 184.444161 | 1.0000x |
| `split26 v2 mixed` | 160.003809 | 1.1527x |
| benchmark full-double reference | 242.820177 | - |

### Speedup vs benchmark full-double reference
- `split30 mixed`: `1.3165x`
- `split26 v2 mixed`: `1.5176x`

## Heaviest `split26 v2` kernels from NSYS

| Kernel | Calls | Total time (ms) |
|---|---:|---:|
| `computeCoulombFockDDDDv2_25_FP32` | 4 | 7.979 |
| `computeCoulombFockDDDDv2_3_FP32` | 4 | 7.951 |
| `computeCoulombFockDDDDv2_7_FP32` | 4 | 7.811 |
| `computeCoulombFockDDDDv2_16_FP32` | 4 | 7.266 |
| `computeCoulombFockDDDDv2_9_FP32` | 4 | 7.203 |


## Conclusion
- `split26 v2` improves the mixed-precision DDDD path by about **1.1215x** on host end-to-end timing.
- The `nsys` reconstructed totals give a consistent result of about **1.1527x**.
- Relative to the benchmark full-double reference path, `split26 v2` reaches about **1.5176x** speedup.
