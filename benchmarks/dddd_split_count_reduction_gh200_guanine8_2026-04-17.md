# DDDD Split-Count Reduction Benchmark on GH200 (guanine-8)

## Goal
- Measure the effect of reducing the DDDD mixed-precision split count from `30` kernels to `26` kernels.
- Treat this as a result about the split policy itself, separate from later IR-pass work.

## Variants
- `split30 mixed`: original mixed-precision DDDD path in [`FockDriverGPU.cu`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu), using the original `3` and `4` kernels separately.
- `split26 v2 mixed`: new mixed-precision DDDD path using the generated `v2` kernels included from [`dddd_split26_v2_fp32.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd_split26_v2_fp32.inc) and [`dddd_split26_v2_fp64.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd_split26_v2_fp64.inc).

## Data sources
- [`ablation_results.log`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/ablation_results.log)
- [`guanine8_split_compare.nsys-rep`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/split26_compare/guanine8_split_compare.nsys-rep)
- [`guanine8_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/nsys_results/split26_compare/guanine8_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv)

## Host Timing Summary
Host timing is the main comparison here because both paths are measured with the same start/end logic around the full DDDD mixed path.

From the latest 4 timing samples in [`ablation_results.log`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/ablation_results.log):

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

Interpretation:
- The new `split26` path stays at essentially the same mixed-result accuracy level as the original `split30` path.
- The direct difference between `split26` and `split30` is very small compared with the overall mixed-precision error versus the full reference.

## NSYS Support
`nsys` is useful here as supporting evidence, but not as the primary metric.

Reason:
- the benchmark run also includes extra single-kernel ablation launches for some old `split30` kernels such as `DDDD3_FP32`, `DDDD4_FP32`, `DDDD21_FP32`, and `DDDD26_FP32`;
- because of that, a raw sum over all old `computeCoulombFockDDDD*_FP32/FP64` rows slightly overcounts the pure `split30 mixed` path;
- in [`FockDriverGPU.cu`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu), these old kernels appear once inside the full `split30 mixed` path and then again in separate ablation timing blocks, so `nsys` sees both launches under the same kernel name.

Using the exported kernel summary:

| Metric | Total GPU time (ms) |
|---|---:|
| `split26 v2` kernels only (`computeCoulombFockDDDDv2_*_FP64/FP32`) | 160.003809 |
| old `split30` raw sum (`computeCoulombFockDDDD*_FP64/FP32`) | 198.111553 |
| old `split30` adjusted estimate | 184.444161 |

The adjusted `nsys` estimate gives an approximate speedup of **1.1527x** for `split26 v2` over `split30`.

This is directionally consistent with the host timing result (`1.1215x`).

## Overall Comparison vs Original Full Double
This is a useful overall implementation comparison, but it should be treated as a secondary result because it mixes two effects:
- split policy change (`30 -> 26`)
- precision change (`full double -> mixed precision`)

A small but important detail: the full-double DDDD kernels also appear twice in the profiled run.
- one set belongs to the normal production path writing into `d_mat_J`;
- one set belongs to the benchmark reference path writing into `d_mat_J2_ref`.

So for the benchmark-level comparison, the fairer `original full double` number is the per-call reconstructed reference-path total rather than the raw `nsys` sum.

From the same `nsys` kernel summary:

| Variant | DDDD total GPU time (ms) | Speedup vs benchmark full-double reference |
|---|---:|---:|
| original full double benchmark reference (reconstructed from per-call averages) | 242.820177 | 1.0000x |
| `split30 mixed` adjusted estimate | 184.444161 | 1.3165x |
| `split26 v2 mixed` | 160.003809 | 1.5176x |

Interpretation:
- relative to the benchmark full-double reference path, the current mixed `split30` path is about `1.32x` faster;
- the new mixed `split26 v2` path reaches about `1.52x` speedup vs the benchmark full-double reference.

## Heaviest `split26 v2` kernels from NSYS

Top `split26 v2` kernels by total GPU time:

| Kernel | Calls | Total time (ms) |
|---|---:|---:|
| `computeCoulombFockDDDDv2_25_FP32` | 4 | 7.979 |
| `computeCoulombFockDDDDv2_3_FP32` | 4 | 7.951 |
| `computeCoulombFockDDDDv2_7_FP32` | 4 | 7.811 |
| `computeCoulombFockDDDDv2_16_FP32` | 4 | 7.266 |
| `computeCoulombFockDDDDv2_9_FP32` | 4 | 7.203 |

## Conclusion
- This result should be treated as a clean benchmark for **reducing the DDDD split count itself**.
- On GH200 with `guanine-8`, moving from `split30 mixed` to `split26 v2 mixed` gives about **1.12x** speedup at the full DDDD-path level while preserving mixed-result accuracy.
- This supports using `split26` as the better base implementation before adding further IR-pass optimizations.
