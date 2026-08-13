# DDDD Split Experiment

## Purpose

The supplied DDDD resplit changes the layout from old19 to RS26 and regresses MP performance. This experiment separates kernel-count effects from grouping/factoring effects and tests whether the existing old layout also benefits from fewer kernels.

## Host Results

Guanine-8 on one GH200 GPU. Each value averages three program runs and four SCF interactions per run. The host timer covers selected kernel launches, GPU execution, and final stream synchronization. It excludes allocation, zeroing, cut construction, copies, and validation.

| Layout | Source grouping | Kernels | MP time (ms) | Speedup vs old19 |
|---|---|---:|---:|---:|
| old19 | old | 19 | 118.959 | 1.0000x |
| RS26 | supplied RS | 26 | 139.353 | 0.8535x |
| RS K19 static | RS | 19 | 122.549 | 0.9692x |
| RS K19 runtime | RS | 19 | 123.265 | 0.9632x |
| RS K19 light | RS | 19 | 122.424 | 0.9697x |
| RS K16 runtime | RS | 16 | 115.078 | 1.0321x |
| old K16 runtime | old | 16 | 113.020 | 1.0525x |

At equal kernel count, old-derived K16 is about `1.0182x` faster than RS-derived K16. Kernel count is therefore important but does not fully determine performance; grouping, factoring, and generated instructions still matter.

The old-derived K16 winner reduces host MP time by `5.939 ms` per interaction relative to old19, corresponding to `1.0525x` speedup. Nsys measures `118.931 ms` for old19 and `113.064 ms` for the winner, corresponding to `1.0519x`.

All candidates passed the existing host validation. The worst observed K16 MP absolute error is `3.298412e-12`; the worst original-precision absolute difference against old original is `8.673617e-19`.

## Nsys Confirmation

| Layout | Original (ms/interaction) | Original speedup | MP (ms/interaction) | MP speedup |
|---|---:|---:|---:|---:|
| RS K16 runtime | 167.487 | 0.9998x | 115.073 | 1.0348x |
| old K16 runtime | 160.326 | 1.0445x | 113.064 | 1.0519x |

Nsys includes only CUDA kernel execution time and confirms the host-timer ranking.

## NCU Explanation

Old-derived K16 merges old groups `[0+1]`, `[8+9]`, and `[13+14]`. The complete FP32 bundle drops from `57,126,151,623` to `54,203,777,700` executed SM instructions, a `5.12%` reduction. The three merged groups individually execute about 28-31% fewer instructions, while every unchanged group remains effectively 1.000x. This directly attributes the improvement to eliminating duplicated per-kernel work at those boundaries.

## Detailed Data

- K19 host: `build_logs/dddd_k19_host_23044072/README.md`
- K16 host: `build_logs/dddd_k16_host_23047976/README.md`
- K16 Nsys: `nsys_results/dddd_k16_23048261/README.md`
- old19 vs old K16 NCU: `ncu_results/dddd_old_k16_23048651/README.md`
