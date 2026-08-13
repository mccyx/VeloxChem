# DPDD Split Experiment

## Candidates

- old7: existing old layout.
- RS5: supplied resplit.
- old-derived K5: `[0] [1+2] [3] [4+5] [6]`.
- RS-derived K4: `[0] [1+2] [3] [4]`.
- old-derived K4: runtime-balanced `[0+1] [2+3] [4] [5+6]`.

## Results

Guanine-8 on one GH200. Host values average three runs and four SCF interactions per run. The timer covers selected launches, GPU execution, and final synchronization; it excludes allocation, zeroing, cut construction, copies, and validation.

| Layout | Host MP (ms) | Host speedup vs old7 | Nsys MP (ms/interaction) | Nsys speedup vs old7 |
|---|---:|---:|---:|---:|
| RS5 | 83.758 | 1.1416x | 83.828 | 1.1412x |
| old-derived K5 | 82.375 | 1.1622x | 82.179 | 1.1647x |
| RS-derived K4 | 72.408 | 1.3232x | 72.348 | 1.3224x |
| old-derived K4 | 73.914 | 1.2927x | 73.897 | 1.2950x |

RS-derived K4 is the fastest tested DPDD layout. At equal K4 count it is about `1.0214x` faster than old-derived K4, unlike DDDD and DDDP where the old-derived candidate won. This confirms that kernel count is not a sufficient split criterion; the expression grouping/factoring must also be measured.

All variants passed validation. The worst observed MP absolute error is `8.472268e-12`; the worst original-precision absolute difference is `1.170938e-17`.

## NCU Explanation

The RS-derived K4 FP32 bundle executes `36,545,483,258` SM instructions versus `49,271,653,599` for old7, a `25.83%` reduction. Because old7 and RS-K4 use different expression groupings, only complete bundles are compared. The instruction ratio is `1.3482x`, close to the Nsys MP runtime speedup of `1.3224x`.

Detailed data:

- Host: `build_logs/dpdd_split_host_23074898/README.md`
- Nsys: `nsys_results/dpdd_split_23084733/README.md`
- NCU: `ncu_results/dpdd_rs_k4_23086485/README.md`
