# DDDP Split Experiment

## Candidates

- old7: existing old layout.
- RS6: supplied resplit layout.
- RS-derived K5: `[0] [1] [2+3] [4] [5]`.
- old-derived K5: `[0] [1+2] [3] [4+5] [6]`.

Only adjacent kernels with the same Boys-function order are merged.

## Results

Guanine-8 on one GH200. Host values average three runs and four SCF interactions per run. The host timer covers selected launches, GPU execution, and final synchronization; it excludes allocation, zeroing, cut construction, copies, and validation.

| Layout | Host MP (ms) | Host speedup vs old7 | Nsys MP (ms/interaction) | Nsys speedup vs old7 |
|---|---:|---:|---:|---:|
| RS6 | 97.598 | 1.0581x | 97.647 | 1.0598x |
| RS-derived K5 | 88.904 | 1.1627x | 88.749 | 1.1657x |
| old-derived K5 | 87.152 | 1.1866x | 87.172 | 1.1863x |

Old-derived K5 is the fastest tested DDDP layout. At the same five-kernel count it is about `1.0201x` faster than RS-derived K5, showing that grouping/factoring still matters after controlling kernel count.

All variants passed host validation. The worst observed MP absolute error is `4.291909e-12`; the worst MP relative error is `1.275721e-09`.

## NCU Explanation

The old-derived K5 FP32 bundle executes `45,057,338,333` SM instructions versus `52,521,219,399` for old7, a `14.21%` reduction. Merged groups `[1+2]` and `[4+5]` reduce executed instructions by about 27% and 24%, respectively, while unchanged groups remain effectively 1.000x. This supports eliminated duplicated setup as the primary benefit.

Detailed data:

- Host: `build_logs/dddp_k5_host_23054258/README.md`
- Nsys: `nsys_results/dddp_k5_23055132/README.md`
- NCU: `ncu_results/dddp_old_k5_23055538/README.md`
