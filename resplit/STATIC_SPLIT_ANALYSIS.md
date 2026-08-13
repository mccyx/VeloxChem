# Static Analysis of Old and Resplit Exchange Kernels

## Scope

This analysis reads the generated `src/gpu/EriExchange.cu` directly. It counts
arithmetic operators in each factored `eri_ijkl` source expression and compares
the sum across all subkernels in an old or resplit (RS) family.

These are static source-expression counts, not compiled instructions or dynamic
FLOPs. They exclude repeated kernel setup, Boys-function evaluation, indexing,
loads/stores, control flow, and output reduction. NCU is required to measure the
compiled and dynamically executed work.

## Family summary

| Family | Old kernels | RS kernels | Old ERI ops | RS ERI ops | RS/old ops | MP old/RS speed ratio |
|---|---:|---:|---:|---:|---:|---:|
| `DDDD` | 19 | 26 | 25281 | 25295 | 1.0006 | 0.8521x |
| `DDDP` | 7 | 6 | 6687 | 6685 | 0.9997 | 1.0582x |
| `DDDS` | 1 | 2 | 2228 | 2269 | 1.0184 | 0.6920x |
| `DPDD` | 7 | 5 | 6623 | 6619 | 0.9994 | 1.1418x |
| `DSDD` | 1 | 2 | 2169 | 2210 | 1.0189 | 0.6761x |
| `PDDD` | 8 | 5 | 6625 | 6619 | 0.9991 | 1.2437x |
| `PPDD` | 1 | 2 | 2164 | 2205 | 1.0189 | 0.6889x |
| `SDDD` | 1 | 2 | 2163 | 2204 | 1.0190 | 0.6833x |

Ratios are `old / RS`, so values above one mean RS is faster.

## Initial interpretation

The total factored ERI arithmetic is effectively unchanged for `DDDD`, `DDDP`,
`DPDD`, and `PDDD`. The four one-to-two splits add about 1.9% static ERI
arithmetic, most likely because factoring opportunities differ across the split
boundary, but that increase is much smaller than their 32-35% runtime regression.

Kernel count currently predicts the direction of every measured result:

- `PDDD` (`8 -> 5`), `DPDD` (`7 -> 5`), and `DDDP` (`7 -> 6`) improve.
- `DDDD` (`19 -> 26`) regresses.
- `DDDS`, `DSDD`, `PPDD`, and `SDDD` (`1 -> 2`) regress.

The leading hypothesis is therefore repeated per-kernel setup and reduction
work, launch overhead, and/or loss of useful compiler optimization across split
boundaries, rather than a major change in the core ERI arithmetic count.

## Next measurements

For `PPDD`, compare the old kernel against the sum of RS0 and RS1 for both FP32
and FP64:

1. Executed and issued instructions.
2. FP arithmetic instructions, with FMA reported separately.
3. Registers per thread, occupancy, and local-memory spills.
4. Memory and atomic instruction counts.
5. Warp stalls and achieved SM throughput.

The same analysis can then be applied to `PDDD` as the representative successful
resplit. This gives one regression and one improvement with which to test the
kernel-count and duplicated-setup hypothesis.
