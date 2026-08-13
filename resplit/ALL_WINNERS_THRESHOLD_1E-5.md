# All-Winner Exchange at Threshold 1e-5

## Configuration

- Date: 2026-08-13
- Hardware: one GH200 node (`nid002892`)
- Input: `guanine-8-hf-k1e-5.inp`
- Exchange threshold: `mixed_precision_threshold_k = 1e-5`
- Coulomb threshold: `mixed_precision_threshold_j = 1e-6`
- Exchange families: all 54
- Samples: three runs, four interactions per run (12 samples)
- Timing method: host timer

The same four selected exchange layouts were enabled together:

| family | selected layout |
|---|---|
| PDDD | `k4_m23` |
| DDDD | `old_k16_runtime` |
| DDDP | `old_k5` |
| DPDD | `rs_k4` |

The other 50 families retain their old layouts. No rebuild was required; the
same executable used for the `1e-6` all-winner benchmark was selected through
the existing environment variables.

## Definitions

For family `i`:

- `O_i` is the old-layout original FP64 time.
- `R_i` is the original FP64 time of the selected layout.
- `M_i` is the selected MP time.
- `C_i` is the GPU cut-building time.

The final old-to-new production speedup is:

`sum(O_i) / sum(M_i)`

The same-layout MP speedup is:

`sum(R_i) / sum(M_i)`

For the complete workload, the selected-layout FP64 aggregate preserves the
same-run comparison context:

`selected FP64 aggregate = old FP64 aggregate + sum(R_i - old comparison FP64_i)`

The other 50 families have no layout replacement, so their delta is zero.
Aggregate speedups are ratios of summed times, not arithmetic means of family
speedups.

The theoretical model assumes FP32 throughput is twice FP64 throughput:

`theoretical speedup = 1 / (f64_weighted + f32_weighted / 2)`

The fractions are weighted by each family's old original FP64 time.

## Results

| K threshold | weighted FP32 | theory kernel | old FP64 (ms) | selected FP64 (ms) | selected MP (ms) | cuts (ms) | vs old FP64 | vs selected FP64 | vs old FP64 + cuts | vs selected FP64 + cuts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-6` | 82.204% | 1.6979x | 4430.465 | 4271.470 | 2606.795 | 22.554 | 1.6996x | 1.6386x | 1.6850x | 1.6245x |
| `1e-5` | 89.828% | 1.8153x | 4440.181 | 4281.170 | 2448.200 | 23.197 | 1.8137x | 1.7487x | 1.7966x | 1.7323x |

`Vs old FP64` is the full old-to-new production speedup. `Vs selected FP64`
isolates the MP benefit after applying the same selected layouts to FP64 and
MP. The `1e-5` theoretical speedup with measured cuts is `1.7983x`.

In direct terms for `1e-5`:

- **`1.8137x`**: complete 54-family old original FP64 versus selected MP.
- **`1.7487x`**: complete 54-family selected-layout FP64 versus selected MP.
- **`1.7966x`**: old original FP64 versus selected MP plus GPU cut building.
- **`1.7323x`**: selected-layout FP64 versus selected MP plus GPU cut building.

Raising the threshold from `1e-6` to `1e-5` increases the weighted FP32
fraction by `7.624` percentage points. The selected MP kernel time falls by
`158.595 ms`, and the old-to-new kernel speedup rises from `1.6996x` to
`1.8137x`.

## Accuracy

All host-side validation blocks completed. Across the selected-winner samples:

| family | worst max absolute error | worst relative error |
|---|---:|---:|
| PDDD | 1.236881e-10 | 4.219275e-07 |
| DDDD | 9.697527e-11 | 2.060275e-07 |
| DDDP | 5.571944e-11 | 1.640586e-08 |
| DPDD | 4.986735e-11 | 1.468279e-08 |

For the complete selected 54-family workload, the worst absolute error is
`1.331139e-09` from SSPS, and the worst relative error is `4.219275e-07` from
PDDD. Compared with `1e-6`, the wider threshold improves performance but raises
the observed errors by roughly one order of magnitude.

## Timing Scope

The host kernel timer includes launches, GPU execution, and the final stream
synchronization. It excludes allocation and zeroing, cut building, device/host
copies, and accuracy validation. The `+ cuts` results add only the measured GPU
cut-building time.

## Artifacts

- Slurm job: `23148501`
- Host summary: `build_logs/exchange_winners_k1e-5_23148501/SUMMARY.md`
- Per-interaction data:
  `build_logs/exchange_winners_k1e-5_23148501/interaction_timings.csv`
- Per-combination data:
  `build_logs/exchange_winners_k1e-5_23148501/per_combination_winners.csv`

Nsys was not run for this threshold point.
