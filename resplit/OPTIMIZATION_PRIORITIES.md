# Exchange MP Optimization Priorities

This note is a checkpoint for deciding which exchange kernel families to
optimize next. It combines the corrected 54-combination MP timings with the
old-versus-resplit (RS) comparison.

## Data sources

- Corrected all-combination timing: Slurm job `22521933`
- Old-versus-RS comparison: Slurm job `22840458`
- All-combination FP64 kernel total: `4414.151 ms`
- All-combination MP kernel total before selecting RS variants: `2705.070 ms`
- Measured all-combination kernel-only MP speedup: `1.6318x`

The two timing sets came from separate runs. Values that combine them are
therefore estimates and should be confirmed by timing the final hybrid
dispatcher in one executable.

## RS selection

Keep RS only where it improved both the original FP64 and MP implementations.
Ratios are `old / RS`, so a value greater than one means RS is faster.

| Family | FP64 ratio | MP ratio | Selected layout |
|---|---:|---:|---|
| `PDDD` | 1.2460x | 1.2437x | RS |
| `DPDD` | 1.1291x | 1.1418x | RS |
| `DDDP` | 1.0344x | 1.0582x | RS |
| `DDDD` | 0.8575x | 0.8521x | Old |
| `DDDS` | 0.6935x | 0.6920x | Old |
| `DSDD` | 0.6521x | 0.6761x | Old |
| `PPDD` | 0.6590x | 0.6889x | Old |
| `SDDD` | 0.6749x | 0.6833x | Old |

RS-original and old-original results agreed to a maximum absolute difference
of `2.255141e-17`. The worst relative error of RS MP against RS original was
`3.059177e-08`.

Selecting the three faster RS MP families is estimated to reduce the complete
54-family MP kernel total from about `2705 ms` to about `2645 ms`, an additional
reduction of approximately `2.2%`.

## Prioritization method

Optimization priority should not be based on per-family speedup alone. Two
quantities are considered:

1. **Current runtime share**: selected MP time divided by the total selected MP
   exchange-kernel time. This identifies current bottlenecks.
2. **Approximate headroom**: `selected MP time - selected FP64 time / 2`. This
   estimates the maximum removable time if MP could reach an ideal `2x` speedup
   from a `2:1` FP32-to-FP64 throughput ratio.

The headroom calculation is an optimistic ranking metric, not a performance
prediction. It assumes the relevant work can benefit fully from FP32 throughput
and ignores memory traffic, launch overhead, dependencies, occupancy, and the
fraction that must remain FP64.

## Priority ranking

The selected timings use RS measurements for `PDDD`, `DPDD`, and `DDDP`, and
the old measurements for the other families. Runtime shares are approximate
because the selected values combine the two runs described above.

| Priority | Family | Layout | Selected MP (ms) | Approx. MP share | Achieved MP speedup | Approx. headroom (ms) |
|---:|---|---|---:|---:|---:|---:|
| 1 | `PDDD` | RS | 165.2 | 6.2% | 1.49x | 42.1 |
| 2 | `DDDD` | Old | 119.1 | 4.5% | 1.40x | 35.5 |
| 3 | `PPDD` | Old | 112.0 | 4.2% | 1.58x | 23.4 |
| 4 | `DDDP` | RS | 97.7 | 3.7% | 1.55x | 21.9 |
| 5 | `PDDP` | Old | 113.6 | 4.3% | 1.63x | 21.3 |
| 6 | `PPPP` | Old | 154.3 | 5.8% | 1.73x | 21.2 |
| 7 | `PPDP` | Old | 186.0 | 7.0% | 1.78x | 20.3 |
| 8 | `DPDD` | RS | 83.6 | 3.2% | 1.56x | 18.4 |

`PPDP` has the largest remaining MP runtime among these families, but its
`1.78x` speedup is already relatively close to the simple `2x` ceiling. In
contrast, `PDDD` and `DDDD` combine substantial runtime with weak achieved MP
speedup, giving them more potential effect on the full exchange calculation.

## Recommended next work

1. Implement a production hybrid dispatcher using RS for `PDDD`, `DPDD`, and
   `DDDP`, with the old layout for all other families.
2. Re-run all 54 combinations in one executable to establish the hybrid
   baseline and remove cross-run uncertainty.
3. Profile `PDDD` and `DDDD` first with Nsight Systems and Nsight Compute.
4. For those families, compare FP64 and FP32 kernel occupancy, register use,
   memory throughput, instruction mix, and per-subkernel load balance.
5. Recalculate this ranking from the hybrid run before optimizing `PPDD` or
   `DDDP`.

These percentages describe total **exchange-kernel time**, not complete SCF
wall time. Overall SCF impact additionally depends on the fraction of SCF time
spent in exchange kernels.
