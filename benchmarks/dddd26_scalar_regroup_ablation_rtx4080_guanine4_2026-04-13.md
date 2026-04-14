# DDDD26 Scalar/Regroup Ablation On RTX 4080 With guanine-4 (2026-04-13)

Source artifacts:

- log: `ablation_results_rtx4080_guanine4_2026-04-13.log`
- csv: `dddd26_scalar_regroup_ablation_rtx4080_guanine4_2026-04-13.csv`
- binary inspected with `cuobjdump`: `build/python/veloxchem/veloxchemlib.so`
- workload: `vlx guanine-4.inp`
- device: `RTX 4080`

## Summary

`DDDD26` is fully integrated and runnable in the current `.inc`-based setup, and on this RTX 4080 + `guanine-4` workload both auto-generated variants beat the baseline kernel.

The current best result is:

- `DDDD26_FP32_auto_scalarized_regroup`

The gap to plain `auto_scalarized` is small, but `regroup` is slightly faster in this run while keeping the same resource usage.

## Timing And Resource Table

| Variant | elapsed ms | REG | STACK | SHARED | LOCAL | max \|ΔJ\| vs old26 | max \|ΔJ\| vs J2_2kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DDDD26_FP32 baseline` | 0.724744 | 52 | 40 | 2188 | 0 | - | - |
| `DDDD26_FP32_auto_scalarized` | 0.682062 | 58 | 0 | 2188 | 0 | 0.000000e+00 | 1.355253e-20 |
| `DDDD26_FP32_auto_scalarized_regroup` | 0.676781 | 58 | 0 | 2188 | 0 | 3.824386e-14 | 3.824388e-14 |

## Interpretation

- both auto-generated `DDDD26` variants beat the baseline:
  - baseline: `0.724744 ms`
  - auto scalarized: `0.682062 ms`
  - auto scalarized + regroup: `0.676781 ms`
- baseline uses `REG 52` with `STACK 40`
- both auto variants move to:
  - `REG 58`
  - `STACK 0`
  - `SHARED 2188`

So this is another case where the faster variant does not come from lower register count alone. Removing stack usage appears to matter more here than the small increase in `REG`.

The current `regroup` pass is also at least not harmful on this workload; in fact it gives a small additional speedup over plain `auto_scalarized` while preserving the same resource profile.

## Correctness

For `DDDD26_FP32_auto_scalarized`:

- contribution vs `old26`:
  - `max |ΔJ| = 0.000000e+00`
- mixed result vs `ref`:
  - `max |ΔJ| = 1.142644e-10`
- mixed result vs `J2_2kernels`:
  - `max |ΔJ| = 1.355253e-20`

For `DDDD26_FP32_auto_scalarized_regroup`:

- contribution vs `old26`:
  - `max |ΔJ| = 3.824386e-14`
- mixed result vs `ref`:
  - `max |ΔJ| = 1.142619e-10`
- mixed result vs `J2_2kernels`:
  - `max |ΔJ| = 3.824388e-14`

Both variants remain numerically well behaved in this run.

## Bottom Line

For this `RTX 4080 + guanine-4` run, the current `DDDD26` ranking is:

1. `DDDD26_FP32_auto_scalarized_regroup`
2. `DDDD26_FP32_auto_scalarized`
3. `DDDD26_FP32 baseline`

That is a nice signal for the current `regroup` pass: on this machine it is small, safe, and directionally beneficial.
