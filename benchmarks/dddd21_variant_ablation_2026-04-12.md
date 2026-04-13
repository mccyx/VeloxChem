# DDDD21 Variant Ablation (2026-04-12)

Source artifacts:

- log: `ablation_results.log`
- binary inspected with `cuobjdump`: `build/python/veloxchem/veloxchemlib.so`
- workload: `vlx guanine-8.inp`

## Summary

For `DDDD21`, the current hand-tuned `scalarized` variant remains the fastest version in this run.

The new auto-generated `scalar`-pipeline variant (historically split as `scalarize+hoist+rewrite`) is also a clear win over baseline and, notably, reproduces the old `DDDD21` contribution exactly in this workload. The first experimental `auto+cse` pass remains correct, but it is slightly slower than plain `auto_scalarized` and does not improve on the hand-tuned kernel.

The `shared_staged` variant is correct and reduces register pressure relative to baseline, but it increases shared-memory usage dramatically and is slower than the plain `scalarized` version.

## Timing And Resource Table

| Variant | elapsed ms | REG | STACK | SHARED | LOCAL | max \|ΔJ\| vs old21 | max \|ΔJ\| vs J2_2kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DDDD21_FP32 baseline` | 0.822469 | 78 | 0 | 2188 | 0 | - | - |
| `DDDD21_FP32_scalarized` | 0.444497 | 64 | 0 | 2188 | 0 | 9.945100e-14 | 9.945101e-14 |
| `DDDD21_FP32_auto_scalarized` | 0.457328 | 48 | 0 | 2188 | 0 | 0.000000e+00 | 5.421011e-20 |
| `DDDD21_FP32_auto_scalarized_cse` | 0.458161 | 48 | 0 | 2188 | 0 | 3.661746e-14 | 3.661746e-14 |
| `DDDD21_FP32_shared_staged` | 0.579660 | 64 | 0 | 18572 | 0 | 9.945100e-14 | 9.945101e-14 |

## Interpretation

- hand `scalarized` is still the fastest current DDDD21 variant:
  - time improves from `0.822469 ms` to `0.444497 ms`
  - register count drops from `78` to `64`
- `auto_scalarized` is very close to hand `scalarized`:
  - `0.457328 ms` vs `0.444497 ms`
  - and it drives register count even lower: `REG 48`
  - in this run it reproduces the old `DDDD21` contribution exactly (`max |ΔJ| = 0`)
- `auto_scalarized_cse` does not buy anything yet:
  - `0.458161 ms`, slightly slower than plain `auto_scalarized`
  - `REG` stays at `48`
  - and it introduces a tiny but measurable difference (`3.661746e-14`) that plain `auto_scalarized` did not
- `shared_staged` keeps the same register count as hand `scalarized`
  - `REG 64`
  - but shared-memory usage jumps from `2188` to `18572`
  - timing worsens relative to `scalarized`: `0.579660 ms` vs `0.444497 ms`

So for the current DDDD21 implementations:

- reducing register pressure is important
- but lower register count alone is not enough to beat the hand-tuned version
- increasing shared-memory footprint this much is not a free optimization
- `shared_staged` should be viewed as a different occupancy/resource tradeoff, not as a clear upgrade over `scalarized`
- the current exact-local `CSE` pass is not yet the source of extra DDDD21 speedup

## Bottom Line

The current best DDDD21 kernel remains:

- `DDDD21_FP32_scalarized`

The most encouraging automation result is:

- `DDDD21_FP32_auto_scalarized`

because it gets very close to the hand-tuned kernel while using only the default source-shaping pipeline (`scalar`).

This also sharpens where future automation effort should go:

- `DDDD34` was mainly about indexed-access hoisting
- `DDDD21` is where more advanced transforms such as stronger CSE, expression splitting, and more semantic regrouping are more likely to matter
