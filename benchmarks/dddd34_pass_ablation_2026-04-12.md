# DDDD34 Pass Ablation (2026-04-12)

Source artifacts:

- log: `ablation_results.log`
- binary inspected with `cuobjdump`: `build/python/veloxchem/veloxchemlib.so`
- workload: `vlx guanine-8.inp`

## Summary

For `DDDD34`, the main performance win comes from `indexed-access hoist`, not from scalar declaration alone and not from direct `[0/1/2]` rewrites alone.

Key observations:

- `auto_scalarize_only` is essentially identical to merged baseline in both time and register count.
- `auto_scalarize_rewrite` is also essentially identical to merged baseline.
- `auto_scalarize_hoist` drops to the same performance class as full `auto_scalarized`.
- `auto_scalarized_cse` remains correct, but does not improve on `auto_scalarized` and raises register count slightly.

## Timing And Resource Table

| Variant | Passes | elapsed ms | REG | STACK | SHARED | LOCAL | max \|ΔJ\| vs separate 3+4 | max \|ΔJ\| vs J2_2kernels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DDDD34_FP32 merged` | none | 1.261718 | 93 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_scalarized` | manual | 0.799366 | 78 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_auto_scalarize_only` | `scalarize` | 1.263031 | 93 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_auto_scalarize_hoist` | `scalarize,hoist` | 0.811621 | 77 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_auto_scalarize_rewrite` | `scalarize,rewrite` | 1.261366 | 93 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_auto_scalarized` | `scalarize,hoist,rewrite` | 0.803846 | 77 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |
| `DDDD34_FP32_auto_scalarized_cse` | `scalarize,hoist,rewrite,cse` | 0.808326 | 80 | 0 | 3248 | 0 | 3.160185e-13 | 3.160185e-13 |

## Interpretation

- `scalarize` alone is not enough for this kernel:
  - time stays at `~1.26 ms`
  - register count stays at `93`
- `rewrite` alone is also not enough:
  - time stays at `~1.26 ms`
  - register count stays at `93`
- `hoist` is the decisive source-shaping pass:
  - `REG` drops from `93` to `77`
  - time drops from `~1.26 ms` to `~0.81 ms`
- adding `rewrite` on top of `hoist` changes little for `DDDD34`
- current exact local `cse` is not a net win here:
  - `REG` rises from `77` to `80`
  - time slightly worsens from `0.803846 ms` to `0.808326 ms`

## Bottom Line

For `DDDD34`, dynamic indexed-access hoisting is the primary performance-enabling pass.

The practical default pipeline choice is therefore:

- `scalar`

with the note that this bundled `scalar` stage corresponds to the historical `scalarize+hoist+rewrite` sequence used in the ablation table above.

while the current exact local `cse` pass should remain experimental rather than part of the default path.
