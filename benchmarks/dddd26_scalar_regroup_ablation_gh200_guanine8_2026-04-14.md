# DDDD26 FP32 Scalar/Regroup Ablation on GH200 (guanine-8)

## Data sources
- `benchmarks/ablation_results_gh200_guanine8_dddd26_auto_2026-04-14.log`
- `benchmarks/cuobjdump_resource_usage_gh200_2026-04-14.txt` (re-generated at 13:39)

## Timing summary (DDDD26 FP32)

From 4 timing samples in the ablation log:

| Variant | Samples (ms) | Avg (ms) | Speedup vs baseline |
|---|---|---:|---:|
| `DDDD26_FP32` baseline | 0.799077, 0.804453, 0.798821, 0.802629 | 0.801245 | 1.0000x |
| `DDDD26_FP32_auto_scalarized` | 0.456817, 0.469040, 0.456721, 0.458673 | 0.460313 | 1.7407x |
| `DDDD26_FP32_auto_scalarized_regroup` | 0.451921, 0.463185, 0.450417, 0.453297 | 0.454705 | 1.7621x |

`auto_scalarized_regroup` is about **1.23%** faster than `auto_scalarized` (`0.460313 / 0.454705 = 1.0123x`).

## Accuracy summary (final reported block)

### `auto_scalarized` vs old26 contribution
- max `|ΔJ|` = `0.000000e+00`
- rel error = `0.000000e+00`

### `auto_scalarized` mixed result (`J26a` vs ref)
- max `|ΔJ|` = `9.355379e-11`
- rel error = `6.978297e-09`

### `auto_scalarized_regroup` vs old26 contribution
- max `|ΔJ|` = `6.633366e-14`
- rel error = `3.426822e-07`

### `auto_scalarized_regroup` mixed result (`J26r` vs ref)
- max `|ΔJ|` = `9.355761e-11`
- rel error = `6.978582e-09`

### Cut stats (same final block)
- FP64 fraction among computed = `7.42753%`
- FP32 fraction among computed = `92.5725%`
- screened fraction among all = `58.2338%`

## cuobjdump resource usage (DDDD26 kernels)

From `cuobjdump --dump-resource-usage`:

| Kernel | REG | STACK | SHARED | LOCAL | CONST[0] |
|---|---:|---:|---:|---:|---:|
| `computeCoulombFockDDDD26_FP32` | 67 | 0 | 3212 | 0 | 656 |
| `computeCoulombFockDDDD26_FP32_auto_scalarized` | 58 | 0 | 3212 | 0 | 656 |
| `computeCoulombFockDDDD26_FP32_auto_scalarized_regroup` | 60 | 0 | 3212 | 0 | 656 |

## Conclusion
- `auto_scalarized` gives the main gain on GH200 (register drop: 67 -> 58, speedup ~1.74x).
- `regroup` keeps accuracy stable at mixed-result level and provides an additional small speedup (~1.2%) with a slight register increase (58 -> 60).
