# DDDD21 Variant Ablation On RTX 4080 With guanine-4 (2026-04-13)

Source artifacts:

- log: `ablation_results_rtx4080_guanine4_2026-04-13.log`
- csv: `dddd21_variant_ablation_rtx4080_guanine4_2026-04-13.csv`
- binary inspected with `cuobjdump`: `build/python/veloxchem/veloxchemlib.so`
- workload: `vlx guanine-4.inp`
- device: `RTX 4080`

## Summary

For `DDDD21` on this RTX 4080 + `guanine-4` run, the current best variant is:

- `DDDD21_FP32_auto_scalarized`

The gap is small, but `auto_scalarized` is slightly faster than the manual `scalarized` kernel and slightly faster than `auto_scalarized_cse`.

Unlike `DDDD34`, this run keeps the same broad conclusion as earlier experiments:

- plain `auto_scalarized` is the most promising automated `DDDD21` variant
- `cse` is correct, but does not improve on plain `auto_scalarized`
- `shared_staged` remains clearly slower because of its large shared-memory footprint

## Timing And Resource Table

| Variant | elapsed ms | REG | STACK | SHARED | LOCAL | max \|ΔJ\| vs old21 | max \|ΔJ\| vs J2_2kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DDDD21_FP32 baseline` | 0.726471 | 52 | 40 | 2188 | 0 | - | - |
| `DDDD21_FP32_scalarized` | 0.674168 | 60 | 0 | 2188 | 0 | 6.768435e-14 | 6.768436e-14 |
| `DDDD21_FP32_auto_scalarized` | 0.669146 | 60 | 0 | 2188 | 0 | 0.000000e+00 | 2.710505e-20 |
| `DDDD21_FP32_auto_scalarized_cse` | 0.672182 | 58 | 0 | 2188 | 0 | 2.904762e-14 | 2.904762e-14 |
| `DDDD21_FP32_shared_staged` | 0.938485 | 45 | 0 | 18572 | 0 | 6.768435e-14 | 6.768436e-14 |

## Interpretation

- `auto_scalarized` is the fastest current `DDDD21` variant in this run:
  - `0.669146 ms`
- manual `scalarized` is very close, but slightly slower:
  - `0.674168 ms`
- `auto_scalarized_cse` is also close, but still not the winner:
  - `0.672182 ms`
- baseline is clearly slower:
  - `0.726471 ms`
- `shared_staged` remains the worst option in this table:
  - `0.938485 ms`
  - `SHARED 18572` vs `2188` for the other variants

Resource usage is also informative here:

- baseline uses `REG 52`, but pays `STACK 40`
- manual `scalarized` and plain `auto_scalarized` both move to `STACK 0`
- current `cse` lowers `REG` slightly to `58`, but still does not beat plain `auto_scalarized`
- `shared_staged` lowers `REG` further to `45`, but shared-memory cost dominates

So on this machine, `DDDD21` again shows that lower register count alone is not enough. The best tradeoff is still the plain auto-generated scalarized form.

## Correctness

For `DDDD21_FP32_auto_scalarized`:

- contribution vs `old21`:
  - `max |ΔJ| = 0.000000e+00`
- mixed result vs `ref`:
  - `max |ΔJ| = 1.142644e-10`
- mixed result vs `J2_2kernels`:
  - `max |ΔJ| = 2.710505e-20`

Manual `scalarized` also remains correct, with only tiny numerical differences:

- contribution vs `old21`:
  - `max |ΔJ| = 6.768435e-14`

## Bottom Line

For this `RTX 4080 + guanine-4` run, the current ranking for `DDDD21` is:

1. `DDDD21_FP32_auto_scalarized`
2. `DDDD21_FP32_auto_scalarized_cse`
3. `DDDD21_FP32_scalarized`
4. `DDDD21_FP32 baseline`
5. `DDDD21_FP32_shared_staged`

The top three are close, but plain `auto_scalarized` is the best current choice.
