# DDDD34 Pass Ablation On RTX 4080 With guanine-4 (2026-04-13)

Source artifacts:

- log: `ablation_results_rtx4080_guanine4_2026-04-13.log`
- csv: `dddd34_pass_ablation_rtx4080_guanine4_2026-04-13.csv`
- binary inspected with `cuobjdump`: `build/python/veloxchem/veloxchemlib.so`
- workload: `vlx guanine-4.inp`
- device: `RTX 4080`

## Summary

This run does not reproduce the earlier GH200 ordering for `DDDD34`.

On this RTX 4080 + `guanine-4` workload:

- `merged` is already much faster than separate `DDDD3 + DDDD4`
- `auto_scalarized` is correct, but slower than `merged`
- `auto_scalarized_cse` is the fastest `DDDD34` variant in this run
- `auto_scalarize_only` and `auto_scalarize_rewrite` stay very close to `merged`
- `auto_scalarize_hoist` and full `auto_scalarized` move into the same slower class as manual `scalarized`

So the machine-dependent picture matters here: the source transformation that won on GH200 is not the winner on this 4080 run.

## Timing And Resource Table

| Variant | Passes | elapsed ms | REG | STACK | SHARED | LOCAL | max \|ΔJ\| vs separate 3+4 | max \|ΔJ\| vs J2_2kernels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DDDD34_FP32 merged` | none | 1.040681 | 64 | 40 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_scalarized` | manual | 1.095649 | 73 | 0 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_auto_scalarize_only` | `scalarize` | 1.042204 | 64 | 40 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_auto_scalarize_hoist` | `scalarize,hoist` | 1.094340 | 73 | 0 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_auto_scalarize_rewrite` | `scalarize,rewrite` | 1.039351 | 64 | 40 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_auto_scalarized` | `scalarize,hoist,rewrite` | 1.090932 | 73 | 0 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |
| `DDDD34_FP32_auto_scalarized_cse` | `scalarize,hoist,rewrite,cse` | 0.998506 | 64 | 0 | 2224 | 0 | 3.456251e-13 | 3.456251e-13 |

## Interpretation

- `merged` already captures most of the gain over separate `DDDD3+4`:
  - `1.800579 ms -> 1.040681 ms`
- on this machine, `scalarize_only` and `scalarize_rewrite` behave like `merged`:
  - timings stay near `1.04 ms`
  - resource usage also matches `merged`: `REG 64`, `STACK 40`
- `hoist` changes the kernel shape:
  - `REG 64 -> 73`
  - `STACK 40 -> 0`
  - but the timing here gets worse rather than better: `1.040681 ms -> 1.094340 ms`
- full `auto_scalarized` follows the same pattern as `auto_scalarize_hoist`
- `auto_scalarized_cse` is the best current result in this run:
  - `0.998506 ms`
  - `REG 64`
  - `STACK 0`

This is a useful counterexample to the earlier GH200 result. Here, lower stack and the exact local `cse` pass together appear to help more than the hoist-heavy variants that previously looked best.

## Correctness

All `DDDD34` variants in this run remain numerically consistent with the previous decomposed reference:

- `max |ΔJ| vs separate 3+4 = 3.456251e-13`
- `max |ΔJ| vs ref = 1.142137e-10`
- `max |ΔJ| vs J2_2kernels = 3.456251e-13`

For `DDDD34_FP32_auto_scalarized`, specifically:

- contribution vs separate `3+4`:
  - `max |ΔJ| = 3.456251e-13`
- mixed result vs `ref`:
  - `max |ΔJ| = 1.142137e-10`
- mixed result vs `J2_2kernels`:
  - `max |ΔJ| = 3.456251e-13`

## Bottom Line

For this `RTX 4080 + guanine-4` run:

- fastest `DDDD34` variant: `DDDD34_FP32_auto_scalarized_cse`
- `merged` is the next best baseline-quality option
- full `auto_scalarized` is correct, but not the best performer here

That means we should treat the current `DDDD34` pass ranking as architecture- and workload-sensitive rather than globally settled.
