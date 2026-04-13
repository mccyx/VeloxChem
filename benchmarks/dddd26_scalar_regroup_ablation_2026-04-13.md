# DDDD26 Scalar/Regroup Ablation (2026-04-13)

## Scope

This note tracks the new auto-generated `DDDD26` FP32 variants integrated on 2026-04-13.

Integrated kernels:

- `computeCoulombFockDDDD26_FP32`
- `computeCoulombFockDDDD26_FP32_auto_scalarized`
- `computeCoulombFockDDDD26_FP32_auto_scalarized_regroup`

Generated source artifacts:

- [`dddd26_auto_scalarized.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd26_auto_scalarized.inc)
- [`dddd26_auto_scalarized_regroup.inc`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/generated/dddd26_auto_scalarized_regroup.inc)

Host experiment hook:

- [`FockDriverGPU.cu`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu)

## What Changed

### `auto_scalarized`

Bundled scalar pipeline:

- scalarize fixed-size arrays
- hoist indexed selections
- rewrite direct array-index uses

For `DDDD26_FP32`, this mainly affects:

- `r_k_f[3]`
- `r_l_f[3]`
- `PQ_f[3]`
- indexed accesses such as `r_l_f[c0]`, `r_k_f[d1]`, `PQ_f[a0]`, `PQ_f[c1]`

### `auto_scalarized_regroup`

Builds on `auto_scalarized` and additionally introduces low-risk power-chain regrouping in the `eri_ijkl_f` prefactors.

Current aliases introduced:

- `inv_S4_f_pow5`
- `S1_f_pow2`
- `S2_f_pow2`

## Intended Comparisons

Contribution-only comparisons:

- `auto26` vs `old26`
- `auto26_regroup` vs `old26`

Mixed-result comparisons:

- replace `DDDD26_FP32` contribution inside `J2_2kernels`
- compare new mixed result against:
  - `ref`
  - `J2_2kernels`

Timing comparisons:

- `DDDD26_FP32 baseline`
- `DDDD26_FP32_auto_scalarized`
- `DDDD26_FP32_auto_scalarized_regroup`

## Run Notes

Suggested workload:

```bash
vlx guanine-8.inp
```

Suggested profiling targets:

```bash
./profile_ncu_fp32.sh computeCoulombFockDDDD26_FP32
./profile_ncu_fp32.sh computeCoulombFockDDDD26_FP32_auto_scalarized
./profile_ncu_fp32.sh computeCoulombFockDDDD26_FP32_auto_scalarized_regroup
```

For walltime summary, use the existing host-side timing printout emitted by [`FockDriverGPU.cu`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/src/gpu/FockDriverGPU.cu).

## Status

Code integration is complete.

Runtime/accuracy numbers are still pending because CUDA compilation was not available in the current shell environment (`nvcc` missing here).
