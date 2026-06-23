# FP32 AST Scalarization for Non-DDDD Coulomb Kernels on GH200

Date: 2026-06-22

Repository: `VeloxChem.mixed-precision-2`

Input: `guanine-8.inp`

Device setup:
- GH200 node `nid002897`
- Accuracy and `nsys` run used the normal `env.sh` 4-GPU setup.
- The `ncu` verification used the existing one-GPU profiling wrapper, which forces `OMP_NUM_THREADS=1`, `OMP_PLACES={0}`, and `CUDA_VISIBLE_DEVICES=0`.

Generated/profiled kernels:
- `computeCoulombFockPPDD_FP32_auto_s_ast`
- `computeCoulombFockDDSD_FP32_auto_s_ast`
- `computeCoulombFockDDPP_FP32_auto_s_ast`
- `computeCoulombFockDDPD0_FP32_auto_s_ast` ... `computeCoulombFockDDPD9_FP32_auto_s_ast`

Key artifacts:
- `nsys_results/guanine8_fp32_scalar_compare.nsys-rep`
- `nsys_results/guanine8_fp32_scalar_compare_kernel_sum_cuda_gpu_kern_sum.csv`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32.summary.txt`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32_auto_s_ast.summary.txt`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32.source.csv`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32_auto_s_ast.source.csv`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32.warpstate.csv`
- `ncu_results/20260622_scalar_verify_ddpd7/guanine-8.ddpd7_fp32_auto_s_ast.warpstate.csv`

## Accuracy

The scalarized mixed paths were checked against the reference J result and compared with the original two-kernel mixed baseline. Here "mixed result" means the final mixed-precision J contribution after combining the FP64 path with the FP32 or scalarized FP32 path. It is not only the isolated scalarized kernel contribution.

All tested scalarized paths matched the baseline error level.

| Kernel family | Scalar mixed max abs error vs ref | Scalar mixed RMS error vs ref | Scalar relative error vs ref | Baseline behavior |
|---|---:|---:|---:|---|
| PPDD | `1.85e-10` to `2.68e-10` | `7.61e-12` to `7.94e-12` | `3.6e-9` to `5.2e-9` | Same error level as two-kernel baseline |
| DDSD | `2.69e-10` to `3.10e-10` | `2.21e-11` to `2.34e-11` | `7.9e-9` to `8.9e-9` | Same error level as two-kernel baseline |
| DDPP | `9.56e-10` to `1.86e-9` | `8.67e-11` to `9.38e-11` | `2.3e-10` to `4.5e-10` | Same error level as two-kernel baseline |
| DDPD | `1.32e-9` to `1.69e-9` | `1.01e-10` to `1.11e-10` | `1.3e-8` to `1.6e-8` | Same error level as two-kernel baseline |
| PDPD FP32 auto_s_ast | `1.38e-9` to `2.26e-9` | `7.43e-11` to `7.88e-11` | `1.6e-8` to `2.4e-8` | Scalar mixed vs two-kernel baseline was only `~2e-13` to `3e-13` max abs |

DDDD reference point from the same run:

| DDDD path | Max abs error vs ref | RMS error vs ref | Relative error vs ref |
|---|---:|---:|---:|
| DDDD two-kernel baseline | `9.37e-11` to `1.45e-10` | `9.0e-12` to `9.4e-12` | `7.0e-9` to `1.1e-8` |
| DDDD split26 v2 mixed | `9.35e-11` to `1.44e-10` | `9.0e-12` to `9.4e-12` | `7.0e-9` to `1.1e-8` |
| DDDD split26 v2 scalar mixed vs split26 baseline | typically `~1e-14`; individual replacement checks can be `~1e-20` | `~1e-15` or lower | `~1e-12` or lower |

Accuracy conclusion: scalarization did not introduce a visible numerical regression. The final mixed-result error is dominated by the mixed-precision approximation itself, not by scalarization.

## Nsight Systems Timing

Source: `nsys_results/guanine8_fp32_scalar_compare_kernel_sum_cuda_gpu_kern_sum.csv`

| Kernel | Baseline avg | Scalar avg | Speedup |
|---|---:|---:|---:|
| `computeCoulombFockPPDD_FP32` | 12.764 ms | 12.782 ms | 0.999x |
| `computeCoulombFockDDSD_FP32` | 6.498 ms | 6.053 ms | 1.073x |
| `computeCoulombFockDDPP_FP32` | 17.334 ms | 15.127 ms | 1.146x |
| `computeCoulombFockPDPD_FP32` | 53.820 ms | 54.972 ms | 0.979x |

For `PDPD`, the baseline had 8 instances and the scalarized variant had 4 instances in this run, so total time is not directly comparable; use average duration for the per-launch comparison.

DDPD split kernels:

| Kernel | Baseline avg | Scalar avg | Speedup |
|---|---:|---:|---:|
| `DDPD0_FP32` | 4.120 ms | 2.908 ms | 1.417x |
| `DDPD1_FP32` | 7.477 ms | 5.163 ms | 1.448x |
| `DDPD2_FP32` | 2.177 ms | 1.754 ms | 1.241x |
| `DDPD3_FP32` | 8.542 ms | 6.570 ms | 1.300x |
| `DDPD4_FP32` | 11.088 ms | 9.084 ms | 1.221x |
| `DDPD5_FP32` | 8.204 ms | 6.323 ms | 1.297x |
| `DDPD6_FP32` | 5.319 ms | 3.448 ms | 1.543x |
| `DDPD7_FP32` | 4.303 ms | 2.455 ms | 1.753x |
| `DDPD8_FP32` | 3.009 ms | 2.838 ms | 1.060x |
| `DDPD9_FP32` | 7.907 ms | 6.059 ms | 1.305x |

Aggregate DDPD split-kernel total:

| Path | Total |
|---|---:|
| Baseline DDPD0..9 | 248.6 ms |
| Scalarized DDPD0..9 | 186.4 ms |
| Aggregate speedup | 1.33x |

Timing conclusion: scalarization is most effective on already-split kernels. For the unsplit kernels, the effect is small or neutral. DDPD split kernels show a consistent improvement, while PPDD is neutral and PDPD is slightly slower per launch.

## NCU Verification: DDPD7

DDPD7 was selected because it had the largest `nsys` speedup among the tested DDPD split kernels.

Profiling command used the existing wrapper:

```bash
tools/profiling/run_ncu_gh200_one_gpu.sh single-fp32 computeCoulombFockDDPD7_FP32 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh single-fp32 computeCoulombFockDDPD7_FP32_auto_s_ast guanine-8.inp
```

Main metrics:

| Metric | Baseline | Scalar |
|---|---:|---:|
| Duration | 21.88 ms | 12.68 ms |
| Executed Instructions | 13.72B | 7.80B |
| Registers/thread | 51 | 48 |
| Theoretical occupancy | 50.0% | 62.5% |
| Achieved occupancy | 49.80% | 62.01% |
| Compute throughput | 77.67% | 76.82% |
| Memory throughput | 23.88% | 41.60% |

Warp state metrics:

| Warp state metric | Baseline | Scalar |
|---|---:|---:|
| Stall Not Selected | 4.11 | 5.05 |
| Stall Math Pipe Throttle | 2.15 | 1.00 |
| Stall Wait | 1.23 | 1.61 |
| Selected | 1.00 | 1.00 |
| Stall Dispatch Stall | 0.70 | 1.63 |
| Stall Long Scoreboard | 0.48 | 1.67 |
| Stall Barrier | 0.28 | 0.34 |
| Stall Short Scoreboard | 0.18 | 0.42 |

Important nuance: the scalarized kernel has higher cycles per executed instruction (`12.91` vs `10.26`) and higher not-selected / scoreboard-like stalls, but it executes far fewer total instructions. The wall-clock duration still drops substantially.

## Instruction Breakdown: DDPD7

High-level FP32 arithmetic counts from NCU:

| Metric | Baseline | Scalar |
|---|---:|---:|
| Fused FP32 instructions | 1.081B | 1.081B |
| Non-fused FP32 instructions | 3.013B | 2.985B |
| Fused FP64 instructions | 117,865 | 117,865 |
| Non-fused FP64 instructions | 33.466M | 33.466M |

SASS opcode aggregation from the NCU source CSV showed that `FFMA`, `FMUL`, `LDG`, and `LDS` were effectively unchanged. The reduction came mostly from predicate, set, integer, and address/control overhead.

Largest decreases by executed instruction count:

| Opcode | Baseline | Scalar | Decrease | Ratio |
|---|---:|---:|---:|---:|
| `ISETP` | 1.755B | 0.521B | 1.233B | 3.37x |
| `UISETP` | 0.877B | 0.00019B | 0.877B | 4651x |
| `PLOP3` | 0.877B | 0.00019B | 0.877B | 4651x |
| `P4` | 0.795B | 0 | 0.795B | removed |
| `P3` | 0.740B | 0 | 0.740B | removed |
| `P2` | 0.548B | 0 | 0.548B | removed |
| `P5` | 0.301B | 0 | 0.301B | removed |
| `P1` | 0.384B | 0.082B | 0.301B | 4.67x |
| `P2R` | 0.247B | 0 | 0.247B | removed |
| `P0` | 0.604B | 0.385B | 0.219B | 1.57x |
| `P6` | 0.192B | 0 | 0.192B | removed |
| `IMAD` | 0.719B | 0.588B | 0.131B | 1.22x |

Selected unchanged arithmetic/memory opcodes:

| Opcode | Baseline | Scalar | Observation |
|---|---:|---:|---|
| `FFMA` | 1.054B | 1.054B | unchanged |
| `FMUL` | 2.547B | 2.547B | unchanged |
| `LDG` | 0.478B | 0.478B | unchanged |
| `LDS` | 0.059B | 0.059B | unchanged |

Some opcodes increased:

| Opcode | Baseline | Scalar | Change |
|---|---:|---:|---:|
| `FSEL` | 0 | 0.548B | increased |
| `IADD3` | 0.195B | 0.265B | increased |
| `LEA` | 0.110B | 0.159B | increased |

Instruction conclusion: for DDPD7, scalarization does not speed up the kernel by increasing FFMA count. Instead, it removes a large amount of predicate/set/control/index overhead. The key reduction is total executed SASS instructions, especially `ISETP`, `UISETP`, `PLOP3`, predicate-register moves, and related integer/address operations.

## Interpretation

The measurements support the hypothesis from the earlier DDDD study:

1. The main scalarization benefit is lower executed instruction count.
2. Register and occupancy behavior is not guaranteed to move uniformly across kernels.
3. Split kernels benefit more clearly because the live set is smaller, making scalarization more likely to remove indexing and predicate overhead without creating excessive register pressure.

For DDPD7, the case is especially clean: instruction count drops by about 43%, registers/thread drops from 51 to 48, and achieved occupancy rises from 49.80% to 62.01%.

For unsplit kernels, scalarization alone is less reliable:

- PPDD is neutral.
- DDSD and DDPP improve modestly.
- PDPD is slightly slower per launch.

This suggests that for DDPP-like kernels, splitting first and scalarizing the split kernels may be a better next experiment than applying scalarization to the large unsplit kernel only.
