# Exchange Mixed-Precision Production Benchmark

## Scope

This report compares the cleaned mixed-precision exchange production path
against the original FP64 implementation for `guanine-8` on one GH200 node
with four GPUs.

- Original FP64: `origin/gpu-dev-4`, commit `509b668b6`
- Mixed precision: local `mixed-precision-exchange-test` branch, based on
  `4b032494b` plus the production cleanup
- Coulomb mixed-precision threshold in both MP runs: `1e-6`
- Exchange thresholds tested: `1e-6` and `1e-5`
- SCF convergence threshold: `2e-4`

The MP production path contains all 54 exchange combinations. It directly
accumulates the selected FP64 and FP32 exchange kernels into the production K
matrix. It does not launch the original exchange kernels, reference kernels,
validation comparisons, or variant selectors.

The four selected resplit layouts are:

| Family | Production layout |
|---|---|
| PDDD | `K4_M23` |
| DDDD | `K16_OLD_RUNTIME` |
| DDDP | `K5_OLD` |
| DPDD | `K4_RS` |

The other 50 combinations retain their previously selected MP layouts.

## Host Timing

The reported host value is the steady-state `K compute` timer. For each run,
the last four Fock builds are averaged across all four GPUs. The final value is
the mean of three independent original runs or two independent MP runs.

In the original executable, `K compute` contains the original FP64 exchange
kernel launches and the final stream synchronization. In the production MP
executable, it contains host cut-layout construction, cut-displacement H2D
copies, GPU cut building, the selected FP64 and FP32 kernel launches, and the
final stream synchronization. Buffer allocation and initialization are done
before this timer.

The achieved host speedup is calculated as:

`speedup = original FP64 K compute / production MP K compute`

| Exchange threshold | Original FP64 (s) | Production MP (s) | MP run stdev (s) | Speedup |
|---|---:|---:|---:|---:|
| `1e-6` | 4.505167 | 2.668500 | 0.000088 | 1.6883x |
| `1e-5` | 4.505167 | 2.507219 | 0.000309 | 1.7969x |

The original run-to-run standard deviation was `0.001656 s`.

## Nsys Timing

For each profile, the average durations of all `computeExchangeFock*` kernel
rows in the Nsys `cuda_gpu_kern_sum` report are summed. This gives a
kernel-only aggregate that uses the same normalization for original and MP.
The GPU cut time is normalized to one Fock build and added separately.

| Exchange threshold | Original FP64 kernels (ms) | MP kernels (ms) | GPU cuts (ms) | Kernel-only speedup | Speedup with GPU cuts |
|---|---:|---:|---:|---:|---:|
| `1e-6` | 3831.339 | 2251.670 | 5.619 | 1.7016x | 1.6973x |
| `1e-5` | 3831.339 | 2114.848 | 5.640 | 1.8116x | 1.8068x |

Nsys confirmed that the production MP profiles contain no unsuffixed original
exchange launches. They contain 79 FP64 and 79 FP32 exchange kernel rows,
including the selected split subkernels.

The original Nsys run used `nid002892`; the MP Nsys runs used `nid002893`.
Both are GH200 nodes, but the Nsys comparison is therefore not a same-physical-
node measurement. The host comparison was measured on `nid002892` for both
executables.

## Accuracy

The original and MP calculations each converged independently in 10 SCF
iterations.

| Exchange threshold | Total energy (a.u.) | Signed difference vs original (a.u.) | Absolute difference (a.u.) | Relative difference |
|---|---:|---:|---:|---:|
| Original FP64 | -4624.0280059125 | 0 | 0 | 0 |
| `1e-6` | -4624.0280060107 | -9.8200e-8 | 9.8200e-8 | 2.1237e-11 |
| `1e-5` | -4624.0280060101 | -9.7601e-8 | 9.7601e-8 | 2.1107e-11 |

The two MP total energies differ from each other by `5.99e-10 a.u.`. These
values describe the final converged SCF energy. Per-combination Fock-matrix
errors are documented in the earlier validation reports and are not recomputed
by the cleaned production executable.

## Artifacts

- Original build: Slurm job `23215747`
- Original host and Nsys benchmark: Slurm job `23215765`
- MP clean build: Slurm job `23215542`
- MP post-cleanup validation: Slurm job `23215583`
- MP Nsys profile: Slurm job `23215584`
