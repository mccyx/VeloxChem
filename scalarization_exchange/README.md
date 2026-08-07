# Exchange Scalarization Experiment

This experiment tested the Coulomb DDDD scalarization pipeline on the five
`PDDD_RS_FP32` exchange kernels. The scalarization tool was read from the full
Panor filesystem with `PYTHONDONTWRITEBYTECODE=1`; no files under
`/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-experiments`
were modified.

## Environment

- Slurm job: `22841881`
- Node: `nid002892` (GH200)
- Input: `guanine-8-hf.inp`
- Baseline commit: `481c6ba15`
- Scalar pass: `kernel_opt_pipeline.py --passes scalar`
- Scalarized functions: `computeExchangeFockPDDD{0..4}_RS_FP32`

The tracked source and executable were restored to the committed baseline at
the end of the experiment.

## Transformation

Each of the five kernels had the same transformation report:

- scalarized `r_l_f[3]` and `PQ_f[3]`
- hoisted 11 dynamic indexed accesses
- rewrote nine fixed-index accesses
- retained ambiguous accesses that failed the tool's safety checks

## CUDA resources

| Kernel | Baseline registers | Scalar registers | Baseline stack | Scalar stack | Shared/local |
|---|---:|---:|---:|---:|---|
| `PDDD0_RS_FP32` | 96 | 96 | 40 B | 16 B | unchanged |
| `PDDD1_RS_FP32` | 128 | 127 | 40 B | 16 B | unchanged |
| `PDDD2_RS_FP32` | 126 | 128 | 40 B | 16 B | unchanged |
| `PDDD3_RS_FP32` | 127 | 128 | 40 B | 16 B | unchanged |
| `PDDD4_RS_FP32` | 96 | 96 | 40 B | 16 B | unchanged |

Scalarization reduced stack-frame size but did not produce a consistent
register reduction.

## Host timing and accuracy

The existing old-versus-RS validation harness ran four SCF measurements with
all five scalarized kernels enabled.

| Measurement | Mean time |
|---|---:|
| Old MP, concurrent control | `205.850 ms` |
| Scalarized RS MP | `166.203 ms` |
| Previous unscalarized RS MP, job `22840458` | `165.224 ms` |

The absolute cross-run difference is a `0.6%` regression. Normalizing each RS
measurement by its concurrently measured old-MP control indicates an
approximately `0.4%` regression. Accuracy was unchanged: the sampled PDDD RS
MP comparison had maximum absolute error `3.899960e-12` and relative error
`1.014893e-08` against RS original.

## Nsight Compute

The profiling scripts under the Panor tools directory assume a
`computeCoulombFock` prefix. `profile_exchange_ncu.sh` adapts only the kernel
naming and output directory for exchange kernels.

An NCU `--set full` attempt stalled during kernel replay at 0% and produced no
report. It is not used as evidence. Matching `--set basic` reports were
successfully collected for the dominant `PDDD1_RS_FP32` kernel:

| Metric | Baseline | Scalarized |
|---|---:|---:|
| Duration | `126.53 ms` | `128.71 ms` |
| Registers/thread | 128 | 127 |
| Theoretical occupancy | 25.00% | 25.00% |
| Achieved occupancy | 24.58% | 24.62% |
| Compute throughput | 75.55% | 75.34% |
| Memory throughput | 34.21% | 26.96% |

NCU therefore measures a `1.7%` regression for the dominant scalarized
subkernel. The one-register reduction does not change its occupancy tier, and
the stack reduction does not translate into a performance gain.

### Focused instruction counters

A separate focused NCU run on the committed, unscalarized baseline collected
the requested instruction metrics under Slurm job `22842505`:

| Metric | Baseline count |
|---|---:|
| SM subpartition instructions executed | `77,079,721,462` |
| SM subpartition instructions issued | `77,080,281,447` |
| FP32 thread instructions | `1,877,901,008,841` |
| FP32 FADD thread instructions | `104,514,328,190` |
| FP32 FMUL thread instructions | `801,686,840,149` |
| FP32 FFMA thread instructions | `953,519,735,141` |

The `smsp__inst_executed` counters count instructions at the SM-subpartition
level, while the SASS operation counters count participating thread
instructions. Their magnitudes are therefore not directly comparable.

A matching run with only `PDDD1_RS_FP32` scalarized was collected under Slurm
job `22842576`:

| Metric | Baseline | Scalarized | Change |
|---|---:|---:|---:|
| SM subpartition instructions executed | `77,079,721,462` | `78,326,159,564` | `+1.617%` |
| SM subpartition instructions issued | `77,080,281,447` | `78,326,719,546` | `+1.617%` |
| FP32 thread instructions | `1,877,901,008,841` | `1,933,188,185,613` | `+2.944%` |
| FP32 FADD thread instructions | `104,514,328,190` | `104,514,328,190` | `0%` |
| FP32 FMUL thread instructions | `801,686,840,149` | `801,686,840,149` | `0%` |
| FP32 FFMA thread instructions | `953,519,735,141` | `953,519,735,141` | `0%` |

The arithmetic instruction counts are identical. The extra `1.246` billion
SM-subpartition instructions and `55.287` billion FP32 thread instructions are
therefore associated with non-arithmetic selection/data movement introduced by
the scalarized indexed-access expressions. This explains why the lower stack
frame and one-register reduction did not improve runtime.

## Conclusion

Do not select the current scalar transformation for production
`PDDD1_RS_FP32`, and do not enable all five transformed PDDD kernels as the
default group. Small-array scalarization is not automatically transferable
from Coulomb DDDD to this exchange layout.

If scalarization is investigated further, test kernels independently. A useful
next candidate would be an exchange kernel where scalarization crosses a
register occupancy boundary or removes actual local-memory traffic, rather
than merely reducing stack-frame metadata.

## Retained renamed variants

The five scalarized kernels are retained alongside their baselines in
`src/gpu/EriExchange.cu` and declared in `src/gpu/EriExchange.hpp`:

```text
computeExchangeFockPDDD0_RS_FP32_scalar
computeExchangeFockPDDD1_RS_FP32_scalar
computeExchangeFockPDDD2_RS_FP32_scalar
computeExchangeFockPDDD3_RS_FP32_scalar
computeExchangeFockPDDD4_RS_FP32_scalar
```

The original `computeExchangeFockPDDD{0..4}_RS_FP32` names are unchanged. A
full project build under Slurm job `22842638` confirmed that all ten symbols
coexist in `EriExchange.o`.

The host selects the FP32 PDDD RS family at runtime:

```bash
# Default; the explicit setting is optional.
VLX_PDDD_RS_FP32_VARIANT=baseline vlx guanine-8-hf.inp

VLX_PDDD_RS_FP32_VARIANT=scalar vlx guanine-8-hf.inp
```

Unset means `baseline`. Any value other than `baseline` or `scalar` is rejected
as a critical configuration error. The selector changes only the five RS FP32
launches; the FP64 kernels, buffers, timing boundary, and validation path are
the same.

Both modes were built and validated from the same executable under Slurm job
`22842821`:

| Mode | Mean PDDD RS MP | Maximum absolute error | Relative error |
|---|---:|---:|---:|
| Baseline | `165.890 ms` | `3.900233e-12` | `1.014964e-08` |
| Scalar | `166.275 ms` | `3.899960e-12` | `1.014893e-08` |

Scalar was `0.23%` slower by raw mean and approximately `0.34%` slower after
normalizing each mode by its concurrent old-MP control.
