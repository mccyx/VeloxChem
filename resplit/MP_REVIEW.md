# Resplit MP Review

## Scope

Reviewed artifacts:

- `EriExchange_resplit_alternatives_MP.cu`
- `EriExchange_resplit_alternatives_MP.hpp`
- `FockDriverGPU_resplit_comparison_helper.cu`
- `FockDriverGPU_resplit_comparison_blocks.cu`
- `re-split-EriExchange.cu`

The resplit artifacts intentionally cover only the eight logical combinations
whose split distribution changed: `SDDD`, `PPDD`, `PDDD`, `DSDD`, `DPDD`,
`DDDS`, `DDDP`, and `DDDD`.

## Static consistency results

| Check | Result |
|---|---:|
| Resplit original kernels | 50 |
| Resplit MP FP64 kernels | 50 |
| Resplit MP FP32 kernels | 50 |
| Total definitions | 150 |
| Total header declarations | 150 |
| Missing or extra declarations | 0 |
| Declaration/definition signature mismatches | 0 |
| Unique resplit kernels called by host blocks | 150 |
| Resplit kernels not called by host blocks | 0 |
| Host calls without declarations | 0 |
| Comparison blocks | 8 |

All 50 `_RS` original kernel bodies are byte-equivalent after whitespace and
the `_RS` name suffix are normalized to the corresponding kernels in the
supervisor-provided `re-split-EriExchange.cu`. This confirms that the comparison
uses the supplied resplit implementation rather than a different regeneration.

Each MP FP64 kernel consumes `n = [0, prec_cut)`. Each MP FP32 kernel consumes
`n = [prec_cut, screen_cut)`. There are 50 occurrences of each partitioning
pattern, matching the 50 resplit kernels.

## Compile results

The following compile checks passed on `nid002892` for `sm_90` with the project
CUDA flags:

1. All 150 resplit kernel definitions as a CUDA translation unit.
2. The supplied header declarations followed by all 150 definitions.
3. A complete temporary `FockDriverGPU` translation unit with the comparison
   helper and all eight blocks inserted at their matching validation markers.

The generated kernels emit unused-intermediate warnings for some high-angular-
momentum expressions. No compile errors were reported. The complete temporary
host unit emitted only a compiler note that detailed variable tracking was
disabled because the OpenMP function is very large.

## What the host comparison checks

For each changed logical combination, the host block allocates four isolated
result buffers and compares:

```text
old original vs resplit original
old MP       vs old original
resplit MP   vs resplit original
```

All four buffers are zeroed on the same stream. The displacement upload and cut
kernel execute on that stream, followed by a synchronization before timing.
Each measured path also ends with a stream synchronization. Device results are
copied to separate host vectors before accuracy checks.

The same precision and screening cuts are used for old and resplit MP kernels,
which is appropriate because resplitting changes algebraic kernel partitioning,
not the logical `(ik,m,n)` cut layout.

## Timing interpretation

The reported fields are kernel-only wall times:

- `old original`: current original FP64 kernel sequence
- `RS original`: supervisor resplit original FP64 kernel sequence
- `old MP`: current MP FP64 and FP32 kernel sequence
- `RS MP`: resplit MP FP64 and FP32 kernel sequence

Allocation, zeroing, host layout construction, displacement upload, and GPU cut
construction occur before these four timers. They are deliberately excluded,
so the comparison isolates the effect of resplitting. Cut construction is
identical for old and resplit MP and does not need to be counted when calculating
the before/after resplit ratio.

The helper reports:

```text
original RS speedup = old original / RS original
MP RS speedup       = old MP / RS MP
```

## Benchmarking caveats

The implementation is suitable for correctness testing and an initial timing
comparison, but each path is measured once per SCF evaluation in a fixed order:

```text
old original -> RS original -> old MP -> RS MP
```

This ordering can introduce warm-cache, clock, and first-launch bias. For a
publication-quality performance result, run several repetitions after one
untimed warm-up and alternate or randomize old/resplit order. Nsight Systems
should also be used to report summed kernel durations for each named family.

The provided `.cu`, `.hpp`, helper, and host blocks are fragments, not standalone
tracked replacements. Integration must retain the normal includes and
`namespace gpu` wrappers from the project files.

## Review conclusion

No structural, declaration, launch-coverage, compilation, or numerical blocker
was found. The artifacts were integrated into the tracked project sources,
built, linked, and run on `nid002892` in Slurm job `22840458`.

## Integrated runtime results

Four SCF samples were recorded for each changed family:

| Combination | Old original (ms) | RS original (ms) | Original RS ratio | Old MP (ms) | RS MP (ms) | MP RS ratio |
|---|---:|---:|---:|---:|---:|---:|
| `DDDD` | 167.415 | 195.247 | 0.8575x | 118.692 | 139.291 | 0.8521x |
| `DDDP` | 156.696 | 151.478 | 1.0344x | 103.349 | 97.668 | 1.0582x |
| `DDDS` | 19.308 | 27.841 | 0.6935x | 12.201 | 17.631 | 0.6920x |
| `DPDD` | 147.079 | 130.263 | 1.1291x | 95.416 | 83.564 | 1.1418x |
| `DSDD` | 18.327 | 28.103 | 0.6521x | 11.701 | 17.306 | 0.6761x |
| `PDDD` | 306.729 | 246.165 | 1.2460x | 205.494 | 165.224 | 1.2437x |
| `PPDD` | 177.179 | 268.855 | 0.6590x | 111.970 | 162.538 | 0.6889x |
| `SDDD` | 35.885 | 53.173 | 0.6749x | 23.004 | 33.666 | 0.6833x |

The runtime-weighted ratios over these eight families are:

```text
original RS ratio = 1028.617 / 1101.125 = 0.9342x
MP RS ratio       =  681.828 /  716.887 = 0.9511x
```

A ratio below one means the resplit version is slower. The resplit improves
`DDDP`, `DPDD`, and `PDDD`, but the regressions in the other five families make
the combined changed-family workload approximately 4.9% slower for MP.

Accuracy results:

```text
resplit original vs old original: worst absolute difference 2.255141e-17
resplit original vs old original: worst relative difference 7.281497e-15
resplit MP vs resplit original:   worst absolute difference 4.703154e-11
resplit MP vs resplit original:   worst relative difference 3.059177e-08
```

Nsight Systems independently gives inferred comparison ratios of `0.9328x`
for original FP64 and `0.9536x` for MP. The inference removes duplicate old
kernel executions from the normal workload and existing validation path.

Runtime artifacts:

- Host results: `build_logs/resplit_comparison_job22840458`
- Nsight report: `nsys_results/resplit_comparison_job22840458/guanine8_resplit.nsys-rep`
- Nsight kernel CSV: `nsys_results/resplit_comparison_job22840458/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
