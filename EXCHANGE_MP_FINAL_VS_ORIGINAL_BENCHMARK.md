# Final Exchange Mixed-Precision Benchmark

## Scope

This benchmark compares the final exchange mixed-precision implementation,
including the FP32 density matrix used by its FP32 kernels, against the
original FP64 implementation:

- original FP64 exchange;
- final MP exchange with selected resplit kernels and FP32 density.

All calculations used one GH200 node with four GPUs, HF/def2-SVP, and an SCF
convergence threshold of `2e-4`. The original executable used FP64 for both J
and K. The final MP executable used a fixed Coulomb threshold of `1e-6` and
exchange thresholds `1e-6`, `1e-5`, and `1e-4`. The final MP runs are Slurm
array job `24258287`; all 30 tasks completed successfully.

The original FP64 values are reused from jobs `23216525`, `23221585`,
`23223787`, and `23224304`. Reusing them avoids an unnecessary FP64 rerun.
The final MP values are single-run measurements on equivalent GH200 nodes.

- Original FP64 executable: `gpu-dev-4`, commit `509b668b6`.
- Final MP executable: the source state committed as `18a4ee704` on
  `mixed-precision-exchange-test`, built in
  `/cfs/klemming/scratch/y/yuch4126/exchange_density_fp32_benchmark/density_fp32_build`.

## Timing Definition

`Fock` is the host `Compute Fockmat` timer averaged over the last four Fock
builds. It covers the parallel J and K compute region. It excludes the earlier
`Prep. SphToCart`, `Prep. sortQD`, `Prep. preLinK`, and `Prep. Fockmat`
regions. In particular, construction of the FP32 density on the host occurs in
`Prep. SphToCart` and is not included in this timer.

`K` is the mean per-GPU `K compute` host timer, averaged over the same last
four Fock builds and all four GPUs. It starts after `K prep`, so it excludes
the primary K buffer allocations and the density, pair-data, and index H2D
copies. It includes K-buffer zeroing; host cut-layout construction; cut
displacement H2D copies; lazy allocation or growth of the reusable cut
workspace; GPU cut building; FP64 and FP32 exchange kernels; result D2H copies
and CPU Cartesian-to-spherical accumulation; and final resource release and
synchronization. It is therefore a production K-compute-region measurement,
not a kernel-only measurement or a complete end-to-end K preparation time.

The reported speedup is:

`FP64/new = original FP64 time / new MP time`

The original values used in that division are:

| System | Original FP64 Fock (s) | Original FP64 K (s) |
|---|---:|---:|
| guanine-4 | 0.847000 | 0.612625 |
| guanine-8 | 5.619250 | 4.304688 |
| guanine-12 | 14.703750 | 11.444250 |
| guanine-sugar | 36.281000 | 27.769000 |
| guanine-sugar-phosphate | 111.661250 | 76.289250 |
| water-006 | 0.311250 | 0.225625 |
| water-010 | 5.969250 | 4.769812 |
| water-014 | 38.994500 | 28.933250 |
| water-017 | 109.409500 | 73.917000 |
| water-019 | 196.782500 | 125.785500 |

The FP32 fraction is the unweighted fraction of all unscreened exchange tile
pairs assigned to FP32, aggregated over the full SCF run. Changing the density
storage precision does not change cut classification, so the fractions from
diagnostic job `23998744` apply to the new runs.

## Results

| System | K threshold | FP32 fraction | Final MP Fock (s) | Fock speedup | Final MP K (s) | K speedup | Energy abs. diff. (a.u.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| guanine-4 | `1e-6` | 75.67% | 0.540750 | 1.5663x | 0.386188 | 1.5863x | 1.650e-08 |
| guanine-4 | `1e-5` | 84.77% | 0.516500 | 1.6399x | 0.362125 | 1.6918x | 1.920e-08 |
| guanine-4 | `1e-4` | 92.00% | 0.502000 | 1.6873x | 0.346438 | 1.7684x | 3.960e-08 |
| guanine-8 | `1e-6` | 83.36% | 3.213250 | 1.7488x | 2.507250 | 1.7169x | 6.410e-08 |
| guanine-8 | `1e-5` | 90.41% | 3.038500 | 1.8494x | 2.335625 | 1.8431x | 5.630e-08 |
| guanine-8 | `1e-4` | 95.30% | 2.943250 | 1.9092x | 2.237938 | 1.9235x | 3.070e-08 |
| guanine-12 | `1e-6` | 85.20% | 8.206000 | 1.7918x | 6.531000 | 1.7523x | 1.019e-07 |
| guanine-12 | `1e-5` | 91.66% | 7.792750 | 1.8868x | 6.136250 | 1.8650x | 9.910e-08 |
| guanine-12 | `1e-4` | 96.00% | 7.573250 | 1.9415x | 5.898375 | 1.9402x | 7.520e-08 |
| guanine-sugar | `1e-6` | 84.56% | 20.197000 | 1.7964x | 15.883500 | 1.7483x | 1.217e-07 |
| guanine-sugar | `1e-5` | 91.16% | 19.236500 | 1.8860x | 15.007500 | 1.8503x | 1.326e-07 |
| guanine-sugar | `1e-4` | 95.71% | 18.668500 | 1.9434x | 14.422250 | 1.9254x | 1.908e-07 |
| guanine-sugar-phosphate | `1e-6` | 83.75% | 61.410750 | 1.8183x | 43.961562 | 1.7354x | 6.254e-07 |
| guanine-sugar-phosphate | `1e-5` | 90.44% | 59.069750 | 1.8903x | 41.572187 | 1.8351x | 6.305e-07 |
| guanine-sugar-phosphate | `1e-4` | 95.19% | 57.191250 | 1.9524x | 39.729500 | 1.9202x | 5.507e-07 |
| water-006 | `1e-6` | 80.39% | 0.221250 | 1.4068x | 0.153438 | 1.4705x | 3.230e-08 |
| water-006 | `1e-5` | 87.69% | 0.210250 | 1.4804x | 0.143000 | 1.5778x | 3.290e-08 |
| water-006 | `1e-4` | 93.20% | 0.207500 | 1.5000x | 0.139625 | 1.6159x | 8.450e-08 |
| water-010 | `1e-6` | 84.84% | 3.757750 | 1.5885x | 3.083500 | 1.5469x | 7.860e-08 |
| water-010 | `1e-5` | 90.89% | 3.636000 | 1.6417x | 2.972250 | 1.6048x | 8.060e-08 |
| water-010 | `1e-4` | 95.14% | 3.552500 | 1.6803x | 2.875500 | 1.6588x | 3.037e-07 |
| water-014 | `1e-6` | 86.64% | 24.156250 | 1.6143x | 18.888000 | 1.5318x | 5.644e-07 |
| water-014 | `1e-5` | 92.19% | 23.520750 | 1.6579x | 18.244750 | 1.5858x | 5.563e-07 |
| water-014 | `1e-4` | 95.95% | 23.021250 | 1.6938x | 17.766000 | 1.6286x | 5.761e-07 |
| water-017 | `1e-6` | 87.29% | 65.578750 | 1.6684x | 47.577000 | 1.5536x | 6.547e-07 |
| water-017 | `1e-5` | 92.66% | 63.804750 | 1.7148x | 45.751750 | 1.6156x | 6.802e-07 |
| water-017 | `1e-4` | 96.24% | 62.599000 | 1.7478x | 44.554000 | 1.6590x | 4.852e-07 |
| water-019 | `1e-6` | 87.57% | 120.707250 | 1.6302x | 83.802250 | 1.5010x | 9.907e-07 |
| water-019 | `1e-5` | 92.86% | 116.536250 | 1.6886x | 80.080750 | 1.5707x | 9.856e-07 |
| water-019 | `1e-4` | 96.36% | 115.632000 | 1.7018x | 79.020000 | 1.5918x | 6.762e-07 |

## Conclusion

Against original FP64, the larger guanine systems reach `1.79-1.82x`
`Compute Fockmat` speedup at `1e-6`, `1.89x` at `1e-5`, and `1.94-1.95x` at
`1e-4`. Water systems remain less favorable, reaching `1.59-1.67x`,
`1.64-1.71x`, and `1.68-1.75x`, respectively, for water-010 through
water-019. These values apply specifically to the timed compute region and
should not be labeled full-SCF or end-to-end ERI speedups.

All cases converged in the expected iteration count. The largest absolute
final-energy difference from original FP64 is `9.907e-7 a.u.`. As in the
earlier benchmark, this total-energy difference includes both the fixed
`1e-6` Coulomb MP contribution and the selected exchange threshold, and can
contain error cancellation.
