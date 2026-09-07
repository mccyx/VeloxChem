# Exchange FP32-Density Benchmark

## Scope

This benchmark measures the exchange MP implementation after converting the
density matrix consumed by the FP32 exchange kernels from FP64 to FP32. It
compares three implementations:

- original FP64 exchange;
- the previous MP implementation with an FP64 density matrix;
- the new MP implementation with an FP32 density matrix.

All calculations used one GH200 node with four GPUs, HF/def2-SVP, an SCF
convergence threshold of `2e-4`, and a fixed Coulomb MP threshold of `1e-6`.
Exchange thresholds `1e-6`, `1e-5`, and `1e-4` were tested. The new runs are
Slurm array job `24258287`; all 30 tasks completed successfully.

The original and previous-MP values are reused from jobs `23216525`,
`23221585`, `23223787`, and `23224304`. Reusing them avoids an unnecessary
FP64 rerun. The new values are single-run measurements on equivalent GH200
nodes, so small old-MP-to-new differences should be interpreted with normal
cross-run timing variation in mind.

## Timing Definition

`Fock` is the host `Compute Fockmat` timer averaged over the last four Fock
builds. `K` is the `K compute` timer averaged over the same four builds and all
four GPUs. The new MP timed path includes host cut-layout construction,
displacement copies, GPU cut building, FP64 and FP32 exchange kernels, and the
final synchronization. Buffer allocation and initialization occur before the
timer.

The reported production speedup is:

`FP64/new = original FP64 time / new MP time`

The incremental density-conversion speedup reported below is:

`old/new = previous MP time with FP64 density / new MP time with FP32 density`

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

## FP64-Density MP Versus FP32-Density MP

This auxiliary table isolates the performance change from converting the
density input of the FP32 exchange kernels. It is retained for development
analysis and is not part of the final production comparison.

| System | K threshold | Previous MP Fock (s) | Fock old/new | Previous MP K (s) | K old/new |
|---|---:|---:|---:|---:|---:|
| guanine-4 | `1e-6` | 0.550250 | 1.0176x | 0.394500 | 1.0215x |
| guanine-4 | `1e-5` | 0.524750 | 1.0160x | 0.368875 | 1.0186x |
| guanine-4 | `1e-4` | 0.501500 | 0.9990x | 0.345875 | 0.9984x |
| guanine-8 | `1e-6` | 3.247500 | 1.0107x | 2.546437 | 1.0156x |
| guanine-8 | `1e-5` | 3.107500 | 1.0227x | 2.400875 | 1.0279x |
| guanine-8 | `1e-4` | 3.011500 | 1.0232x | 2.302188 | 1.0287x |
| guanine-12 | `1e-6` | 8.390750 | 1.0225x | 6.696000 | 1.0253x |
| guanine-12 | `1e-5` | 8.004750 | 1.0272x | 6.336750 | 1.0327x |
| guanine-12 | `1e-4` | 7.761250 | 1.0248x | 6.088750 | 1.0323x |
| guanine-sugar | `1e-6` | 20.744500 | 1.0271x | 16.495625 | 1.0385x |
| guanine-sugar | `1e-5` | 19.817500 | 1.0302x | 15.577000 | 1.0379x |
| guanine-sugar | `1e-4` | 19.318250 | 1.0348x | 14.988250 | 1.0392x |
| guanine-sugar-phosphate | `1e-6` | 62.605250 | 1.0195x | 45.137250 | 1.0267x |
| guanine-sugar-phosphate | `1e-5` | 59.974500 | 1.0153x | 42.551313 | 1.0236x |
| guanine-sugar-phosphate | `1e-4` | 58.235000 | 1.0183x | 40.818625 | 1.0274x |
| water-006 | `1e-6` | 0.222750 | 1.0068x | 0.156688 | 1.0212x |
| water-006 | `1e-5` | 0.215000 | 1.0226x | 0.147063 | 1.0284x |
| water-006 | `1e-4` | 0.209500 | 1.0096x | 0.143500 | 1.0278x |
| water-010 | `1e-6` | 3.868500 | 1.0295x | 3.196625 | 1.0367x |
| water-010 | `1e-5` | 3.733000 | 1.0267x | 3.063500 | 1.0307x |
| water-010 | `1e-4` | 3.646750 | 1.0265x | 2.970250 | 1.0330x |
| water-014 | `1e-6` | 24.596250 | 1.0182x | 19.385000 | 1.0263x |
| water-014 | `1e-5` | 24.029250 | 1.0216x | 18.781750 | 1.0294x |
| water-014 | `1e-4` | 23.552750 | 1.0231x | 18.298250 | 1.0300x |
| water-017 | `1e-6` | 67.116000 | 1.0234x | 49.170750 | 1.0335x |
| water-017 | `1e-5` | 65.612500 | 1.0283x | 47.580625 | 1.0400x |
| water-017 | `1e-4` | 64.448000 | 1.0295x | 46.550375 | 1.0448x |
| water-019 | `1e-6` | 122.503500 | 1.0149x | 86.306250 | 1.0299x |
| water-019 | `1e-5` | 121.152000 | 1.0396x | 84.749500 | 1.0583x |
| water-019 | `1e-4` | 118.391250 | 1.0239x | 82.251500 | 1.0409x |

## Conclusion

For the medium and large systems, FP32 density usually adds about `2-4%`
Fock speedup and about `2-6%` K speedup over the previous MP path. The small
guanine-4 `1e-4` difference is below one millisecond and is dominated by
single-run timing resolution.

Against original FP64, the larger guanine systems reach `1.79-1.82x` overall
ERI speedup at `1e-6`, `1.89x` at `1e-5`, and `1.94-1.95x` at `1e-4`. Water
systems remain less favorable, reaching `1.59-1.67x`, `1.64-1.71x`, and
`1.68-1.75x`, respectively, for water-010 through water-019.

All cases converged in the expected iteration count. The largest absolute
final-energy difference from original FP64 is `9.907e-7 a.u.`. As in the
earlier benchmark, this total-energy difference includes both the fixed
`1e-6` Coulomb MP contribution and the selected exchange threshold, and can
contain error cancellation.

The largest energy change between the previous MP calculation and the new
FP32-density calculation is `9.50e-9 a.u.` across all 30 cases. The density
conversion therefore adds no material final-energy error in this test set.
