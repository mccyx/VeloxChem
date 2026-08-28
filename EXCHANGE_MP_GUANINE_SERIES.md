# Exchange Mixed-Precision Guanine-Series Benchmark

## Configuration

- Date: 2026-08-14
- Hardware: one GH200 node with four GPUs per system
- Method: HF/def2-SVP
- SCF convergence threshold: `2e-4`
- Original FP64: `origin/gpu-dev-4`, commit `509b668b6`
- Mixed precision: local `mixed-precision-exchange-test` production cleanup
- Coulomb MP threshold: fixed at `1e-6`
- Exchange MP thresholds: `1e-6`, `1e-5`, and `1e-4`
- Guanine-series Slurm array job: `23216525`
- Guanine-sugar-phosphate Slurm array job: `23221585`

The geometries came from `guanine-series.zip`. The BLYP inputs in that archive
were not used because pure BLYP does not provide a meaningful exact-exchange
K benchmark. New HF inputs were generated with otherwise identical settings.

For guanine-4 through guanine-sugar, each system ran original FP64 followed by
all three MP thresholds on the same physical node. The larger
guanine-sugar-phosphate cases ran as separate array tasks on equivalent GH200
nodes. This is a single-run threshold screen. Important points should be
repeated before treating small timing differences as final.

## Overall ERI Timing Results

The primary performance metric is the host `Compute Fockmat` timer, averaged
over the last four Fock builds. It includes the shared Boys/GTO preparation,
Coulomb J preparation and kernels, Exchange K preparation and kernels, MP cut
construction, data transfers inside the timed region, and synchronization.
It therefore measures the complete ERI Fock-build path rather than only the
Exchange kernels.

`overall ERI speedup = original FP64 Compute Fockmat / MP Compute Fockmat`

| System | K threshold | Original FP64 (s) | MP ERI (s) | Overall ERI speedup | SCF time (s) | Iterations |
|---|---:|---:|---:|---:|---:|---:|
| guanine-4 | `1e-6` | 0.847000 | 0.550250 | 1.5393x | 7.42 | 11 |
| guanine-4 | `1e-5` | 0.847000 | 0.524750 | 1.6141x | 7.05 | 11 |
| guanine-4 | `1e-4` | 0.847000 | 0.501500 | 1.6889x | 6.84 | 11 |
| guanine-8 | `1e-6` | 5.619250 | 3.247500 | 1.7303x | 44.21 | 12 |
| guanine-8 | `1e-5` | 5.619250 | 3.107500 | 1.8083x | 42.24 | 12 |
| guanine-8 | `1e-4` | 5.619250 | 3.011500 | 1.8659x | 41.03 | 12 |
| guanine-12 | `1e-6` | 14.703750 | 8.390750 | 1.7524x | 110.88 | 12 |
| guanine-12 | `1e-5` | 14.703750 | 8.004750 | 1.8369x | 106.20 | 12 |
| guanine-12 | `1e-4` | 14.703750 | 7.761250 | 1.8945x | 103.38 | 12 |
| guanine-sugar | `1e-6` | 36.281000 | 20.744500 | 1.7489x | 273.91 | 12 |
| guanine-sugar | `1e-5` | 36.281000 | 19.817500 | 1.8308x | 262.57 | 12 |
| guanine-sugar | `1e-4` | 36.281000 | 19.318250 | 1.8781x | 256.35 | 12 |
| guanine-sugar-phosphate | `1e-6` | 111.661250 | 62.605250 | 1.7836x | 915.94 | 13 |
| guanine-sugar-phosphate | `1e-5` | 111.661250 | 59.974500 | 1.8618x | 882.45 | 13 |
| guanine-sugar-phosphate | `1e-4` | 111.661250 | 58.235000 | 1.9174x | 859.98 | 13 |

The original FP64 SCF times were `10.98`, `72.73`, `187.31`, `462.22`, and
`1556.47 s`, in increasing system-size order.

## Exchange K Timing Details

`K compute` is averaged over the last four Fock builds and all four GPUs. For
the MP executable it includes host cut-layout construction, displacement H2D
copies, GPU cut building, FP64 and FP32 exchange kernel launches, and final
stream synchronization. The speedup is:

`original FP64 K compute / production MP K compute`

| System | K threshold | Original FP64 (s) | Production MP (s) | Speedup | SCF time (s) | Iterations |
|---|---:|---:|---:|---:|---:|---:|
| guanine-4 | `1e-6` | 0.612625 | 0.394500 | 1.5529x | 7.42 | 11 |
| guanine-4 | `1e-5` | 0.612625 | 0.368875 | 1.6608x | 7.05 | 11 |
| guanine-4 | `1e-4` | 0.612625 | 0.345875 | 1.7712x | 6.84 | 11 |
| guanine-8 | `1e-6` | 4.304688 | 2.546437 | 1.6905x | 44.21 | 12 |
| guanine-8 | `1e-5` | 4.304688 | 2.400875 | 1.7930x | 42.24 | 12 |
| guanine-8 | `1e-4` | 4.304688 | 2.302188 | 1.8698x | 41.03 | 12 |
| guanine-12 | `1e-6` | 11.444250 | 6.696000 | 1.7091x | 110.88 | 12 |
| guanine-12 | `1e-5` | 11.444250 | 6.336750 | 1.8060x | 106.20 | 12 |
| guanine-12 | `1e-4` | 11.444250 | 6.088750 | 1.8796x | 103.38 | 12 |
| guanine-sugar | `1e-6` | 27.769000 | 16.495625 | 1.6834x | 273.91 | 12 |
| guanine-sugar | `1e-5` | 27.769000 | 15.577000 | 1.7827x | 262.57 | 12 |
| guanine-sugar | `1e-4` | 27.769000 | 14.988250 | 1.8527x | 256.35 | 12 |
| guanine-sugar-phosphate | `1e-6` | 76.289250 | 45.137250 | 1.6902x | 915.94 | 13 |
| guanine-sugar-phosphate | `1e-5` | 76.289250 | 42.551313 | 1.7929x | 882.45 | 13 |
| guanine-sugar-phosphate | `1e-4` | 76.289250 | 40.818625 | 1.8690x | 859.98 | 13 |

This K-only table is retained as an auxiliary breakdown. The overall ERI
table above is the primary production result because shared MP preparation and
Coulomb MP are part of the intended workflow.

## Energy Results

| System | Original energy (a.u.) | K threshold | MP energy (a.u.) | Absolute difference (a.u.) |
|---|---:|---:|---:|---:|
| guanine-4 | -2312.0195575770 | `1e-6` | -2312.0195575606 | 1.640e-8 |
| guanine-4 | -2312.0195575770 | `1e-5` | -2312.0195575582 | 1.880e-8 |
| guanine-4 | -2312.0195575770 | `1e-4` | -2312.0195575382 | 3.880e-8 |
| guanine-8 | -4624.0003661977 | `1e-6` | -4624.0003662602 | 6.250e-8 |
| guanine-8 | -4624.0003661977 | `1e-5` | -4624.0003662558 | 5.810e-8 |
| guanine-8 | -4624.0003661977 | `1e-4` | -4624.0003662295 | 3.180e-8 |
| guanine-12 | -6935.9792761075 | `1e-6` | -6935.9792762095 | 1.020e-7 |
| guanine-12 | -6935.9792761075 | `1e-5` | -6935.9792762049 | 9.740e-8 |
| guanine-12 | -6935.9792761075 | `1e-4` | -6935.9792761837 | 7.620e-8 |
| guanine-sugar | -11486.2211885228 | `1e-6` | -11486.2211883986 | 1.242e-7 |
| guanine-sugar | -11486.2211885228 | `1e-5` | -11486.2211883919 | 1.309e-7 |
| guanine-sugar | -11486.2211885228 | `1e-4` | -11486.2211883286 | 1.942e-7 |
| guanine-sugar-phosphate | -30517.3986504645 | `1e-6` | -30517.3986510953 | 6.308e-7 |
| guanine-sugar-phosphate | -30517.3986504645 | `1e-5` | -30517.3986510927 | 6.282e-7 |
| guanine-sugar-phosphate | -30517.3986504645 | `1e-4` | -30517.3986510147 | 5.502e-7 |

Every threshold converged in the same number of SCF iterations as its
original FP64 calculation. The final SCF energy is not a monotonic error
measure: cancellation can make a wider threshold appear more accurate for a
particular system. In addition, these energy differences include the effect of
the fixed `1e-6` Coulomb MP threshold as well as the exchange MP threshold.
They therefore do not isolate exchange error.

## Initial Conclusion

For the four larger systems, the overall ERI speedup is `1.73-1.78x` at
`1e-6`, `1.81-1.86x` at `1e-5`, and `1.87-1.92x` at `1e-4`. The largest
observed final-energy difference is `6.31e-7 a.u.` for
guanine-sugar-phosphate, and every threshold converges in the same number of
SCF iterations as original FP64.

The current results make `1e-4` a viable performance candidate for further
system-level testing. They do not prove that it is intrinsically more accurate
than tighter thresholds: its smaller GSP energy difference is consistent with
error cancellation in the converged total energy.
