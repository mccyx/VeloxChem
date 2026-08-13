# All-Winner Exchange Aggregate Benchmark

## Configuration

- Date: 2026-08-13
- Hardware: one GH200 node (`nid002892`)
- Input: `guanine-8-hf.inp`
- Exchange families included: all 54
- Host samples: three runs, four interactions per run (12 samples)
- Nsys samples: one profiled run, normalized to one interaction

The selected split variants were enabled together in one executable:

| family | selected variant |
|---|---|
| PDDD | `k4_m23` |
| DDDD | `old_k16_runtime` |
| DDDP | `old_k5` |
| DPDD | `rs_k4` |

The remaining 50 exchange families retain their old split. Thus,
`all-winner MP` means the complete 54-family MP workload with these four old
MP layouts replaced by their winners. It is not a sum of separate benchmark
runs.

## Baseline and Speedup Definitions

The performance baseline is always the **old-layout original FP64** kernel,
including for the four families whose MP layout was resplit. A selected/resplit
original FP64 variant is measured only to study the effect of layout changes
and to verify that resplitting preserves the original result. It is not the
denominator of the final achieved MP speedup.

For family `i`, define:

- `O_i`: old-layout original FP64 kernel time.
- `M_i`: selected MP kernel time. This is the winner MP time for PDDD, DDDD,
  DDDP, and DPDD, and the old-layout MP time for every other family.
- `C_i`: GPU cut-building time.
- `f32_i` and `f64_i`: fractions of non-screened work assigned to FP32 and
  FP64, with `f32_i + f64_i = 1`.

The per-family achieved speedups are:

`achieved kernel speedup_i = O_i / M_i`

`achieved speedup with cuts_i = O_i / (M_i + C_i)`

For example, the PDDD result is calculated as:

`306.969 ms old original FP64 / 142.314 ms selected k4_m23 MP = 2.1570x`

The 54-family aggregate speedups are ratios of summed times, not arithmetic
means of the 54 individual speedups:

`aggregate achieved kernel speedup = sum(O_i) / sum(M_i)`

`aggregate achieved speedup with cuts = sum(O_i) / sum(M_i + C_i)`

Consequently, the host aggregate is:

`4430.465 / 2606.795 = 1.6996x` for kernel only, and

`4430.465 / 2629.349 = 1.6850x` when GPU cut building is included.

The simplified per-family theoretical speedup assumes a 2:1 FP32-to-FP64
throughput ratio and equal cost per unit of non-screened work:

`theoretical kernel speedup_i = 1 / (f64_i + f32_i / 2)`

For the aggregate theoretical estimate, the fractions are weighted by each
family's old original FP64 time:

`f32_weighted = sum(O_i * f32_i) / sum(O_i)`

`f64_weighted = sum(O_i * f64_i) / sum(O_i)`

`aggregate theoretical kernel speedup = 1 / (f64_weighted + f32_weighted / 2)`

This gives `f32_weighted = 82.204%`, `f64_weighted = 17.796%`, and an
aggregate theoretical kernel speedup of `1.6979x`. The theoretical MP time is
`sum(O_i) / 1.6979`; the theoretical estimate with measured cuts is:

`sum(O_i) / (sum(O_i) / 1.6979 + sum(C_i)) = 1.6833x`

### Same-Layout MP Speedup

A second ratio isolates the mixed-precision benefit within the same selected
split. For each of the four winner families, let `R_i` be the original FP64
time of the selected/resplit layout. Its same-layout MP speedup is:

`same-layout MP speedup_i = R_i / M_i`

Unlike the achieved production speedup `O_i / M_i`, this ratio does not use the
old-layout original FP64 baseline. It should therefore be reported separately
and should not be used as the final old-to-new performance result.

| family | selected layout | FP32 | selected original FP64 (ms) | selected MP (ms) | theory kernel speedup | same-layout MP speedup |
|---|---|---:|---:|---:|---:|---:|
| PDDD | `k4_m23` | 81.053% | 215.731 | 142.314 | 1.6814x | 1.5159x |
| DDDD | `old_k16_runtime` | 77.438% | 160.357 | 113.276 | 1.6318x | 1.4156x |
| DDDP | `old_k5` | 79.234% | 132.045 | 87.252 | 1.6561x | 1.5134x |
| DPDD | `rs_k4` | 80.655% | 112.220 | 72.475 | 1.6758x | 1.5484x |

For these four winners together, times are summed before division:

`sum(R_i) / sum(M_i) = 620.353 / 415.317 = 1.4937x`

For a complete 54-family same-layout comparison, the other 50 families retain
the old layout, so their selected original FP64 time equals their old original
FP64 time. Replacing the four old original times by the four selected original
times gives:

`selected-layout original FP64 aggregate = 4273.055 ms`

`4273.055 / 2606.795 = 1.6392x` for kernel only, and

`4273.055 / 2629.349 = 1.6251x` when GPU cut building is included.

These values answer a different question from the primary `1.6996x` and
`1.6850x` results: they measure MP against FP64 after applying the same selected
layouts to both precision modes, whereas the primary results measure the full
improvement from the old original FP64 implementation to the selected MP
implementation.

In direct terms:

- **`1.4937x`**: selected original FP64 versus selected MP, summed over only
  the four winner families.
- **`1.6392x`**: selected-layout original FP64 versus selected MP for the full
  54-family workload after selecting the four winners. The other 50 families
  retain their old layouts in both precision modes.
- **`1.6996x`**: old-layout original FP64 versus selected MP for the full
  54-family workload. This is the final old-to-new production kernel speedup.

The corresponding full-workload values with GPU cut building included are
`1.6251x` for the same-layout comparison and `1.6850x` for the final old-to-new
comparison.

## Aggregate Results

### Host Timer

| metric | average (ms) | speedup vs old original FP64 |
|---|---:|---:|
| old original FP64 kernels | 4430.465 | 1.0000x |
| old-split MP kernels | 2715.698 | 1.6314x |
| all-winner MP kernels | 2606.795 | 1.6996x |
| old-split MP kernels + cut building | 2738.251 | 1.6180x |
| all-winner MP kernels + cut building | 2629.349 | 1.6850x |

The original-runtime-weighted precision mix and its theoretical estimates are
defined explicitly in the preceding section.

The average GPU cut-building time was `22.554 ms`. The all-winner MP kernel
standard deviation over the 12 interaction samples was `12.452 ms`.

The host kernel timer includes kernel launches, GPU execution, and the final
stream synchronization. It excludes buffer allocation and zeroing, cut
building, device/host copies, and accuracy validation. The `+ cut building`
rows add the measured GPU cut-building kernel time; allocation and zeroing
remain outside the timer.

### Nsys Kernel Timing

| metric | Nsys time (ms) | speedup vs old original FP64 |
|---|---:|---:|
| old original FP64 kernels | 4423.338 | 1.0000x |
| old-split MP kernels | 2709.919 | 1.6323x |
| all-winner MP kernels | 2599.218 | 1.7018x |

Nsys reports CUDA kernel execution only. Per-kernel average durations were
summed and normalized to one interaction, so repeated validation and comparison
launches do not inflate the aggregate.

| replaced family | old MP (ms) | winner MP (ms) | reduction (ms) |
|---|---:|---:|---:|
| PDDD | 207.334 | 142.471 | 64.863 |
| DDDD | 119.039 | 112.939 | 6.100 |
| DDDP | 103.574 | 87.279 | 16.295 |
| DPDD | 95.876 | 72.433 | 23.443 |

Together, the four winner layouts reduce the Nsys MP kernel time by
`110.701 ms`, or `4.085%` relative to the old-split MP aggregate.

## Per-Combination Results

The following host-timer table uses the selected layout for each family. The
four optimized families name their winner; `old` means that resplitting was not
selected. `Theory kernel` assumes FP32 throughput is twice FP64 throughput:

`theoretical speedup = 1 / (FP64 fraction + FP32 fraction / 2)`

The theory therefore models only the precision mix. It does not model changes
in instruction count, register pressure, occupancy, launch overhead, or load
balance caused by resplitting. Times and fractions are means over 12 interaction
samples. All achieved speedups use the old original FP64 time as their
numerator, as defined above.

| family | layout | FP32 | old original FP64 (ms) | selected MP (ms) | cuts (ms) | theory kernel | achieved kernel | achieved + cuts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DDDD | `old_k16_runtime` | 77.438% | 167.259 | 113.276 | 0.132 | 1.6318x | 1.4766x | 1.4748x |
| DDDP | `old_k5` | 79.234% | 156.868 | 87.252 | 0.137 | 1.6561x | 1.7979x | 1.7951x |
| DDDS | `old` | 82.063% | 19.192 | 12.321 | 0.139 | 1.6958x | 1.5578x | 1.5404x |
| DPDD | `rs_k4` | 80.655% | 146.667 | 72.475 | 0.133 | 1.6758x | 2.0237x | 2.0200x |
| DPDP | `old` | 80.654% | 100.564 | 60.167 | 0.140 | 1.6758x | 1.6714x | 1.6675x |
| DPDS | `old` | 83.760% | 33.645 | 19.132 | 0.141 | 1.7206x | 1.7586x | 1.7457x |
| DSDD | `old` | 82.544% | 18.191 | 11.792 | 0.139 | 1.7028x | 1.5427x | 1.5247x |
| DSDP | `old` | 82.947% | 34.398 | 19.029 | 0.140 | 1.7086x | 1.8077x | 1.7945x |
| DSDS | `old` | 79.812% | 11.130 | 7.417 | 0.135 | 1.6641x | 1.5006x | 1.4737x |
| PDDD | `k4_m23` | 81.053% | 306.969 | 142.314 | 0.499 | 1.6814x | 2.1570x | 2.1495x |
| PDDP | `old` | 82.479% | 184.680 | 113.692 | 0.506 | 1.7018x | 1.6244x | 1.6172x |
| PDDS | `old` | 85.196% | 65.599 | 36.639 | 0.490 | 1.7421x | 1.7904x | 1.7668x |
| PDPD | `old` | 84.235% | 93.556 | 58.052 | 0.546 | 1.7276x | 1.6116x | 1.5966x |
| PDPP | `old` | 83.633% | 165.919 | 91.584 | 0.563 | 1.7187x | 1.8117x | 1.8006x |
| PDPS | `old` | 85.831% | 49.447 | 30.656 | 0.551 | 1.7518x | 1.6130x | 1.5845x |
| PPDD | `old` | 80.677% | 177.166 | 112.549 | 0.497 | 1.6761x | 1.5741x | 1.5672x |
| PPDP | `old` | 80.811% | 332.102 | 186.523 | 0.519 | 1.6780x | 1.7805x | 1.7756x |
| PPDS | `old` | 83.699% | 104.736 | 61.176 | 0.505 | 1.7197x | 1.7120x | 1.6980x |
| PPPD | `old` | 82.761% | 155.556 | 93.072 | 0.555 | 1.7059x | 1.6714x | 1.6615x |
| PPPP | `old` | 80.997% | 267.919 | 155.408 | 0.581 | 1.6806x | 1.7240x | 1.7176x |
| PPPS | `old` | 83.606% | 99.083 | 62.139 | 0.840 | 1.7183x | 1.5945x | 1.5733x |
| PSDD | `old` | 82.531% | 58.632 | 32.211 | 0.490 | 1.7026x | 1.8202x | 1.7930x |
| PSDP | `old` | 83.025% | 96.811 | 60.080 | 0.508 | 1.7098x | 1.6114x | 1.5979x |
| PSDS | `old` | 79.922% | 45.748 | 28.570 | 0.499 | 1.6656x | 1.6012x | 1.5737x |
| PSPD | `old` | 84.729% | 45.211 | 29.548 | 0.544 | 1.7350x | 1.5301x | 1.5024x |
| PSPP | `old` | 83.433% | 101.126 | 58.936 | 0.572 | 1.7157x | 1.7158x | 1.6993x |
| PSPS | `old` | 80.063% | 51.864 | 35.027 | 0.552 | 1.6675x | 1.4807x | 1.4577x |
| SDDD | `old` | 82.770% | 35.699 | 23.174 | 0.305 | 1.7060x | 1.5405x | 1.5205x |
| SDDP | `old` | 84.131% | 64.008 | 35.661 | 0.311 | 1.7261x | 1.7949x | 1.7794x |
| SDDS | `old` | 86.598% | 18.211 | 11.582 | 0.300 | 1.7636x | 1.5724x | 1.5327x |
| SDPD | `old` | 85.588% | 54.581 | 37.096 | 0.659 | 1.7481x | 1.4713x | 1.4457x |
| SDPP | `old` | 84.209% | 94.207 | 58.318 | 0.701 | 1.7272x | 1.6154x | 1.5962x |
| SDPS | `old` | 86.426% | 38.251 | 22.883 | 0.677 | 1.7610x | 1.6716x | 1.6236x |
| SDSD | `old` | 87.239% | 9.594 | 6.384 | 0.215 | 1.7737x | 1.5029x | 1.4540x |
| SDSP | `old` | 85.908% | 19.977 | 12.425 | 0.218 | 1.7530x | 1.6078x | 1.5800x |
| SDSS | `old` | 85.906% | 9.781 | 5.825 | 0.212 | 1.7529x | 1.6790x | 1.6201x |
| SPDD | `old` | 81.989% | 58.272 | 34.007 | 0.305 | 1.6948x | 1.7135x | 1.6983x |
| SPDP | `old` | 82.128% | 100.607 | 59.785 | 0.318 | 1.6968x | 1.6828x | 1.6739x |
| SPDS | `old` | 84.896% | 39.562 | 23.790 | 0.308 | 1.7376x | 1.6630x | 1.6417x |
| SPPD | `old` | 84.719% | 96.088 | 59.873 | 0.680 | 1.7349x | 1.6049x | 1.5868x |
| SPPP | `old` | 82.201% | 207.176 | 121.611 | 0.720 | 1.6978x | 1.7036x | 1.6936x |
| SPPS | `old` | 84.865% | 99.415 | 57.224 | 0.912 | 1.7371x | 1.7373x | 1.7100x |
| SPSD | `old` | 85.959% | 18.853 | 11.839 | 0.216 | 1.7538x | 1.5925x | 1.5639x |
| SPSP | `old` | 83.482% | 51.288 | 28.757 | 0.217 | 1.7165x | 1.7835x | 1.7702x |
| SPSS | `old` | 83.947% | 22.229 | 13.740 | 0.228 | 1.7233x | 1.6179x | 1.5915x |
| SSDD | `old` | 82.016% | 17.840 | 11.860 | 0.301 | 1.6951x | 1.5042x | 1.4670x |
| SSDP | `old` | 82.442% | 40.679 | 24.340 | 0.310 | 1.7013x | 1.6713x | 1.6503x |
| SSDS | `old` | 79.409% | 19.631 | 12.728 | 0.302 | 1.6585x | 1.5424x | 1.5066x |
| SSPD | `old` | 84.405% | 38.899 | 24.558 | 0.666 | 1.7302x | 1.5839x | 1.5421x |
| SSPP | `old` | 82.478% | 90.815 | 56.720 | 0.710 | 1.7018x | 1.6011x | 1.5813x |
| SSPS | `old` | 79.288% | 52.847 | 33.103 | 0.690 | 1.6568x | 1.5964x | 1.5638x |
| SSSD | `old` | 86.731% | 8.324 | 5.853 | 0.210 | 1.7657x | 1.4222x | 1.3730x |
| SSSP | `old` | 84.702% | 22.657 | 13.718 | 0.221 | 1.7346x | 1.6516x | 1.6254x |
| SSSS | `old` | 79.601% | 10.936 | 7.048 | 0.590 | 1.6611x | 1.5516x | 1.4317x |

## Accuracy

All selected original variants matched their old original references, and all
selected MP variants passed the existing host-side validation against their
selected original variants. Across the 48 selected-winner validation samples:

| family | worst max absolute error | worst relative error |
|---|---:|---:|
| PDDD | 7.383008e-12 | 3.051338e-08 |
| DDDD | 3.274553e-12 | 7.597036e-09 |
| DDDP | 4.270059e-12 | 1.269227e-09 |
| DPDD | 8.455160e-12 | 2.489511e-09 |

The aggregate worst absolute error is `8.455160e-12`; the aggregate worst
relative error is `3.051338e-08`.

## Artifacts

- Host job: `23144562`
- Host summary: `build_logs/exchange_winners_host_23144562/SUMMARY.md`
- Per-interaction host data:
  `build_logs/exchange_winners_host_23144562/interaction_timings.csv`
- Per-combination source data:
  `build_logs/exchange_winners_host_23144562/per_combination_winners.csv`
- Nsys job: `23143889`
- Nsys summary: `nsys_results/exchange_winners_23143889/SUMMARY.md`
- Nsys report: `nsys_results/exchange_winners_23143889/all_winners.nsys-rep`
