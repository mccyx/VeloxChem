# Mixed-Precision Exchange: Current Results

## Executive summary

The original-layout mixed-precision (MP) implementation covers all 54 exchange
combinations. On one GH200 for `guanine-8-hf.inp` at an MP threshold of `1e-6`,
the aggregate exchange-kernel speedup is `1.6316x` with Nsight Systems and
`1.6318x` with the host timer. Including the complete MP cut-construction
overhead (host-side setup, data transfer, GPU cut kernel, and synchronization)
gives `1.6176x`. All 216 validation records (54 combinations x 4 SCF
iterations) passed; the largest absolute MP-reference difference is
`1.156485e-10` for `SSPS`.

The resplit (RS) implementation changes eight kernel families. RS improves both
FP64 and MP performance for `PDDD`, `DPDD`, and `DDDP`, but regresses the other
five. Combining RS for those three families with the old layout for the other
five is estimated to reduce the complete MP exchange-kernel time by another
`2.2%`.

## Configuration and timing definitions

- Hardware: one NVIDIA GH200 (`nid002892`).
- Input: `guanine-8-hf.inp`; MP threshold J/K: `1e-6`.
- Old-layout timing: Slurm job `22521933`, commit `d4e455b01`, four SCF iterations.
- Old-layout Nsys run: Slurm job `22520603`.
- RS comparison: Slurm job `22840458`, commit `4b032494b`, four SCF iterations.

The host `cuts` interval includes host cut-layout construction, reusable device
workspace growth when needed, the `displ_cuts` H2D copy,
`build_exchange_cuts_kernel`, and stream synchronization. MP/reference output
buffer allocation and zeroing occur before this timer and are excluded. The MP
kernel interval contains the FP64 and FP32 exchange kernels and synchronization.

`displ_cuts` is an offset array describing where each shell-pair/tile group's
cut entries begin in the flattened device cut arrays. The GPU kernel uses these
offsets to locate the entries belonging to each group.

Nsys reports the exchange kernels alone. It cannot measure the complete host
cut-construction interval, so the host-timed `1.6176x` is used for the
cut-inclusive comparison.

## Original-layout aggregate result

| Measurement | Original FP64 | MP | Speedup |
|---|---:|---:|---:|
| Nsys, exchange kernels only | 4411.386 ms | 2703.735 ms | `1.6316x` |
| Host timer, exchange kernels only | 4414.151 ms | 2705.070 ms | `1.6318x` |
| Host timer, MP kernels + cuts | 4414.151 ms | 2728.902 ms | `1.6176x` |

All times in this table are means per SCF iteration. Nsys recorded totals over
four iterations; the displayed Nsys times are those totals divided by four.

The measured host cut cost is `23.831 ms` per SCF iteration, or about `0.88%`
of MP kernel time. Speedups are ratios of aggregate times, rather than the
arithmetic mean of per-combination speedups.

## Accuracy summary

- 54 combinations x 4 SCF iterations = 216 validation records.
- Worst absolute difference: `1.156485e-10` (`SSPS`).
- No missing validation blocks, CUDA errors, or execution failures.
- Validation copies, comparisons, reference-buffer work, and log output are not
  included in either production speedup.

## Original-layout per-combination results

The `Actual kernel` and `Actual + cuts` columns below are host-timer results.
The current Nsys report supplies the aggregate kernel-only cross-check above,
not a separate 54-row per-combination table.

The theoretical model assumes FP32 throughput is twice FP64 throughput and
equal cost per unit of non-screened work:

`theoretical speedup = 1 / (f64 + f32 / 2)`

The runtime-weighted fractions are approximately `82.204%` FP32 and `17.796%`
FP64. They predict `1.6979x` kernel-only, or `1.6824x` after adding the measured
host cut cost.

Fractions and timings in the table are means over four SCF iterations.

| Combination | FP32 | FP64 | Theory kernel | Theory + measured cuts | Actual kernel | Actual + cuts |
|---|---:|---:|---:|---:|---:|---:|
| DDDD | 77.438% | 22.562% | 1.6318x | 1.6298x | 1.4034x | 1.4019x |
| DDDP | 79.234% | 20.766% | 1.6561x | 1.6538x | 1.5084x | 1.5065x |
| DDDS | 82.063% | 17.937% | 1.6958x | 1.6758x | 1.5593x | 1.5423x |
| DPDD | 80.655% | 19.345% | 1.6758x | 1.6733x | 1.5293x | 1.5272x |
| DPDP | 80.654% | 19.346% | 1.6758x | 1.6719x | 1.6669x | 1.6630x |
| DPDS | 83.760% | 16.240% | 1.7206x | 1.7084x | 1.7606x | 1.7478x |
| DSDD | 82.544% | 17.456% | 1.7028x | 1.6814x | 1.5407x | 1.5232x |
| DSDP | 82.947% | 17.053% | 1.7086x | 1.6967x | 1.8126x | 1.7992x |
| DSDS | 79.812% | 20.188% | 1.6641x | 1.6315x | 1.5030x | 1.4764x |
| PDDD | 81.053% | 18.947% | 1.6814x | 1.6631x | 1.4728x | 1.4588x |
| PDDP | 82.479% | 17.521% | 1.7018x | 1.6942x | 1.6251x | 1.6181x |
| PDDS | 85.196% | 14.804% | 1.7421x | 1.7202x | 1.7780x | 1.7551x |
| PDPD | 84.235% | 15.765% | 1.7276x | 1.7109x | 1.6146x | 1.5999x |
| PDPP | 83.633% | 16.367% | 1.7187x | 1.7086x | 1.8080x | 1.7968x |
| PDPS | 85.831% | 14.169% | 1.7518x | 1.7186x | 1.6179x | 1.5896x |
| PPDD | 80.677% | 19.323% | 1.6761x | 1.6684x | 1.5811x | 1.5742x |
| PPDP | 80.811% | 19.189% | 1.6780x | 1.6737x | 1.7822x | 1.7773x |
| PPDS | 83.699% | 16.301% | 1.7197x | 1.7060x | 1.7086x | 1.6951x |
| PPPD | 82.761% | 17.239% | 1.7059x | 1.6959x | 1.6725x | 1.6629x |
| PPPP | 80.997% | 19.003% | 1.6806x | 1.6744x | 1.7256x | 1.7191x |
| PPPS | 83.606% | 16.394% | 1.7183x | 1.6943x | 1.5998x | 1.5790x |
| PSDD | 82.531% | 17.469% | 1.7026x | 1.6793x | 1.8206x | 1.7940x |
| PSDP | 83.025% | 16.975% | 1.7098x | 1.6947x | 1.6116x | 1.5982x |
| PSDS | 79.922% | 20.078% | 1.6656x | 1.6371x | 1.6011x | 1.5747x |
| PSPD | 84.729% | 15.271% | 1.7350x | 1.6984x | 1.5260x | 1.4976x |
| PSPP | 83.433% | 16.567% | 1.7157x | 1.6995x | 1.7071x | 1.6910x |
| PSPS | 80.063% | 19.937% | 1.6675x | 1.6394x | 1.4848x | 1.4624x |
| SDDD | 82.770% | 17.230% | 1.7060x | 1.6813x | 1.5432x | 1.5229x |
| SDDP | 84.131% | 15.869% | 1.7261x | 1.7110x | 1.8008x | 1.7844x |
| SDDS | 86.598% | 13.402% | 1.7636x | 1.7114x | 1.5666x | 1.5252x |
| SDPD | 85.588% | 14.412% | 1.7481x | 1.7142x | 1.4708x | 1.4468x |
| SDPP | 84.209% | 15.791% | 1.7272x | 1.7066x | 1.6276x | 1.6093x |
| SDPS | 86.426% | 13.574% | 1.7610x | 1.7101x | 1.6655x | 1.6199x |
| SDSD | 87.239% | 12.761% | 1.7737x | 1.7046x | 1.5030x | 1.4531x |
| SDSP | 85.908% | 14.092% | 1.7530x | 1.7202x | 1.6301x | 1.6017x |
| SDSS | 85.906% | 14.094% | 1.7529x | 1.6887x | 1.6755x | 1.6168x |
| SPDD | 81.989% | 18.011% | 1.6948x | 1.6794x | 1.7275x | 1.7116x |
| SPDP | 82.128% | 17.872% | 1.6968x | 1.6877x | 1.6960x | 1.6869x |
| SPDS | 84.896% | 15.104% | 1.7376x | 1.7144x | 1.6655x | 1.6442x |
| SPPD | 84.719% | 15.281% | 1.7349x | 1.7149x | 1.6179x | 1.6004x |
| SPPP | 82.201% | 17.799% | 1.6978x | 1.6883x | 1.7053x | 1.6957x |
| SPPS | 84.865% | 15.135% | 1.7371x | 1.7126x | 1.7305x | 1.7062x |
| SPSD | 85.959% | 14.041% | 1.7538x | 1.7192x | 1.5899x | 1.5615x |
| SPSP | 83.482% | 16.518% | 1.7165x | 1.7042x | 1.7815x | 1.7683x |
| SPSS | 83.947% | 16.053% | 1.7233x | 1.6941x | 1.6147x | 1.5890x |
| SSDD | 82.016% | 17.984% | 1.6951x | 1.5675x | 1.4997x | 1.3989x |
| SSDP | 82.442% | 17.558% | 1.7013x | 1.6794x | 1.6649x | 1.6440x |
| SSDS | 79.409% | 20.591% | 1.6585x | 1.6171x | 1.5379x | 1.5022x |
| SSPD | 84.405% | 15.595% | 1.7302x | 1.6829x | 1.5789x | 1.5394x |
| SSPP | 82.478% | 17.522% | 1.7018x | 1.6806x | 1.6040x | 1.5852x |
| SSPS | 79.288% | 20.712% | 1.6568x | 1.6237x | 1.5979x | 1.5670x |
| SSSD | 86.731% | 13.269% | 1.7657x | 1.6906x | 1.4363x | 1.3862x |
| SSSP | 84.702% | 15.298% | 1.7346x | 1.7073x | 1.6561x | 1.6311x |
| SSSS | 79.601% | 20.399% | 1.6611x | 1.5551x | 1.5584x | 1.4647x |

Measured kernel-only speedups range from `1.4034x` (`DDDD`) to `1.8206x`
(`PSDD`). With cuts, they range from `1.3862x` (`SSSD`) to `1.7992x` (`DSDP`).
Some measured values exceed the simple theoretical estimate because the model
does not represent differences in instruction mix, memory traffic, occupancy,
kernel structure, or launch overhead.

## Resplit results

Ratios below are `old / RS`, so a value greater than one means RS is faster.
The table contains the eight families changed by resplitting and is the relevant
part of the RS comparison.

| Family | Old FP64 (ms) | RS FP64 (ms) | FP64 ratio | Old MP (ms) | RS MP (ms) | MP ratio | Choice |
|---|---:|---:|---:|---:|---:|---:|---|
| `DDDD` | 167.415 | 195.247 | 0.8575x | 118.692 | 139.291 | 0.8521x | Old |
| `DDDP` | 156.696 | 151.478 | 1.0344x | 103.349 | 97.668 | 1.0582x | RS |
| `DDDS` | 19.308 | 27.841 | 0.6935x | 12.201 | 17.631 | 0.6920x | Old |
| `DPDD` | 147.079 | 130.263 | 1.1291x | 95.416 | 83.564 | 1.1418x | RS |
| `DSDD` | 18.327 | 28.103 | 0.6521x | 11.701 | 17.306 | 0.6761x | Old |
| `PDDD` | 306.729 | 246.165 | 1.2460x | 205.494 | 165.224 | 1.2437x | RS |
| `PPDD` | 177.179 | 268.855 | 0.6590x | 111.970 | 162.538 | 0.6889x | Old |
| `SDDD` | 35.885 | 53.173 | 0.6749x | 23.004 | 33.666 | 0.6833x | Old |

Across all eight changed families, always using RS gives `0.9342x` for FP64
and `0.9511x` for MP, so it is about `4.9%` slower for MP in aggregate. Nsys
independently gives `0.9328x` for FP64 and `0.9536x` for MP. The useful result
is therefore selective: keep RS for `PDDD`, `DPDD`, and `DDDP`, and keep the old
layout for the other five.

RS-original and old-original agree to `2.255141e-17` maximum absolute
difference. Against RS original, RS MP has a worst absolute error of
`8.448223e-12` (`DPDD`) and a worst relative error of `3.051688e-08` (`PDDD`),
with no CUDA or runtime failures. The two maxima occur in different families
because relative error normalizes by the largest reference magnitude.

The estimated hybrid changes the complete 54-family MP kernel total from about
`2705 ms` to `2645 ms`, an additional reduction of about `2.2%`. This estimate
combines two runs and must be remeasured after implementing the hybrid selector.

## Optimization priority

The ranking combines current selected-layout MP runtime with optimistic
headroom `MP time - FP64 time / 2`. Headroom is a prioritization metric, not a
prediction: it ignores the mandatory FP64 fraction and other hardware effects.

| Priority | Family | Layout | Selected MP (ms) | MP time share | Achieved MP speedup | Approx. headroom (ms) |
|---:|---|---|---:|---:|---:|---:|
| 1 | `PDDD` | RS | 165.2 | 6.2% | 1.49x | 42.1 |
| 2 | `DDDD` | Old | 119.1 | 4.5% | 1.40x | 35.5 |
| 3 | `PPDD` | Old | 112.0 | 4.2% | 1.58x | 23.4 |
| 4 | `DDDP` | RS | 97.7 | 3.7% | 1.55x | 21.9 |
| 5 | `PDDP` | Old | 113.6 | 4.3% | 1.63x | 21.3 |
| 6 | `PPPP` | Old | 154.3 | 5.8% | 1.73x | 21.2 |
| 7 | `PPDP` | Old | 186.0 | 7.0% | 1.78x | 20.3 |
| 8 | `DPDD` | RS | 83.6 | 3.2% | 1.56x | 18.4 |

`PPDP` has the largest time share in this table, but its `1.78x` speedup is
already relatively close to the simple `2x` ceiling. `PDDD` and `DDDD` combine
substantial runtime with weaker achieved speedup, so they are the first NCU
profiling targets. The next measurements should compare FP64 and FP32 register
use, occupancy, executed instruction mix, memory throughput, and subkernel load
balance.

## Data provenance

- Corrected old-layout report:
  `build_logs/all_exchange_timing_corrected_job22521933/README.md`
- Old-layout per-combination CSV:
  `build_logs/all_exchange_timing_corrected_job22521933/per_combination_theoretical_speedup.csv`
- Nsys old-layout report:
  `nsys_results/all_exchange_validation_job22520603/guanine8_all_exchange.nsys-rep`
- RS comparison report:
  `build_logs/resplit_comparison_job22840458/README.md`
- Nsys RS report:
  `nsys_results/resplit_comparison_job22840458/guanine8_resplit.nsys-rep`
- Detailed priority analysis: `resplit/OPTIMIZATION_PRIORITIES.md`

All reported percentages describe exchange-kernel time, not complete SCF wall
time. Overall SCF impact also depends on the fraction of SCF runtime spent in
exchange.
