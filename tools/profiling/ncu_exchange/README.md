# Typical Exchange NCU Profiles

This local wrapper adapts the read-only helpers in
`VeloxChem.mixed-precision-experiments/tools/profiling/ncu` to exact exchange
kernel names. It does not modify or write into that source directory.

Run it inside a one-GH200 allocation after sourcing `env.sh`:

```bash
source env.sh
tools/profiling/ncu_exchange/profile_typical_exchange.sh guanine-8-hf.inp
```

The selected comparisons are:

| Family | Old/source launches | Selected resplit launch | Reason |
|---|---|---|---|
| PDDD | `PDDD2_RS_FP32` + `PDDD3_RS_FP32` | `PDDD2_K4_M23_FP32` | Clear RS-derived merge win |
| DPDD | `DPDD1_FP32` + `DPDD2_FP32` | `DPDD1_K4_RS_FP32` | Strongest complete-layout instruction reduction |
| DDDD | `DDDD0_FP32` + `DDDD1_FP32` | `DDDD0_K16_OLD_RUNTIME_FP32` | Marginal-win control |

The winner kernel replaces two source launches in each row. Additive metrics
such as duration and executed instructions must therefore compare the winner
against the sum of both source launches. Registers, occupancy, throughput, and
stall percentages remain per-launch properties and should be shown separately.

For non-resplit optimization candidates, profile the six heaviest original
FP32 exchange kernels from the all-exchange Nsys baseline:

```bash
tools/profiling/ncu_exchange/profile_typical_old_exchange.sh guanine-8-hf.inp
```

Profile both directly comparable merges in the selected DDDP K5 layout with:

```bash
tools/profiling/ncu_exchange/profile_dddp_merges.sh guanine-8-hf.inp
```

Profile the supplied RS5 source pair merged into selected DPDD K4 group 1:

```bash
tools/profiling/ncu_exchange/profile_dpdd_merge_sources.sh guanine-8-hf.inp
```
