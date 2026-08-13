# Exchange Resplit Optimization Experiments

The final simultaneous all-winner benchmark is recorded in
[`ALL_WINNERS_AGGREGATE_BENCHMARK.md`](ALL_WINNERS_AGGREGATE_BENCHMARK.md).
It measures a `1.6996x` aggregate host-timer kernel speedup and a `1.7018x`
aggregate Nsys kernel speedup over original FP64. Including GPU cut building,
the host-timer speedup is `1.6850x`.

## 1. Objective

These experiments investigate why the supplied exchange resplit (RS) kernels improve some integral families but regress others, and how to select better subkernel boundaries.

The original observation was:

- RS improves `PDDD`, `DPDD`, and `DDDP`.
- RS regresses `DDDD`, `DDDS`, `DSDD`, `PPDD`, and `SDDD`.
- The complete mathematical ERI contribution is unchanged, while the summed operator count of the separately factored, generated `eri_ijkl` source expressions changes slightly because the split boundaries change the available factoring opportunities. The number of subkernels changes substantially.

This suggested that repeated per-kernel work, rather than a large change in the mathematical ERI work, was a major performance factor. The experiments therefore vary both:

1. The number of subkernels.
2. The assignment of ERI expression groups to those subkernels.

## 2. Terminology

### 2.1 Expression group

Each generated exchange subkernel contains one assignment of the form:

```cpp
const double eri_ijkl = common_prefactor * (
    factored_expression_group
);
```

The FP32 version has the corresponding `float eri_ijkl_f` assignment. In this report, an **expression group** is the already-generated arithmetic inside the outer parentheses of one such assignment.

The complete family result is the sum of all expression groups across its subkernels. Changing the split changes which groups are evaluated together in one CUDA kernel, but must not omit, duplicate, or reorder the source groups.

The complete unfactored mathematical work is therefore the same. However, the static count used in this experiment counts `+`, `-`, `*`, and `/` operators **after each subkernel has already been factored independently**. Moving a split boundary can expose or remove common factors within a subkernel, so the summed factored operator count is not required to be exactly identical. For example:

```text
DDDD old19: 25,281 factored source operators
DDDD RS26:  25,295 factored source operators
```

This 14-operator difference is only `0.055%`. It does not mean that RS computes a different ERI contribution; it means the same contribution is represented by a slightly different factored source expression.

### 2.2 Grouping

**Grouping** means the assignment of ordered expression groups to CUDA subkernels.

For example, an RS layout with five source subkernels can be written as:

```text
[0] [1] [2] [3] [4]
```

Merging source groups 2 and 3 produces:

```text
[0] [1] [2+3] [4]
```

The second layout launches four kernels. Its third kernel evaluates both expression groups that were previously evaluated by RS kernels 2 and 3.

Grouping affects performance because every CUDA subkernel repeats work such as:

- shell-pair and primitive indexing;
- coordinate and exponent loads;
- screening and cut-index handling;
- Boys-function preparation;
- intermediate-variable setup;
- synchronization and output reduction;
- kernel launch and scheduling overhead.

Fewer kernels can reduce this duplicated work, but larger kernels can also increase register pressure or reduce occupancy. Therefore, fewer kernels are not automatically better.

### 2.3 Factoring

**Factoring** is the algebraic structure of the generated `eri_ijkl` expression. For example:

```cpp
A * B + A * C
```

may be represented as:

```cpp
A * (B + C)
```

Two layouts containing the same mathematical ERI terms may expose different common subexpressions, intermediate variables, additions, multiplications, and FMA opportunities. NVCC can consequently generate different instruction counts, register use, and schedules.

The candidates in this experiment do **not** rerun a symbolic factorization pass. They start from the already-factored CUDA expressions in either the old or RS source. When groups are merged, their factored inner expressions are placed under one compatible outer prefactor. NVCC may then perform normal compiler optimization across the combined function, but this is not equivalent to globally regenerating and refactoring the family.

### 2.4 Old-derived and RS-derived

The word **derived** identifies which existing CUDA layout supplies the source expression groups and kernel body templates.

- **Old-derived**: extract groups from `computeExchangeFock<FAMILY><index>` and its `_FP64` and `_FP32` forms.
- **RS-derived**: extract groups from `computeExchangeFock<FAMILY><index>_RS` and its `_FP64` and `_FP32` forms.

For example, DDDP old-derived K5 is:

```text
old7 source:       [0] [1] [2] [3] [4] [5] [6]
old-derived K5:    [0] [1+2] [3] [4+5] [6]
```

DDDP RS-derived K5 is:

```text
RS6 source:        [0] [1] [2] [3] [4] [5]
RS-derived K5:     [0] [1] [2+3] [4] [5]
```

Both candidates have five kernels, but they contain different source expression groups because old7 and RS6 already use different boundaries and factoring. Comparing them controls the kernel count while testing the effect of grouping/factoring.

## 3. How a Derived Kernel Is Constructed

The generation scripts perform the following operations for original, FP64, and FP32 independently.

1. Locate each source function by its exact kernel name.
2. Extract the complete CUDA function body using balanced braces.
3. Locate the `eri_ijkl` or `eri_ijkl_f` assignment.
4. Separate its common outer prefactor from the inner factored expression.
5. Verify that all groups being merged have compatible outer prefixes.
6. Preserve the source-group order and concatenate the inner expressions.
7. Insert the combined expression into one source kernel body used as the template.
8. Rename the generated function, for example:

```text
computeExchangeFockDPDD1_K4_RS_FP32
```

9. Generate matching header declarations and a JSON manifest recording the source-group mapping.
10. Verify the expected number of functions and reject duplicated expression signs or missing groups.

The template body matters because different source kernels may declare different setup variables. For RS-derived adjacent groups, the later or higher-order body is normally used. For old-derived cross-order merges, the generator selects the most complete compatible body, currently approximated by the longest source function. Compilation is then used as an additional dependency check. For example, an early DDDD old-derived build exposed a missing `delta` declaration; selecting the more complete template fixed it.

This is a practical experimental construction method. A production generator should calculate the union of required setup dependencies explicitly rather than infer completeness from function length.

## 4. Candidate Selection

Candidates were selected using three forms of evidence:

1. **Static source arithmetic count**: a rough measure of the already-factored `eri_ijkl` expression.
2. **Boys-function order**: merges within the same order are the simplest and safest candidates; cross-order merges require a sufficiently high-order and complete template.
3. **Nsys per-subkernel runtime**: used to avoid severely imbalanced contiguous groups.

Static count is only a candidate-generation heuristic. It excludes repeated setup, loads/stores, control flow, screening behavior, compiled instructions, register pressure, and occupancy.

The final selection rule is:

```text
static count and Boys order -> generate a small candidate set
host timing and Nsys        -> select the fastest candidate
NCU dynamic instructions    -> explain the result
host validation             -> confirm numerical correctness
```

## 5. Runtime Selection and One Executable

All candidates are compiled into one executable. Environment variables select exactly one layout per family at runtime:

```bash
export VLX_EXCHANGE_PDDD_SPLIT=k4_m23
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
export VLX_EXCHANGE_DDDP_SPLIT=old_k5
export VLX_EXCHANGE_DPDD_SPLIT=rs_k4
```

The selector controls original, FP64, and FP32 launch groups together. A run never mixes the original kernel from one grouping with the MP kernel from another grouping.

The defaults preserve the previously integrated RS layouts:

```text
PDDD: rs5
DDDD: rs26
DDDP: rs6
DPDD: rs5
```

Compiling all candidates once avoids comparing different executables or rebuilds.

## 6. Measurement Definitions

### Host timer

The host timer covers:

- selected CUDA kernel launches;
- GPU execution;
- final stream synchronization.

It excludes:

- device-buffer allocation;
- buffer zeroing;
- host cut-layout construction;
- GPU cut construction;
- copies used for validation;
- numerical comparison and logging.

Host results use three program runs and four SCF interactions per run, giving 12 samples per candidate.

### Nsys

Nsys reports CUDA execution time for the named family kernel group only. Times are normalized to one SCF interaction. It excludes host launch and synchronization overhead as well as allocation, cuts, copies, and validation.

### NCU

NCU profiles one selected launch of each FP32 subkernel and collects:

```text
smsp__inst_executed.sum
smsp__inst_issued.sum
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
```

Dynamic instruction counts explain performance changes, but runtime remains the final optimization criterion because instructions do not capture all register, occupancy, memory, and scheduling effects.

## 7. Results

All measurements use `guanine-8-hf.inp` on one NVIDIA GH200 GPU.

| Family | Previous selected layout | Best tested layout | Previous MP (ms) | Best host MP (ms) | Best Nsys MP (ms) | Nsys speedup vs old |
|---|---|---|---:|---:|---:|---:|
| PDDD | RS5 | RS-derived K4 M23 | 165.5 | 141.955 | 142.084 | 1.4562x |
| DDDD | old19 | old-derived K16 | 118.959 | 113.020 | 113.064 | 1.0519x |
| DDDP | RS6 | old-derived K5 | 97.598 | 87.152 | 87.172 | 1.1863x |
| DPDD | RS5 | RS-derived K4 | 83.758 | 72.408 | 72.348 | 1.3224x |

The speedup column uses the corresponding old-layout MP kernels as the baseline, not FP64.

### 7.1 PDDD

Best mapping:

```text
RS5:       [0] [1] [2]   [3] [4]
K4 M23:    [0] [1] [2+3]     [4]
```

- Host MP: `141.955 ms`
- Nsys MP: `142.084 ms`
- Nsys speedup versus old MP: `1.4562x`
- FP32 executed-instruction reduction versus RS5: `14.66%`
- Worst observed MP absolute error: `7.383008e-12`

### 7.2 DDDD

The supplied resplit changed old19 into RS26 and regressed MP performance:

```text
old19 baseline:     118.959 ms
RS26:             139.353 ms
RS-derived K19:   about 122.4 ms
RS-derived K16:   115.078 ms
old-derived K16:  113.020 ms
```

The old19 host baseline and old-derived K16 winner were measured in the same 12-sample benchmark: `118.959 ms` and `113.020 ms`, respectively. The winner saves `5.939 ms` per interaction and gives a host speedup of `1.0525x`. Nsys independently measures old19 at `118.931 ms` and the winner at `113.064 ms`, giving `1.0519x`.

Reducing `26 -> 19` removes most of the regression, showing that repeated per-kernel work is a major cause. However, old19 remains about 3% faster than the tested RS-derived K19 layouts at the same kernel count. The old-derived K16 winner reduces FP32 executed instructions by `5.12%` relative to old19.

### 7.3 DDDP

Mappings:

```text
RS-derived K5:    [0] [1]   [2+3] [4]   [5]
old-derived K5:   [0] [1+2] [3]   [4+5] [6]
```

- RS6 MP: `97.598 ms`
- RS-derived K5 MP: `88.904 ms`
- old-derived K5 MP: `87.152 ms`
- Winner Nsys speedup versus old7: `1.1863x`
- Winner FP32 executed-instruction reduction versus old7: `14.21%`
- Worst observed MP absolute error: `4.291909e-12`

At equal K5 count, the old-derived grouping is about 2% faster.

### 7.4 DPDD

Mappings:

```text
old-derived K5:   [0]   [1+2] [3]   [4+5] [6]
RS-derived K4:    [0]   [1+2] [3]   [4]
old-derived K4:   [0+1] [2+3] [4]   [5+6]
```

- RS5 MP: `83.758 ms`
- old-derived K5 MP: `82.375 ms`
- RS-derived K4 MP: `72.408 ms`
- old-derived K4 MP: `73.914 ms`
- Winner Nsys speedup versus old7: `1.3224x`
- Winner FP32 executed-instruction reduction versus old7: `25.83%`
- Worst observed MP absolute error: `8.472268e-12`

At equal K4 count, the RS-derived grouping is about 2.1% faster. Old-derived is therefore not universally better.

## 8. Main Conclusions

1. Kernel count is an important performance variable because every subkernel repeats setup and reduction work.
2. Kernel count alone is not a sufficient split criterion. At equal count, old-derived and RS-derived layouts differ by approximately 2-3% in several experiments.
3. Static arithmetic count is useful for screening candidates, not selecting the winner.
4. Dynamic executed instructions consistently explain a large part of the measured runtime changes.
5. The best grouping is family-dependent:
   - old-derived wins for DDDD and DDDP;
   - RS-derived wins for PDDD and DPDD.
6. Host and Nsys rankings agree closely, indicating that the improvements are primarily GPU execution improvements rather than host launch-timing artifacts.
7. All tested candidates preserve the expected numerical accuracy.

The four winners save roughly `51 ms` per SCF interaction compared with the previously selected layouts. This suggests about a further 2% reduction in aggregate exchange MP kernel time, but that value combines separate runs and is only an estimate. The winners must be selected together and measured in one complete hybrid benchmark before reporting an aggregate production result.

## 9. Reproduction Guide

### Generate candidates

Run from the repository root:

```bash
python3 tools/pddd_split_search/generate_k4_variants.py
python3 tools/dddd_split_search/design_candidates.py
python3 tools/dddd_split_search/generate_k19_variants.py
python3 tools/dddd_split_search/design_k16_candidates.py
python3 tools/dddd_split_search/generate_k16_variants.py
python3 tools/dddp_split_search/generate_k5_variants.py
python3 tools/dpdd_split_search/generate_variants.py
```

The generated `.cu.inc` files are included inside `EriExchange.cu`; the `.hpp.inc` files are included inside `EriExchange.hpp`. They are part of the existing translation unit and do not require separate Makefile targets.

### Build

```bash
rm -f src/gpu/EriExchange.d src/gpu/FockDriverGPU.d
sbatch tools/pddd_split_search/submit_build.sbatch
```

The `.d` files are generated dependency caches. Removing them is necessary after changing included `.inc` paths or content dependencies.

### Run the current winners

```bash
source env.sh
export VLX_EXCHANGE_PDDD_SPLIT=k4_m23
export VLX_EXCHANGE_DDDD_SPLIT=old_k16_runtime
export VLX_EXCHANGE_DDDP_SPLIT=old_k5
export VLX_EXCHANGE_DPDD_SPLIT=rs_k4
export VLX_ABLATION_LOG=ablation_results.log
vlx guanine-8-hf.inp
```

### Reproduce individual benchmark stages

```bash
# PDDD
sbatch tools/pddd_split_search/submit_host_benchmark.sbatch
sbatch tools/pddd_split_search/submit_nsys_benchmark.sbatch

# DDDD
sbatch tools/dddd_split_search/submit_k16_host_benchmark.sbatch
sbatch tools/dddd_split_search/submit_k16_nsys.sbatch

# DDDP
sbatch tools/dddp_split_search/submit_host_benchmark.sbatch
sbatch tools/dddp_split_search/submit_nsys.sbatch

# DPDD
sbatch tools/dpdd_split_search/submit_host_benchmark.sbatch
sbatch tools/dpdd_split_search/submit_nsys.sbatch
```

NCU scripts are also present in the same family-specific directories. NCU is expensive because exact kernel names are profiled one at a time, with a separate program execution and metric replay for each subkernel.

## 10. Applying the Method to Another Family

The following is the practical workflow for extending the experiment to another exchange family.

### Step 1: inspect the existing layouts

```bash
python3 tools/profiling/analyze_exchange_cuda_splits.py \
    src/gpu/EriExchange.cu --families FAMILY

python3 tools/profiling/summarize_exchange_resplit_nsys.py \
    nsys_results/resplit_comparison_job22840458/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv \
    --families FAMILY --precision fp32
```

Record, for every old and RS subkernel:

- its source index;
- Boys-function order;
- static factored arithmetic count;
- average FP32 runtime;
- the number of old and RS kernels.

### Step 2: write explicit contiguous mappings

Represent every candidate as a list of source-index lists. For example:

```python
groups = [[0], [1, 2], [3], [4, 5], [6]]
```

Before generating code, verify:

```python
flat = [index for group in groups for index in group]
assert flat == list(range(source_kernel_count))
```

This proves that every source group appears exactly once and that the original order is preserved.

Prefer same-Boys-order adjacent merges first. Use a cross-order merge only when the selected template computes a sufficiently high Boys order and contains every setup dependency required by all merged expressions.

### Step 3: generate all three precision forms

For each mapping, generate:

```text
original
FP64
FP32
```

Do not generate only FP32. Accuracy validation needs the candidate's corresponding original-precision implementation, and the MP execution needs both FP64 and FP32 forms using identical boundaries.

The family generators in `tools/*_split_search/` demonstrate the extraction and merge procedure. Always emit a manifest such as:

```text
src/gpu/<FAMILY>_split_variants.manifest.json
```

The manifest is the auditable definition of what each generated kernel contains.

### Step 4: include and compile once

Include generated definitions before the closing `gpu` namespace in `EriExchange.cu`, and declarations in `EriExchange.hpp`. Keep `.inc` files directly under `src/gpu/`; creating arbitrary subdirectories under `src/` causes this repository's recursive Makefile to treat them as libraries.

Add a runtime selector in the family comparison block. Read the environment variable before starting the kernel timer and validate its value. The same branch must control original, FP64, and FP32 launches.

Then rebuild after removing stale dependency caches:

```bash
rm -f src/gpu/EriExchange.d src/gpu/FockDriverGPU.d
sbatch tools/pddd_split_search/submit_build.sbatch
```

A successful compile is necessary but not sufficient. A merged template can compile while retaining unnecessary setup, which may still hurt runtime.

### Step 5: validate before profiling

Run every selector with the existing host validation and check:

- candidate original versus old original;
- candidate MP versus candidate original;
- no CUDA errors or missing validation blocks.

Do not compare a candidate MP result against an original result from a different grouping when deciding whether the generated candidate is internally correct.

### Step 6: measure in this order

1. Run three host-timed program repetitions.
2. Reject inaccurate or clearly slower candidates.
3. Run one Nsys profile for the surviving candidates.
4. Run NCU only for the winner and its relevant baseline.

Use runtime to choose the winner. Use NCU to determine whether the speedup came from reduced dynamic instructions or whether other GPU effects must be investigated.

### Step 7: stop when the return becomes small

Do not exhaustively test every possible boundary. Continue reducing the kernel count only while runtime improves materially. If adjacent counts differ by less than measurement variability, keep the simpler or better-balanced layout and move to a higher-impact family.

## 11. Detailed Reports

- `resplit/PDDD_K4_SPLIT_EXPERIMENT.md`
- `resplit/DDDD_SPLIT_EXPERIMENT.md`
- `resplit/DDDP_SPLIT_EXPERIMENT.md`
- `resplit/DPDD_SPLIT_EXPERIMENT.md`
- `build_logs/pddd_k4_host_23003736/README.md`
- `build_logs/dddd_k16_host_23047976/README.md`
- `build_logs/dddp_k5_host_23054258/README.md`
- `build_logs/dpdd_split_host_23074898/README.md`
- `nsys_results/pddd_k4_23004036/README.md`
- `nsys_results/dddd_k16_23048261/README.md`
- `nsys_results/dddp_k5_23055132/README.md`
- `nsys_results/dpdd_split_23084733/README.md`
