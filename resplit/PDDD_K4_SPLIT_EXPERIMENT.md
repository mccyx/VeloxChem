# PDDD K4 Split Experiment

## Setup

- System: guanine-8
- Hardware: one GH200 GPU
- Executable: one build containing `rs5`, `k4_m01`, `k4_m12`, `k4_m23`, and `k4_m34`
- Selector: `VLX_EXCHANGE_PDDD_SPLIT`
- K4 candidates merge one adjacent pair from the supplied RS5 split:
  - `k4_m01`: `[0+1] [2] [3] [4]`
  - `k4_m12`: `[0] [1+2] [3] [4]`
  - `k4_m23`: `[0] [1] [2+3] [4]`
  - `k4_m34`: `[0] [1] [2] [3+4]`

## Results

| variant | host MP (ms) | host speedup vs old MP | Nsys MP (ms/interaction) | Nsys speedup vs old MP |
|---|---:|---:|---:|---:|
| rs5 | 165.492 | 1.2433x | 165.125 | 1.2518x |
| k4_m01 | 150.493 | 1.3699x | 150.552 | 1.3752x |
| k4_m12 | 143.929 | 1.4345x | 143.747 | 1.4391x |
| k4_m23 | 141.955 | 1.4515x | 142.084 | 1.4562x |
| k4_m34 | 144.486 | 1.4268x | 144.524 | 1.4322x |

Host results are averages over three program runs and four SCF interactions per run. The host timer covers the selected kernel launch group, GPU execution, and final stream synchronization. It excludes allocation, zeroing, host cut-layout construction, GPU cut construction, copies, and validation.

Nsys results contain only CUDA execution time for the selected PDDD kernel groups. They exclude host launch and synchronization overhead in addition to the work excluded from the host timer.

`k4_m23` is the fastest MP candidate in both measurements. Its worst observed MP error against its corresponding original-precision implementation is `7.383008e-12` absolute and `3.051338e-08` relative. The worst original-precision difference against old original is `1.246832e-18` absolute.

## NCU Explanation

For FP32, merging RS2 and RS3 reduces the complete bundle's executed SM instructions from `84,759,603,423` to `72,337,115,296`, a `14.66%` reduction. The merged K4 kernel 2 executes `1.4980x` fewer instructions than RS2 and RS3 summed. The ERI expression groups are retained, so this reduction is consistent with eliminating duplicated per-kernel setup work.

Detailed results:

- Host: `build_logs/pddd_k4_host_23003736/README.md`
- Nsys: `nsys_results/pddd_k4_23004036/README.md`
- NCU: `ncu_results/pddd_k4_winner_23004462/README.md`
