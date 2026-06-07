# Nsight Compute Helpers

This directory now supports:
- single-kernel `profile -> export -> assess`
- batch profiling for `DDDD0..25` in both `FP64` and `FP32`
- reproducible single-GH200-GPU runs by default
- timestamped output directories when invoked through the top-level wrapper

## Files
- [`profile_ncu_fp32.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_ncu_fp32.sh)
  Runs `ncu` for one FP32 kernel and produces:
  - `*.ncu-rep`
  - `*.txt` details dump

- [`profile_ncu_fp64.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_ncu_fp64.sh)
  Same as above, but for one FP64 kernel.

- [`export_ncu_artifacts.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/export_ncu_artifacts.sh)
  Imports an existing `*.ncu-rep` and exports:
  - `*.summary.txt`
  - `*.source.csv`
  - `*.warpstate.csv`

- [`profile_and_export_ncu_fp32.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_and_export_ncu_fp32.sh)
  Convenience wrapper that runs both steps above for FP32.

- [`profile_and_export_ncu_fp64.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_and_export_ncu_fp64.sh)
  FP64 version of the same convenience wrapper.

- [`profile_all_dddd_ncu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_all_dddd_ncu.sh)
  Batch driver for all `DDDD0..25`, running both `FP64` and `FP32`.

- [`profile_all_dddd_v2_fp32_ncu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_all_dddd_v2_fp32_ncu.sh)
  Batch driver for the `split26 v2` FP32 kernels:
  - `computeCoulombFockDDDDv2_0_FP32`
  - ...
  - `computeCoulombFockDDDDv2_25_FP32`

- [`profile_all_dddd_v2_fp32_scalar_compare_ncu.sh`](/cfs/klemming/home/y/yuch4126/repo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_all_dddd_v2_fp32_scalar_compare_ncu.sh)
  Batch driver for paired baseline/scalar NCU profiling:
  - `computeCoulombFockDDDDv2_0..25_FP32`
  - `computeCoulombFockDDDDv2_0..25_FP32_auto_s_ast`

- [`profile_all_dddd_v2_fp64_ncu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_all_dddd_v2_fp64_ncu.sh)
  Batch driver for the `split26 v2` FP64 kernels:
  - `computeCoulombFockDDDDv2_0_FP64`
  - ...
  - `computeCoulombFockDDDDv2_25_FP64`

- [`profile_all_dddd_v2_ncu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/profile_all_dddd_v2_ncu.sh)
  Batch driver for the full `split26 v2` family, running:
  - all `FP64`
  - then all `FP32`

- [`extract_summary_metrics.py`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/extract_summary_metrics.py)
  Prints a compact table from one or more `*.summary.txt` files.

- [`top_kernel_metrics.py`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/top_kernel_metrics.py)
  Reads a `*.source.csv` file and reports:
  - top instructions by executed count
  - top stall reasons
  - top not-issued stall reasons

- [`top_warpstate_metrics.py`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/top_warpstate_metrics.py)
  Reads a `*.warpstate.csv` file and reports the top warp-state metrics.

- [`assess_kernels.py`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/ncu/assess_kernels.py)
  High-level comparison script.
  Input is one or more `*.summary.txt` files; it auto-infers the matching
  `*.source.csv` and `*.warpstate.csv` files in the same directory.

## Default profiling environment
The shell wrappers now default to the single-GH200-GPU environment recommended by your advisor:

```bash
export OMP_NUM_THREADS=1
export OMP_PLACES="{0}"
export CUDA_VISIBLE_DEVICES=0
```

You can still override any of those variables before running the scripts.

## Quick start

### 1. Profile and export one FP32 kernel
```bash
cd /cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2
source env.sh

tools/profiling/ncu/profile_and_export_ncu_fp32.sh DDDD26 guanine-8.inp
```

This creates files like:

```text
ncu_results/20260421_143015/
  guanine-8.dddd26_fp32.ncu-rep
  dddd26_fp32.txt
  guanine-8.dddd26_fp32.summary.txt
  guanine-8.dddd26_fp32.source.csv
  guanine-8.dddd26_fp32.warpstate.csv
```

### 2. Profile and export one FP64 kernel
```bash
tools/profiling/ncu/profile_and_export_ncu_fp64.sh DDDD26 guanine-8.inp
```

This creates files like:

```text
ncu_results/20260421_143015/
  guanine-8.dddd26_fp64.ncu-rep
  dddd26_fp64.txt
  guanine-8.dddd26_fp64.summary.txt
  guanine-8.dddd26_fp64.source.csv
  guanine-8.dddd26_fp64.warpstate.csv
```

### 3. Inspect one kernel
```bash
python3 tools/profiling/ncu/extract_summary_metrics.py \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32.summary.txt

python3 tools/profiling/ncu/top_kernel_metrics.py \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32.source.csv

python3 tools/profiling/ncu/top_warpstate_metrics.py \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32.warpstate.csv
```

### 4. Compare multiple kernels
```bash
python3 tools/profiling/ncu/assess_kernels.py \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32.summary.txt \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32_auto_scalarized.summary.txt \
  ncu_results/20260421_143015/guanine-8.dddd26_fp32_auto_scalarized_regroup.summary.txt
```

### 5. Batch-profile all `DDDD0..25`
This is the closest match to your advisor's loop.

```bash
cd /cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2
source env.sh
tools/profiling/ncu/profile_all_dddd_ncu.sh guanine-8 guanine-8.inp
```

This runs:
- `computeCoulombFockDDDD0..25_FP64`
- `computeCoulombFockDDDD0..25_FP32`

and exports for each kernel:
- `.ncu-rep`
- `.summary.txt`
- `.source.csv`
- `.warpstate.csv`

### 6. Batch-profile only `split26 v2` FP32 kernels
Use this mode when you want to analyze only the new split-26 implementation rather than the old `DDDD{i}_FP32` family.

```bash
cd /cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2
source env.sh
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2-fp32 guanine-8 guanine-8.inp
```

This runs:
- `computeCoulombFockDDDDv2_0_FP32`
- ...
- `computeCoulombFockDDDDv2_25_FP32`

### 7. Batch-profile only `split26 v2` FP64 kernels
```bash
cd /cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2
source env.sh
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2-fp64 guanine-8 guanine-8.inp
```

This runs:
- `computeCoulombFockDDDDv2_0_FP64`
- ...
- `computeCoulombFockDDDDv2_25_FP64`

### 8. Batch-profile all `split26 v2` kernels (`FP64` + `FP32`)
```bash
cd /cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2
source env.sh
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2 guanine-8 guanine-8.inp
```

This runs:
- `computeCoulombFockDDDDv2_0..25_FP64`
- `computeCoulombFockDDDDv2_0..25_FP32`

## Export details
The exported files now match the structure expected by the Python scripts and are closer to your advisor's original commands:

- `summary.txt`
  - generated by `ncu --import <report>`
- `source.csv`
  - generated by `ncu --import <report> --page source --print-source sass --csv`
- `warpstate.csv`
  - generated by `ncu --import <report> --section WarpStateStats --page details --print-details all --csv`

## Notes
- The helper scripts expect file names ending in:
  - `.summary.txt`
  - `.source.csv`
  - `.warpstate.csv`
- `assess_kernels.py` assumes those three companion files share the same stem.
- `job` is inferred from the input file name, so `guanine-8.inp` produces files prefixed with `guanine-8.`.
- The top-level wrapper `tools/profiling/run_ncu_gh200_one_gpu.sh` sets `VLX_NCU_OUT_DIR` to a fresh timestamped directory by default.
