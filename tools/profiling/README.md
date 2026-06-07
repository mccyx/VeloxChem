# Profiling Entrypoints

Use these top-level wrappers when you want a fully automated GH200 single-GPU workflow.

They all do the same environment setup:
- `cd` into the repo root
- `source env.sh`
- then override profiling-specific settings:
  - `OMP_NUM_THREADS=1`
  - `OMP_PLACES="{0}"`
  - `CUDA_VISIBLE_DEVICES=0`

This avoids the `OMP_NUM_THREADS=4` setting from `env.sh` leaking into profiling runs.

Each wrapper now also creates a timestamped profiling session by default:
- `ncu_results/<YYYYMMDD_HHMMSS>/...`
- `nsys_results/<ablation_stem>/<YYYYMMDD_HHMMSS>/...`

That keeps repeated runs from overwriting older profiling artifacts.

## NCU
- [`run_ncu_gh200_one_gpu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/run_ncu_gh200_one_gpu.sh)
- [`submit_ncu_gh200_one_gpu.sbatch`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/submit_ncu_gh200_one_gpu.sbatch)

Examples:

```bash
tools/profiling/run_ncu_gh200_one_gpu.sh single-fp32 DDDD26 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh single-fp64 DDDD26 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd guanine-8 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2 guanine-8 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2-fp64 guanine-8 guanine-8.inp
tools/profiling/run_ncu_gh200_one_gpu.sh all-dddd-v2-fp32 guanine-8 guanine-8.inp
```

Typical output:

```text
ncu_results/20260421_143015/
  guanine-8.dddd26_fp32.ncu-rep
  guanine-8.dddd26_fp32.summary.txt
  ...
```

Batch submission example:

```bash
sbatch tools/profiling/submit_ncu_gh200_one_gpu.sbatch -- all-dddd-v2-fp32 guanine-8 guanine-8.inp
```

If you omit the arguments, the batch script defaults to:
- mode: `all-dddd-v2-fp32`
- job: `guanine-8`
- input: `guanine-8.inp`

## NSYS
- [`run_nsys_gh200_single_gpu.sh`](/cfs/klemming/projects/supr/panor/yuxiao/gitrepo/VeloxChem.mixed-precision-2/tools/profiling/run_nsys_gh200_single_gpu.sh)

Examples:

```bash
tools/profiling/run_nsys_gh200_single_gpu.sh profile guanine-8.inp split_compare
tools/profiling/run_nsys_gh200_single_gpu.sh summarize benchmarks/ablation_results_gh200_guanine8_split26_2026-04-17.log
tools/profiling/run_nsys_gh200_single_gpu.sh full benchmarks/ablation_results_gh200_guanine8_split26_2026-04-17.log guanine-8.inp split_compare
```

Typical output:

```text
nsys_results/ablation_results_gh200_guanine8_split26_2026-04-17/20260421_143122/
  ablation_results_gh200_guanine8_split26_2026-04-17_split_compare.nsys-rep
  ablation_results_gh200_guanine8_split26_2026-04-17_split_compare_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
```

## Lower-level scripts
If you already have the environment set exactly the way you want, you can still call the lower-level scripts directly:
- `tools/profiling/ncu/...`
- `tools/profiling/nsys/...`
