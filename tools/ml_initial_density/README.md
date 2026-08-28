# PET initial-density experiments

This directory contains the small, reproducible front end for the pretrained
PET electron-density model used in the Atomistic Cookbook. Large and generated
artifacts stay under:

```text
/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density
```

The first experiment deliberately runs on eight CPU cores in Dardel's shared
partition. It checks the scientific workflow without reserving a full GH200
node. It compares PySCF PBE/def2-SVP SCF starting from SAD, reference RI, and
PET-predicted RI densities, and writes timings plus all three density matrices.

Downloaded upstream artifacts (2026-08-28):

```text
09e1e2c0f13f6bd1d63d4377f375a60101500c9d67faa883d2fe57f74f3bbe40  ml-density.zip
f27bf90619797e15749107fd98fe6b9f62415ec10c63d326d0db22f4c13a1246  pet-density.pt
```

Submit it with:

```bash
sbatch tools/ml_initial_density/submit_reproduce_pet_cpu.sbatch
```

The job creates an isolated Python 3.11 environment in scratch on first use.
Expected upstream results for the included SCFBench molecule are approximately
12 SAD cycles, 8 reference-RI cycles, and 9 PET cycles, with a converged energy
of -302.5314352046 Hartree.

Outputs are written to:

```text
$VLX_ML_DENSITY_SCRATCH/benchmark_results/pet_pyscf_reproduction/result.json
$VLX_ML_DENSITY_SCRATCH/benchmark_results/pet_pyscf_reproduction/initial_densities_pyscf.npz
```

After this reproduction passes, add a separate CUDA environment for the GH200
node and benchmark only model graph construction and inference there. Do not
mix the AI environment into the VeloxChem build environment.

The GH200 environment is created from `logingh` so its compiled packages use
the correct aarch64 architecture:

```bash
ssh logingh
cd /cfs/klemming/home/y/yuch4126/work/VeloxChem.ml-initial-density-pet
tools/ml_initial_density/bootstrap_gh200_env.sh
```

It pins the official aarch64 CUDA 13.0 build of PyTorch 2.10.0, which satisfies
the PyTorch and metatomic version ranges of `metatrain==2026.2`. CUDA execution
has been verified on the compute-capability 12.1 GPU exposed by `logingh`. A
formal one-inference GH200 smoke job uses the PDC test GH account:

```bash
sbatch tools/ml_initial_density/submit_pet_gh200_smoke.sbatch
```

The end-to-end PySCF reproduction can then run on one GH compute node:

```bash
sbatch tools/ml_initial_density/submit_reproduce_pet_gh200.sbatch
```

Measure cold-start and steady-state inference separately with:

```bash
sbatch tools/ml_initial_density/submit_pet_gh200_inference_benchmark.sbatch
```

Repeat density construction and interleave SAD/reference-RI/PET-started SCF
timings with:

```bash
sbatch tools/ml_initial_density/submit_pet_repeated_timing_gh200.sbatch
```

The default is five repeats using eight CPU threads. The SCF execution order is
rotated each repeat to reduce cache and order bias. Override the repetition
count at submission time with, for example, `--export=ALL,BENCH_REPEATS=8`.

The AO handoff utilities are:

```text
probe_ao_conventions.py              export/compare PySCF and VeloxChem overlap
transform_density_for_veloxchem.py   permute and convert RKS density to one spin
reproduce_pet_veloxchem.py           run VeloxChem SAD/PET/reference-RI SCF
```

`ScfDriver.compute(..., initial_density=D)` now accepts a symmetric restricted
single-spin AO density in VeloxChem ordering when using DIIS. A spin-summed
PySCF RKS density must first be permuted and multiplied by 0.5.

## Current status

- Official recipe and checkpoint downloaded and checksummed in scratch.
- CPU reproduction code and isolated-environment bootstrap are ready.
- GH environment uses PyTorch 2.10.0+cu130 on aarch64.
- Compute-node smoke job 24001957 completed successfully on an NVIDIA GH200
  120GB: 13 output blocks and 267 RI coefficients; initial model/calculator
  load took 2.07 s and the first CUDA forward took 8.22 s.
- The `sum28-gpugh` reservation expired at 12:00 on 2026-08-28; subsequent
  scripts use the normal `gpugh` queue with account `pdc-software-test-gh`.
- Full reproduction job 24007798 passed: SAD/reference-RI/PET required 12/8/9
  SCF cycles and converged energies agreed within 2e-11 Hartree. This cold run
  was not an end-to-end speedup: PET forward took 7.80 s, PET-to-DM conversion
  0.44 s, and PET-started SCF 8.25 s, versus 1.98 s for SAD guess plus SCF.
- The next benchmark measures warm steady-state model inference separately from
  CUDA/JIT cold-start cost.
- Repeated inference additionally requires the CUDA 13 wheel's `nvidia/cu13/lib`
  directory in `LD_LIBRARY_PATH`, because the second forward triggers NVRTC
  fusion and dynamically loads `libnvrtc-builtins.so.13.0`.
- Inference benchmark job 24007953 passed. Model/calculator loading took 1.52 s
  and the first CUDA forward took 10.94 s, but 20 post-warm-up forwards had a
  15.73 ms median (16.94 ms mean, 15.24--21.71 ms range). Optimization should
  therefore target cold-start compilation/caching and persistent-model reuse
  before attempting to accelerate steady-state kernels.
- Repeated end-to-end jobs 24008018 (8 threads) and 24008057 (1 thread)
  eliminated the anomalous one-shot 8.25 s PET-started SCF result. Five
  interleaved repeats gave stable SAD/RI/PET cycle counts of 12/8/9. At eight
  threads the warm median totals were 2.977 s for SAD and 2.700 s for PET
  (about 9.3% faster); at one thread they were 4.839 s and 4.417 s (about 8.7%
  faster). A second forward still took 2.5--2.7 s after the 7.4--8.9 s cold
  forward, so robust timing requires two untimed warm-up calls.
- PySCF-to-VeloxChem AO mapping was validated using the full def2-SVP overlap
  matrix: the maximum absolute difference after permutation was 3.48e-11. The
  transformed PET restricted density had 18.999999999921 alpha electrons.
- The first native VeloxChem handoff passed on nid002897. SAD/PET/reference-RI
  required 15/13/11 iterations and converged energies agreed within 1.97e-11
  Hartree. The 4.37/0.73/0.63 s one-shot timings include execution-order and
  cold-start effects and are not yet a performance comparison.
- Five warm, order-rotated native VeloxChem repeats gave median SAD/PET/RI
  timings of 0.8272/0.7089/0.6199 s with stable 15/13/11 iterations. PET thus
  reduced the native SCF part by about 14.3%. Adding the separately measured
  0.017 s steady model forward and current 0.361 s PySCF RI-to-DM bridge gives
  about 1.087 s, however, so PET is still about 31% slower end-to-end than
  native SAD for this seven-atom system. The bridge, not steady model
  inference, is now the primary optimization target.
