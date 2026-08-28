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
