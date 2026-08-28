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

## Current status

- Official recipe and checkpoint downloaded and checksummed in scratch.
- CPU reproduction code and isolated-environment bootstrap are ready.
- The first Slurm submission on 2026-08-28 could not reach the Slurm
  controller; resubmit the same job once the controller is available.
