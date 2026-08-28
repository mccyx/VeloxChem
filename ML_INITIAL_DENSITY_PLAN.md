# PET Initial-Density Integration Plan

## Objective

Integrate a pretrained PET electron-density model with VeloxChem as an optional
SCF initial guess, initially for restricted closed-shell PBE/def2-SVP
calculations. Determine whether it reduces end-to-end SCF wall time after PET
inference, RI conversion, data transfer, and fallback costs are included.

Development branch:

```text
ml-initial-density-pet
```

Current status (2026-08-28): milestones 1--3 and the first small-system part of
milestone 4 are complete. The pretrained model is reproduced, PySCF-to-
VeloxChem AO/spin conversion is overlap-validated, an external restricted
density API is implemented, and native VeloxChem SAD/PET/reference-RI results
agree in energy. The current bottleneck is the external PySCF RI-to-DM bridge.
The actionable next-stage plan is in `NATIVE_RI_FIRST_FOCK_PLAN.md`.

## Initial scope

Supported first:

- pretrained PET checkpoint; no model training;
- PBE/def2-SVP;
- restricted closed-shell molecules in the PET/SCFBench chemical domain;
- file-based PET-to-VeloxChem density handoff;
- single-node correctness and performance evaluation.

Deferred:

- HF and hybrid DFT;
- open-shell calculations;
- arbitrary basis sets;
- differentiable SCF;
- replacement of J, K, XC, or ERI kernels;
- C++/LibTorch integration;
- custom AI kernels and multi-GPU model parallelism.

## Existing model

Start from the pretrained PET density model demonstrated in the Atomistic
Cookbook:

<https://atomistic-cookbook.org/examples/ml-density/ml-density.html>

The model was trained on SCFBench PBE/def2-SVP calculations and predicts
overlap-metric RI coefficients in the def2-universal-jfit auxiliary basis.
Reproduce the published PySCF workflow before changing the model or integrating
it with VeloxChem.

## Storage layout

Keep source and compact results in this repository. Store environments,
checkpoints, datasets, exported density matrices, and profiler traces in:

```text
/cfs/klemming/scratch/y/yuch4126/dev/ml_initial_density/
├── env/
├── checkpoints/
├── datasets/
├── exported_densities/
├── benchmark_results/
└── profiler_results/
```

Proposed repository layout:

```text
tools/ml_initial_density/
├── README.md
├── reproduce_pet_pyscf.py
├── export_pet_density.py
├── compare_density_conventions.py
├── benchmark_initial_guesses.py
└── inputs/

tests/
├── test_external_density_guess.py
└── test_density_convention.py
```

## Milestone 0: VeloxChem baselines

1. Add deterministic H2O, ethanol, and benzene geometries.
2. Run VeloxChem PBE/def2-SVP with the existing SAD guess.
3. Record revision, environment, input hash, SCF iterations, total SCF time,
   FockERI time, and FockXC time.
4. Repeat after warm-up and report median timing.

Use a method block such as:

```text
@method_settings
basis: def2-svp
xcfun: pbe
grid_level: 4
@end
```

Exit criterion: reproducible SAD baseline results for at least three molecules.

## Milestone 1: reproduce pretrained PET

1. Create an isolated AI environment on the target GH node; do not modify the
   working VeloxChem environment.
2. Install and record exact versions of PyTorch, metatensor, metatomic, ASE,
   PySCF, and the cookbook dependencies.
3. Download the PET checkpoint to scratch and record its checksum.
4. Reproduce the upstream PySCF energy and SCF-cycle result.
5. Measure graph construction, PET forward, RI-to-density conversion, and SCF
   time separately.
6. Export the predicted density and metadata for controlled test molecules.

Exit criterion: the unmodified pretrained model produces a verified PySCF
initial guess before VeloxChem integration starts.

## Milestone 2: external-density API

Add an optional Python-level argument:

```python
def compute(
    self,
    molecule,
    ao_basis,
    min_basis=None,
    initial_density=None,
):
    ...
```

Selection order:

```text
explicit initial_density
    else restart density
    else SAD density
```

Validate:

- number of matrices for the SCF type;
- AO dimensions;
- floating-point dtype and finite values;
- symmetry;
- electron count in the AO overlap metric;
- MPI root ownership and broadcast.

No AI package may become a VeloxChem dependency in this milestone.

Required tests:

- explicitly passing the SAD density reproduces the normal SAD result;
- invalid shape, NaN/Inf, and incorrect electron count are rejected;
- existing restart and SAD behavior remains unchanged;
- MPI ranks receive identical initial density.

Exit criterion: VeloxChem starts and converges from a supplied NumPy density,
with existing tests still passing.

## Milestone 3: density convention and AO mapping

PySCF and VeloxChem may differ in AO ordering, spherical-harmonic ordering,
phase, normalization, and restricted-density convention even when both use
def2-SVP.

Compare AO overlap matrices and derive a transformation `U` such that:

```text
S_VeloxChem ~= U.T @ S_PySCF @ U
```

Apply the corresponding mathematically correct density transformation only
after determining whether `U` is a permutation/phase matrix or a more general
basis transformation.

Validate progressively using simple systems, p blocks, and d-polarization
blocks. Check:

- overlap matrices;
- matrix symmetry;
- real-space density at selected points;
- one-electron expectation values;
- initial Fock/energy behavior;
- final converged-energy equality.

Restricted-density convention is critical. Existing VeloxChem tests use
per-spin density, so closed-shell water satisfies:

```text
Tr(D_VeloxChem S) = 5
```

PySCF restricted density is commonly spin-summed:

```text
Tr(D_PySCF S) = 10
```

The adapter may therefore require `D_VeloxChem = 0.5 * D_PySCF`, but the factor
must be guarded by an explicit electron-count assertion.

Every exported density should include method, orbital basis, auxiliary basis,
atom order, coordinate units, density convention, model identifier, and
checkpoint checksum.

Exit criterion: converted PET/PySCF density passes overlap, electron-count, and
SCF convergence checks for all small validation molecules.

## Milestone 4: end-to-end evaluation

Compare:

1. VeloxChem SAD;
2. pretrained PET;
3. previous-geometry converged density;
4. simple trajectory density extrapolation where applicable.

Record:

- Fock-build count and SCF iterations;
- model graph construction and inference time;
- RI/density conversion and transfer time;
- total SCF, FockERI, and FockXC time;
- first-step commutator residual;
- convergence and fallback rate;
- converged energy and gradient agreement.

Primary speedup:

```text
T_SAD / (T_PET_inference + T_conversion + T_PET_started_SCF)
```

Iteration reduction alone is not sufficient.

Fall back to SAD when metadata is incompatible, the species is unsupported,
the density is malformed, the electron count is wrong, or the first exact SCF
evaluation is unsafe.

Exit criterion: PET converges to the same result and reduces end-to-end time on
a meaningful representative workload.

## Milestone 5: AI-infrastructure optimization

Profile only after scientific correctness and utility are established.

Potential bottlenecks:

- GPU neighbor-list and graph construction;
- attention/message-passing kernels;
- gather/scatter and segmented reductions;
- dynamic shapes and padding;
- CPU/GPU synchronization;
- `torch.compile` graph breaks;
- CUDA Graph size bucketing;
- FP32/BF16 mixed precision;
- RI-to-density conversion and data movement;
- batched molecule inference.

Compare eager PyTorch, `torch.compile`, precision policies, and batch sizes 1,
8, and 32. Measure both model-only and PET-to-SCF time.

Use four GH200 GPUs for independent molecules or trajectory replicas, not for
splitting one small molecule initially.

Exit criterion: optimize only a bottleneck shown to matter end to end.

## Conditional HF/hybrid extension

PET predicts an auxiliary representation of PBE real-space density, not a
unique full one-particle density matrix for exact exchange. Test transfer only
after the PBE path works.

If PET is inadequate for HF/hybrid initial guesses:

1. reuse an AO-matrix architecture such as dm-PhiSNet/PhiSNet/QHNet/SPHNet;
2. generate a fixed-method/fixed-basis VeloxChem dataset;
3. fine-tune before training from scratch;
4. enforce symmetry, electron number, and generalized idempotency;
5. evaluate Fock-build count and wall time rather than matrix MSE;
6. consider solver-aligned training only after a supervised baseline.

## Success definition

The first stage succeeds only if:

1. no new model is trained;
2. PET output becomes a validated VeloxChem initial density;
3. PET- and SAD-started PBE/def2-SVP calculations converge to the same result;
4. fallback is safe and existing workflows are unchanged;
5. end-to-end time improves after all AI/conversion overhead is included;
6. the environment and benchmark reproduce on Dardel GH nodes.
