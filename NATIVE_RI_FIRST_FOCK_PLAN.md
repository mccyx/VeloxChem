# VeloxChem Native RI First-Fock Plan

## Goal

Replace the external PySCF RI-to-density bridge with a native VeloxChem path:

```text
PET RI coefficients
  -> RI density and gradient on the VeloxChem grid
  -> native Coulomb J and PBE Vxc
  -> first Fock diagonalization
  -> restricted VeloxChem AO density
  -> ordinary SCF
```

Initial scope is deliberately restricted to neutral, closed-shell PBE/
def2-SVP calculations with def2-universal-jfit, one MPI rank, one GH node, and
the existing pretrained PET model. FP64 correctness comes before mixed
precision or Tensor Core optimization.

## Baseline already established

- PET steady forward: approximately 17--19 ms.
- Current PySCF RI-to-DM bridge: approximately 361 ms median.
- Native VeloxChem repeated medians on the 7-atom validation molecule:
  - SAD: 0.8272 s and 15 iterations;
  - PET-started SCF: 0.7089 s and 13 iterations;
  - reference-RI-started SCF: 0.6199 s and 11 iterations.
- Current warm PET total: approximately 1.087 s versus 0.827 s for SAD.
- PySCF/VeloxChem def2-SVP overlap mapping agrees to `3.48e-11` maximum error.
- Transformed PET density has `Tr(D_alpha S) = 18.999999999921`.

## Phase 0: Freeze references and profile the bridge

Tasks:

1. Save PySCF reference arrays for the validation molecule:
   - auxiliary coefficients in PySCF and metatensor order;
   - grid coordinates and weights;
   - auxiliary AO values and x/y/z derivatives;
   - `rho` and `grad rho`;
   - `J`, `Vxc`, `Hcore`, first Fock, eigenvalues and first density.
2. Add timers around molecule/basis construction, grid construction, AO
   evaluation, auxiliary AO evaluation, LibXC, three-center integrals,
   contraction, eigensolve and density construction.
3. Record dimensions and memory use as functions of atoms, orbital AOs,
   auxiliary functions and grid points.

Exit criteria:

- the approximately 361 ms bridge is explained by named components;
- all reference arrays are reproducible and checksummed;
- no optimization begins without a component-level baseline.

Estimated effort: 1--2 days.

## Phase 1: Auxiliary-basis coefficient ordering

Tasks:

1. Load `def2-universal-jfit` as a VeloxChem molecular basis.
2. Define an explicit descriptor for every coefficient:

   ```text
   atom, element, angular momentum, radial function, magnetic component
   ```

3. Convert metatensor PET blocks into VeloxChem auxiliary AO order.
4. Validate the mapping against PySCF auxiliary overlap matrices and selected
   auxiliary AO values at asymmetric points.
5. Reject unsupported elements and mismatched radial channels with clear
   errors.

Exit criteria:

- every coefficient is mapped exactly once;
- auxiliary overlap and point-value comparisons pass at FP64 tolerance;
- H/C/N/O validation molecules pass electron-count checks.

Estimated effort: 2--4 days.

## Phase 2: Native RI density and gradient on the grid

Implement, initially in FP64:

```text
rho[g]    = sum_P X[g,P]    c[P]
grad_k[g] = sum_P dX_k[g,P] c[P], k=x,y,z
```

Tasks:

1. Reuse VeloxChem molecular-grid blocking and auxiliary GTO evaluation.
2. Add an interface that evaluates auxiliary basis values and first
   derivatives on each grid block.
3. Contract coefficients without materializing unnecessary global arrays.
4. Compare every grid block and integrated electron count with PySCF.

Exit criteria:

- `rho` and all three gradient components match the PySCF reference;
- integrated PET/reference-RI electron counts meet the chosen tolerance;
- memory is bounded by grid-block size rather than the full grid matrix.

Estimated effort: 3--7 days.

## Phase 3: Native PBE Vxc from RI density

Tasks:

1. Feed native `rho` and `grad rho` into the existing LibXC PBE path.
2. Reuse orbital AO values/derivatives and grid weights.
3. Accumulate a symmetric orbital-basis `Vxc` matrix.
4. Keep the existing AO-density XC path unchanged; add a separate RI-density
   entry point until validation is complete.

Exit criteria:

- native `Vxc` agrees with the PySCF/reference implementation at FP64
  tolerance appropriate to the two grid implementations;
- matrix symmetry and MPI/GPU reductions are verified;
- no regression in the existing XC integrator tests.

Estimated effort: 3--7 days.

## Phase 4: Three-center Coulomb J

Required expression:

```text
J_mn = sum_P (mn|P) c_P
```

Tasks:

1. Write a CPU or otherwise simple FP64 reference for small s/p systems.
2. Determine which existing VeloxChem integral recurrences and generated code
   can be reused for mixed orbital/auxiliary shells.
3. Implement a true three-center driver; do not assume a four-center kernel
   with a dummy function is equivalent without proof.
4. Define shell-pair/auxiliary-shell partitioning, screening and GPU memory
   layout.
5. Compare individual integral tiles, contracted `J`, symmetry and Coulomb
   energy with PySCF.

Exit criteria:

- s, p and d orbital/auxiliary shell tests pass;
- contracted `J` passes matrix and energy comparisons;
- the implementation does not require storing the full three-center tensor for
  production-sized systems.

Estimated effort: 1--3 weeks; this is the highest-risk phase.

## Phase 5: First Fock and density generation

Construct:

```text
F_RI = Hcore + J_RI + Vxc_RI
F_RI C = S C epsilon
D_alpha = C_occ C_occ^T
```

Reuse the existing VeloxChem overlap, orthogonalization, eigensolver and
occupation code.

Validate:

- Fock symmetry;
- `C.T S C = I`;
- eigensolver residual `F C - S C epsilon`;
- `Tr(D_alpha S) = N_alpha`;
- generalized idempotency;
- agreement with saved PySCF first-Fock eigenvalues/density;
- final SCF energy and convergence.

Exit criteria:

- the validation molecule reproduces the existing 13 PET-started iterations or
  improves them;
- the external PySCF bridge is absent from the runtime path;
- fallback to SAD is available on validation failure.

Estimated effort: 2--4 days after phases 2--4.

## Phase 6: Size-scaling and crossover

Controlled scaling:

- 1/2/4/8 copies of the validated 7-atom fragment;
- approximately 80/160/320/640 orbital AOs.

Real-molecule validation, initially restricted to supported neutral
closed-shell C/H/N/O chemistry:

- caffeine;
- aspirin;
- glucose or a nucleobase;
- alanine dipeptide.

Measure cold and warm paths separately. Record model, RI first-Fock, SCF and
total times, exact Fock-build count, energy, failures and peak memory.

Exit criteria:

- identify the atom/AO size where native PET becomes faster than SAD, or show
  that no crossover exists in the tested range;
- explain scaling by measured components rather than iteration count alone.

Estimated effort: 3--5 days once the native path works.

## Phase 7: Tensor Core and mixed precision

Use the FP64 implementation as the permanent reference. Optimize in this
order:

1. FP32 auxiliary AO/derivative coefficient contractions.
2. Batch density and three gradient channels, molecules or geometries into
   Tensor-Core-friendly GEMMs.
3. TF32/BF16 input with FP32 accumulation for suitable `A.T @ W @ A` Vxc
   blocks, followed by FP64 accumulation/reduction.
4. Bound-driven FP64/FP32/TF32 partitioning for three-center J tiles.
5. Only then evaluate a lower-precision eigensolver with FP64 residual checks
   and iterative refinement.

For every precision policy record:

```text
rho and gradient error
electron-count error
J, Vxc and F matrix error
eigensolver residual and S-orthogonality
initial-density error
SCF iterations and failures
final energy
kernel and end-to-end time
```

Reject an optimization if saved kernel time is outweighed by an extra SCF
iteration or fallback.

Estimated effort: 3--8 weeks depending on how many kernels are replaced.

## Phase 8: Optional model fine-tuning

Do this only after the native pipeline provides a stable target and benchmark.
Generate labels using converged PBE/def2-SVP densities fitted in the
def2-universal-jfit overlap metric. Begin with supported H/C/N/O chemistry,
larger molecules and geometry trajectories. Evaluate SCF iterations and wall
time in addition to coefficient loss.

Adding a new element such as phosphorus is a separate extension requiring
checkpoint capability inspection, element-specific output channels and a
chemically diverse training set.

## Immediate next actions

1. Implement Phase 0 bridge profiling and save FP64 reference intermediates.
2. Inspect PET checkpoint supported species explicitly.
3. Implement Phase 1 auxiliary coefficient descriptors and mapping.
4. Add the smallest Phase 2 test: auxiliary `rho` at selected asymmetric grid
   points, before integrating with the full molecular grid.
5. Review Phase 0--2 results before starting three-center J.

The first review checkpoint is intentionally before Phase 4 because the
three-center GPU implementation is the largest commitment and should only
start after coefficient ordering and RI density evaluation are proven.
