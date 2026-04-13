# Kernel Optimization Tool

`tools/kernel_opt_pipeline.py` is a small source-to-source helper for the DDDD-style CUDA kernels in `src/gpu/EriCoulomb.cu`.

It currently automates the safest part of the manual tuning workflow:

- run a single `scalar` pipeline that:
  - scalarizes `const float name[3] = {...};` into three scalar locals
  - hoists repeated dynamic accesses like `PQ_f[a0]` into local aliases such as `PQ_a0_f`
  - rewrites direct element reads like `r_k_f[0]` to scalar names
- run a first `regroup` pass that folds repeated multiplicative power chains inside long `eri_ijkl_f` expressions
- extract exact repeated parenthesized subexpressions inside long `eri_ijkl_f` assignments

This is intentionally a v1 pass. It does not yet:

- extract larger semantic common subexpressions such as the full hand-written `pa_pb_mix` / `pqab_mix` groups
- split long expressions into staged temporaries
- rewrite the whole file in place

It is also intentionally conservative:

- hoisting is limited to the same lexical block
- a dynamic index must have exactly one local definition in that block
- if an index may be redefined later, the tool skips that hoist and reports why
- shared indices defined in an ancestor block are only accepted when a synchronizing `__syncthreads()` appears before the rewritten region

## Example

```bash
python3 tools/kernel_opt_pipeline.py \
  --source src/gpu/EriCoulomb.cu \
  --function computeCoulombFockDDDD21_FP32 \
  --passes scalar \
  --output /tmp/DDDD21_FP32.scalarized.cu \
  --report
```

## Intended workflow

1. Run the tool on a baseline kernel.
2. Diff the generated output against the hand-tuned version.
3. Keep the safe scalarization/hoisting pieces.
4. Add follow-up manual or scripted CSE/splitting passes.

The goal is to move the repeated “small-array scalarize + indexed-access hoist + direct-index cleanup” work into one reusable default `scalar` step before deeper tuning.

## Current Pass Pipeline

The current script behaves like a small fixed pass pipeline.

When `scalar` is enabled, each matching `const float name[3] = {...};` declaration goes through one bundled source-shaping stage:

1. scalarize
   Replace the array with three scalar locals such as `PQ0_f`, `PQ1_f`, `PQ2_f`.
2. indexed-access hoist
   Insert aliases such as `PQ_a0_f` when the index is proven stable enough.
3. direct-index rewrite
   Rewrite fixed reads like `PQ_f[0]` to the scalar names.

After the bundled `scalar` stage finishes, the optional follow-up passes are:

4. regroup
   Fold repeated multiplicative power chains such as `S1_f * S1_f` or `inv_S4_f * ... * inv_S4_f` into local aliases like `S1_f_pow2` and `inv_S4_f_pow5`.
5. exact local CSE
   Scan long `eri_ijkl_f` assignments and extract exact repeated parenthesized subexpressions.

The current implementation lives in `tools/kernel_opt_pipeline.py`.

## Default vs Experimental Use

The recommended default pipeline is:

- `scalar`

This is the current "normal" source-shaping path.

For `DDDD34`, the current ablation results show:

- `hoist` is the main source of performance gain
- `rewrite` has little standalone effect, but it belongs naturally in the cleaned-up final form
- the current exact local `cse` pass is correct, but not beneficial enough to be part of the default pipeline

So the default CLI behavior intentionally does **not** enable `cse`.

The public CLI now exposes three pass names:

- `scalar`
- `regroup`
- `cse`

For example:

- `--passes scalar`
- `--passes scalar,regroup`
- `--passes scalar,cse`
- `--passes scalar,regroup,cse`

For backwards compatibility, legacy spellings such as `scalarize,hoist,rewrite` are still accepted and normalized to `scalar`.

## Pass Dependencies

The recommended mental model is now simple:

- run `scalar` as the default source-shaping pipeline
- optionally add `regroup` next
- optionally add `cse` afterward for experiments

Internally, `scalar` still performs the old ordered steps:

- scalarization
- indexed-access hoisting
- direct-index rewrite

This keeps the implementation modular while presenting a cleaner public interface.

The current `regroup` pass is intentionally narrow. It only handles repeated power chains that commonly appear in ERI prefactors, for example:

- `S1_f * S1_f`
- `S2_f * S2_f * S2_f`
- `inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f`

That makes it a low-risk first step toward more semantic regrouping without turning it into another broad, register-hungry CSE pass.

Historical ablations that compared `scalarize`, `scalarize+hoist`, and `scalarize+hoist+rewrite` are still useful for understanding which substep mattered, but they are no longer the recommended public API.

## Why Generated Kernels Live In `.inc` Files

The current auto-generated DDDD kernel variants are emitted as source fragments under:

- `src/gpu/generated/*.inc`

and then included from `src/gpu/EriCoulomb.cu`.

This is intentional.

The generated variants are not meant to be independent CUDA translation units yet. They are better viewed as:

- generated kernel bodies / definitions
- kept separate from the handwritten `EriCoulomb.cu`
- easy to regenerate, diff, and replace

So `.inc` here means:

- an include fragment
- source that is pulled into another file with `#include`
- not a standalone file that is directly compiled into its own `.o`

This keeps the handwritten source cleaner while still letting the generated kernels participate in the normal `EriCoulomb.cu` build.

## Why `src/gpu/generated` Also Needs A `Makefile` And A Stub `.cpp`

This is not because the `.inc` files themselves need compiling.

It is because the current top-level VeloxChem build system assumes that every source subdirectory behaves like a normal buildable module:

- the top-level `src/Makefile` recursively enters every subdirectory with `make --directory=...`
- the final shared-library link step also expects object files matching patterns such as `gpu/generated/*.o`

If `src/gpu/generated/` only contains `.inc` files, two problems appear:

1. recursive `make` fails if the directory has no `Makefile`
2. final linking may fail because `gpu/generated/*.o` matches nothing

To make the existing build system accept this directory, we add:

- `src/gpu/generated/Makefile`
- `src/gpu/generated/generated_stub.cpp`

The stub source is only a build-system placeholder. Its job is to produce one harmless object file so the link step can continue normally.

So the current structure should be read as:

- `.inc` files are the real generated kernel content
- `generated/Makefile` and `generated_stub.cpp` are build-system adapters
- they exist to fit the current VeloxChem directory-wide make/link conventions

In a future, more compiler-like setup, generated kernels might instead be:

- emitted as standalone `.cu` files with dedicated compilation rules, or
- inserted through a more explicit code-generation stage

But for the current repository structure, `.inc` fragments plus a minimal generated-directory adapter are the lowest-friction solution.

## Why The Safety Checks Exist

The risky part is not scalarizing `float x[3]` into `x0/x1/x2`. That rewrite is local and value-preserving.

The risky part is hoisting a dynamic read like:

```cpp
PQ_f[c0]
```

into:

```cpp
const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
```

This is only correct if the value of `c0` does not change before the uses replaced by `PQ_c0_f`.

That is why the tool now checks that:

- the index has a single local definition in the same block
- the index is not redefined later in that block
- the alias is inserted after the index definition

If those conditions are not met, the tool keeps the original indexed access.

There is one important extension for the DDDD kernels:

- some indices such as `a0/a1/b0/b1` are stored in `__shared__` variables
- they are assigned in an earlier initialization block
- the kernel executes `__syncthreads()` before the later loop body that uses them

The current pass treats this as a valid ancestor-definition pattern for hoisting. In other words, it allows a limited “definition in one block, use in a later sibling region” case when the source structure matches:

1. `__shared__` variable
2. one visible defining statement in an ancestor region
3. synchronization before the rewritten use region

## Shared-Memory Risk Boundary

The `__shared__` rule is a CUDA-aware approximation, not a full correctness proof.

It is intended for the current DDDD kernel structure where:

- a small set of shared indices is initialized once
- a block-wide `__syncthreads()` follows
- a later loop body repeatedly reads those indices

This approximation can become unsafe if any of the following happens after the synchronization point and before the rewritten use region finishes:

- the same shared variable is written again
- different control-flow paths write different values
- additional synchronization structure changes which writes are guaranteed visible
- the rewritten region spans code where the shared value is no longer invariant

In short:

- `shared + __syncthreads()` is not automatically sufficient
- it is only a reasonable rule when the synchronized shared value remains stable for the whole rewritten region

The current pass does not yet prove that stability globally. It only recognizes the common “ancestor definition + barrier + later use” pattern and assumes no later conflicting writes inside the relevant region.

## Compiler Concepts Behind This

These checks are a lightweight version of several compiler analyses.

### `def-use`

`def-use` means “where is a value defined, and where is that same definition used?”

For `c0`, we want to know:

- where `c0` gets its value
- which reads of `PQ_f[c0]` depend on that definition

If we cannot identify a stable definition for `c0`, we should not hoist.

### `dominance`

A definition `dominates` a use if every path to that use goes through the definition first.

For an alias like `PQ_c0_f`, we need:

- the definition of `c0` to dominate the alias
- the alias to dominate every rewritten use

The current script does not do full control-flow dominance analysis. Instead, it uses a much simpler rule:

- stay inside one lexical block
- only hoist after the unique local definition

That is weaker than a real compiler pass, but much safer than unrestricted text rewriting.

### `invariance`

An expression is invariant over a region if its inputs do not change in that region.

For example, if `c0`, `PQ0_f`, `PQ1_f`, and `PQ2_f` stay fixed, then `PQ_c0_f` is invariant and can be reused.

If any of those values can change, hoisting may be wrong.

### `SSA`

`SSA` means `Static Single Assignment`: each variable version is assigned once.

Instead of:

```cpp
c0 = ...
x = PQ[c0]
c0 = ...
y = PQ[c0]
```

SSA-like reasoning separates them conceptually into:

```cpp
c0_1 = ...
x = PQ[c0_1]
c0_2 = ...
y = PQ[c0_2]
```

That makes it much easier to see whether a hoisted alias still refers to the right value.

This tool does not build full SSA form, but its “single definition, no redefinition” rule is essentially a restricted SSA-style safety condition.

## Why This Matters For VeloxChem

The DDDD kernels are full of patterns like:

- `r_k_f[c0]`
- `r_l_f[d1]`
- `PQ_f[a0]`

These are good scalarization candidates because:

- the arrays are tiny
- the indices are usually local cartesian-component selectors
- repeated indexed reads can become expensive and can lengthen value live ranges

But those same patterns are only safe to hoist when the selector values stay fixed.

So this tooling is not just “pretty-printing” the code. It is already doing a small compiler-style transformation, and compiler-style transformations need semantic guards.

## Current CSE Scope

The first CSE pass is intentionally narrow.

It only looks inside long `const float eri_ijkl_f = ...;` assignments and extracts exact repeated parenthesized subexpressions, for example:

```cpp
(PB_0_f * pq_b1 + PB_1_f * pq_b0)
```

when the same parenthesized expression appears multiple times in the same RHS.

## Reading `ptxas` / Resource Usage

When a scalarization or CSE pass "works", the effect is often visible in CUDA resource usage before it is visible in SASS.

The most useful fields for the current kernels are:

- `REG`: registers used per thread
- `STACK`: stack-frame size
- `LOCAL`: thread-local memory usage
- `SHARED`: shared memory usage per block

For these DDDD kernels, a good first rule of thumb is:

- lower `REG` is often good when it does not increase `LOCAL`
- nonzero `LOCAL` or `STACK` is usually a warning sign for spills or extra per-thread state
- much larger `SHARED` can also hurt occupancy even if `REG` improves

### Current Observations

From `cuobjdump --dump-resource-usage` on the current FP32 DDDD variants:

- `DDDD34_FP32`: `REG 93`, `STACK 0`, `SHARED 3248`, `LOCAL 0`
- `DDDD34_FP32_scalarized`: `REG 78`, `STACK 0`, `SHARED 3248`, `LOCAL 0`
- `DDDD34_FP32_auto_scalarized`: `REG 77`, `STACK 0`, `SHARED 3248`, `LOCAL 0`
- `DDDD34_FP32_auto_scalarized_cse`: `REG 80`, `STACK 0`, `SHARED 3248`, `LOCAL 0`
- `DDDD21_FP32`: `REG 78`, `STACK 0`, `SHARED 3212`, `LOCAL 0`
- `DDDD21_FP32_scalarized`: `REG 64`, `STACK 0`, `SHARED 3212`, `LOCAL 0`
- `DDDD21_FP32_shared_staged`: `REG 64`, `STACK 0`, `SHARED 19596`, `LOCAL 0`

These numbers already show several useful conclusions:

- scalarization is clearly changing the low-level kernel shape
- for `DDDD34`, scalarization reduces register pressure sharply: `93 -> 78 -> 77`
- the current exact local CSE for `DDDD34` slightly raises register usage: `77 -> 80`
- no current FP32 DDDD variant shows `STACK` or `LOCAL` growth, so these changes are not caused by obvious spills to local memory
- `DDDD21_shared_staged` keeps the same register count as `DDDD21_scalarized`, but pays a much larger shared-memory cost

So for the current kernels, `ptxas` already supports the benchmark results:

- `scalarize` is effective at the low level
- the current `DDDD34` CSE pass is correct but not beneficial
- shared staging should be treated as a different occupancy/resource tradeoff, not just a register optimization

### Why Lower `REG` Can Help

For these kernels, a lower register count often means:

- shorter live ranges
- fewer simultaneously live temporaries
- better occupancy headroom
- less stress on register allocation and instruction scheduling

That is exactly what we expect from:

- small-array scalarization
- hoisting repeated indexed accesses once instead of re-materializing them many times

### Why CSE Can Raise `REG`

CSE is not automatically a win.

If a repeated subexpression is extracted into a temporary such as `cse0_f`, the compiler may need to keep that value alive across a larger region. In other words:

- repeated computation may go down
- but live range may go up
- and register pressure may increase

That appears to be what happens in the current `DDDD34_FP32_auto_scalarized_cse` variant.

The practical rule is:

- if CSE lowers repeated work and does not raise register pressure too much, it may help
- if CSE creates long-lived temporaries, it can be neutral or slower even when it reduces source-level duplication

This is useful because it captures the easiest, lowest-risk repeated pieces after scalarization, but it is not yet a full algebraic optimizer.

In particular, it does not yet infer larger semantic groups like:

- `pa_pb_mix`
- `qc_qd_mix`
- `pqab_mix`
- `pqcd_mix`

unless those larger groups appear as exact repeated subexpressions in the source.

## What Still Is Not Covered

The current pass still does not perform:

- full control-flow analysis
- full dominance analysis across branches/loops
- global value numbering
- common subexpression elimination
- expression splitting based on register-pressure heuristics

Those are the natural next steps if we want to evolve this into a more compiler-like optimization pipeline.
