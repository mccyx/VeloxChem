# Kernel Scalarization Tool

`tools/scalarize_kernel.py` is a small source-to-source helper for the DDDD-style CUDA kernels in `src/gpu/EriCoulomb.cu`.

It currently automates the safest part of the manual tuning workflow:

- scalarize `const float name[3] = {...};` into three scalar locals
- hoist repeated dynamic accesses like `PQ_f[a0]` into local aliases such as `PQ_a0_f`
- rewrite direct element reads like `r_k_f[0]` to scalar names

This is intentionally a v1 pass. It does not yet:

- extract larger common subexpressions
- split long expressions into staged temporaries
- rewrite the whole file in place

It is also intentionally conservative:

- hoisting is limited to the same lexical block
- a dynamic index must have exactly one local definition in that block
- if an index may be redefined later, the tool skips that hoist and reports why
- shared indices defined in an ancestor block are only accepted when a synchronizing `__syncthreads()` appears before the rewritten region

## Example

```bash
python3 tools/scalarize_kernel.py \
  --source src/gpu/EriCoulomb.cu \
  --function computeCoulombFockDDDD21_FP32 \
  --output /tmp/DDDD21_FP32.scalarized.cu \
  --report
```

## Intended workflow

1. Run the tool on a baseline kernel.
2. Diff the generated output against the hand-tuned version.
3. Keep the safe scalarization/hoisting pieces.
4. Add follow-up manual or scripted CSE/splitting passes.

The goal is to move the repeated “small-array scalarize + indexed-access hoist” work into a reusable step before deeper tuning.

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

## What Still Is Not Covered

The current pass still does not perform:

- full control-flow analysis
- full dominance analysis across branches/loops
- global value numbering
- common subexpression elimination
- expression splitting based on register-pressure heuristics

Those are the natural next steps if we want to evolve this into a more compiler-like optimization pipeline.
