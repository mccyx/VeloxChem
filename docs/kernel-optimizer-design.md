# Kernel Optimizer Design Notes

This note accompanies:

- `tools/kernel_opt_pipeline.py`
- `tools/kernel_ir.py`
- `tools/README.kernel-opt.md`

The current goal is to move from regex-only rewriting toward a small compiler-style optimizer for generated CUDA kernels.

## The LLVM Hierarchy To Remember

LLVM users usually think in this order:

- `Module`
- `Function`
- `BasicBlock`
- `Instruction`
- `Value`

### `Module`

The top-level IR container.

It usually contains:

- function definitions
- global variables
- declarations

### `Function`

One function inside the module.

For this project, a kernel such as `computeCoulombFockDDDD21_FP32` is the natural function-level unit.

### `BasicBlock`

This is not just a `{ ... }` block in source code.

A basic block is:

- one straight-line sequence of operations
- one entry
- one exit
- ending in a control-flow terminator like `br` or `ret`

Basic blocks are the nodes of a control-flow graph.

### `Instruction`

A single IR operation, for example:

- `load`
- `store`
- `fadd`
- `fmul`
- `call`

### `Value`

Many LLVM entities are values:

- constants
- function arguments
- results of instructions

`Value` is the generic “thing another instruction can use”.

## Three Meanings Of “Block”

This is easy to mix up.

### Lexical block

A source-level `{ ... }` scope.

This controls:

- where local names are visible
- how nested scopes behave

### Basic block

A control-flow block in compiler IR.

This controls:

- dominance
- CFG structure
- branch/merge reasoning

### Analysis region

The region where we assume a value is stable enough for a transformation.

Right now our optimizer uses lexical blocks as its first analysis region because they are easier to recover directly from source.

## Why We Are Starting With A Source-Level IR

The current work is transforming generated CUDA source, not LLVM IR.

What we need first is:

- lexical block structure
- statement boundaries
- local definitions and uses
- simple lifetime checks

So instead of starting with LLVM or MLIR, we start with a smaller source-level IR in `tools/kernel_ir.py`.

That IR currently models:

- functions
- lexical blocks
- statements inside each block
- block-local `defs`
- block-local `uses`

## Concepts The Current Pass Depends On

### `def-use`

`def-use` means:

- where a value is defined
- where that definition is used

Example:

```cpp
const auto c0 = d_cart_inds[k % 6][0];
const auto x = PQ_f[c0];
```

The definition of `c0` feeds the use in `PQ_f[c0]`.

### `dominance`

A definition dominates a use if every path to that use goes through the definition first.

For a hoisted alias like:

```cpp
const float PQ_c0_f = (c0 == 0 ? PQ0_f : (c0 == 1 ? PQ1_f : PQ2_f));
```

we need the definition of `c0` to dominate that alias, and the alias to dominate the uses it replaces.

The current tool does not compute full CFG dominance. It uses a conservative approximation:

- stay inside one lexical block
- require exactly one definition statement
- reject later redefinitions

### `invariance`

An expression is invariant over a region if its inputs do not change in that region.

For example, `PQ_c0_f` is only reusable if:

- `c0`
- `PQ0_f`
- `PQ1_f`
- `PQ2_f`

stay unchanged over the rewritten region.

### `SSA`

`SSA` means `Static Single Assignment`.

Instead of:

```cpp
c0 = ...
x = PQ[c0]
c0 = ...
y = PQ[c0]
```

SSA-style reasoning separates them conceptually:

```cpp
c0_1 = ...
x = PQ[c0_1]
c0_2 = ...
y = PQ[c0_2]
```

This makes it obvious that the two uses do not depend on the same value.

The current source-level pass does not build full SSA form, but it already uses SSA-like safety rules:

- prefer a single local definition
- reject transforms when the same name is redefined later

## What `tools/kernel_ir.py` Gives Us

The new helper gives us:

- the innermost lexical block for a given position
- the ordered statements in that block
- statement-level defs/uses
- simple queries such as:
  - definition sites of a variable
  - first use after a point
  - later redefinitions

That is enough for a first real analysis-based pass.

There is also one project-specific rule on top of the pure block-local analysis:

- if an index is a `__shared__` variable
- and it is defined in an earlier ancestor-region statement
- and a `__syncthreads()` is observed before the rewritten region

then the pass may treat that definition as visible and stable for later hoisting.

This is not full CFG dominance. It is a CUDA-aware source-level approximation tailored to the current kernel structure.

## Shared-Memory Approximation And Its Risks

The current optimizer has a special-case rule for shared-memory indices such as `a0/a1/b0/b1`:

- define them in an earlier ancestor region
- synchronize with `__syncthreads()`
- reuse them in a later loop region

This rule is useful for the present kernels, but it comes with an explicit assumption:

- the shared value used as an index remains stable throughout the rewritten region

That assumption may fail if:

- the same shared variable is written again after the barrier
- different paths assign different values before later uses
- further barriers or control-flow structure change visibility/lifetime in a way the source-level analysis does not model
- the alias is reused beyond the region in which the synchronized value is actually invariant

So the current rule should be read as:

- a practical CUDA-aware dominance approximation
- not a complete shared-memory correctness proof

Future compiler-like upgrades should add stronger reasoning for:

- later shared-memory writes
- path-sensitive control flow
- barrier-aware region boundaries
- value invalidation after synchronization points

## Why Scalarization Often Helps More Than Naive CSE

The current DDDD results are a good example of an important optimizer lesson:

- reducing source-level duplication is not the same thing as reducing runtime cost
- a transformation is only useful if it improves the compiled kernel, not just the source text

### The Bundled Scalar Pipeline

The current public source-shaping entry point is the bundled `scalar` pipeline.

Internally it still performs three ordered transformations:

- scalarization
- indexed-access hoisting
- direct-index rewrite

That public bundling matches the current empirical picture:

- scalarization is the enabling first step
- indexed-access hoisting is usually the dominant performance contributor
- direct-index rewrite is mostly cleanup and canonicalization

So the implementation remains factored, but the user-facing model is now:

- run `scalar` by default
- optionally add `regroup` next
- optionally add `cse` afterward for experiments

Within that bundled stage, scalarization helps the current kernels in two related ways.

First, it replaces tiny arrays like:

```cpp
const float PQ_f[3] = {...};
```

with scalar values such as:

```cpp
const float PQ0_f = ...;
const float PQ1_f = ...;
const float PQ2_f = ...;
```

Second, indexed-access hoisting replaces repeated dynamic reads like:

```cpp
PQ_f[a0]
```

with one alias such as:

```cpp
const float PQ_a0_f = ...;
```

and then reuses that alias.

The bundled scalar pipeline often helps because it reduces:

- repeated index-selection logic
- repeated address-like expression materialization
- the number of distinct subexpressions the backend must reason about

It can also shorten some live ranges, because the backend sees a simpler value graph.

### Why Regroup Is Separate From CSE

The current pass stack now distinguishes:

- `regroup`
- `cse`

because they optimize for different outcomes.

The first `regroup` pass is intentionally narrow and structural. It targets repeated multiplicative power chains such as:

- `S1_f * S1_f`
- `S2_f * S2_f * S2_f`
- `inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f * inv_S4_f`

and rewrites them into local aliases like:

- `S1_f_pow2`
- `S2_f_pow3`
- `inv_S4_f_pow5`

This is meant as a low-risk source-shaping step for long ERI prefactors.

By contrast, the current `cse` pass extracts exact repeated parenthesized subexpressions. That can reduce source-level duplication, but it can also extend live ranges and increase register pressure.

So the intended public hierarchy is:

- `scalar`: default and broadly useful
- `regroup`: narrow structural cleanup
- `cse`: experimental exact-subexpression extraction

For the current FP32 kernels, that effect is visible in resource usage:

- `DDDD34_FP32`: `REG 93`
- `DDDD34_FP32_scalarized`: `REG 78`
- `DDDD34_FP32_auto_scalarized`: `REG 77`
- `DDDD21_FP32`: `REG 78`
- `DDDD21_FP32_scalarized`: `REG 64`

### Why Naive CSE Can Be Neutral Or Worse

Naive CSE usually assumes:

- fewer repeated operations is better

But on GPUs, another cost matters just as much:

- how many values must remain live at the same time

If a pass extracts a repeated expression into:

```cpp
const float cse0_f = ...;
```

then `cse0_f` may need to stay live across a larger slice of the kernel than the original inlined pieces did.

That means:

- fewer source-level repeats
- but potentially longer live ranges
- and therefore more registers

This is exactly why a small CSE pass can sometimes make performance worse even when it "looks cleaner."

In the current `DDDD34` experiment:

- `DDDD34_FP32_auto_scalarized`: `REG 77`
- `DDDD34_FP32_auto_scalarized_cse`: `REG 80`

with no `STACK` or `LOCAL` growth.

That suggests the current exact local CSE is not causing obvious spills. Instead, it is likely extending live ranges enough to slightly worsen register pressure and scheduling/occupancy behavior.

### The Low-Level Reading Rule

For the current kernels, a practical interpretation rule is:

- if a transform lowers `REG` without increasing `LOCAL`, it is usually promising
- if `LOCAL` or `STACK` becomes nonzero, inspect for spills first
- if `SHARED` jumps sharply, treat it as an occupancy tradeoff rather than a free optimization
- if CSE raises `REG`, it may still help, but it now has to compensate for that extra pressure with a larger reduction in real work

This is why `ptxas` resource usage is a useful intermediate checkpoint between source rewriting and full performance measurements.

## Why This Matters For VeloxChem

The generated kernels contain many expressions like:

- `r_k_f[c0]`
- `r_l_f[d1]`
- `PQ_f[a0]`

These are attractive scalarization targets because:

- the arrays are tiny
- the index values are often repeated
- the rewritten form can reduce repeated indexed access and simplify long expressions

But they are only safe to hoist when the controlling index values are stable.

That is exactly why we need analysis instead of blind rewriting.

## Current Limitation

This is still not a full compiler IR.

It does not yet model:

- true CFG basic blocks
- branch merges
- phi-like value merging
- global value numbering
- full common subexpression elimination

Those are later steps.

For now, the design goal is:

- a trustworthy local analysis layer
- a safe default scalar pipeline for tiny arrays and indexed accesses
- a foundation for later regrouping, CSE, and expression-splitting passes
