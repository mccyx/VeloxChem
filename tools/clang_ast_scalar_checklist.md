# Clang AST Scalar Checklist

## Goal
Replace the regex-based matching layer in `tools/kernel_opt_pipeline.py` for the `scalar` pass with an AST-based matcher, while keeping the overall source-to-source workflow.

## Why
- Current scalar matching depends on a narrow regex for `const float name[3] = {...};`.
- It misses `split26_v2` arrays such as `r_k[3]`, `r_l[3]`, and `PQ[3]` because formatting differs.
- It does not support arrays like `F8_t[5]`.
- Old kernels such as `DDDD21` and `DDDD26` benefited because their declarations happened to match the regex exactly.

## Phase 0: Environment probe
- [ ] On a Clang-capable node, verify that `cc -Xclang -ast-dump=json` works.
- [ ] Prefer `nid002892` with:
  - `ml swap PrgEnv-nvidia PrgEnv-cray`
  - `ml load cce/19.0.0`
- [ ] Record the minimal compile flags needed to parse one target kernel.
- [ ] Confirm whether parsing the `.inc` content directly is easier than parsing the full `.cu` translation unit.

## Phase 1: Minimal AST prototype
- [ ] Add a small helper script, e.g. `tools/clang_ast_dump.py`.
- [ ] Input: a source file plus a function name.
- [ ] Output: JSON AST for the function region or a filtered AST subtree.
- [ ] First test target:
  - `computeCoulombFockDDDDv2_16_FP32`
  - source file: `src/gpu/generated/dddd_split26_v2_fp32.inc`
- [ ] Verify that the AST contains:
  - `FunctionDecl`
  - `VarDecl`
  - `ArraySubscriptExpr`
  - expression nodes needed for rewrite anchoring

## Phase 2: AST matcher for scalar targets
- [ ] Add an AST-side collector, e.g. `tools/clang_ast_scalar.py`.
- [ ] Detect local array declarations for scalarization:
  - `const float name[3] = {...}`
  - later extend to `const float name[5] = {...}`
- [ ] Record for each target array:
  - source range of the declaration
  - array name
  - element count
  - initializer element source ranges
- [ ] Confirm this detects:
  - `r_k[3]`
  - `r_l[3]`
  - `PQ[3]`
- [ ] Confirm whether `F8_t[5]` appears in a form we can safely scalarize in the first iteration.

## Phase 3: AST-based source rewrite for scalar
- [ ] Implement declaration rewrite using source ranges instead of regex.
- [ ] Generate scalar declarations like:
  - `PQ0`, `PQ1`, `PQ2`
- [ ] Keep the rest of the existing pass pipeline unchanged for the first iteration.
- [ ] Wire the new path behind a flag first, e.g. `--matcher ast`.
- [ ] Preserve the old regex path as fallback during migration.

## Phase 4: Hoist integration
- [ ] Reuse AST results or source scans to collect `ArraySubscriptExpr` uses.
- [ ] For arrays already scalarized, generate hoisted aliases for dynamic indices.
- [ ] Keep the current scalar -> hoist -> rewrite ordering.
- [ ] First support `[3]` arrays only.
- [ ] Defer `[5]` hoist support until scalarization for `[5]` is validated.

## Phase 5: Validation on known cases
- [ ] Re-run the pipeline on old kernels that previously benefited:
  - `DDDD21`
  - `DDDD26`
  - `DDDD34` if useful
- [ ] Confirm the AST path reproduces the old successful scalarization.
- [ ] Re-run on `v2` kernels:
  - `v2_2`
  - `v2_16`
  - `v2_25`
- [ ] Manually inspect rewritten source to confirm small arrays are truly scalarized.

## Phase 6: Performance validation
- [ ] Regenerate one or two variants only at first.
- [ ] Rebuild and run `vlx guanine-8.inp`.
- [ ] Compare:
  - baseline
  - `auto_s`
- [ ] Only add `regroup` back after AST-based scalar is confirmed to materially change source.

## Notes
- Keep the scope on `scalar` first; do not migrate `regroup` or `cse` yet.
- The immediate success condition is not speedup yet; it is proving that AST-based scalar actually changes `v2` source in the intended way.
- Once that is true, timing data becomes meaningful again.
