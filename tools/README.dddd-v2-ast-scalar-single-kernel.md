# DDDD v2 AST-Based Scalarization for a Single Kernel

This document describes the single-kernel AST-based scalarization workflow used
for DDDD v2 Coulomb kernels in VeloxChem.

The tool reads one CUDA/C++ source fragment containing a selected kernel,
parses the kernel with a Clang AST frontend, detects local size-3 arrays such as

```cpp
const double PQ[3] = {...};
```

and rewrites them into scalar local variables. It also rewrites safe direct or
dynamic indexed accesses when the existing source-analysis checks prove that the
index value is stable enough.

The tool writes one transformed kernel implementation file. It does not modify
the original source tree in place.

## Files To Include

For the single-kernel workflow, keep these five files under the repository
`tools/` directory:

```text
tools/clang_ast_dump.py
tools/clang_ast_scalar.py
tools/kernel_ir.py
tools/kernel_opt_pipeline.py
tools/run_kernel_opt_pipeline_ast.sh
```

The recommended directory layout is:

```text
VeloxChem.mixed-precision-2/
  tools/
    clang_ast_dump.py
    clang_ast_scalar.py
    kernel_ir.py
    kernel_opt_pipeline.py
    run_kernel_opt_pipeline_ast.sh
```

The scripts assume this layout. The shell wrapper locates
`kernel_opt_pipeline.py` relative to the repository `tools/` directory, and the
Python files import each other as same-directory modules.

## Input Source Files

The workflow needs a source file containing the original kernel definition. In
the current DDDD v2 experiments, the inputs are:

```text
src/gpu/generated/dddd_split26_v2_fp32.inc
src/gpu/generated/dddd_split26_v2_fp64.inc
```

These files were prepared from the original DDDD v2 test CUDA files:

```text
v2_test_coulomb_dddd_fp32.cu
v2_test_coulomb_dddd_fp64.cu
```

and placed under `src/gpu/generated/` as `.inc` fragments.

The tool does not require the `.inc` extension specifically. The `--source`
argument can point to any file that contains the requested kernel function
definition.

## Dependencies

Required:

- Python 3.
- A Clang-compatible C++ frontend that supports `-Xclang -ast-dump=json`.
- The VeloxChem source tree.
- The baseline DDDD v2 kernel source fragments listed above.

On the PDC GH200 system, the normal VeloxChem build uses the compiler settings
from `src/Makefile.setup`, for example:

```make
USE_CUDA := true
CXX    := nvc++
DEVCC  := nvcc -gencode arch=compute_90,code=sm_90 -lineinfo --use_fast_math
```

The AST scalarization tool uses a separate Clang frontend only for parsing and
AST dumping. It does not change the normal VeloxChem build compiler.

## Environment Setup

From the repository root, load the project environment first:

```bash
cd /path/to/VeloxChem.mixed-precision-2
source env.sh
```

The `source` command runs `env.sh` in the current shell, so exported variables,
loaded modules, virtual environments, and library paths remain available for the
following commands.

## Clang Frontend Selection

The Clang frontend is selected in:

```text
tools/run_kernel_opt_pipeline_ast.sh
```

The selection order is:

1. Use `VLX_AST_COMPILER` if it is set and executable.
2. Otherwise use `clang++` from `PATH`, if available.
3. Otherwise try known Cray CCE Clang frontend paths:

```text
/opt/cray/pe/cce/19.0.0/cce-clang/aarch64/bin/clang++
/opt/cray/pe/cce/19.0.0/cce-clang/x86_64/bin/clang++
/opt/cray/pe/cce/18.0.1/cce-clang/aarch64/bin/clang++
/opt/cray/pe/cce/18.0.1/cce-clang/x86_64/bin/clang++
/opt/cray/pe/cce/18.0.0/cce-clang/aarch64/bin/clang++
/opt/cray/pe/cce/18.0.0/cce-clang/x86_64/bin/clang++
```

To set the frontend manually:

```bash
export VLX_AST_COMPILER=/path/to/clang++
```

For example, on a GH200 node the following CCE Clang frontend was used
successfully:

```bash
export VLX_AST_COMPILER=/opt/cray/pe/cce/19.0.0/cce-clang/aarch64/bin/clang++
```

The tool intentionally avoids using the Cray `cc` compiler wrapper for AST
dumping. The wrapper may try to resolve MPI/LibSci/pkg-config dependencies that
are not needed for this source-to-source parsing step.

## Usage Examples

Scalarize one FP64 DDDD v2 kernel:

```bash
cd /path/to/VeloxChem.mixed-precision-2
source env.sh

tools/run_kernel_opt_pipeline_ast.sh \
  --source src/gpu/generated/dddd_split26_v2_fp64.inc \
  --function computeCoulombFockDDDDv2_16_FP64 \
  --passes scalar \
  --output ddddv2_16_fp64_scalarized.cu
```

Scalarize one FP32 DDDD v2 kernel:

```bash
cd /path/to/VeloxChem.mixed-precision-2
source env.sh

tools/run_kernel_opt_pipeline_ast.sh \
  --source src/gpu/generated/dddd_split26_v2_fp32.inc \
  --function computeCoulombFockDDDDv2_16_FP32 \
  --passes scalar \
  --output ddddv2_16_fp32_scalarized.cu
```

The `--output` path controls where the transformed kernel is written.

For example:

```text
--output ddddv2_16_fp64_scalarized.cu
```

writes to the current working directory, while

```text
--output /tmp/ddddv2_16_fp64_scalarized.cu
```

writes to `/tmp`.

## Internal Workflow

The execution path is:

```text
run_kernel_opt_pipeline_ast.sh
  -> kernel_opt_pipeline.py --matcher ast
      -> clang_ast_dump.py
      -> clang_ast_scalar.py
      -> kernel_ir.py
```

More specifically:

1. `run_kernel_opt_pipeline_ast.sh` finds the repository root and selects the
   Clang AST frontend.
2. `kernel_opt_pipeline.py` reads the `--source` file and extracts the selected
   kernel function.
3. `clang_ast_dump.py` invokes Clang with `-Xclang -ast-dump=json`.
4. `clang_ast_scalar.py` finds local `const float name[3]` and
   `const double name[3]` declarations through the AST.
5. `kernel_opt_pipeline.py` rewrites those arrays into scalar locals.
6. The existing source-analysis logic in `kernel_ir.py` is used to rewrite safe
   indexed accesses.
7. The transformed single-kernel implementation is written to `--output`.

## Quick Checks

Check that the output file exists:

```bash
ls -lh ddddv2_16_fp64_scalarized.cu
```

Check whether size-3 local array declarations remain:

```bash
rg "const (float|double) [A-Za-z_][A-Za-z0-9_]*\\[3\\]" \
  ddddv2_16_fp64_scalarized.cu
```

No output from this command means that this class of local size-3 arrays was
removed from the transformed kernel.

Inspect the beginning of the transformed kernel:

```bash
head -n 60 ddddv2_16_fp64_scalarized.cu
```

## Notes On Paths

`run_kernel_opt_pipeline_ast.sh` uses:

```bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
```

This means the wrapper determines the repository root from its own location.
Therefore, the five scripts should remain in the repository `tools/` directory.

If they are moved elsewhere, the wrapper path logic and/or Python import path
would need to be adjusted.
