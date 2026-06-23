#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPLIT_SOURCE = REPO_ROOT / "re-split-EriCoulomb.cu"
DEFAULT_GEN_DIR = REPO_ROOT / "src/gpu/generated"
AST_PIPELINE_WRAPPER = REPO_ROOT / "tools/run_kernel_opt_pipeline_ast.sh"
REGEX_PIPELINE = REPO_ROOT / "tools/kernel_opt_pipeline.py"

TARGETS = [
    ("PPDD0", False),
    ("PPDD1", False),
    ("DDSD0", True),
    ("DDSD1", True),
    ("DDPP0", True),
    ("DDPP1", True),
    ("DDPD0", True),
    ("DDPD1", True),
    ("DDPD2", True),
    ("DDPD3", True),
    ("DDPD4", True),
    ("DDPD5", True),
]


def extract_function(source_text, function_name):
    prefix_pattern = re.compile(
        r"__global__\s+void\s+__launch_bounds__\s*\([^)]*\)\s*\n\s*"
        + re.escape(function_name)
        + r"\s*\(",
        re.MULTILINE,
    )
    match = prefix_pattern.search(source_text)
    if not match:
        raise ValueError("Could not locate %s" % function_name)

    start = match.start()
    brace = source_text.find("{", match.end())
    if brace == -1:
        raise ValueError("Could not locate body for %s" % function_name)

    depth = 0
    for idx in range(brace, len(source_text)):
        char = source_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source_text[start : idx + 1]

    raise ValueError("Could not locate end of %s" % function_name)


def rename_function(function_text, old_name, new_name):
    pattern = r"\b" + re.escape(old_name) + r"(?=\s*\()"
    return re.sub(pattern, new_name, function_text, count=1)


def declaration_from_function(function_text):
    brace = function_text.find("{")
    if brace == -1:
        raise ValueError("Function text does not contain a body")
    return function_text[:brace].rstrip() + "\n;"



def extract_kernel_prefix(source_text, function_name):
    pattern = re.compile(
        r"(__global__\s+void\s+__launch_bounds__\s*\([^)]*\)\s*)\n\s*"
        + re.escape(function_name)
        + r"\s*\(",
        re.MULTILINE,
    )
    match = pattern.search(source_text)
    if not match:
        raise ValueError("Could not locate kernel prefix for %s" % function_name)
    return match.group(1).rstrip()


def ensure_full_kernel_text(source_text, function_name, transformed_text):
    stripped = transformed_text.lstrip()
    if stripped.startswith("__global__"):
        return transformed_text
    prefix = extract_kernel_prefix(source_text, function_name)
    return prefix + "\n" + stripped

def generate_fp64_code(base_lines, kernel_token, is_dd):
    out = []
    tile_name = "TILE_DIM_LARGE" if is_dd else "TILE_DIM"
    for line in base_lines:
        if "computeCoulombFock%s(" % kernel_token in line:
            line = line.replace(
                "computeCoulombFock%s(" % kernel_token,
                "computeCoulombFock%s_FP64(" % kernel_token,
            )

        if "_mat_Q_local" in line and "const double*" in line:
            continue
        if "_mat_Q," in line and "const double*" in line:
            continue

        if "const double    eri_threshold)" in line:
            line = line.replace(
                "const double    eri_threshold)",
                "const uint32_t* prec_cut_ij_tile)",
            )

        if "const uint32_t ij = blockDim.x" in line:
            out.append(line)
            out.append("    const uint32_t ij_tile = blockIdx.x;")
            out.append("    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];")
            continue

        if "for (uint32_t m = 0; m <" in line and tile_name in line:
            line = "    for (uint32_t m = 0; m < prec_cut; m++)"

        if "fabs(" in line and "_mat_Q_local" in line and "<= eri_threshold" in line:
            idx = line.find("|| (fabs")
            if idx != -1:
                line = line[:idx].rstrip() + ")"

        out.append(line)
    return out


def generate_fp32_code(base_lines, kernel_token, is_dd):
    out = []
    tile_name = "TILE_DIM_LARGE" if is_dd else "TILE_DIM"
    eri_accum = "ERIs[threadIdx.y] += eri_ijkl_f *" if is_dd else "ERIs[threadIdx.y][threadIdx.x] += eri_ijkl_f *"
    eri_store = "        ERIs[threadIdx.y] += (double)contrib_f;" if is_dd else "        ERIs[threadIdx.y][threadIdx.x] += (double)contrib_f;"

    vars_to_f = [
        "a_i", "a_j", "r_i", "r_j", "S_ij_00", "S1", "inv_S1", "PA_0", "PA_1", "PB_0", "PB_1",
        "a_k", "a_l", "r_k", "r_l", "S_kl_00", "S2", "inv_S2", "inv_S4", "PQ",
        "r2_PQ", "Lambda", "QC_0", "QC_1", "QD_0", "QD_1", "delta", "eri_ijkl",
    ]

    for line in base_lines:
        if "computeCoulombFock%s(" % kernel_token in line:
            line = line.replace(
                "computeCoulombFock%s(" % kernel_token,
                "computeCoulombFock%s_FP32(" % kernel_token,
            )
            out.append(line)
            continue

        if "_mat_Q_local" in line and "const double*" in line:
            continue
        if "_mat_Q," in line and "const double*" in line:
            continue

        if "const double*" in line and "mat_J" not in line:
            line = line.replace("const double*", "const float* ")
            line = re.sub(r"([a-z]_prim_info)", r"\1_f", line)
            line = re.sub(r"([a-z]{2}_mat_D)", r"\1_f", line)
            line = re.sub(r"([a-z]{2}_pair_data_local)\b", r"\1_f", line)
            line = re.sub(r"([a-z]{2}_pair_data)(?!_local)\b", r"\1_f", line)
            line = re.sub(r"(boys_func_table)", r"\1_f", line)
            line = re.sub(r"(boys_func_ft)", r"\1_f", line)

        if "const double    eri_threshold)" in line:
            line = "                       const uint32_t* prec_cut_ij_tile,\n                       const uint32_t* screen_cut_ij_tile)"

        if is_dd:
            line = line.replace("__shared__ double   delta", "__shared__ float   delta_f")
            line = line.replace("__shared__ double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;", "__shared__ float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;")
            line = line.replace("__shared__ double PA_0, PA_1;", "__shared__ float PA_0_f, PA_1_f;")
            line = line.replace("__shared__ double PA_0, PA_1, PB_0, PB_1;", "__shared__ float PA_0_f, PA_1_f, PB_0_f, PB_1_f;")
            line = line.replace("__shared__ double PA_0, PB_0;", "__shared__ float PA_0_f, PB_0_f;")
            line = line.replace("__shared__ double PA_0, PB_0, PB_1;", "__shared__ float PA_0_f, PB_0_f, PB_1_f;")
            line = line.replace("__shared__ double PB_0, PB_1;", "__shared__ float PB_0_f, PB_1_f;")
            line = line.replace("__shared__ double PB_0;", "__shared__ float PB_0_f;")
        else:
            line = line.replace("__shared__ double   delta", "__shared__ float   delta_f")
            line = line.replace("double a_i, a_j, r_i[3], r_j[3], S_ij_00, S1, inv_S1;", "float a_i_f, a_j_f, r_i_f[3], r_j_f[3], S_ij_00_f, S1_f, inv_S1_f;")
            line = line.replace("double PA_0, PA_1;", "float PA_0_f, PA_1_f;")
            line = line.replace("double PA_0, PB_0;", "float PA_0_f, PB_0_f;")
            line = line.replace("double PA_0, PB_0, PB_1;", "float PA_0_f, PB_0_f, PB_1_f;")
            line = line.replace("double PB_0, PB_1;", "float PB_0_f, PB_1_f;")
            line = line.replace("double PB_0;", "float PB_0_f;")

        line = line.replace("const double r_k[3]", "const float r_k_f[3]")
        line = line.replace("const double r_l[3]", "const float r_l_f[3]")
        line = line.replace("const double PQ[3]", "const float PQ_f[3]")
        line = line.replace("const double eri_ijkl", "const float eri_ijkl_f")
        line = re.sub(r"double\s+(F\d+_t)\[", r"float \1_f[", line)

        for var_name in vars_to_f:
            line = re.sub(r"\b%s\b" % re.escape(var_name), "%s_f" % var_name, line)

        line = re.sub(r"\b(F\d+_t)\b", r"\1_f", line)
        line = line.replace("MATH_CONST_INV_PI", "MATH_CONST_INV_PI_F")
        line = line.replace("sqrt(", "sqrtf(")
        line = line.replace("gpu::computeBoysFunction(", "gpu::computeBoysFunction_f(")

        line = re.sub(r"([a-z]_prim_info)\[", r"\1_f[", line)
        line = re.sub(r"([a-z]{2}_pair_data_local)\[", r"\1_f[", line)
        line = re.sub(r"([a-z]{2}_pair_data)\[", r"\1_f[", line)
        line = re.sub(r"([a-z]{2}_mat_D)\[", r"\1_f[", line)
        line = line.replace("boys_func_table,", "boys_func_table_f,")
        line = line.replace("boys_func_ft)", "boys_func_ft_f)")

        if "const uint32_t ij = blockDim.x" in line:
            out.append(line)
            out.append("    const uint32_t ij_tile = blockIdx.x;")
            out.append("    const uint32_t prec_cut = prec_cut_ij_tile[ij_tile];")
            out.append("    const uint32_t screen_cut = screen_cut_ij_tile[ij_tile];")
            continue

        if "for (uint32_t m = 0; m <" in line and tile_name in line:
            line = "    for (uint32_t m = prec_cut; m < screen_cut; m++)"

        if "fabs(" in line and "_mat_Q_local" in line and "<= eri_threshold" in line:
            idx = line.find("|| (fabs")
            if idx != -1:
                line = line[:idx].rstrip() + ")"

        if "inv_S1_f = " in line and "1.0" in line:
            line = "            inv_S1_f = (float) (1.0 / (double)S1_f);" if is_dd else "        inv_S1_f = (float) (1.0 / (double)S1_f);"

        if eri_accum in line:
            d_match = re.search(r"([a-z]{2}_mat_D_f)", line)
            d_arr = d_match.group(1) if d_match else "mat_D_f"
            out.append("        const float D_f = %s[kl];" % d_arr)
            if "* 2.0" in line or "* 2.0f" in line:
                out.append("        const float contrib_f = eri_ijkl_f * D_f * 2.0f;")
            else:
                out.append("        const float sym_f = (k != l) ? 2.0f : 1.0f;")
                out.append("        const float contrib_f = eri_ijkl_f * D_f * sym_f;")
            out.append(eri_store)
            continue

        if "mat_J" not in line and "ERIs" not in line and "inv_S1_f =" not in line:
            line = re.sub(r"(?<![a-zA-Z0-9_])(\d+\.\d+)(?![fF0-9])", r"\1f", line)
            line = line.replace("double J_ij = 0.0f;", "double J_ij = 0.0;")

        out.append(line)

    return out


def make_resplit_variants(source_text, token, is_dd):
    old_name = "computeCoulombFock%s" % token
    new_base = "computeCoulombFock%s_resplit" % token
    base_text = extract_function(source_text, old_name)
    base_lines = base_text.splitlines()

    fp64 = "\n".join(generate_fp64_code(base_lines, token, is_dd))
    fp32 = "\n".join(generate_fp32_code(base_lines, token, is_dd))

    return {
        "base": rename_function(base_text, old_name, new_base),
        "fp64": rename_function(fp64, old_name + "_FP64", new_base + "_FP64"),
        "fp32": rename_function(fp32, old_name + "_FP32", new_base + "_FP32"),
    }


def insert_generated_includes(path, include_lines):
    text = path.read_text()
    pending = [line for line in include_lines if line not in text]
    if not pending:
        return False

    matches = list(re.finditer(r'^#include "generated/[^"]+"$', text, re.MULTILINE))
    if not matches:
        raise ValueError("Could not find generated include block in %s" % path)
    insert_pos = matches[-1].end()
    path.write_text(text[:insert_pos] + "\n" + "\n".join(pending) + text[insert_pos:])
    return True


def run_scalar_pipeline(source_path, function_name, matcher):
    if matcher == "ast":
        cmd = [
            str(AST_PIPELINE_WRAPPER),
            "--source",
            str(source_path),
            "--function",
            function_name,
            "--passes",
            "scalar",
        ]
    else:
        cmd = [
            sys.executable,
            str(REGEX_PIPELINE),
            "--source",
            str(source_path),
            "--function",
            function_name,
            "--passes",
            "scalar",
            "--matcher",
            matcher,
        ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Scalar pipeline failed for %s\nCommand: %s\nstdout:\n%s\nstderr:\n%s"
            % (function_name, " ".join(cmd), result.stdout, result.stderr)
        )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Generate non-overwriting resplit MP kernels and FP32 scalar variants.")
    parser.add_argument("--resplit-source", default=str(DEFAULT_RESPLIT_SOURCE))
    parser.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR))
    parser.add_argument("--matcher", choices=["ast", "regex"], default="ast")
    parser.add_argument("--skip-scalar", action="store_true")
    parser.add_argument("--patch-includes", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.resplit_source)
    gen_dir = Path(args.gen_dir)
    source_text = source_path.read_text()
    gen_dir.mkdir(parents=True, exist_ok=True)

    mp_sections = []
    scalar_sections = []
    declarations = []
    scalar_declarations = []
    generated_fp32_names = []

    for token, is_dd in TARGETS:
        variants = make_resplit_variants(source_text, token, is_dd)
        for label in ["base", "fp64", "fp32"]:
            function_text = variants[label]
            mp_sections.append("// %s %s\n%s" % (token, label.upper(), function_text.rstrip()))
            declarations.append(declaration_from_function(function_text))
        generated_fp32_names.append("computeCoulombFock%s_resplit_FP32" % token)

    mp_path = gen_dir / "coulomb_resplit_mp.inc"
    decl_path = gen_dir / "coulomb_resplit_mp_decl.inc"
    mp_path.write_text("\n\n".join(mp_sections) + "\n")
    decl_path.write_text("\n\n".join(declarations) + "\n")

    if not args.skip_scalar:
        for function_name in generated_fp32_names:
            transformed = run_scalar_pipeline(mp_path, function_name, args.matcher)
            transformed = ensure_full_kernel_text(mp_path.read_text(), function_name, transformed)
            new_name = function_name + "_auto_s_ast"
            transformed = rename_function(transformed, function_name, new_name)
            scalar_sections.append("// %s\n%s" % (new_name, transformed.rstrip()))
            scalar_declarations.append(declaration_from_function(transformed))

        scalar_path = gen_dir / "coulomb_resplit_mp_auto_s_ast.inc"
        scalar_decl_path = gen_dir / "coulomb_resplit_mp_auto_s_ast_decl.inc"
        scalar_path.write_text("\n\n".join(scalar_sections) + "\n")
        scalar_decl_path.write_text("\n\n".join(scalar_declarations) + "\n")

    if args.patch_includes:
        insert_generated_includes(
            REPO_ROOT / "src/gpu/EriCoulomb.cu",
            [
                '#include "generated/coulomb_resplit_mp.inc"',
                '#include "generated/coulomb_resplit_mp_auto_s_ast.inc"',
            ]
            if not args.skip_scalar
            else ['#include "generated/coulomb_resplit_mp.inc"'],
        )
        insert_generated_includes(
            REPO_ROOT / "src/gpu/EriCoulomb.hpp",
            [
                '#include "generated/coulomb_resplit_mp_decl.inc"',
                '#include "generated/coulomb_resplit_mp_auto_s_ast_decl.inc"',
            ]
            if not args.skip_scalar
            else ['#include "generated/coulomb_resplit_mp_decl.inc"'],
        )

    print("Wrote %s" % mp_path)
    print("Wrote %s" % decl_path)
    if not args.skip_scalar:
        print("Wrote %s" % (gen_dir / "coulomb_resplit_mp_auto_s_ast.inc"))
        print("Wrote %s" % (gen_dir / "coulomb_resplit_mp_auto_s_ast_decl.inc"))


if __name__ == "__main__":
    main()
