#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

from generate_resplit_mp_bundle import (
    DEFAULT_GEN_DIR,
    REPO_ROOT,
    AST_PIPELINE_WRAPPER,
    declaration_from_function,
    ensure_full_kernel_text,
    extract_function,
    generate_fp32_code,
    generate_fp64_code,
    insert_generated_includes,
    rename_function,
)


DEFAULT_SOURCE = REPO_ROOT / "src/gpu/EriCoulomb.cu"

SPECS = {
    "DDSD": {
        "is_dd": True,
        "f_array": "F6_t",
        "variants": {
            "altA": [[0, 1], [2, 3, 4, 5, 6]],
            "altB": [[0, 1], [2, 3], [4, 5, 6]],
        },
    },
    "DDPP": {
        "is_dd": True,
        "f_array": "F6_t",
        "variants": {
            "altA": [[0, 1], [2, 3, 4, 5, 6]],
            "altB": [[0, 1], [2, 3], [4, 5, 6]],
        },
    },
}


def matching_paren(text, open_pos):
    depth = 0
    for pos in range(open_pos, len(text)):
        if text[pos] == "(":
            depth += 1
        elif text[pos] == ")":
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError("Could not find matching parenthesis")


def split_top_level_terms(expr):
    terms = []
    depth = 0
    start = 0
    for pos, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0 and pos > start:
            prev = expr[pos - 1]
            if prev not in "eE":
                terms.append(expr[start:pos].strip())
                start = pos
    terms.append(expr[start:].strip())
    return [term for term in terms if term]


def replace_boys_order(function_text, f_array, max_index):
    pattern = re.compile(
        r"double\s+%s\s*\[\s*\d+\s*\]\s*;\s*\n\s*gpu::computeBoysFunction\(\s*%s\s*,([^;]+?),\s*\d+\s*,"
        % (re.escape(f_array), re.escape(f_array)),
        re.DOTALL,
    )

    def repl(match):
        return (
            "double %s[%d];\n\n        gpu::computeBoysFunction(%s,%s, %d,"
            % (f_array, max_index + 1, f_array, match.group(1), max_index)
        )

    new_text, count = pattern.subn(repl, function_text, count=1)
    if count != 1:
        raise ValueError("Could not rewrite Boys order for %s" % f_array)
    return new_text


def build_split_kernel(function_text, family, variant, part_index, f_indices, f_array):
    assign = "const double eri_ijkl = Lambda * S_ij_00 * S_kl_00 * ("
    start = function_text.find(assign)
    if start < 0:
        raise ValueError("Could not find eri_ijkl assignment in %s" % family)
    open_pos = function_text.find("(", start + len("const double eri_ijkl"))
    close_pos = matching_paren(function_text, open_pos)
    inner = function_text[open_pos + 1 : close_pos]
    terms = split_top_level_terms(inner)

    selected = []
    for term in terms:
        for idx in f_indices:
            if re.search(r"\b%s\s*\[\s*%d\s*\]" % (re.escape(f_array), idx), term):
                selected.append(term)
                break
    if not selected:
        raise ValueError("No terms selected for %s %s part %d" % (family, variant, part_index))

    new_inner = "\n\n                " + "\n\n                ".join(selected) + "\n\n            "
    new_text = function_text[: open_pos + 1] + new_inner + function_text[close_pos:]
    new_text = replace_boys_order(new_text, f_array, max(f_indices))

    old_name = "computeCoulombFock%s" % family
    new_name = "computeCoulombFock%s_%s%d" % (family, variant, part_index)
    return rename_function(new_text, old_name, new_name), new_name


def make_family_variants(source_text, family, spec):
    full_text = extract_function(source_text, "computeCoulombFock%s" % family)
    out = []
    for variant, groups in spec["variants"].items():
        for part_index, f_indices in enumerate(groups):
            base, name = build_split_kernel(full_text, family, variant, part_index, f_indices, spec["f_array"])
            base_lines = base.splitlines()
            fp64 = "\n".join(generate_fp64_code(base_lines, "%s_%s%d" % (family, variant, part_index), spec["is_dd"]))
            fp32 = "\n".join(generate_fp32_code(base_lines, "%s_%s%d" % (family, variant, part_index), spec["is_dd"]))
            out.append(
                {
                    "family": family,
                    "variant": variant,
                    "part": part_index,
                    "name": name,
                    "base": base,
                    "fp64": fp64,
                    "fp32": fp32,
                }
            )
    return out


def run_scalar_pipeline(source_path, function_name):
    cmd = [
        str(AST_PIPELINE_WRAPPER),
        "--source",
        str(source_path),
        "--function",
        function_name,
        "--passes",
        "scalar",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Scalar pipeline failed for %s\nCommand: %s\nstdout:\n%s\nstderr:\n%s"
            % (function_name, " ".join(cmd), result.stdout, result.stderr)
        )
    return result.stdout


def launch_lines(names, suffix, args, indent="            "):
    lines = []
    for name in names:
        lines.append(
            indent
            + "gpu::%s%s<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(\n%s);"
            % (name, suffix, args)
        )
    return "\n".join(lines)


def host_snippet_for_family(family, variants):
    if family == "DDSD":
        fp64_args = """                               d_mat_J2_2kernels, d_s_prim_info, static_cast<uint32_t>(s_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_sd_first_inds, d_sd_second_inds, d_sd_pair_data, static_cast<uint32_t>(sd_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile"""
        fp32_args = """                               d_mat_J2_2kernels, d_s_prim_info_f, static_cast<uint32_t>(s_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_sd_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_sd_first_inds, d_sd_second_inds, d_sd_pair_data_f, static_cast<uint32_t>(sd_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile"""
    elif family == "DDPP":
        fp64_args = """                               d_mat_J2_2kernels, d_p_prim_info, static_cast<uint32_t>(p_prim_count), d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_mat_D2,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile"""
        fp32_args = """                               d_mat_J2_2kernels, d_p_prim_info_f, static_cast<uint32_t>(p_prim_count), d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_pp_mat_D_f,
                               d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
                               d_pp_first_inds, d_pp_second_inds, d_pp_pair_data_f, static_cast<uint32_t>(pp_prim_pair_count),
                               d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile"""
    else:
        raise ValueError(family)

    by_variant = {}
    for item in variants:
        by_variant.setdefault(item["variant"], []).append(item["name"])

    blocks = []
    for variant in sorted(by_variant):
        names = by_variant[variant]
        label = "computeCoulombFock%s_%s_split_FP32_auto_s_ast compare" % (family, variant)
        check = "computeCoulombFock%s_%s_split FP32_auto_s_ast mixed result (vs ref)" % (family, variant)
        blocks.append(
            """
            gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
                d_mat_J2_2kernels, static_cast<uint32_t>(dd_prim_pair_count_local));
            gpuSafe(gpuStreamSynchronize(stream));
%s
            {
                GpuEventHandle start_event;
                GpuEventHandle stop_event;
                createGpuEvent(&start_event);
                createGpuEvent(&stop_event);
                const auto start = std::chrono::steady_clock::now();
                recordGpuEvent(start_event, stream);
%s
                recordGpuEvent(stop_event, stream);
                synchronizeGpuEvent(stop_event);
                const auto end = std::chrono::steady_clock::now();
                append_kernel_timing("%s",
                    std::chrono::duration<double, std::milli>(end - start).count(),
                    elapsedGpuEventMs(start_event, stop_event));
                destroyGpuEvent(start_event);
                destroyGpuEvent(stop_event);
            }
            gpuSafe(gpuMemcpyAsync(h_mat_J2_resplit_auto_s_ast.data(), d_mat_J2_2kernels,
                dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
            gpuSafe(gpuStreamSynchronize(stream));
            check_J_against_ref("%s", h_mat_J2_resplit_auto_s_ast, h_mat_J2_ref,
                (uint32_t)dd_prim_pair_count_local);
"""
            % (
                launch_lines(names, "_FP64", fp64_args),
                launch_lines(names, "_FP32_auto_s_ast", fp32_args),
                label,
                check,
            )
        )
    return "\n".join(blocks).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate alternative manual split candidates for DDSD/DDPP.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR))
    parser.add_argument("--skip-scalar", action="store_true")
    parser.add_argument("--patch-includes", action="store_true")
    args = parser.parse_args()

    source_text = Path(args.source).read_text()
    gen_dir = Path(args.gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    for family, spec in SPECS.items():
        all_items.extend(make_family_variants(source_text, family, spec))

    mp_sections = []
    declarations = []
    fp32_names = []
    for item in all_items:
        for label in ["base", "fp64", "fp32"]:
            text = item[label].rstrip()
            mp_sections.append("// %s %s %s%d %s\n%s" % (item["family"], item["variant"], item["family"], item["part"], label.upper(), text))
            declarations.append(declaration_from_function(text))
        fp32_names.append(item["name"] + "_FP32")

    mp_path = gen_dir / "coulomb_alt_split_mp.inc"
    decl_path = gen_dir / "coulomb_alt_split_mp_decl.inc"
    mp_path.write_text("\n\n".join(mp_sections) + "\n")
    decl_path.write_text("\n\n".join(declarations) + "\n")

    if not args.skip_scalar:
        scalar_sections = []
        scalar_declarations = []
        for function_name in fp32_names:
            transformed = run_scalar_pipeline(mp_path, function_name)
            transformed = ensure_full_kernel_text(mp_path.read_text(), function_name, transformed)
            new_name = function_name + "_auto_s_ast"
            transformed = rename_function(transformed, function_name, new_name)
            scalar_sections.append("// %s\n%s" % (new_name, transformed.rstrip()))
            scalar_declarations.append(declaration_from_function(transformed))

        scalar_path = gen_dir / "coulomb_alt_split_mp_auto_s_ast.inc"
        scalar_decl_path = gen_dir / "coulomb_alt_split_mp_auto_s_ast_decl.inc"
        scalar_path.write_text("\n\n".join(scalar_sections) + "\n")
        scalar_decl_path.write_text("\n\n".join(scalar_declarations) + "\n")

    for family in SPECS:
        host_path = gen_dir / ("coulomb_alt_split_%s_host.inc" % family.lower())
        host_path.write_text(host_snippet_for_family(family, [item for item in all_items if item["family"] == family]))

    if args.patch_includes:
        insert_generated_includes(
            REPO_ROOT / "src/gpu/EriCoulomb.cu",
            [
                '#include "generated/coulomb_alt_split_mp.inc"',
                '#include "generated/coulomb_alt_split_mp_auto_s_ast.inc"',
            ]
            if not args.skip_scalar
            else ['#include "generated/coulomb_alt_split_mp.inc"'],
        )
        insert_generated_includes(
            REPO_ROOT / "src/gpu/EriCoulomb.hpp",
            [
                '#include "generated/coulomb_alt_split_mp_decl.inc"',
                '#include "generated/coulomb_alt_split_mp_auto_s_ast_decl.inc"',
            ]
            if not args.skip_scalar
            else ['#include "generated/coulomb_alt_split_mp_decl.inc"'],
        )

    print("Wrote %s" % mp_path)
    print("Wrote %s" % decl_path)
    if not args.skip_scalar:
        print("Wrote %s" % (gen_dir / "coulomb_alt_split_mp_auto_s_ast.inc"))
        print("Wrote %s" % (gen_dir / "coulomb_alt_split_mp_auto_s_ast_decl.inc"))
    print("Wrote host snippets for %s" % ", ".join(SPECS))


if __name__ == "__main__":
    main()
