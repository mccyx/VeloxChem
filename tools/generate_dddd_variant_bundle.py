#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "src/gpu/EriCoulomb.cu"
DEFAULT_HPP = REPO_ROOT / "src/gpu/EriCoulomb.hpp"
DEFAULT_GEN_DIR = REPO_ROOT / "src/gpu/generated"
PIPELINE = REPO_ROOT / "tools/kernel_opt_pipeline.py"
AST_PIPELINE_WRAPPER = REPO_ROOT / "tools/run_kernel_opt_pipeline_ast.sh"
DEFAULT_DDDD_V2_INDICES = range(26)


def sanitize_tag(tag):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", tag.strip())


def parse_indices(indices_text):
    indices = []
    seen = set()
    for raw_part in indices_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            if hi < lo:
                raise ValueError("Invalid descending index range: %s" % part)
            values = range(lo, hi + 1)
        else:
            values = [int(part)]
        for value in values:
            if value < 0:
                raise ValueError("Kernel index must be non-negative: %d" % value)
            if value not in seen:
                seen.add(value)
                indices.append(value)
    if not indices:
        raise ValueError("No kernel indices selected")
    return indices


def functions_from_set(function_set, indices_text):
    indices = parse_indices(indices_text)
    precisions = []
    if function_set in {"dddd-v2-fp32", "dddd-v2-all"}:
        precisions.append("FP32")
    if function_set in {"dddd-v2-fp64", "dddd-v2-all"}:
        precisions.append("FP64")
    return [
        "computeCoulombFockDDDDv2_%d_%s" % (idx, precision)
        for precision in precisions
        for idx in indices
    ]


def function_to_file_stem(function_name):
    stem = function_name
    if stem.startswith("computeCoulombFock"):
        stem = stem[len("computeCoulombFock") :]
    return stem.lower()


def rename_function(function_text, old_name, new_name):
    pattern = r"\b" + re.escape(old_name) + r"(?=\s*\()"
    return re.sub(pattern, new_name, function_text, count=1)


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


def declaration_from_function(function_text):
    brace = function_text.find("{")
    if brace == -1:
        raise ValueError("Function text does not contain a body")
    decl = function_text[:brace].rstrip() + "\n;"
    return decl


def insert_generated_includes(path, include_lines):
    text = path.read_text()
    pending = [line for line in include_lines if line not in text]
    if not pending:
        return False

    matches = list(re.finditer(r'^#include "generated/[^"]+"$', text, re.MULTILINE))
    if not matches:
        raise ValueError("Could not find generated include block in %s" % path)
    last = matches[-1]
    insert_pos = last.end()
    new_text = text[:insert_pos] + "\n" + "\n".join(pending) + text[insert_pos:]
    path.write_text(new_text)
    return True


def resolve_source_path(explicit_source, function_name):
    if explicit_source:
        return Path(explicit_source)

    needle = function_name + "("
    candidates = []
    for path in sorted((REPO_ROOT / "src/gpu").rglob("*")):
        if path.suffix not in {".cu", ".inc"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if needle in text:
            if "{" in text[text.find(needle) : text.find(needle) + 4000]:
                candidates.append(path)

    if not candidates:
        raise ValueError("Could not locate source file for %s" % function_name)

    preferred = [path for path in candidates if "generated" in str(path)]
    return preferred[0] if preferred else candidates[0]


def run_pipeline(source_path, function_name, passes, matcher):
    if matcher == "ast":
        cmd = [
            str(AST_PIPELINE_WRAPPER),
            "--source",
            str(source_path),
            "--function",
            function_name,
            "--passes",
            passes,
        ]
    else:
        cmd = [
            sys.executable,
            str(PIPELINE),
            "--source",
            str(source_path),
            "--function",
            function_name,
            "--passes",
            passes,
            "--matcher",
            matcher,
        ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kernel optimization pipeline failed for %s with exit code %d.\n"
            "Command: %s\n"
            "stdout:\n%s\n"
            "stderr:\n%s"
            % (
                function_name,
                result.returncode,
                " ".join(cmd),
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )
    return result.stdout


def benchmark_buffer_name(function_name, tag):
    stem = function_to_file_stem(function_name)
    return "%s_%s" % (stem, sanitize_tag(tag))


def dddd_launch_args(function_name, target_name):
    if function_name.endswith("_FP64"):
        return """        {target}, d_d_prim_info, static_cast<uint32_t>(d_prim_count), d_dd_mat_D,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table, d_boys_func_ft, d_prec_cut_ij_tile""".format(target=target_name)

    return """        {target}, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile""".format(target=target_name)


def emit_benchmark_snippet(function_name, new_name, tag):
    return emit_benchmark_markdown(function_name, new_name, tag)


def emit_host_snippet(function_name, new_name, tag):
    old_buf = benchmark_buffer_name(function_name, "old")
    new_buf = benchmark_buffer_name(function_name, tag)
    mix_buf = benchmark_buffer_name(function_name, "%s_mix" % tag)
    label = new_name.replace("computeCoulombFock", "")
    old_args = dddd_launch_args(function_name, "d_mat_J3_%s" % old_buf)
    new_args = dddd_launch_args(function_name, "d_mat_J3_%s" % new_buf)
    return """// {label}
double* d_mat_J3_{old_buf} = nullptr;
double* d_mat_J3_{new_buf} = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_{old_buf}, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_{new_buf}, dd_prim_pair_count_local * sizeof(double)));

std::vector<double> h_mat_J3_{old_buf}(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_{new_buf}(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_{mix_buf}(dd_prim_pair_count_local, 0.0);

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_{old_buf}, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::{function_name}<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
{old_args});
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("{function_name} baseline",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}}

gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_{new_buf}, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{{
    GpuEventHandle start_event;
    GpuEventHandle stop_event;
    createGpuEvent(&start_event);
    createGpuEvent(&stop_event);
    const auto start = std::chrono::steady_clock::now();
    recordGpuEvent(start_event, stream);
    gpu::{new_name}<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
{new_args});
    recordGpuEvent(stop_event, stream);
    synchronizeGpuEvent(stop_event);
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("{new_name}",
        std::chrono::duration<double, std::milli>(end - start).count(),
        elapsedGpuEventMs(start_event, stop_event));
    destroyGpuEvent(start_event);
    destroyGpuEvent(stop_event);
}}

gpuSafe(gpuMemcpyAsync(h_mat_J3_{old_buf}.data(), d_mat_J3_{old_buf},
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_{new_buf}.data(), d_mat_J3_{new_buf},
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_{mix_buf}.size(); ++idx)
{{
    h_mat_J3_{mix_buf}[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_{old_buf}[idx] + h_mat_J3_{new_buf}[idx];
}}

check_J_against_ref("{new_name} contribution", h_mat_J3_{new_buf}, h_mat_J3_{old_buf},
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("{new_name} mixed result (vs ref)", h_mat_J3_{mix_buf}, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("{new_name} mixed result (vs split26 v2 baseline)", h_mat_J3_{mix_buf}, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);

gpuSafe(gpuFree(d_mat_J3_{old_buf}));
gpuSafe(gpuFree(d_mat_J3_{new_buf}));
""".format(
        label=label,
        old_buf=old_buf,
        new_buf=new_buf,
        mix_buf=mix_buf,
        function_name=function_name,
        new_name=new_name,
        old_args=old_args,
        new_args=new_args,
    )


def emit_benchmark_markdown(function_name, new_name, tag):
    old_buf = benchmark_buffer_name(function_name, "old")
    new_buf = benchmark_buffer_name(function_name, tag)
    mix_buf = benchmark_buffer_name(function_name, "%s_mix" % tag)
    label = new_name.replace("computeCoulombFock", "")
    host_snippet = emit_host_snippet(function_name, new_name, tag)
    return """### {label}

Device buffers:
```cpp
double* d_mat_J3_{old_buf} = nullptr;
double* d_mat_J3_{new_buf} = nullptr;
gpuSafe(gpuMalloc(&d_mat_J3_{old_buf}, dd_prim_pair_count_local * sizeof(double)));
gpuSafe(gpuMalloc(&d_mat_J3_{new_buf}, dd_prim_pair_count_local * sizeof(double)));
```

Host buffers:
```cpp
std::vector<double> h_mat_J3_{old_buf}(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_{new_buf}(dd_prim_pair_count_local, 0.0);
std::vector<double> h_mat_J3_{mix_buf}(dd_prim_pair_count_local, 0.0);
```

Old-kernel timing block:
```cpp
gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_{old_buf}, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{{
    const auto start = std::chrono::steady_clock::now();
    gpu::{function_name}<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_{old_buf}, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    gpuSafe(gpuStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("{function_name} baseline",
        std::chrono::duration<double, std::milli>(end - start).count());
}}
```

New-kernel timing block:
```cpp
gpu::zeroData<<<zero_num_blocks, zero_threads_per_block, 0, stream>>>(
    d_mat_J3_{new_buf}, static_cast<uint32_t>(dd_prim_pair_count_local));
gpuSafe(gpuStreamSynchronize(stream));
{{
    const auto start = std::chrono::steady_clock::now();
    gpu::{new_name}<<<dd_dispatch_num_blocks, dd_dispatch_threads_per_block, 0, stream>>>(
        d_mat_J3_{new_buf}, d_d_prim_info_f, static_cast<uint32_t>(d_prim_count), d_dd_mat_D_f,
        d_dd_first_inds_local, d_dd_second_inds_local, d_dd_pair_data_local_f, static_cast<uint32_t>(dd_prim_pair_count_local),
        d_dd_first_inds, d_dd_second_inds, d_dd_pair_data_f, static_cast<uint32_t>(dd_prim_pair_count),
        d_boys_func_table_f, d_boys_func_ft_f, d_prec_cut_ij_tile, d_screen_cut_ij_tile);
    gpuSafe(gpuStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    append_kernel_timing("{new_name}",
        std::chrono::duration<double, std::milli>(end - start).count());
}}
```

Memcpy + reconstruction:
```cpp
gpuSafe(gpuMemcpyAsync(h_mat_J3_{old_buf}.data(), d_mat_J3_{old_buf},
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuMemcpyAsync(h_mat_J3_{new_buf}.data(), d_mat_J3_{new_buf},
    dd_prim_pair_count_local * sizeof(double), gpuMemcpyDeviceToHost, stream));
gpuSafe(gpuStreamSynchronize(stream));

for (size_t idx = 0; idx < h_mat_J3_{mix_buf}.size(); ++idx)
{{
    h_mat_J3_{mix_buf}[idx] = h_mat_J2_v2_2kernels[idx] - h_mat_J3_{old_buf}[idx] + h_mat_J3_{new_buf}[idx];
}}
```

Checks:
```cpp
check_J_against_ref("{new_name} contribution", h_mat_J3_{new_buf}, h_mat_J3_{old_buf},
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("{new_name} mixed result (vs ref)", h_mat_J3_{mix_buf}, h_mat_J2_ref,
    (uint32_t)dd_prim_pair_count_local);
check_J_against_ref("{new_name} mixed result (vs split26 v2 baseline)", h_mat_J3_{mix_buf}, h_mat_J2_v2_2kernels,
    (uint32_t)dd_prim_pair_count_local);
```

Complete host block with CUDA/HIP event timing and gpuFree:
```cpp
{host_snippet}
```
""".format(
        label=label,
        old_buf=old_buf,
        new_buf=new_buf,
        mix_buf=mix_buf,
        function_name=function_name,
        new_name=new_name,
        host_snippet=host_snippet.rstrip(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate DDDD kernel variants, declaration fragments, and benchmark snippets."
    )
    parser.add_argument(
        "--functions",
        default="",
        help="Comma-separated kernel names, e.g. computeCoulombFockDDDDv2_2_FP32,computeCoulombFockDDDDv2_9_FP32",
    )
    parser.add_argument(
        "--function-set",
        choices=["dddd-v2-fp32", "dddd-v2-fp64", "dddd-v2-all"],
        help="Convenience selector for DDDD v2 kernels. Uses --indices, default 0-25.",
    )
    parser.add_argument(
        "--indices",
        default="0-25",
        help="Comma-separated indices/ranges used with --function-set, e.g. 0-25 or 2,4,16.",
    )
    parser.add_argument("--passes", required=True, help="Pass list for kernel_opt_pipeline.py")
    parser.add_argument("--tag", required=True, help="Suffix tag for generated variant names")
    parser.add_argument(
        "--source",
        help="Optional source file containing the baseline kernels. If omitted, the script auto-locates the function.",
    )
    parser.add_argument("--generated-dir", default=str(DEFAULT_GEN_DIR), help="Directory for generated .inc fragments")
    parser.add_argument("--patch-includes", action="store_true", help="Patch EriCoulomb.cu/EriCoulomb.hpp include lists")
    parser.add_argument(
        "--matcher",
        choices=["regex", "ast"],
        default="regex",
        help="Matcher backend used by kernel_opt_pipeline.py",
    )
    parser.add_argument(
        "--combine-output",
        action="store_true",
        help="Write one implementation, one declaration, and one host snippet file instead of one set per kernel.",
    )
    parser.add_argument(
        "--combined-prefix",
        help="Output prefix for --combine-output. Defaults to dddd_v2_<precision>_<tag> when all kernels share one precision.",
    )
    args = parser.parse_args()

    tag = sanitize_tag(args.tag)
    generated_dir = Path(args.generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    function_names = []
    if args.function_set:
        function_names.extend(functions_from_set(args.function_set, args.indices))
    if args.functions:
        function_names.extend([name.strip() for name in args.functions.split(",") if name.strip()])
    if not function_names:
        parser.error("Either --functions or --function-set is required")

    # decl_blocks = []
    # benchmark_sections = []
    # include_files = []
    impl_include_files = []
    decl_include_files = []
    snippet_files = []
    host_include_files = []
    combined_impl_blocks = []
    combined_decl_blocks = []
    combined_host_blocks = []

    for function_name in function_names:

        source_path = resolve_source_path(args.source, function_name)
        source_text = source_path.read_text()
        transformed = run_pipeline(source_path, function_name, args.passes, args.matcher)
        transformed = ensure_full_kernel_text(source_text, function_name, transformed)
        new_name = "%s_%s" % (function_name, tag)
        renamed = rename_function(transformed, function_name, new_name)
        decl_text = declaration_from_function(renamed)
        # benchmark_sections.append(emit_benchmark_snippet(function_name, new_name, tag))

        stem = function_to_file_stem(function_name)

        if args.combine_output:
            combined_impl_blocks.append(renamed.rstrip())
            combined_decl_blocks.append(decl_text.rstrip())
            combined_host_blocks.append(emit_host_snippet(function_name, new_name, tag).rstrip())
        else:
            snippet_name = "%s_%s_benchmark.md" % (stem, tag)
            snippet_path = generated_dir / snippet_name
            snippet_path.write_text(
                "# DDDD Variant Benchmark Snippets\n\n"
                "Baseline for reconstruction: `h_mat_J2_v2_2kernels`\n\n"
                + emit_benchmark_snippet(function_name, new_name, tag)
                + "\n"
            )
            snippet_files.append(snippet_name)

            host_name = "%s_%s_host.inc" % (stem, tag)
            host_path = generated_dir / host_name
            host_path.write_text(emit_host_snippet(function_name, new_name, tag) + "\n")
            host_include_files.append(host_name)

            impl_path = generated_dir / ("%s_%s.inc" % (stem, tag))
            impl_path.write_text(renamed)
            impl_include_files.append(impl_path.name)

            decl_path = generated_dir / ("%s_%s_decl.inc" % (stem, tag))
            decl_path.write_text(decl_text + "\n")
            decl_include_files.append(decl_path.name)

    if args.combine_output:
        precisions = sorted(
            {
                match.group(1).lower()
                for name in function_names
                for match in [re.search(r"_(FP32|FP64)$", name)]
                if match
            }
        )
        precision_label = precisions[0] if len(precisions) == 1 else "mixed"
        combined_prefix = args.combined_prefix or "dddd_v2_%s_%s" % (precision_label, tag)

        impl_name = "%s.inc" % combined_prefix
        decl_name = "%s_decl.inc" % combined_prefix
        host_name = "%s_host.inc" % combined_prefix

        (generated_dir / impl_name).write_text("\n\n".join(combined_impl_blocks) + "\n")
        (generated_dir / decl_name).write_text("\n\n".join(combined_decl_blocks) + "\n")
        (generated_dir / host_name).write_text("\n\n".join(combined_host_blocks) + "\n")

        impl_include_files.append(impl_name)
        decl_include_files.append(decl_name)
        host_include_files.append(host_name)

    # decl_name = "dddd_variant_decls_%s.inc" % tag
    # decl_path = generated_dir / decl_name
    # decl_path.write_text("\n\n".join(decl_blocks) + "\n")

    # snippet_name = "dddd_variant_benchmark_snippets_%s.md" % tag
    # snippet_path = generated_dir / snippet_name
    # snippet_path.write_text(
    #     "# DDDD Variant Benchmark Snippets\n\n"
    #     "Baseline for reconstruction: `h_mat_J2_v2_2kernels`\n\n"
    #     + "\n\n".join(benchmark_sections)
    #     + "\n"
    # )

    if args.patch_includes:
        eri_coulomb_cu = REPO_ROOT / "src/gpu/EriCoulomb.cu"
        eri_coulomb_hpp = REPO_ROOT / "src/gpu/EriCoulomb.hpp"
        insert_generated_includes(
            eri_coulomb_cu,
            ['#include "generated/%s"' % include_name for include_name in impl_include_files],
        )
        insert_generated_includes(
            eri_coulomb_hpp,
            ['#include "generated/%s"' % include_name for include_name in decl_include_files],
        )

    print("Generated implementation fragments:")
    for include_name in impl_include_files:
        print("  src/gpu/generated/%s" % include_name)

    print("Generated declaration fragments:")
    for include_name in decl_include_files:
        print("  src/gpu/generated/%s" % include_name)

    print("Generated benchmark snippet guides:")
    for snippet_name in snippet_files:
        print("  src/gpu/generated/%s" % snippet_name)

    print("Generated host benchmark fragments:")
    for host_name in host_include_files:
        print("  src/gpu/generated/%s" % host_name)


if __name__ == "__main__":
    main()
