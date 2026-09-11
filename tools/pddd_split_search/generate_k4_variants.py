#!/usr/bin/env python3

"""Generate PDDD K4 candidates by merging adjacent supplied RS kernels."""

import argparse
import json
import re
from pathlib import Path


VARIANTS = {
    "M01": {"merge": 0, "boundaries": [32, 40, 48, 63]},
    "M12": {"merge": 1, "boundaries": [16, 40, 48, 63]},
    "M23": {"merge": 2, "boundaries": [16, 32, 48, 63]},
    "M34": {"merge": 3, "boundaries": [16, 32, 40, 63]},
}
PRECISION_SUFFIXES = ("", "_FP64", "_FP32")
FUNCTION_PREFIX = "computeExchangeFockPDDD"
ERI_ASSIGNMENT_RE = re.compile(
    r"const\s+(?:double|float)\s+eri_ijkl(?:_f)?\s*=\s*"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("src/gpu/EriExchange.cu"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/gpu"),
    )
    return parser.parse_args()


def matching_delimiter(text, start, opening, closing):
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError(f"Unterminated {opening}{closing} block")


def extract_function(text, name):
    name_pos = text.find(f"\n{name}(")
    if name_pos < 0:
        raise ValueError(f"Missing function: {name}")
    start = text.rfind("__global__ void", 0, name_pos)
    if start < 0:
        raise ValueError(f"Missing declaration start: {name}")
    body_start = text.find("{", name_pos)
    end = matching_delimiter(text, body_start, "{", "}")
    return text[start : end + 1]


def extract_expression(function):
    match = ERI_ASSIGNMENT_RE.search(function)
    if not match:
        raise ValueError("Missing eri_ijkl assignment")
    end = function.find(";", match.end())
    if end < 0:
        raise ValueError("Missing eri_ijkl terminator")
    return match.end(), end, function[match.end() : end]


def outer_expression(expression):
    stripped = expression.strip()
    opening = stripped.find("(")
    if opening < 0 or not stripped.endswith(")"):
        raise ValueError("Unexpected eri_ijkl expression shape")
    closing = matching_delimiter(stripped, opening, "(", ")")
    if closing != len(stripped) - 1:
        raise ValueError("The outer eri_ijkl group is not the final expression")
    return stripped[: opening + 1], stripped[opening + 1 : closing]


def normalized(text):
    return re.sub(r"\s+", "", text)


def renamed(function, old_name, new_name):
    if function.count(old_name) != 1:
        raise ValueError(f"Expected one function-name occurrence: {old_name}")
    return function.replace(old_name, new_name, 1)


def merged_function(left, right, left_name, right_name, new_name):
    _, _, left_expression = extract_expression(left)
    right_start, right_end, right_expression = extract_expression(right)
    left_prefix, left_inner = outer_expression(left_expression)
    right_prefix, right_inner = outer_expression(right_expression)
    if normalized(left_prefix) != normalized(right_prefix):
        raise ValueError(f"Incompatible expression prefixes: {left_name}, {right_name}")

    right_inner = right_inner.strip()
    separator = "" if right_inner.startswith(("+", "-")) else "+ "
    combined = (
        "\n"
        + right_prefix
        + "\n\n"
        + left_inner.strip()
        + "\n\n                        "
        + separator
        + right_inner
        + "\n\n                    )"
    )
    result = right[:right_start] + combined + right[right_end:]
    return renamed(result, right_name, new_name)


def declaration(function):
    body_start = function.find("{")
    if body_start < 0:
        raise ValueError("Missing function body")
    return function[:body_start].rstrip() + ";"


def source_name(index, suffix):
    return f"{FUNCTION_PREFIX}{index}_RS{suffix}"


def target_name(variant, index, suffix):
    return f"{FUNCTION_PREFIX}{index}_K4_{variant}{suffix}"


def generate_variant(source_functions, variant, merge_index, suffix):
    groups = []
    source_index = 0
    target_index = 0
    while source_index < 5:
        new_name = target_name(variant, target_index, suffix)
        if source_index == merge_index:
            left_name = source_name(source_index, suffix)
            right_name = source_name(source_index + 1, suffix)
            function = merged_function(
                source_functions[left_name],
                source_functions[right_name],
                left_name,
                right_name,
                new_name,
            )
            sources = [source_index, source_index + 1]
            source_index += 2
        else:
            old_name = source_name(source_index, suffix)
            function = renamed(source_functions[old_name], old_name, new_name)
            sources = [source_index]
            source_index += 1
        groups.append({"name": new_name, "sources": sources, "function": function})
        target_index += 1

    if len(groups) != 4:
        raise ValueError(f"{variant}{suffix} generated {len(groups)} kernels")
    return groups


def main():
    args = parse_args()
    source_text = args.source.read_text()
    source_functions = {}
    for suffix in PRECISION_SUFFIXES:
        for index in range(5):
            name = source_name(index, suffix)
            source_functions[name] = extract_function(source_text, name)

    cuda_sections = []
    header_sections = []
    manifest = {"source": str(args.source), "variants": {}}
    generated_names = set()

    for variant, config in VARIANTS.items():
        manifest["variants"][variant] = {
            "merge": [config["merge"], config["merge"] + 1],
            "boundaries": config["boundaries"],
            "precisions": {},
        }
        cuda_sections.append(f"// ===== PDDD K4 {variant} =====")
        header_sections.append(f"// ===== PDDD K4 {variant} =====")
        for suffix in PRECISION_SUFFIXES:
            groups = generate_variant(
                source_functions, variant, config["merge"], suffix
            )
            precision = suffix[1:].lower() if suffix else "original"
            manifest["variants"][variant]["precisions"][precision] = []
            for group in groups:
                if group["name"] in generated_names:
                    raise ValueError(f'Duplicate generated name: {group["name"]}')
                generated_names.add(group["name"])
                cuda_sections.append(group["function"])
                header_sections.append(declaration(group["function"]))
                manifest["variants"][variant]["precisions"][precision].append(
                    {"kernel": group["name"], "rs_sources": group["sources"]}
                )

    expected_functions = len(VARIANTS) * len(PRECISION_SUFFIXES) * 4
    if len(generated_names) != expected_functions:
        raise ValueError(
            f"Expected {expected_functions} functions, got {len(generated_names)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_path = args.output_dir / "PDDD_K4_variants.cu.inc"
    header_path = args.output_dir / "PDDD_K4_variants.hpp.inc"
    manifest_path = args.output_dir / "PDDD_K4_variants.manifest.json"
    cuda_path.write_text("\n\n".join(cuda_sections) + "\n")
    header_path.write_text("\n\n".join(header_sections) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if re.search(r"\+\s*\+\s*F\d+_t", cuda_path.read_text()):
        raise ValueError("Generated source contains a duplicated expression sign")

    print(f"Generated functions: {expected_functions}")
    print(cuda_path)
    print(header_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
