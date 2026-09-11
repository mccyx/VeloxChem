#!/usr/bin/env python3

"""Generate DDDD K19 candidates by merging supplied RS26 kernels."""

import argparse
import json
import re
from pathlib import Path


VARIANT_TAGS = {
    "static_balanced": "K19_STATIC",
    "runtime_balanced": "K19_RUNTIME",
    "light_setup": "K19_LIGHT",
}
SUFFIXES = ("", "_FP64", "_FP32")
PREFIX = "computeExchangeFockDDDD"
ERI_RE = re.compile(r"const\s+(?:double|float)\s+eri_ijkl(?:_f)?\s*=\s*")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("src/gpu/EriExchange.cu"))
    parser.add_argument("--candidates", type=Path, default=Path("tools/dddd_split_search/candidates.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/gpu"))
    return parser.parse_args()


def matching(text, start, opening, closing):
    depth = 0
    for pos in range(start, len(text)):
        depth += text[pos] == opening
        depth -= text[pos] == closing
        if depth == 0:
            return pos
    raise ValueError("Unterminated block")


def extract_function(text, name):
    name_pos = text.find(f"\n{name}(")
    if name_pos < 0:
        raise ValueError(f"Missing {name}")
    start = text.rfind("__global__ void", 0, name_pos)
    body = text.find("{", name_pos)
    return text[start : matching(text, body, "{", "}") + 1]


def expression(function):
    match = ERI_RE.search(function)
    if not match:
        raise ValueError("Missing ERI expression")
    end = function.find(";", match.end())
    return match.end(), end, function[match.end() : end]


def outer(value):
    stripped = value.strip()
    opening = stripped.find("(")
    closing = matching(stripped, opening, "(", ")")
    if closing != len(stripped) - 1:
        raise ValueError("Unexpected outer expression")
    return stripped[: opening + 1], stripped[opening + 1 : closing]


def normalized(value):
    return re.sub(r"\s+", "", value)


def renamed(function, old, new):
    if function.count(old) != 1:
        raise ValueError(f"Unexpected name count for {old}")
    return function.replace(old, new, 1)


def merge_group(functions, source_names, target_name):
    template = functions[source_names[-1]]
    start, end, template_expression = expression(template)
    common_prefix, _ = outer(template_expression)
    inners = []
    for name in source_names:
        _, _, value = expression(functions[name])
        prefix, inner = outer(value)
        if normalized(prefix) != normalized(common_prefix):
            raise ValueError(f"Incompatible prefixes in {source_names}")
        inner = inner.strip()
        if inners and not inner.startswith(("+", "-")):
            inner = "+ " + inner
        inners.append(inner)
    combined = "\n" + common_prefix + "\n\n" + "\n\n                        ".join(inners) + "\n\n                    )"
    result = template[:start] + combined + template[end:]
    return renamed(result, source_names[-1], target_name)


def declaration(function):
    return function[: function.find("{")].rstrip() + ";"


def main():
    args = parse_args()
    text = args.source.read_text()
    groups_by_variant = json.loads(args.candidates.read_text())["candidates"]
    functions = {}
    for suffix in SUFFIXES:
        for index in range(26):
            name = f"{PREFIX}{index}_RS{suffix}"
            functions[name] = extract_function(text, name)

    cuda = []
    header = []
    manifest = {"source": str(args.source), "variants": {}}
    names = set()
    for variant, tag in VARIANT_TAGS.items():
        groups = groups_by_variant[variant]
        manifest["variants"][variant] = {"tag": tag, "groups": groups}
        cuda.append(f"// ===== DDDD {tag} =====")
        header.append(f"// ===== DDDD {tag} =====")
        for suffix in SUFFIXES:
            for target_index, group in enumerate(groups):
                source_names = [f"{PREFIX}{index}_RS{suffix}" for index in group]
                target_name = f"{PREFIX}{target_index}_{tag}{suffix}"
                if len(group) == 1:
                    function = renamed(functions[source_names[0]], source_names[0], target_name)
                else:
                    function = merge_group(functions, source_names, target_name)
                if target_name in names:
                    raise ValueError(f"Duplicate {target_name}")
                names.add(target_name)
                cuda.append(function)
                header.append(declaration(function))

    expected = len(VARIANT_TAGS) * len(SUFFIXES) * 19
    if len(names) != expected:
        raise ValueError(f"Expected {expected}, got {len(names)}")
    cuda_text = "\n\n".join(cuda) + "\n"
    if re.search(r"\+\s*\+\s*F\d+_t", cuda_text):
        raise ValueError("Duplicated expression sign")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DDDD_K19_variants.cu.inc").write_text(cuda_text)
    (args.output_dir / "DDDD_K19_variants.hpp.inc").write_text("\n\n".join(header) + "\n")
    (args.output_dir / "DDDD_K19_variants.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated functions: {expected}")


if __name__ == "__main__":
    main()
