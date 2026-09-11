#!/usr/bin/env python3

"""Generate runtime-balanced DDDD K16 candidates from RS26 and old19."""

import json
import re
from pathlib import Path


SOURCE = Path("src/gpu/EriExchange.cu")
CANDIDATES = Path("tools/dddd_split_search/k16_candidates.json")
OUTPUT = Path("src/gpu")
VARIANTS = {"rs_k16_runtime": "K16_RS_RUNTIME", "old_k16_runtime": "K16_OLD_RUNTIME"}
SUFFIXES = ("", "_FP64", "_FP32")
PREFIX = "computeExchangeFockDDDD"
ERI_RE = re.compile(r"const\s+(?:double|float)\s+eri_ijkl(?:_f)?\s*=\s*")


def matching(text, start, opening, closing):
    depth = 0
    for pos in range(start, len(text)):
        depth += text[pos] == opening
        depth -= text[pos] == closing
        if depth == 0:
            return pos
    raise ValueError("Unterminated block")


def extract_function(text, name):
    pos = text.find(f"\n{name}(")
    if pos < 0:
        raise ValueError(f"Missing {name}")
    start = text.rfind("__global__ void", 0, pos)
    body = text.find("{", pos)
    return text[start : matching(text, body, "{", "}") + 1]


def expression(function):
    match = ERI_RE.search(function)
    end = function.find(";", match.end())
    return match.end(), end, function[match.end() : end]


def outer(value):
    stripped = value.strip()
    opening = stripped.find("(")
    closing = matching(stripped, opening, "(", ")")
    if closing != len(stripped) - 1:
        raise ValueError("Unexpected outer expression")
    return stripped[: opening + 1], stripped[opening + 1 : closing]


def rename(function, old, new):
    if function.count(old) != 1:
        raise ValueError(f"Unexpected name count for {old}")
    return function.replace(old, new, 1)


def merge(functions, names, target, prefer_complete_setup=False):
    template_name = max(names, key=lambda name: len(functions[name])) if prefer_complete_setup else names[-1]
    template = functions[template_name]
    start, end, value = expression(template)
    common, _ = outer(value)
    inners = []
    for name in names:
        _, _, value = expression(functions[name])
        prefix, inner = outer(value)
        if re.sub(r"\s+", "", prefix) != re.sub(r"\s+", "", common):
            raise ValueError(f"Incompatible prefixes: {names}")
        inner = inner.strip()
        if inners and not inner.startswith(("+", "-")):
            inner = "+ " + inner
        inners.append(inner)
    combined = "\n" + common + "\n\n" + "\n\n                        ".join(inners) + "\n\n                    )"
    return rename(template[:start] + combined + template[end:], template_name, target)


def declaration(function):
    return function[: function.find("{")].rstrip() + ";"


def main():
    source = SOURCE.read_text()
    specs = json.loads(CANDIDATES.read_text())
    cuda, header, names = [], [], set()
    manifest = {"source": str(SOURCE), "variants": {}}
    for variant, tag in VARIANTS.items():
        spec = specs[variant]
        source_layout = spec["suffix"]
        groups = spec["groups"]
        manifest["variants"][variant] = {"tag": tag, "source_layout": source_layout or "old", "groups": groups}
        cuda.append(f"// ===== DDDD {tag} =====")
        header.append(f"// ===== DDDD {tag} =====")
        for suffix in SUFFIXES:
            functions = {}
            for index in range(spec["count"]):
                name = f"{PREFIX}{index}{source_layout}{suffix}"
                functions[name] = extract_function(source, name)
            for target_index, group in enumerate(groups):
                sources = [f"{PREFIX}{index}{source_layout}{suffix}" for index in group]
                target = f"{PREFIX}{target_index}_{tag}{suffix}"
                function = rename(functions[sources[0]], sources[0], target) if len(sources) == 1 else merge(
                    functions, sources, target, prefer_complete_setup=(source_layout == "")
                )
                if target in names:
                    raise ValueError(f"Duplicate {target}")
                names.add(target)
                cuda.append(function)
                header.append(declaration(function))
    expected = len(VARIANTS) * len(SUFFIXES) * 16
    if len(names) != expected:
        raise ValueError(f"Expected {expected}, got {len(names)}")
    cuda_text = "\n\n".join(cuda) + "\n"
    if re.search(r"\+\s*\+\s*F\d+_t", cuda_text):
        raise ValueError("Duplicated expression sign")
    (OUTPUT / "DDDD_K16_variants.cu.inc").write_text(cuda_text)
    (OUTPUT / "DDDD_K16_variants.hpp.inc").write_text("\n\n".join(header) + "\n")
    (OUTPUT / "DDDD_K16_variants.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated functions: {expected}")


if __name__ == "__main__":
    main()
