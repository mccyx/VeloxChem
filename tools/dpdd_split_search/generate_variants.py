#!/usr/bin/env python3

"""Generate DPDD old-derived K5 and old/RS-derived K4 candidates."""

import json
import re
from pathlib import Path


SOURCE = Path("src/gpu/EriExchange.cu")
OUTPUT = Path("src/gpu")
VARIANTS = {
    "old_k5": {"tag": "K5_OLD", "layout": "", "count": 7, "groups": [[0], [1, 2], [3], [4, 5], [6]]},
    "rs_k4": {"tag": "K4_RS", "layout": "_RS", "count": 5, "groups": [[0], [1, 2], [3], [4]]},
    "old_k4": {"tag": "K4_OLD", "layout": "", "count": 7, "groups": [[0, 1], [2, 3], [4], [5, 6]]},
}
SUFFIXES = ("", "_FP64", "_FP32")
PREFIX = "computeExchangeFockDPDD"
ERI_RE = re.compile(r"const\s+(?:double|float)\s+eri_ijkl(?:_f)?\s*=\s*")


def matching(text, start, opening, closing):
    depth = 0
    for pos in range(start, len(text)):
        depth += text[pos] == opening
        depth -= text[pos] == closing
        if depth == 0:
            return pos
    raise ValueError("Unterminated block")


def extract(text, name):
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
    value = value.strip(); opening = value.find("("); closing = matching(value, opening, "(", ")")
    if closing != len(value) - 1:
        raise ValueError("Unexpected expression shape")
    return value[: opening + 1], value[opening + 1 : closing]


def rename(function, old, new):
    if function.count(old) != 1:
        raise ValueError(f"Unexpected name count: {old}")
    return function.replace(old, new, 1)


def merge(functions, names, target, old_layout):
    # Cross-order old groups need the highest-order, most complete setup body.
    template_name = max(names, key=lambda name: len(functions[name])) if old_layout else names[-1]
    template = functions[template_name]
    start, end, value = expression(template); common, _ = outer(value); inners = []
    for name in names:
        _, _, value = expression(functions[name]); prefix, inner = outer(value)
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
    source = SOURCE.read_text(); cuda = []; header = []; names = set()
    for _, spec in VARIANTS.items():
        tag, layout, groups = spec["tag"], spec["layout"], spec["groups"]
        cuda.append(f"// ===== DPDD {tag} ====="); header.append(f"// ===== DPDD {tag} =====")
        for suffix in SUFFIXES:
            functions = {f"{PREFIX}{i}{layout}{suffix}": extract(source, f"{PREFIX}{i}{layout}{suffix}") for i in range(spec["count"])}
            for target_index, group in enumerate(groups):
                sources = [f"{PREFIX}{i}{layout}{suffix}" for i in group]; target = f"{PREFIX}{target_index}_{tag}{suffix}"
                function = rename(functions[sources[0]], sources[0], target) if len(group) == 1 else merge(functions, sources, target, old_layout=(layout == ""))
                if target in names: raise ValueError(f"Duplicate {target}")
                names.add(target); cuda.append(function); header.append(declaration(function))
    expected = sum(len(spec["groups"]) for spec in VARIANTS.values()) * len(SUFFIXES)
    if len(names) != expected: raise ValueError(f"Expected {expected}, got {len(names)}")
    cuda_text = "\n\n".join(cuda) + "\n"
    if re.search(r"\+\s*\+\s*F\d+_t", cuda_text): raise ValueError("Duplicated sign")
    (OUTPUT / "DPDD_split_variants.cu.inc").write_text(cuda_text)
    (OUTPUT / "DPDD_split_variants.hpp.inc").write_text("\n\n".join(header) + "\n")
    (OUTPUT / "DPDD_split_variants.manifest.json").write_text(json.dumps({"source": str(SOURCE), "variants": VARIANTS}, indent=2) + "\n")
    print(f"Generated functions: {expected}")


if __name__ == "__main__": main()
