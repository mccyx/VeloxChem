#!/usr/bin/env python3

"""Wrap the PDDD RS launch groups with a runtime K4 split selector."""

import re
from pathlib import Path


SOURCE = Path("src/gpu/FockDriverGPU.cu")
SELECTOR = "VLX_EXCHANGE_PDDD_SPLIT"
VARIANTS = {
    "k4_m01": "K4_M01",
    "k4_m12": "K4_M12",
    "k4_m23": "K4_M23",
    "k4_m34": "K4_M34",
}


def launch_span(text, name):
    start = text.find(f"            gpu::{name}<<<")
    if start < 0:
        raise ValueError(f"Missing launch: {name}")
    end = text.find(");\n", start)
    if end < 0:
        raise ValueError(f"Unterminated launch: {name}")
    return start, end + 2


def launch_group(text, suffix):
    names = [f"computeExchangeFockPDDD{i}_RS{suffix}" for i in range(5)]
    spans = [launch_span(text, name) for name in names]
    for left, right in zip(spans, spans[1:]):
        if text[left[1] : right[0]].strip():
            raise ValueError(f"Non-whitespace between {left} and {right}")
    return spans[0][0], spans[-1][1], [text[a:b] for a, b in spans]


def indent_launches(launches, spaces=4):
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for launch in launches for line in launch.splitlines())


def selected_group(launches, suffix):
    sections = [
        '            if (pddd_split_variant == "rs5")\n            {\n'
        + indent_launches(launches)
        + "\n            }"
    ]
    for selector, kernel_tag in VARIANTS.items():
        renamed = []
        for index, launch in enumerate(launches[:4]):
            old = f"computeExchangeFockPDDD{index}_RS{suffix}"
            new = f"computeExchangeFockPDDD{index}_{kernel_tag}{suffix}"
            renamed.append(launch.replace(old, new, 1))
        sections.append(
            f'            else if (pddd_split_variant == "{selector}")\n'
            "            {\n"
            + indent_launches(renamed)
            + "\n            }"
        )
    return "\n".join(sections)


def main():
    text = SOURCE.read_text()
    if "const std::string pddd_split_variant" in text:
        raise ValueError("PDDD split selector is already integrated")

    anchor = "            double rs_time_mark = omp_get_wtime();\n"
    block_start = text.find("// BEGIN GENERATED EXCHANGE RESPLIT COMPARISON PDDD")
    anchor_pos = text.find(anchor, block_start)
    if block_start < 0 or anchor_pos < 0:
        raise ValueError("Missing PDDD comparison timing anchor")
    selector_code = (
        "            const char* pddd_split_env = std::getenv(\""
        + SELECTOR
        + "\");\n"
        "            const std::string pddd_split_variant = pddd_split_env ? pddd_split_env : \"rs5\";\n"
        "            if (pddd_split_variant != \"rs5\" && pddd_split_variant != \"k4_m01\" &&\n"
        "                pddd_split_variant != \"k4_m12\" && pddd_split_variant != \"k4_m23\" &&\n"
        "                pddd_split_variant != \"k4_m34\")\n"
        "            {\n"
        "                throw std::runtime_error(\"Invalid VLX_EXCHANGE_PDDD_SPLIT: \" + pddd_split_variant);\n"
        "            }\n"
    )
    text = text[:anchor_pos] + selector_code + text[anchor_pos:]

    for suffix in ("_FP32", "_FP64", ""):
        start, end, launches = launch_group(text, suffix)
        text = text[:start] + selected_group(launches, suffix) + text[end:]

    text = text.replace(
        'print_exchange_resplit_timing("PDDD", old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);',
        'print_exchange_resplit_timing(("PDDD " + pddd_split_variant).c_str(), old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);',
        1,
    )
    selector_references = text.count("pddd_split_variant")
    if selector_references != 23:
        raise ValueError(f"Unexpected selector reference count: {selector_references}")
    SOURCE.write_text(text)


if __name__ == "__main__":
    main()
