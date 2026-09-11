#!/usr/bin/env python3

"""Add a runtime selector for DPDD split candidates."""

from pathlib import Path


SOURCE = Path("src/gpu/FockDriverGPU.cu")
VARIANTS = {"old_k5": ("K5_OLD", 5), "rs_k4": ("K4_RS", 4), "old_k4": ("K4_OLD", 4)}


def launch_span(text, name):
    start = text.find(f"            gpu::{name}<<<")
    if start < 0: raise ValueError(f"Missing {name}")
    return start, text.find(");\n", start) + 2


def launch_group(text, suffix):
    spans = [launch_span(text, f"computeExchangeFockDPDD{i}_RS{suffix}") for i in range(5)]
    for left, right in zip(spans, spans[1:]):
        if text[left[1] : right[0]].strip(): raise ValueError("Unexpected text between launches")
    return spans[0][0], spans[-1][1], [text[a:b] for a,b in spans]


def indent(launches):
    return "\n".join("    " + line if line else line for launch in launches for line in launch.splitlines())


def selection(launches, suffix):
    sections = ['            if (dpdd_split_variant == "rs5")\n            {\n' + indent(launches) + "\n            }"]
    template = launches[0]
    for selector, (tag, count) in VARIANTS.items():
        candidates = [template.replace(f"computeExchangeFockDPDD0_RS{suffix}", f"computeExchangeFockDPDD{i}_{tag}{suffix}", 1) for i in range(count)]
        sections.append(f'            else if (dpdd_split_variant == "{selector}")\n            {{\n' + indent(candidates) + "\n            }")
    return "\n".join(sections)


def main():
    text = SOURCE.read_text()
    if "const std::string dpdd_split_variant" in text: raise ValueError("DPDD selector already integrated")
    block = text.find("// BEGIN GENERATED EXCHANGE RESPLIT COMPARISON DPDD"); anchor = "            double rs_time_mark = omp_get_wtime();\n"; timing = text.find(anchor, block)
    if block < 0 or timing < 0: raise ValueError("Missing DPDD comparison block")
    selector = (
        '            const char* dpdd_split_env = std::getenv("VLX_EXCHANGE_DPDD_SPLIT");\n'
        '            const std::string dpdd_split_variant = dpdd_split_env ? dpdd_split_env : "rs5";\n'
        '            if (dpdd_split_variant != "rs5" && dpdd_split_variant != "old_k5" &&\n'
        '                dpdd_split_variant != "rs_k4" && dpdd_split_variant != "old_k4")\n'
        "            {\n"
        '                throw std::runtime_error("Invalid VLX_EXCHANGE_DPDD_SPLIT: " + dpdd_split_variant);\n'
        "            }\n"
    )
    text = text[:timing] + selector + text[timing:]
    for suffix in ("_FP32", "_FP64", ""):
        start,end,launches=launch_group(text,suffix); text=text[:start]+selection(launches,suffix)+text[end:]
    old='print_exchange_resplit_timing("DPDD", old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    new='print_exchange_resplit_timing(("DPDD " + dpdd_split_variant).c_str(), old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    if text.count(old)!=1: raise ValueError("Unexpected timing label")
    SOURCE.write_text(text.replace(old,new,1))


if __name__ == "__main__": main()
