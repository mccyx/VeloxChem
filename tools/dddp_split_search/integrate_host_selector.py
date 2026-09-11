#!/usr/bin/env python3

"""Add a runtime selector for DDDP RS6 and K5 variants."""

from pathlib import Path


SOURCE = Path("src/gpu/FockDriverGPU.cu")
VARIANTS = {"rs_k5": "K5_RS", "old_k5": "K5_OLD"}


def launch_span(text, name):
    start = text.find(f"            gpu::{name}<<<")
    if start < 0:
        raise ValueError(f"Missing {name}")
    end = text.find(");\n", start)
    return start, end + 2


def group(text, suffix):
    spans = [launch_span(text, f"computeExchangeFockDDDP{i}_RS{suffix}") for i in range(6)]
    for left, right in zip(spans, spans[1:]):
        if text[left[1] : right[0]].strip():
            raise ValueError("Unexpected text between launches")
    return spans[0][0], spans[-1][1], [text[a:b] for a, b in spans]


def indent(launches):
    return "\n".join("    " + line if line else line for launch in launches for line in launch.splitlines())


def selection(launches, suffix):
    sections = ['            if (dddp_split_variant == "rs6")\n            {\n' + indent(launches) + "\n            }"]
    template = launches[0]
    for selector, tag in VARIANTS.items():
        candidate = []
        for index in range(5):
            candidate.append(template.replace(f"computeExchangeFockDDDP0_RS{suffix}", f"computeExchangeFockDDDP{index}_{tag}{suffix}", 1))
        sections.append(f'            else if (dddp_split_variant == "{selector}")\n            {{\n' + indent(candidate) + "\n            }")
    return "\n".join(sections)


def main():
    text = SOURCE.read_text()
    if "const std::string dddp_split_variant" in text:
        raise ValueError("DDDP selector already integrated")
    block = text.find("// BEGIN GENERATED EXCHANGE RESPLIT COMPARISON DDDP")
    anchor = "            double rs_time_mark = omp_get_wtime();\n"
    timing = text.find(anchor, block)
    if block < 0 or timing < 0:
        raise ValueError("Missing DDDP comparison block")
    selector = (
        '            const char* dddp_split_env = std::getenv("VLX_EXCHANGE_DDDP_SPLIT");\n'
        '            const std::string dddp_split_variant = dddp_split_env ? dddp_split_env : "rs6";\n'
        '            if (dddp_split_variant != "rs6" && dddp_split_variant != "rs_k5" && dddp_split_variant != "old_k5")\n'
        "            {\n"
        '                throw std::runtime_error("Invalid VLX_EXCHANGE_DDDP_SPLIT: " + dddp_split_variant);\n'
        "            }\n"
    )
    text = text[:timing] + selector + text[timing:]
    for suffix in ("_FP32", "_FP64", ""):
        start, end, launches = group(text, suffix)
        text = text[:start] + selection(launches, suffix) + text[end:]
    old = 'print_exchange_resplit_timing("DDDP", old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    new = 'print_exchange_resplit_timing(("DDDP " + dddp_split_variant).c_str(), old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    if text.count(old) != 1:
        raise ValueError("Unexpected timing label")
    SOURCE.write_text(text.replace(old, new, 1))


if __name__ == "__main__":
    main()
