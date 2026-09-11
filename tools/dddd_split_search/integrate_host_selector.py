#!/usr/bin/env python3

"""Wrap DDDD RS26 launches with a runtime K19 selector."""

from pathlib import Path


SOURCE = Path("src/gpu/FockDriverGPU.cu")
VARIANTS = {
    "k19_static": "K19_STATIC",
    "k19_runtime": "K19_RUNTIME",
    "k19_light": "K19_LIGHT",
}


def launch_span(text, name):
    start = text.find(f"            gpu::{name}<<<")
    if start < 0:
        raise ValueError(f"Missing launch {name}")
    end = text.find(");\n", start)
    if end < 0:
        raise ValueError(f"Unterminated launch {name}")
    return start, end + 2


def launch_group(text, suffix):
    names = [f"computeExchangeFockDDDD{i}_RS{suffix}" for i in range(26)]
    spans = [launch_span(text, name) for name in names]
    for left, right in zip(spans, spans[1:]):
        if text[left[1] : right[0]].strip():
            raise ValueError("Unexpected text between launches")
    return spans[0][0], spans[-1][1], [text[a:b] for a, b in spans]


def indent(launches):
    return "\n".join("    " + line if line else line for launch in launches for line in launch.splitlines())


def selected_group(launches, suffix):
    sections = [
        '            if (dddd_split_variant == "rs26")\n            {\n'
        + indent(launches)
        + "\n            }"
    ]
    template = launches[0]
    for selector, tag in VARIANTS.items():
        candidate = []
        for index in range(19):
            old_name = f"computeExchangeFockDDDD0_RS{suffix}"
            new_name = f"computeExchangeFockDDDD{index}_{tag}{suffix}"
            candidate.append(template.replace(old_name, new_name, 1))
        sections.append(
            f'            else if (dddd_split_variant == "{selector}")\n'
            "            {\n"
            + indent(candidate)
            + "\n            }"
        )
    return "\n".join(sections)


def main():
    text = SOURCE.read_text()
    if "const std::string dddd_split_variant" in text:
        raise ValueError("DDDD selector already integrated")
    block = text.find("// BEGIN GENERATED EXCHANGE RESPLIT COMPARISON DDDD")
    anchor = "            double rs_time_mark = omp_get_wtime();\n"
    timing = text.find(anchor, block)
    if block < 0 or timing < 0:
        raise ValueError("Missing DDDD comparison block")
    selector = (
        '            const char* dddd_split_env = std::getenv("VLX_EXCHANGE_DDDD_SPLIT");\n'
        '            const std::string dddd_split_variant = dddd_split_env ? dddd_split_env : "rs26";\n'
        '            if (dddd_split_variant != "rs26" && dddd_split_variant != "k19_static" &&\n'
        '                dddd_split_variant != "k19_runtime" && dddd_split_variant != "k19_light")\n'
        "            {\n"
        '                throw std::runtime_error("Invalid VLX_EXCHANGE_DDDD_SPLIT: " + dddd_split_variant);\n'
        "            }\n"
    )
    text = text[:timing] + selector + text[timing:]
    for suffix in ("_FP32", "_FP64", ""):
        start, end, launches = launch_group(text, suffix)
        text = text[:start] + selected_group(launches, suffix) + text[end:]
    old = 'print_exchange_resplit_timing("DDDD", old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    new = 'print_exchange_resplit_timing(("DDDD " + dddd_split_variant).c_str(), old_ref_seconds, rs_ref_seconds, old_mp_seconds, rs_mp_seconds);'
    if text.count(old) != 1:
        raise ValueError("Unexpected DDDD timing label count")
    text = text.replace(old, new, 1)
    SOURCE.write_text(text)


if __name__ == "__main__":
    main()
