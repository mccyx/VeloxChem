#!/usr/bin/env python3

"""Add the two DDDD K16 variants to the existing runtime selector."""

from pathlib import Path


SOURCE = Path("src/gpu/FockDriverGPU.cu")
VARIANTS = {"rs_k16_runtime": "K16_RS_RUNTIME", "old_k16_runtime": "K16_OLD_RUNTIME"}


def matching_brace(text, start):
    depth = 0
    for pos in range(start, len(text)):
        depth += text[pos] == "{"
        depth -= text[pos] == "}"
        if depth == 0:
            return pos
    raise ValueError("Unterminated branch")


def launch(text, name, start):
    begin = text.find(f"gpu::{name}<<<", start)
    if begin < 0:
        raise ValueError(f"Missing {name}")
    begin = text.rfind("            ", start, begin)
    end = text.find(");\n", begin)
    return text[begin : end + 2]


def indent_launches(launches):
    return "\n".join("    " + line if line else line for item in launches for line in item.splitlines())


def main():
    text = SOURCE.read_text()
    if "K16_RS_RUNTIME<<<" in text:
        raise ValueError("K16 selector already integrated")
    text = text.replace(
        'dddd_split_variant != "k19_runtime" && dddd_split_variant != "k19_light")',
        'dddd_split_variant != "k19_runtime" && dddd_split_variant != "k19_light" &&\n'
        '                dddd_split_variant != "rs_k16_runtime" && dddd_split_variant != "old_k16_runtime")',
        1,
    )
    search = 0
    inserted = 0
    for suffix in ("", "_FP64", "_FP32"):
        branch = text.find('            else if (dddd_split_variant == "k19_light")', search)
        if branch < 0:
            raise ValueError(f"Missing k19_light branch for {suffix}")
        opening = text.find("{", branch)
        closing = matching_brace(text, opening)
        template = launch(text, f"computeExchangeFockDDDD0_K19_LIGHT{suffix}", opening)
        sections = []
        for selector, tag in VARIANTS.items():
            launches = []
            for index in range(16):
                old = f"computeExchangeFockDDDD0_K19_LIGHT{suffix}"
                new = f"computeExchangeFockDDDD{index}_{tag}{suffix}"
                launches.append(template.replace(old, new, 1))
            sections.append(
                f'\n            else if (dddd_split_variant == "{selector}")\n'
                "            {\n" + indent_launches(launches) + "\n            }"
            )
        addition = "".join(sections)
        text = text[: closing + 1] + addition + text[closing + 1 :]
        search = closing + 1 + len(addition)
        inserted += 1
    if inserted != 3:
        raise ValueError(f"Expected 3 precision groups, got {inserted}")
    SOURCE.write_text(text)


if __name__ == "__main__":
    main()
