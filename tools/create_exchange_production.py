#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


WINNERS = {
    "PDDD": ("PDDD_K4_variants", "K4_M23", 4),
    "DDDD": ("DDDD_K16_variants", "K16_OLD_RUNTIME", 16),
    "DDDP": ("DDDP_K5_variants", "K5_OLD", 5),
    "DPDD": ("DPDD_split_variants", "K4_RS", 4),
}


def function_spans(text, names, header):
    spans = []
    for name in names:
        match = re.search(
            r"__global__ void __launch_bounds__\(TILE_SIZE_K\)\s*\n"
            + re.escape(name) + r"\(", text)
        if not match:
            raise ValueError("missing {}".format(name))
        start = match.start()
        if header:
            end = text.find(";", match.end()) + 1
        else:
            brace = text.find("{", match.end())
            depth = 0
            end = None
            for index in range(brace, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                raise ValueError("unclosed {}".format(name))
        spans.append(text[start:end].rstrip() + "\n")
    return spans


def generate_winner_includes(target, experiment):
    definitions = ["// Selected exchange mixed-precision split variants.\n"]
    declarations = ["// Selected exchange mixed-precision split variants.\n"]
    for family, (source_stem, tag, count) in WINNERS.items():
        names = [
            "computeExchangeFock{}{}_{}_{}".format(family, index, tag, precision)
            for precision in ("FP64", "FP32")
            for index in range(count)
        ]
        source = (experiment / "src/gpu/{}.cu.inc".format(source_stem)).read_text()
        header = (experiment / "src/gpu/{}.hpp.inc".format(source_stem)).read_text()
        definitions.append("\n// {}: {}\n".format(family, tag))
        definitions.extend(function_spans(source, names, False))
        declarations.append("\n// {}: {}\n".format(family, tag))
        declarations.extend(function_spans(header, names, True))

    (target / "src/gpu/ExchangeMPWinners.cu.inc").write_text("\n".join(definitions))
    (target / "src/gpu/ExchangeMPWinners.hpp.inc").write_text("\n".join(declarations))

    for relative, include in (
        ("src/gpu/EriExchange.cu", '#include "ExchangeMPWinners.cu.inc"'),
        ("src/gpu/EriExchange.hpp", '#include "ExchangeMPWinners.hpp.inc"'),
    ):
        path = target / relative
        text = path.read_text()
        if include not in text:
            marker = "\n}  // namespace gpu"
            text = text.replace(marker, "\n{}\n{}".format(include, marker), 1)
            path.write_text(text)


def launch_spans(text, family, precision):
    pattern = re.compile(
        r"gpu::computeExchangeFock" + family + r"\d*_(?:[A-Z0-9_]+_)?"
        + precision + r"<<<.*?\n\s*d_exchange_displ_cuts\);", re.S)
    return list(pattern.finditer(text))


def dedent_four(text):
    return "\n".join(line[4:] if line.startswith("    ") else line
                     for line in text.splitlines())


def production_block(validation, family):
    cut_start = validation.index("            const auto exchange_cuts =")
    cut_call = validation.index("            build_exchange_cuts_device", cut_start)
    cut_end = validation.index("stream);", cut_call) + len("stream);")
    cuts = validation[cut_start:cut_end]

    fp64 = launch_spans(validation, family, "FP64")
    fp32 = launch_spans(validation, family, "FP32")
    if not fp64 or not fp32:
        raise ValueError("missing MP launches for {}".format(family))

    if family in WINNERS:
        _, tag, count = WINNERS[family]
        fp64_template = fp64[0].group(0)
        fp32_template = fp32[0].group(0)
        launches = []
        for precision, template in (("FP64", fp64_template), ("FP32", fp32_template)):
            old_name = re.search(r"computeExchangeFock\w+<<<", template).group(0)[:-3]
            for index in range(count):
                new_name = "computeExchangeFock{}{}_{}_{}".format(
                    family, index, WINNERS[family][1], precision)
                launches.append(template.replace(old_name, new_name, 1))
    else:
        launches = [match.group(0) for match in fp64 + fp32]

    body = cuts + "\n" + "\n".join(launches)
    body = body.replace("d_mat_K_mp", "d_mat_K")
    return "        {\n" + dedent_four(body) + "\n        }"


def remove_resplit_comparisons(text):
    pattern = re.compile(
        r"\n\s*// BEGIN GENERATED EXCHANGE RESPLIT COMPARISON ([SPDF]{4})"
        r".*?// END GENERATED EXCHANGE RESPLIT COMPARISON \1\s*", re.S)
    text, count = pattern.subn("\n", text)
    if count != 8:
        raise ValueError("expected 8 resplit blocks, found {}".format(count))
    return text


def promote_pppp(text):
    section_start = text.index("        // K: (PP|PP)")
    section_end = text.index("        // K: (PS|PD)", section_start)
    section = text[section_start:section_end]
    cut_start = section.index("            auto pp_cuts =")
    cut_call = section.index("            build_exchange_cuts_device(", cut_start)
    cut_end = section.index("stream);", cut_call) + len("stream);")
    cuts = section[cut_start:cut_end]
    launch_pattern = re.compile(
        r"gpu::computeExchangeFockPPPP_(?:FP64|FP32)<<<.*?"
        r"\n\s*d_pp_displ_cuts\);", re.S)
    launches = [match.group(0).replace("d_mat_K_mp", "d_mat_K")
                for match in launch_pattern.finditer(section)]
    if len(launches) != 2:
        raise ValueError("expected two PPPP MP launches, found {}".format(len(launches)))
    body = cuts + "\n" + "\n".join(launches)
    replacement = (
        "        // K: (PP|PP)\n"
        "        //     *  *\n\n"
        "        {\n" + dedent_four(body) + "\n        }\n\n")
    return text[:section_start] + replacement + text[section_end:]


def promote_mp_to_production(path):
    text = path.read_text()
    marker = re.compile(
        r"\s*// BEGIN GENERATED EXCHANGE MP VALIDATION (?P<family>[SPDF]{4})"
        r"(?P<body>.*?)// END GENERATED EXCHANGE MP VALIDATION (?P=family)", re.S)
    matches = list(marker.finditer(text))
    if len(matches) != 53:
        raise ValueError("expected 53 generated validation blocks, found {}".format(len(matches)))

    for match in reversed(matches):
        family = match.group("family")
        validation = match.group(0)
        replacement = production_block(validation, family)

        comment_start = text.rfind("        // K:", 0, match.start())
        if comment_start < 0:
            raise ValueError("missing K comment for {}".format(family))
        prefix = text[comment_start:match.start()]
        original_pattern = re.compile(
            r"        gpu::computeExchangeFock" + family + r"\d*<<<.*?"
            r"\n\s*eri_threshold\);", re.S)
        originals = list(original_pattern.finditer(prefix))
        if not originals:
            raise ValueError("missing original production launch for {}".format(family))
        original_start = comment_start + originals[0].start()
        text = text[:original_start] + replacement + text[match.end():]

    text = promote_pppp(text)
    text = remove_resplit_comparisons(text)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    generate_winner_includes(args.target, args.experiment)
    promote_mp_to_production(args.target / "src/gpu/FockDriverGPU.cu")


if __name__ == "__main__":
    main()
