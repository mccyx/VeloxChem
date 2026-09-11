#!/usr/bin/env python3

import re
from pathlib import Path


PATH = Path("src/gpu/FockDriverGPU.cu")


def main():
    text = PATH.read_text()
    marker = "if (use_scalar_pddd_rs_fp32())"
    if marker in text:
        raise ValueError("PDDD scalar selector is already present")

    start_token = "            gpu::computeExchangeFockPDDD0_RS_FP32<<<"
    start = text.find(start_token)
    if start < 0:
        raise ValueError("PDDD0 RS FP32 launch not found")

    pos = start
    for index in range(5):
        launch = "gpu::computeExchangeFockPDDD{}_RS_FP32<<<".format(index)
        launch_pos = text.find(launch, pos)
        if launch_pos < 0:
            raise ValueError("launch not found: " + launch)
        end = text.find("d_exchange_displ_cuts);", launch_pos)
        if end < 0:
            raise ValueError("launch terminator not found: " + launch)
        pos = end + len("d_exchange_displ_cuts);")

    baseline = text[start:pos]
    scalar = re.sub(
        r"computeExchangeFockPDDD([0-4])_RS_FP32",
        r"computeExchangeFockPDDD\1_RS_FP32_scalar",
        baseline,
    )
    indent = "            "
    replacement = (
        indent + "if (use_scalar_pddd_rs_fp32())\n"
        + indent + "{\n"
        + scalar
        + "\n" + indent + "}\n"
        + indent + "else\n"
        + indent + "{\n"
        + baseline
        + "\n" + indent + "}"
    )
    PATH.write_text(text[:start] + replacement + text[pos:])


if __name__ == "__main__":
    main()
