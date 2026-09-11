#!/usr/bin/env python3

from pathlib import Path


SOURCE = Path("src/gpu/EriExchange.cu")
HEADER = Path("src/gpu/EriExchange.hpp")
SCALAR_DIR = Path("scalarization_exchange")
NAMESPACE_END = "\n\n}  // namespace gpu"


def declaration(text, function):
    name_pos = text.find(function + "(")
    if name_pos < 0:
        raise ValueError("missing declaration: " + function)
    start = text.rfind("__global__", 0, name_pos)
    end = text.find(");", name_pos)
    if start < 0 or end < 0:
        raise ValueError("malformed declaration: " + function)
    return text[start:end + 2]


def insert_before_namespace_end(text, addition):
    pos = text.rfind(NAMESPACE_END)
    if pos < 0:
        raise ValueError("namespace terminator not found")
    return text[:pos] + "\n\n" + addition.rstrip() + text[pos:]


def main():
    source = SOURCE.read_text()
    header = HEADER.read_text()
    definitions = []
    declarations = []

    for index in range(5):
        baseline = "computeExchangeFockPDDD{}_RS_FP32".format(index)
        scalar = baseline + "_scalar"
        if scalar + "(" in source or scalar + "(" in header:
            raise ValueError("variant already integrated: " + scalar)

        fragment = (SCALAR_DIR / ("PDDD{}_RS_FP32.scalar.cu".format(index))).read_text()
        if not fragment.startswith(baseline + "("):
            raise ValueError("unexpected fragment function: " + baseline)
        fragment = scalar + fragment[len(baseline):]
        definitions.append(
            "__global__ void __launch_bounds__(TILE_SIZE_K)\n" + fragment.rstrip()
        )
        declarations.append(declaration(header, baseline).replace(baseline, scalar, 1))

    source = insert_before_namespace_end(source, "\n\n".join(definitions))
    header = insert_before_namespace_end(header, "\n\n".join(declarations))
    SOURCE.write_text(source)
    HEADER.write_text(header)


if __name__ == "__main__":
    main()
