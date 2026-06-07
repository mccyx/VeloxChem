#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


CUDA_PREAMBLE = """#include <stdint.h>
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#define __launch_bounds__(x)
#ifndef TILE_SIZE_J
#define TILE_SIZE_J 1
#endif
"""


def find_function_span(text: str, func_name: str):
    start = text.find(func_name + "(")
    if start == -1:
        raise ValueError(f"Function '{func_name}' not found")

    brace_start = text.find("{", start)
    if brace_start == -1:
        raise ValueError(f"Function '{func_name}' has no body")

    depth = 0
    for pos in range(brace_start, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, pos + 1

    raise ValueError(f"Function '{func_name}' body is not balanced")


def build_wrapper(function_text: str) -> str:
    return CUDA_PREAMBLE + '\n#line 1 "kernel_fragment"\n' + function_text + "\n"


def run_ast_dump(compiler: str, function_text: str, function_name: str, language: str):
    wrapper = build_wrapper(function_text)
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as tmp:
        tmp.write(wrapper)
        tmp_path = Path(tmp.name)

    cmd = [
        compiler,
        "-x",
        language,
        "-std=c++17",
        "-fsyntax-only",
        "-Wno-everything",
        "-Xclang",
        "-ast-dump=json",
        "-Xclang",
        "-ast-dump-filter",
        "-Xclang",
        function_name,
        str(tmp_path),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(
            "AST dump produced no JSON output. stderr:\n%s" % proc.stderr.strip()
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to decode AST JSON. stderr:\n%s\nstdout-prefix:\n%s"
            % (proc.stderr.strip(), stdout[:1000])
        ) from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    return payload, proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Clang AST JSON for a single kernel function")
    parser.add_argument("--source", required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--language", default="c++")
    parser.add_argument("--output")
    parser.add_argument("--stderr", action="store_true", help="Print compiler stderr to stderr")
    args = parser.parse_args()

    source_text = Path(args.source).read_text()
    start, end = find_function_span(source_text, args.function)
    function_text = source_text[start:end]
    payload, stderr_text = run_ast_dump(args.compiler, function_text, args.function, args.language)

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    if args.stderr and stderr_text.strip():
        print(stderr_text, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
