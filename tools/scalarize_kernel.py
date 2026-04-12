#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

from kernel_ir import (
    build_ir,
    definition_sites,
    enclosing_child_index,
    find_statement_index,
    first_use_after,
    innermost_block,
    redefinitions_after,
    walk_blocks,
)


DECL_RE = re.compile(
    r"(?P<indent>[ \t]*)const float (?P<name>[A-Za-z_]\w*)\[3\] = \{"
    r"(?P<expr0>[^{};]+?),\s*"
    r"(?P<expr1>[^{};]+?),\s*"
    r"(?P<expr2>[^{};]+?)\};"
)


def find_function_span(text: str, func_name: str) -> Tuple[int, int]:
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


def scalar_names(array_name: str) -> Tuple[str, str, str]:
    if array_name.endswith("_f"):
        stem = array_name[:-2]
        suffix = "_f"
    else:
        stem = array_name
        suffix = ""
    return (f"{stem}0{suffix}", f"{stem}1{suffix}", f"{stem}2{suffix}")


def selector_expr(base_names: Tuple[str, str, str], index_name: str) -> str:
    return (
        f"({index_name} == 0 ? {base_names[0]} : "
        f"({index_name} == 1 ? {base_names[1]} : {base_names[2]}))"
    )


def hoisted_alias_name(array_name: str, index_name: str) -> str:
    if array_name.endswith("_f"):
        stem = array_name[:-2]
        suffix = "_f"
    else:
        stem = array_name
        suffix = ""
    return f"{stem}_{index_name}{suffix}"


def shared_variables(function_text):
    names = set()
    for match in re.finditer(r"__shared__\s+[^;]+;", function_text):
        stmt = match.group(0)
        for name in re.findall(r"\b([A-Za-z_]\w*)\b(?=\s*(?:\[|,|;))", stmt):
            if name not in {"__shared__", "double", "float", "uint32_t", "int"}:
                names.add(name)
    return names


def ancestor_visible_definition(block, name, shared_names):
    current = block
    while current.parent is not None:
        parent = current.parent
        child_idx = enclosing_child_index(parent, current)
        if child_idx is None:
            break
        candidate_stmt_idx = None
        sync_seen = False
        limit_pos = current.start
        for idx, stmt in enumerate(parent.statements):
            if stmt.end > limit_pos:
                break
            if "__syncthreads" in stmt.text:
                sync_seen = True
            if name in stmt.defs:
                candidate_stmt_idx = idx
        if candidate_stmt_idx is not None:
            if name in shared_names:
                if sync_seen:
                    return parent, candidate_stmt_idx, "shared_sync"
            else:
                return parent, candidate_stmt_idx, "ancestor"
        current = parent
    return None


def safe_dynamic_indices(block, start_stmt_idx: int, region: str, array_name: str, function_text: str):
    alias_map: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    insert_after_stmt: Dict[str, Tuple[object, int]] = {}
    suffix_region = "".join(stmt.text for stmt in block.statements[start_stmt_idx + 1 :])
    uses = list(re.finditer(rf"\b{re.escape(array_name)}\[(\w+)\]", suffix_region))
    dynamic_indices = []
    shared_names = shared_variables(function_text)
    for match in uses:
        idx = match.group(1)
        if idx not in {"0", "1", "2"} and idx not in dynamic_indices:
            dynamic_indices.append(idx)

    for idx in dynamic_indices:
        def_sites = definition_sites(block, idx)
        if len(def_sites) != 1:
            ancestor = ancestor_visible_definition(block, idx, shared_names)
            if ancestor is None:
                skipped[idx] = "index must have exactly one definition statement in the same block"
                continue
            def_block, def_idx, reason = ancestor
            alias_map[idx] = hoisted_alias_name(array_name, idx)
            insert_after_stmt[idx] = (def_block, def_idx)
            continue
        else:
            def_idx = def_sites[0]
            use_idx = first_use_after(block, start_stmt_idx, idx)
            if use_idx is None:
                skipped[idx] = "no use of index found after scalarized declaration in this block"
                continue
            if def_idx > use_idx:
                skipped[idx] = "definition does not dominate rewritten uses in this block"
                continue
            if redefinitions_after(block, def_idx, idx):
                skipped[idx] = "index is redefined later in the same block"
                continue
            alias_map[idx] = hoisted_alias_name(array_name, idx)
            insert_after_stmt[idx] = (block, def_idx)

    return alias_map, skipped, insert_after_stmt


def find_declaration_statement_index(block, pos, array_name):
    stmt_idx = find_statement_index(block, pos)
    if stmt_idx is not None:
        return stmt_idx
    for idx, stmt in enumerate(block.statements):
        if (
            ("const float %s[3]" % array_name) in stmt.text
            or ("const float %s0" % array_name[:-2] if array_name.endswith("_f") else "const float %s0" % array_name) in stmt.text
        ):
            return idx
    return None


def scalarize_function(function_text: str) -> Tuple[str, List[str]]:
    report: List[str] = []
    out = function_text
    cursor = 0

    while True:
        match = DECL_RE.search(out, cursor)
        if not match:
            break

        indent = match.group("indent")
        array_name = match.group("name")
        exprs = tuple(match.group(f"expr{i}").strip() for i in range(3))
        scalars = scalar_names(array_name)

        replacement_lines = [
            f"{indent}const float {scalars[0]} = {exprs[0]};",
            f"{indent}const float {scalars[1]} = {exprs[1]};",
            f"{indent}const float {scalars[2]} = {exprs[2]};",
        ]
        out = out[: match.start()] + "\n".join(replacement_lines) + out[match.end() :]

        function_ir = build_ir(out, 0)
        start_search = match.start()
        block = innermost_block(function_ir, start_search)
        stmt_idx = find_declaration_statement_index(block, start_search, array_name)
        if stmt_idx is None:
            raise ValueError("Could not locate declaration statement in block IR")

        block_start, block_end = block.start, block.end
        region = out[block_start:block_end]
        alias_map, skipped, insert_after_stmt = safe_dynamic_indices(block, stmt_idx, region, array_name, out)

        if alias_map:
            replacement_stmt_end = start_search + len("\n".join(replacement_lines))
            absolute_insert_pos = replacement_stmt_end
            for idx in alias_map:
                insert_block, insert_stmt_idx = insert_after_stmt[idx]
                absolute_insert_pos = max(absolute_insert_pos, insert_block.statements[insert_stmt_idx].end)

            alias_lines = [
                f"{indent}const float {alias} = {selector_expr(scalars, idx)};"
                for idx, alias in sorted(alias_map.items())
            ]
            out = out[:absolute_insert_pos] + "\n" + "\n".join(alias_lines) + out[absolute_insert_pos:]

            function_ir = build_ir(out, 0)
            block = innermost_block(function_ir, start_search)
            block_start, block_end = block.start, block.end

        block_region = out[block_start:block_end]
        block_region = re.sub(rf"\b{re.escape(array_name)}\[0\]", scalars[0], block_region)
        block_region = re.sub(rf"\b{re.escape(array_name)}\[1\]", scalars[1], block_region)
        block_region = re.sub(rf"\b{re.escape(array_name)}\[2\]", scalars[2], block_region)

        for idx, alias in alias_map.items():
            block_region = re.sub(rf"\b{re.escape(array_name)}\[{re.escape(idx)}\]", alias, block_region)

        out = out[:block_start] + block_region + out[block_end:]

        report.append(
            f"{array_name}: scalarized to {', '.join(scalars)}; hoisted {len(alias_map)} dynamic indexed accesses"
        )
        for idx, reason in sorted(skipped.items()):
            report.append(f"{array_name}[{idx}]: skipped hoist because {reason}")

        cursor = start_search + 1

    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scalarize small float[3] arrays inside a CUDA kernel body."
    )
    parser.add_argument("--source", required=True, help="Path to the source file")
    parser.add_argument("--function", required=True, help="Kernel/function name")
    parser.add_argument("--output", help="Optional output path")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a short transformation report to stderr",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    text = source_path.read_text()
    start, end = find_function_span(text, args.function)
    function_text = text[start:end]
    transformed, report = scalarize_function(function_text)

    if args.output:
        Path(args.output).write_text(transformed)
    else:
        print(transformed)

    if args.report:
        import sys

        for line in report:
            print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
