#!/usr/bin/env python3

import argparse
import re
from collections import Counter
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
ERI_ASSIGN_RE = re.compile(
    r"^(?P<indent>[ \t]*)const float (?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>.*);\s*$",
    re.DOTALL,
)

PUBLIC_PASS_ORDER = ["scalar", "regroup", "cse"]
LEGACY_SCALAR_PASSES = ["scalarize", "hoist", "rewrite"]


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


def line_indent_at(text: str, pos: int) -> str:
    line_start = text.rfind("\n", 0, pos)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = pos
    while line_end < len(text) and text[line_end] in " \t":
        line_end += 1
    return text[line_start:line_end]


def _normalized_space(text: str) -> str:
    return " ".join(text.split())


def _operator_count(text: str) -> int:
    return sum(text.count(op) for op in ["+", "-", "*", "/"])


def repeated_parenthesized_subexpressions(rhs: str, min_length: int = 18) -> List[str]:
    stack: List[int] = []
    raw_by_norm: Dict[str, str] = {}
    counts: Counter = Counter()

    for pos, ch in enumerate(rhs):
        if ch == "(":
            stack.append(pos)
        elif ch == ")" and stack:
            start = stack.pop()
            raw = rhs[start : pos + 1]
            inner = raw[1:-1].strip()
            norm = _normalized_space(inner)
            if len(norm) < min_length:
                continue
            if "?" in norm or ":" in norm:
                continue
            if _operator_count(norm) < 2:
                continue
            counts[norm] += 1
            raw_by_norm.setdefault(norm, raw)

    candidates = [norm for norm, count in counts.items() if count >= 2]
    candidates.sort(key=lambda norm: (-len(norm), norm))
    return [raw_by_norm[norm] for norm in candidates]


POW_CHAIN_TOKEN_RE = re.compile(r"^[A-Za-z_]\w*$")


def split_top_level_sum_terms(expr: str) -> List[str]:
    terms: List[str] = []
    depth = 0
    term_start = 0

    for pos, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "+-" and depth == 0 and pos > term_start:
            prev = expr[pos - 1]
            if prev not in "eE":
                terms.append(expr[term_start:pos].strip())
                term_start = pos

    terms.append(expr[term_start:].strip())
    return [term for term in terms if term]


def repeated_power_chains(rhs: str, min_power: int = 2, min_total_uses: int = 2) -> List[Tuple[str, int]]:
    counter: Counter = Counter()

    for term in split_top_level_sum_terms(rhs):
        pieces = [piece.strip() for piece in term.split("*")]
        if len(pieces) < min_power:
            continue

        run_token = None
        run_len = 0
        for piece in pieces + ["__END__"]:
            if piece == run_token:
                run_len += 1
                continue

            if (
                run_token is not None
                and run_len >= min_power
                and POW_CHAIN_TOKEN_RE.match(run_token)
            ):
                for power in range(min_power, run_len + 1):
                    counter[(run_token, power)] += run_len - power + 1

            run_token = piece
            run_len = 1

    candidates = [
        (token, power)
        for (token, power), count in counter.items()
        if count >= min_total_uses
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates


def power_alias_name(token: str, power: int) -> str:
    return f"{token}_pow{power}"


def replace_power_chain_occurrences(text: str, token: str, power: int, alias: str) -> Tuple[str, int]:
    pattern = r"\b" + re.escape(token) + r"\b"
    replacement_count = 0

    while True:
        matches = list(re.finditer(pattern, text))
        changed = False
        for start_idx in range(len(matches) - power + 1):
            group = matches[start_idx : start_idx + power]
            if any(group[idx].start() >= group[idx + 1].start() for idx in range(len(group) - 1)):
                continue

            segment_start = group[0].start()
            segment_end = group[-1].end()
            segment = text[segment_start:segment_end]
            compact = re.sub(r"\s+", "", segment)
            expected = "*".join([token] * power)
            if compact != expected:
                continue

            text = text[:segment_start] + alias + text[segment_end:]
            replacement_count += 1
            changed = True
            break

        if not changed:
            break

    return text, replacement_count


def extract_exact_cse(function_text: str) -> Tuple[str, List[str]]:
    function_ir = build_ir(function_text, 0)
    replacements: List[Tuple[int, int, str]] = []
    report: List[str] = []
    temp_counter = 0

    for block in walk_blocks(function_ir):
        for stmt in block.statements:
            if "const float eri_ijkl_f =" not in stmt.text:
                continue

            match = ERI_ASSIGN_RE.match(stmt.text)
            if not match:
                continue

            indent = line_indent_at(function_text, stmt.start)
            rhs = match.group("rhs")
            candidates = repeated_parenthesized_subexpressions(rhs)
            if not candidates:
                continue

            new_rhs = rhs
            decl_exprs: List[Tuple[str, str]] = []
            extracted = 0

            for raw in candidates:
                occurrence_count = new_rhs.count(raw)
                if occurrence_count < 2:
                    continue

                alias = f"cse{temp_counter}_f"
                temp_counter += 1
                decl_exprs.append((alias, raw[1:-1].strip()))
                new_rhs = new_rhs.replace(raw, alias)
                extracted += 1
                report.append(
                    f"eri_ijkl_f: extracted {alias} from repeated subexpression used {occurrence_count} times"
                )

            if extracted == 0:
                continue

            replacement_lines = []
            for alias, expr in decl_exprs:
                replacement_lines.append(f"{indent}const float {alias} = {expr};")
            replacement_lines.append("")
            replacement_lines.append(f"{indent}const float eri_ijkl_f = {new_rhs};")
            replacement = "\n".join(replacement_lines)
            line_start = function_text.rfind("\n", 0, stmt.start)
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1
            replacements.append((line_start, stmt.end, replacement))

    out = function_text
    for start, end, replacement in reversed(replacements):
        out = out[:start] + replacement + out[end:]

    return out, report


def apply_power_chain_regroup(function_text: str) -> Tuple[str, List[str]]:
    function_ir = build_ir(function_text, 0)
    replacements: List[Tuple[int, int, str]] = []
    report: List[str] = []

    for block in walk_blocks(function_ir):
        for stmt in block.statements:
            if "const float eri_ijkl_f =" not in stmt.text:
                continue

            match = ERI_ASSIGN_RE.match(stmt.text)
            if not match:
                continue

            indent = line_indent_at(function_text, stmt.start)
            rhs = match.group("rhs")
            candidates = repeated_power_chains(rhs)
            if not candidates:
                continue

            new_rhs = rhs
            decl_lines: List[str] = []
            extracted = 0

            for token, power in candidates:
                alias = power_alias_name(token, power)
                new_rhs, replacements_made = replace_power_chain_occurrences(
                    new_rhs, token, power, alias
                )
                if replacements_made < 2:
                    continue
                decl_lines.append(f"{indent}const float {alias} = {' * '.join([token] * power)};")
                report.append(
                    f"eri_ijkl_f: regrouped repeated power chain {' * '.join([token] * power)} into {alias} ({replacements_made} uses)"
                )
                extracted += 1

            if extracted == 0:
                continue

            replacement_lines = decl_lines + ["", f"{indent}const float eri_ijkl_f = {new_rhs};"]
            replacement = "\n".join(replacement_lines)
            line_start = function_text.rfind("\n", 0, stmt.start)
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1
            replacements.append((line_start, stmt.end, replacement))

    out = function_text
    for start, end, replacement in reversed(replacements):
        out = out[:start] + replacement + out[end:]

    return out, report


def apply_scalarize_for_array(function_text: str, match, keep_array_fallback: bool) -> Tuple[str, Dict[str, object]]:
    indent = match.group("indent")
    array_name = match.group("name")
    exprs = tuple(match.group(f"expr{i}").strip() for i in range(3))
    scalars = scalar_names(array_name)
    replacement_lines = [
        f"{indent}const float {scalars[0]} = {exprs[0]};",
        f"{indent}const float {scalars[1]} = {exprs[1]};",
        f"{indent}const float {scalars[2]} = {exprs[2]};",
    ]
    if keep_array_fallback:
        replacement_lines.append(
            f"{indent}const float {array_name}[3] = {{{scalars[0]}, {scalars[1]}, {scalars[2]}}};"
        )
    out = function_text[: match.start()] + "\n".join(replacement_lines) + function_text[match.end() :]
    info = {
        "array_name": array_name,
        "scalars": scalars,
        "indent": indent,
        "start_search": match.start(),
        "replacement_len": len("\n".join(replacement_lines)),
        "kept_array_fallback": keep_array_fallback,
    }
    return out, info


def apply_indexed_hoist_for_array(function_text: str, info: Dict[str, object]) -> Tuple[str, List[str]]:
    report: List[str] = []
    array_name = info["array_name"]
    scalars = info["scalars"]
    indent = info["indent"]
    start_search = info["start_search"]
    replacement_len = info["replacement_len"]

    function_ir = build_ir(function_text, 0)
    block = innermost_block(function_ir, start_search)
    stmt_idx = find_declaration_statement_index(block, start_search, array_name)
    if stmt_idx is None:
        raise ValueError("Could not locate declaration statement in block IR")

    block_start, block_end = block.start, block.end
    region = function_text[block_start:block_end]
    alias_map, skipped, insert_after_stmt = safe_dynamic_indices(block, stmt_idx, region, array_name, function_text)

    out = function_text
    if alias_map:
        absolute_insert_pos = start_search + replacement_len
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
    for idx, alias in alias_map.items():
        block_region = re.sub(rf"\b{re.escape(array_name)}\[{re.escape(idx)}\]", alias, block_region)
    out = out[:block_start] + block_region + out[block_end:]

    report.append(f"{array_name}: hoisted {len(alias_map)} dynamic indexed accesses")
    for idx, reason in sorted(skipped.items()):
        report.append(f"{array_name}[{idx}]: skipped hoist because {reason}")
    return out, report


def apply_direct_index_rewrite_for_array(function_text: str, info: Dict[str, object]) -> Tuple[str, List[str]]:
    array_name = info["array_name"]
    scalars = info["scalars"]
    pattern_count = 0
    out = function_text

    count = len(re.findall(rf"\b{re.escape(array_name)}\[0\]", out))
    out = re.sub(rf"\b{re.escape(array_name)}\[0\]", scalars[0], out)
    pattern_count += count
    count = len(re.findall(rf"\b{re.escape(array_name)}\[1\]", out))
    out = re.sub(rf"\b{re.escape(array_name)}\[1\]", scalars[1], out)
    pattern_count += count
    count = len(re.findall(rf"\b{re.escape(array_name)}\[2\]", out))
    out = re.sub(rf"\b{re.escape(array_name)}\[2\]", scalars[2], out)
    pattern_count += count

    report: List[str] = []
    if pattern_count:
        report.append(f"{array_name}: rewrote {pattern_count} direct constant-index accesses")
    return out, report


def normalize_requested_passes(passes_arg: str) -> List[str]:
    requested = []
    for raw_name in passes_arg.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name in LEGACY_SCALAR_PASSES:
            name = "scalar"
        if name not in PUBLIC_PASS_ORDER:
            raise ValueError(
                f"Unknown pass '{raw_name}'. Valid passes: {', '.join(PUBLIC_PASS_ORDER)}"
            )
        if name not in requested:
            requested.append(name)

    if not requested:
        raise ValueError("At least one pass must be selected")

    return [name for name in PUBLIC_PASS_ORDER if name in requested]


def run_pass_pipeline(function_text: str, enabled_passes: List[str]) -> Tuple[str, List[str]]:
    report: List[str] = []
    out = function_text

    if "scalar" in enabled_passes:
        cursor = 0
        while True:
            match = DECL_RE.search(out, cursor)
            if not match:
                break

            keep_array_fallback = False
            out, info = apply_scalarize_for_array(out, match, keep_array_fallback)
            report.append(
                f"{info['array_name']}: scalar pipeline emitted {', '.join(info['scalars'])}"
            )

            out, pass_report = apply_indexed_hoist_for_array(out, info)
            report.extend(pass_report)

            out, pass_report = apply_direct_index_rewrite_for_array(out, info)
            report.extend(pass_report)

            cursor = info["start_search"] + info["replacement_len"]

    if "regroup" in enabled_passes:
        out, pass_report = apply_power_chain_regroup(out)
        report.extend(pass_report)

    if "cse" in enabled_passes:
        out, pass_report = extract_exact_cse(out)
        report.extend(pass_report)

    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the source-level kernel optimization pass pipeline on a CUDA kernel body."
    )
    parser.add_argument("--source", required=True, help="Path to the source file")
    parser.add_argument("--function", required=True, help="Kernel/function name")
    parser.add_argument("--output", help="Optional output path")
    parser.add_argument(
        "--passes",
        default="scalar",
        help=(
            "Comma-separated pass list. Valid passes: "
            + ", ".join(PUBLIC_PASS_ORDER)
            + ". Legacy names scalarize/hoist/rewrite are accepted as aliases for scalar."
        ),
    )
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
    enabled_passes = normalize_requested_passes(args.passes)
    transformed, report = run_pass_pipeline(function_text, enabled_passes)

    if args.output:
        Path(args.output).write_text(transformed)
    else:
        print(transformed)

    if args.report:
        import sys

        print(f"passes: {', '.join(enabled_passes)}", file=sys.stderr)
        for line in report:
            print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
