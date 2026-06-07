#!/usr/bin/env python3

import json
from pathlib import Path


VALID_EXPR_KINDS = {
    "DeclRefExpr",
    "ImplicitCastExpr",
    "BinaryOperator",
    "ParenExpr",
    "UnaryOperator",
    "CallExpr",
    "MemberExpr",
    "CStyleCastExpr",
    "IntegerLiteral",
    "FloatingLiteral",
    "ArraySubscriptExpr",
    "ConditionalOperator",
}


def _line_starts(text: str):
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _loc_line_col(node):
    if not isinstance(node, dict):
        return None
    for key in ("spellingLoc", "expansionLoc"):
        loc = node.get(key)
        if isinstance(loc, dict):
            line = loc.get("presumedLine", loc.get("line"))
            col = loc.get("col")
            if line is not None and col is not None:
                return line, col
    line = node.get("presumedLine", node.get("line"))
    col = node.get("col")
    if line is not None and col is not None:
        return line, col
    return None


def _range_begin_end(node):
    if not isinstance(node, dict):
        return None
    rng = node.get("range")
    if not isinstance(rng, dict):
        return None
    begin = _loc_line_col(rng.get("begin", {})) or _loc_line_col(node.get("loc", {}))
    end = _loc_line_col(rng.get("end", {}))
    if begin and not end:
        end_col = rng.get("end", {}).get("col")
        if end_col is not None:
            end = (begin[0], end_col)
    if begin and end:
        return begin, end
    return None


def _range_begin_end_offsets(node):
    if not isinstance(node, dict):
        return None
    rng = node.get("range")
    if not isinstance(rng, dict):
        return None
    begin = rng.get("begin", {}).get("offset")
    end = rng.get("end", {}).get("offset")
    if begin is not None and end is not None:
        return begin, end
    return None


def _offset_from_line_col(line_starts, line, col):
    if line < 1 or line > len(line_starts):
        raise ValueError("line out of range: %s" % line)
    return line_starts[line - 1] + col - 1


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _expand_token_end(text: str, end: int) -> int:
    if end < 0:
        return end
    if end >= len(text):
        return len(text) - 1

    # Clang ranges for subexpressions sometimes land on the first character of the
    # last token instead of the final character. Extend identifier-like tails so a
    # range ending at `inv_S1` doesn't truncate to just `i`.
    if _is_ident_char(text[end]):
        while end + 1 < len(text) and _is_ident_char(text[end + 1]):
            end += 1
    return end


def slice_from_node(text: str, node: dict, offset_base: int = 0) -> str:
    offset_range = _range_begin_end_offsets(node)
    if offset_range:
        start, end = offset_range
        start -= offset_base
        end -= offset_base
        if 0 <= start <= end < len(text):
            end = _expand_token_end(text, end)
            return text[start : end + 1]
    rng = _range_begin_end(node)
    if not rng:
        return ""
    line_starts = _line_starts(text)
    (b_line, b_col), (e_line, e_col) = rng
    start = _offset_from_line_col(line_starts, b_line, b_col)
    end = _offset_from_line_col(line_starts, e_line, e_col)
    end = _expand_token_end(text, end)
    return text[start : end + 1]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for child in node.get("inner", []):
            yield from _walk(child)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _strip_wrappers(node):
    current = node
    while isinstance(current, dict) and current.get("kind") in {"ImplicitCastExpr", "ParenExpr"}:
        children = current.get("inner", [])
        if len(children) != 1:
            break
        current = children[0]
    return current


def _find_function(ast_payload, function_name):
    for node in _walk(ast_payload):
        if node.get("kind") == "FunctionDecl" and node.get("name") == function_name:
            return node
    return None


def _find_init_list(var_decl):
    for child in var_decl.get("inner", []):
        if child.get("kind") == "InitListExpr":
            return child
    return None


def _array_size_from_type(type_text: str):
    marker = "["
    if marker not in type_text or "]" not in type_text:
        return None
    try:
        return int(type_text.rsplit("[", 1)[1].split("]", 1)[0])
    except ValueError:
        return None


def _scalar_type_from_qual_type(qual_type: str):
    if qual_type.startswith("const float["):
        return "float"
    if qual_type.startswith("const double["):
        return "double"
    return None


def collect_scalar_targets(function_text: str, ast_payload: dict, function_name: str, supported_sizes=(3,)):
    function_node = _find_function(ast_payload, function_name)
    if function_node is None:
        raise ValueError("FunctionDecl for %s not found in AST" % function_name)

    line_starts = _line_starts(function_text)
    function_offset_range = _range_begin_end_offsets(function_node)
    function_offset_base = function_offset_range[0] if function_offset_range else 0
    targets = []
    seen = set()

    for node in _walk(function_node):
        if node.get("kind") != "VarDecl":
            continue

        type_info = node.get("type", {})
        qual_type = type_info.get("qualType", "")
        scalar_type = _scalar_type_from_qual_type(qual_type)
        if scalar_type is None:
            continue

        array_size = _array_size_from_type(qual_type)
        if array_size not in supported_sizes:
            continue

        init_list = _find_init_list(node)
        if init_list is None:
            continue

        expr_nodes = []
        for child in init_list.get("inner", []):
            stripped = _strip_wrappers(child)
            if stripped.get("kind") not in VALID_EXPR_KINDS and child.get("kind") not in VALID_EXPR_KINDS:
                expr_nodes.append(child)
            else:
                expr_nodes.append(stripped)

        if len(expr_nodes) != array_size:
            continue

        name = node.get("name")
        if not name:
            continue

        rng = _range_begin_end(node)
        if not rng:
            continue
        offset_rng = _range_begin_end_offsets(node)
        if not offset_rng:
            continue
        (b_line, _), _ = rng
        replace_start = offset_rng[0] - function_offset_base
        replace_end = offset_rng[1] - function_offset_base + 1
        # Clang's end offset for VarDecl often stops at the initializer rather
        # than the trailing ';'. Consume the declaration terminator as well so
        # source-to-source replacement does not leave behind an orphan semicolon.
        while replace_end < len(function_text) and function_text[replace_end] in " \t":
            replace_end += 1
        if replace_end < len(function_text) and function_text[replace_end] == ";":
            replace_end += 1
        if replace_start < 0 or replace_end > len(function_text) or replace_start >= replace_end:
            continue
        if replace_start in seen:
            continue
        seen.add(replace_start)

        line_start = line_starts[b_line - 1]
        line_end = function_text.find("\n", line_start)
        if line_end == -1:
            line_end = len(function_text)
        line_text = function_text[line_start:line_end]
        indent = line_text[: len(line_text) - len(line_text.lstrip(" \t"))]
        exprs = [slice_from_node(function_text, expr, function_offset_base).strip() for expr in expr_nodes]

        targets.append(
            {
                "array_name": name,
                "array_size": array_size,
                "scalar_type": scalar_type,
                "indent": indent,
                "line": b_line,
                "replace_start": replace_start,
                "replace_end": replace_end,
                "exprs": exprs,
            }
        )

    targets.sort(key=lambda item: item["replace_start"])
    return targets


def load_ast_json(path):
    return json.loads(Path(path).read_text())
