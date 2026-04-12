#!/usr/bin/env python3

import re


IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")

KEYWORDS = {
    "alignas",
    "auto",
    "bool",
    "break",
    "case",
    "char",
    "class",
    "const",
    "constexpr",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "false",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "namespace",
    "nullptr",
    "operator",
    "private",
    "protected",
    "public",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "true",
    "typedef",
    "typename",
    "uint32_t",
    "union",
    "unsigned",
    "using",
    "void",
    "volatile",
    "while",
}

DECL_HEAD_RE = re.compile(
    r"^\s*(?:const\s+)?(?:auto|bool|double|float|int|uint32_t)\s+([A-Za-z_]\w*)\b",
    re.DOTALL,
)
FOR_DECL_RE = re.compile(
    r"^\s*for\s*\(\s*(?:const\s+)?(?:auto|bool|double|float|int|uint32_t)\s+([A-Za-z_]\w*)\b",
    re.DOTALL,
)
ASSIGN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:[+\-*/%&|^]?=)(?!=)")
PREFIX_INCDEC_RE = re.compile(r"\b(?:\+\+|--)\s*([A-Za-z_]\w*)\b")


class Statement(object):
    def __init__(self, start, end, text, defs=None, uses=None):
        self.start = start
        self.end = end
        self.text = text
        self.defs = defs or set()
        self.uses = uses or set()


class Block(object):
    def __init__(self, start, end, parent=None):
        self.start = start
        self.end = end
        self.parent = parent
        self.children = []
        self.statements = []

    def contains(self, pos):
        return self.start <= pos < self.end


def walk_blocks(block):
    yield block
    for child in block.children:
        for item in walk_blocks(child):
            yield item


def _strip_comments_and_strings(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            out.extend("  ")
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.extend("  ")
                i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(" ")
            i += 1
            while i < n:
                cur = text[i]
                if cur == "\\" and i + 1 < n:
                    out.extend("  ")
                    i += 2
                    continue
                out.append("\n" if cur == "\n" else " ")
                i += 1
                if cur == quote:
                    break
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_brace_tree(clean_text, start, end):
    root = Block(start, end, None)
    stack = [root]
    for pos in range(start, end):
        ch = clean_text[pos]
        if ch == "{":
            block = Block(pos, -1, stack[-1])
            stack[-1].children.append(block)
            stack.append(block)
        elif ch == "}":
            if len(stack) == 1:
                continue
            stack[-1].end = pos + 1
            stack.pop()
    return root


def _top_level_statement_ranges(clean_text, block):
    start = block.start + 1 if block.start > 0 else block.start
    end = block.end - 1 if block.end > 0 else block.end
    stmt_start = None
    paren_depth = 0
    brace_depth = 0
    ranges = []

    pos = start
    while pos < end:
        ch = clean_text[pos]
        if stmt_start is None and not ch.isspace():
            stmt_start = pos
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == ";" and paren_depth == 0 and brace_depth == 0 and stmt_start is not None:
            ranges.append((stmt_start, pos + 1))
            stmt_start = None
        pos += 1
    return ranges


def _statement_defs(text):
    defs = set()
    decl = DECL_HEAD_RE.match(text)
    if decl:
        defs.add(decl.group(1))
    for_decl = FOR_DECL_RE.match(text)
    if for_decl:
        defs.add(for_decl.group(1))
    for match in ASSIGN_RE.finditer(text):
        defs.add(match.group(1))
    for match in PREFIX_INCDEC_RE.finditer(text):
        defs.add(match.group(1))
    return defs


def _statement_uses(text, defs):
    uses = set()
    for token in IDENT_RE.findall(text):
        if token in KEYWORDS or token in defs:
            continue
        uses.add(token)
    return uses


def _populate_statements(block, full_text, clean_text):
    for start, end in _top_level_statement_ranges(clean_text, block):
        text = full_text[start:end]
        defs = _statement_defs(text)
        uses = _statement_uses(text, defs)
        block.statements.append(Statement(start, end, text, defs, uses))
    for child in block.children:
        _populate_statements(child, full_text, clean_text)


def build_ir(function_text, global_offset=0):
    clean = _strip_comments_and_strings(function_text)
    root = _extract_brace_tree(clean, 0, len(function_text))
    _populate_statements(root, function_text, clean)

    def shift(block):
        block.start += global_offset
        block.end += global_offset
        for stmt in block.statements:
            stmt.start += global_offset
            stmt.end += global_offset
        for child in block.children:
            shift(child)

    shift(root)
    return root


def innermost_block(block, pos):
    for child in block.children:
        if child.contains(pos):
            return innermost_block(child, pos)
    return block


def find_statement_index(block, pos):
    for idx, stmt in enumerate(block.statements):
        if stmt.start <= pos < stmt.end:
            return idx
    return None


def definition_sites(block, name):
    return [idx for idx, stmt in enumerate(block.statements) if name in stmt.defs]


def first_use_after(block, start_idx, name):
    for idx in range(start_idx, len(block.statements)):
        if name in block.statements[idx].uses:
            return idx
    return None


def redefinitions_after(block, start_idx, name):
    return [idx for idx in range(start_idx + 1, len(block.statements)) if name in block.statements[idx].defs]


def enclosing_child_index(parent, child):
    for idx, candidate in enumerate(parent.children):
        if candidate is child:
            return idx
    return None
