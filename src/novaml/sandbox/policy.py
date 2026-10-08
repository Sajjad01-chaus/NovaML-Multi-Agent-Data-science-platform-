"""Static policy for LLM-written analysis code.

This is the first of two layers (the second is process isolation in
`executor.py`). It rejects code before it runs: imports outside an allowlist,
dangerous builtins, dunder attribute access (the classic ``().__class__.__mro__``
escape) and pandas/numpy I/O methods that could read or write files.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = frozenset({"pandas", "numpy", "math", "statistics", "re", "datetime", "collections", "itertools", "scipy"})

FORBIDDEN_NAMES = frozenset(
    {
        "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
        "memoryview", "object", "type", "super", "classmethod", "staticmethod", "property",
    }
)

# pandas/numpy attributes that touch the filesystem, network or evaluate strings.
FORBIDDEN_ATTRS = frozenset(
    {"eval", "query", "load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile",
     "genfromtxt", "memmap", "system", "popen", "ctypeslib", "lib", "f2py", "testing"}
)
SAFE_TO_ATTRS = frozenset({"to_string", "to_dict", "to_list", "to_numpy", "to_frame", "to_period", "to_timestamp", "to_datetime", "to_numeric", "to_timedelta"})

MAX_CODE_CHARS = 6000


class PolicyViolation(ValueError):
    pass


def check(code: str) -> None:
    if len(code) > MAX_CODE_CHARS:
        raise PolicyViolation(f"code longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PolicyViolation(f"syntax error: {e.msg} (line {e.lineno})") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                parts = n.split(".")
                if parts[0] not in ALLOWED_IMPORTS or any(p in FORBIDDEN_ATTRS or p.startswith("_") for p in parts[1:]):
                    raise PolicyViolation(f"import of {n!r} is not allowed")
            if isinstance(node, ast.ImportFrom) and node.level:
                raise PolicyViolation("relative imports are not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise PolicyViolation(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                raise PolicyViolation(f"private/dunder attribute {a!r} is not allowed")
            if a in FORBIDDEN_ATTRS or a.startswith("read_") or (a.startswith("to_") and a not in SAFE_TO_ATTRS):
                raise PolicyViolation(f"attribute {a!r} is not allowed (I/O or dynamic evaluation)")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            raise PolicyViolation("global/nonlocal are not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise PolicyViolation("string constants containing '__' are not allowed")
