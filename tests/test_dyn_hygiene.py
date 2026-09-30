"""Determinism-hygiene guards (regression boundary).

Two process flaws recurred during development and were caught by forensics
instead of by tests:
  1. hash() on str is salted per process (PYTHONHASHSEED), which once made a
     full sweep irreproducible. Seeding must never use bare hash().
  2. Duplicate method definitions silently shadow earlier ones in the same
     class body (an edit artifact resumed an obsolete test version).

Both are now asserted structurally.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DYN = ROOT / "apedp_dyn"


def _modules():
    return sorted(DYN.rglob("*.py")) + sorted(
        p for p in (ROOT / "tests").glob("test_dyn_*.py"))


class TestNoBareHash:
    def test_no_bare_hash_calls_in_dyn(self):
        offenders = []
        for path in _modules():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "hash":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        assert offenders == [], f"bare hash() seeding paths: {offenders}"

    def test_cell_seeds_pinned(self):
        from apedp_dyn.dyn_sweep import cell_seed

        assert cell_seed("DUFF", 0, 1) == 1563138
        assert cell_seed("STAB", 2, 10) == 1631941
        assert cell_seed("COUP", 1, 5) == 1820834
        assert cell_seed("DUFF_C7BIG", 0, 3) == 1032805


class TestNoDuplicateDefinitions:
    def test_no_shadowed_defs(self):
        from collections import Counter

        offenders = []
        for path in _modules():
            tree = ast.parse(path.read_text())
            # Module level + each class body (nested functions are closures,
            # not shadowing hazards for test collection).
            seen: dict[str, int] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seen[node.name] = seen.get(node.name, 0) + 1
                elif isinstance(node, ast.ClassDef):
                    inner = Counter(
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    for name, count in inner.items():
                        if count > 1:
                            offenders.append(f"{path.relative_to(ROOT)}:{node.name}.{name}")
            for name, count in seen.items():
                if count > 1:
                    offenders.append(f"{path.relative_to(ROOT)}:module.{name}")
        assert offenders == [], f"shadowed definitions: {offenders}"
