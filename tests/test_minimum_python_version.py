"""The code runs on the oldest Python the package says it supports.

``pyproject.toml`` declares ``requires-python = ">=3.10"`` and CI has a 3.10
leg. ``apedp/ingestion/ingestion_academic_apis.py`` imported ``datetime.UTC``,
which exists from 3.11, and ``apedp/__init__.py`` imports that module, so on
3.10 ``import apedp`` raised ``ImportError`` before anything ran. Nobody saw
it, because the CI workflow was not running
(``docs/SYSTEM_FLAWS_AUDIT.md``, E-07); on its first real run the 3.10 leg
failed at the import and, with ``fail-fast``, took the 3.11 and 3.12 legs down
with it.

The 3.10 CI leg is the real check. This is the fast one, and it works on any
interpreter: every source file must parse under the declared minimum's
grammar, and must not use a standard-library name that the minimum does not
have. The name list is a curated subset, not a proof; it holds what has
actually been seen here plus the additions people most often reach for.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREES = ("apedp", "apedp_dyn", "tests", "scripts", "benchmarks", "tools")
# Vendored upstream checkout: not this repository's code, and not imported.
EXCLUDED_PARTS = {"_qualtran_src", "__pycache__", ".venv", "node_modules"}

#: (module, name) -> first Python minor version that has it.
INTRODUCED: dict[tuple[str, str], int] = {
    ("datetime", "UTC"): 11,
    ("typing", "Self"): 11,
    ("typing", "LiteralString"): 11,
    ("typing", "Never"): 11,
    ("typing", "assert_never"): 11,
    ("typing", "assert_type"): 11,
    ("typing", "reveal_type"): 11,
    ("typing", "TypeVarTuple"): 11,
    ("typing", "Unpack"): 11,
    ("typing", "NotRequired"): 11,
    ("typing", "Required"): 11,
    ("typing", "dataclass_transform"): 11,
    ("typing", "override"): 12,
    ("typing", "TypeAliasType"): 12,
    ("enum", "StrEnum"): 11,
    ("enum", "ReprEnum"): 11,
    ("enum", "verify"): 11,
    ("contextlib", "chdir"): 11,
    ("hashlib", "file_digest"): 11,
    ("math", "cbrt"): 11,
    ("math", "exp2"): 11,
    ("operator", "call"): 11,
    ("asyncio", "TaskGroup"): 11,
    ("asyncio", "timeout"): 11,
    ("itertools", "batched"): 12,
    ("pathlib", "UnsupportedOperation"): 13,
}
#: Whole modules -> first minor version that ships them.
MODULE_INTRODUCED = {"tomllib": 11}
#: Builtins -> first minor version that has them.
BUILTIN_INTRODUCED = {"ExceptionGroup": 11, "BaseExceptionGroup": 11}


def declared_minimum() -> int:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*">=\s*3\.(\d+)"', text, re.M)
    assert match, "pyproject.toml no longer declares requires-python as '>=3.N'"
    return int(match.group(1))


def source_files() -> list[Path]:
    files = []
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if not EXCLUDED_PARTS & set(path.parts):
                files.append(path)
    return files


def too_new(tree: ast.AST, minimum: int) -> list[str]:
    """Uses of names newer than ``3.minimum`` that are not guarded.

    An import inside ``try:`` (the usual ``except ImportError`` fallback) or
    under ``if sys.version_info ...`` / ``if TYPE_CHECKING`` is a guarded use
    and is not reported.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.If)):
            for child in ast.walk(node):
                guarded.add(id(child))
    aliases: dict[str, str] = {}  # local name -> stdlib module it is bound to
    found: list[str] = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                aliases[alias.asname or top] = top
                if MODULE_INTRODUCED.get(alias.name, 0) > minimum:
                    found.append(
                        f"line {node.lineno}: import {alias.name} (3.{MODULE_INTRODUCED[alias.name]}+)"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if MODULE_INTRODUCED.get(node.module, 0) > minimum:
                found.append(
                    f"line {node.lineno}: from {node.module} import ... (3.{MODULE_INTRODUCED[node.module]}+)"
                )
            for alias in node.names:
                since = INTRODUCED.get((node.module, alias.name), 0)
                if since > minimum:
                    found.append(
                        f"line {node.lineno}: from {node.module} import {alias.name} (3.{since}+)"
                    )
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id)
            since = INTRODUCED.get((module, node.attr), 0) if module else 0
            if since > minimum:
                found.append(f"line {node.lineno}: {module}.{node.attr} (3.{since}+)")
        elif isinstance(node, ast.Name) and BUILTIN_INTRODUCED.get(node.id, 0) > minimum:
            found.append(f"line {node.lineno}: {node.id} (3.{BUILTIN_INTRODUCED[node.id]}+)")
    return found


def test_the_declared_minimum_is_what_ci_tests() -> None:
    minimum = declared_minimum()
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert f"'3.{minimum}'" in workflow or f'"3.{minimum}"' in workflow, (
        f"requires-python says 3.{minimum} but no CI leg runs it"
    )


def test_every_source_file_parses_under_the_minimum_grammar() -> None:
    minimum = declared_minimum()
    failures = []
    for path in source_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, minimum))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not failures, f"not valid Python 3.{minimum}:\n" + "\n".join(failures)


def test_no_unguarded_standard_library_name_newer_than_the_minimum() -> None:
    minimum = declared_minimum()
    failures = []
    for path in source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        failures += [f"{path.relative_to(ROOT)}: {use}" for use in too_new(tree, minimum)]
    assert not failures, f"needs a Python newer than 3.{minimum}:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from datetime import UTC, datetime\n", ["line 1: from datetime import UTC (3.11+)"]),
        ("import datetime as dt\nx = dt.UTC\n", ["line 2: datetime.UTC (3.11+)"]),
        ("import tomllib\n", ["line 1: import tomllib (3.11+)"]),
        ("from enum import StrEnum\n", ["line 1: from enum import StrEnum (3.11+)"]),
        ("raise ExceptionGroup('g', [])\n", ["line 1: ExceptionGroup (3.11+)"]),
        # Guarded uses and the portable spelling are not reported.
        ("try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n", []),
        ("import sys\nif sys.version_info >= (3, 11):\n    from datetime import UTC\n", []),
        ("from datetime import datetime, timezone\nnow = datetime.now(timezone.utc)\n", []),
    ],
)
def test_the_detector_detects(source: str, expected: list[str]) -> None:
    """A check that cannot fail is not a check: the defect itself, replayed."""
    assert too_new(ast.parse(source), 10) == expected


def test_a_newer_floor_would_accept_the_newer_names() -> None:
    assert too_new(ast.parse("from datetime import UTC\n"), 11) == []
