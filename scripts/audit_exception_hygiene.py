#!/usr/bin/env python3
"""Blanket-``except`` hygiene and module-size ratchet.

Two structural weaknesses in this repository are cheap to measure and easy to
regress silently:

1. **Silver-bullet handlers.** A blanket ``except Exception`` (or a bare
   ``except:``) that neither re-raises nor carries a written rationale turns a
   failure into an ordinary return value. Downstream, "the probe crashed" and
   "the probe succeeded and returned nothing" become the same observation --
   which is the exact confusion this repository's integrity rules exist to
   prevent (``AGENTS.md`` §1.4).

2. **Module size.** A 3,000-line module cannot be reviewed in one sitting, so
   its behaviour is asserted by tests rather than read by a person.

The repository already has the right convention for (1): the handler line
carries ``# noqa: BLE001 - <why this cannot be a typed exception>``, or the
handler re-raises. This script counts the handlers that follow it and the ones
that do not, pins both against a committed baseline, and fails when the tree
gets worse. Counts are allowed to fall without any bookkeeping.

Usage::

    python scripts/audit_exception_hygiene.py            # report
    python scripts/audit_exception_hygiene.py --check    # gate (exit 1 on regression)
    python scripts/audit_exception_hygiene.py --write-baseline

The scan is deliberately scoped to first-party packaged code (``apedp``,
``apedp_dyn``, ``src``). ``sandbox`` and vendored trees are excluded: they are
harnesses and third-party sources, not the library surface.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "apedp.exception-hygiene.v1"

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "benchmarks" / "exception_hygiene_baseline.json"

SCAN_ROOTS = ("apedp", "apedp_dyn", "src")

#: Not policed: harness trees, vendored sources, and build/VCS noise.
EXCLUDED_DIRS = frozenset(
    {
        "sandbox",
        "_qualtran_src",
        "__pycache__",
        "node_modules",
        ".git",
        ".venv",
        "site-packages",
    }
)

#: Exception names that make a handler a silver bullet by construction.
BLANKET_NAMES = frozenset({"Exception", "BaseException"})

#: The specific `noqa` code that records "this catch-all is deliberate".
NOQA_CODE = "BLE001"


@dataclass(frozen=True)
class HandlerRecord:
    """One ``except`` clause and how it was classified."""

    path: str
    line: int
    silenced: bool
    reason: str

    def describe(self) -> str:
        return f"{self.path}:{self.line}  {'SILENT' if self.silenced else 'annotated'}  {self.reason}"


def ruff_ble_exemptions(root: Path = ROOT) -> list[str]:
    """Glob patterns that ``pyproject.toml`` exempts from BLE001 at file level.

    A per-file ignore in the linter config is a stronger statement than an
    inline ``noqa``: it is one reviewable decision covering the whole module,
    recorded where the linter reads it. ``apedp/physics_constraints.py`` uses
    this on purpose (every handler there binds the exception and reports it as
    an ``_unverified`` violation, so the catch-all is the design), and adding
    inline noqas on top only produced ``RUF100`` "unused directive".

    Parsed with a regex rather than ``tomllib`` because this script also runs
    under Python 3.10, where ``tomllib`` does not exist.
    """
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    text = pyproject.read_text(encoding="utf-8")
    marker = "[tool.ruff.lint.per-file-ignores]"
    if marker not in text:
        return []
    section = text.split(marker, 1)[1]
    section = re.split(r"^\[", section, maxsplit=1, flags=re.MULTILINE)[0]
    patterns = []
    for glob, codes in re.findall(r'"([^"]+)"\s*=\s*\[([^\]]*)\]', section, flags=re.DOTALL):
        if "BLE001" in codes:
            patterns.append(glob)
    return patterns


def _exempt(rel: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:]):
            return True
    return False


def iter_python_files(root: Path = ROOT) -> list[Path]:
    """Every first-party Python file in the scanned roots, sorted for stability."""
    found: list[Path] = []
    for name in SCAN_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if EXCLUDED_DIRS.intersection(path.parts):
                continue
            found.append(path)
    return sorted(found)


def _handler_header_lines(tree: ast.Module, handler: ast.ExceptHandler, source: str) -> str:
    """The source text spanning one handler's ``except`` header.

    The rationale comment sits on the ``except`` line itself or on the
    continuation of a parenthesised tuple, so the header is read from the
    ``except`` keyword up to the first body statement.
    """
    lines = source.splitlines()
    start = handler.lineno - 1
    if not handler.body:
        return lines[start] if 0 <= start < len(lines) else ""
    end = min(getattr(stmt, "lineno", start + 1) for stmt in handler.body) - 1
    return "\n".join(lines[start : max(start + 1, end)])


def _is_blanket(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    node = handler.type
    if isinstance(node, ast.Name):
        return node.id in BLANKET_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in BLANKET_NAMES
    if isinstance(node, ast.Tuple):
        return any(
            (isinstance(elt, ast.Name) and elt.id in BLANKET_NAMES)
            or (isinstance(elt, ast.Attribute) and elt.attr in BLANKET_NAMES)
            for elt in node.elts
        )
    return False


def _reraises(handler: ast.ExceptHandler) -> bool:
    """Whether this handler always ends by re-raising.

    A nested ``raise`` counts: the common shape is
    ``except Exception as e: if strict: raise ... from e; log(...)``, where the
    guard decides *whether* to re-raise. Requiring the raise at the handler's
    top level classified those as silent, which understated the tree.
    """
    return any(isinstance(node, ast.Raise) for node in ast.walk(handler))


def classify_source(path_label: str, source: str, exempt: bool = False) -> list[HandlerRecord]:
    """Classify every blanket handler in one module's source."""
    tree = ast.parse(source, filename=path_label)
    records: list[HandlerRecord] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_blanket(node):
            continue
        if exempt:
            records.append(
                HandlerRecord(path_label, node.lineno, False, "file-level BLE001 exemption")
            )
            continue
        header = _handler_header_lines(tree, node, source)
        has_noqa = "noqa" in header and NOQA_CODE in header
        if _reraises(node):
            records.append(HandlerRecord(path_label, node.lineno, False, "re-raises"))
        elif has_noqa:
            records.append(HandlerRecord(path_label, node.lineno, False, f"annotated {NOQA_CODE}"))
        elif "noqa" in header:
            # A `noqa` that names a different code is not a rationale for
            # swallowing everything; the convention is BLE001 or nothing.
            records.append(
                HandlerRecord(path_label, node.lineno, True, "noqa present but not BLE001")
            )
        else:
            records.append(HandlerRecord(path_label, node.lineno, True, "no rationale, no re-raise"))
    return records


def scan(root: Path = ROOT) -> tuple[list[HandlerRecord], dict[str, int]]:
    """Classify every blanket handler and measure every module."""
    records: list[HandlerRecord] = []
    sizes: dict[str, int] = {}
    exemptions = ruff_ble_exemptions(root)
    for path in iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        sizes[rel] = len(text.splitlines())
        try:
            records.extend(classify_source(rel, text, exempt=_exempt(rel, exemptions)))
        except SyntaxError as exc:  # a file we cannot parse cannot be audited
            records.append(HandlerRecord(rel, exc.lineno or 0, True, f"unparseable: {exc.msg}"))
    return records, sizes


def summarise(records: list[HandlerRecord]) -> dict[str, dict[str, int]]:
    """Per-module counts, restricted to modules that have any blanket handler."""
    per_module: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = per_module.setdefault(record.path, {"silent": 0, "annotated": 0})
        bucket["silent" if record.silenced else "annotated"] += 1
    return dict(sorted(per_module.items()))


def build_baseline(records: list[HandlerRecord], sizes: dict[str, int]) -> dict[str, object]:
    """The committed pin: silent counts per module plus a size ceiling."""
    per_module = summarise(records)
    silent_total = sum(bucket["silent"] for bucket in per_module.values())
    annotated_total = sum(bucket["annotated"] for bucket in per_module.values())
    return {
        "schema": SCHEMA,
        "note": (
            "Pinned by scripts/audit_exception_hygiene.py. Silent counts may fall "
            "freely; a rise fails CI. Delete a module's entry once its silent count "
            "reaches zero."
        ),
        "totals": {"silent": silent_total, "annotated": annotated_total},
        "modules": {path: bucket["silent"] for path, bucket in per_module.items() if bucket["silent"]},
        "size_budget": dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:25]),
    }


def check(records: list[HandlerRecord], sizes: dict[str, int], baseline: dict[str, object]) -> list[str]:
    """Return human-readable regressions; an empty list means the gate passes."""
    problems: list[str] = []
    pinned_modules = {str(k): int(v) for k, v in dict(baseline.get("modules") or {}).items()}
    per_module = summarise(records)
    for path, bucket in per_module.items():
        allowed = pinned_modules.get(path, 0)
        if bucket["silent"] > allowed:
            problems.append(
                f"{path}: {bucket['silent']} silent blanket handler(s), pinned at {allowed} "
                f"(+{bucket['silent'] - allowed})"
            )
    silent_total = sum(bucket["silent"] for bucket in per_module.values())
    pinned_total = int((baseline.get("totals") or {}).get("silent", 0))
    if silent_total > pinned_total:
        problems.append(f"tree-wide silent handlers: {silent_total}, pinned at {pinned_total}")

    pinned_sizes = {str(k): int(v) for k, v in dict(baseline.get("size_budget") or {}).items()}
    for path, ceiling in pinned_sizes.items():
        current = sizes.get(path)
        if current is None:
            problems.append(f"{path}: in the size budget but no longer exists; drop its entry")
        elif current > ceiling:
            problems.append(f"{path}: {current} lines, ceiling {ceiling} (+{current - ceiling})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="gate mode: exit 1 on any regression")
    parser.add_argument("--write-baseline", action="store_true", help="re-pin the baseline from the tree")
    parser.add_argument("--verbose", action="store_true", help="list every silent handler with its line")
    args = parser.parse_args(argv)

    records, sizes = scan()
    per_module = summarise(records)
    silent_total = sum(bucket["silent"] for bucket in per_module.values())
    annotated_total = sum(bucket["annotated"] for bucket in per_module.values())

    if args.write_baseline:
        payload = build_baseline(records, sizes)
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE_PATH.relative_to(ROOT)}: {silent_total} silent / {annotated_total} annotated")
        return 0

    print(f"blanket handlers: {silent_total + annotated_total} total, {annotated_total} annotated, {silent_total} silent")
    top = sorted(((p, b["silent"]) for p, b in per_module.items() if b["silent"]), key=lambda kv: -kv[1])[:12]
    for path, count in top:
        print(f"  {count:3d}  {path}")

    if args.verbose:
        for record in records:
            if record.silenced:
                print(record.describe())

    if args.check:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        problems = check(records, sizes, baseline)
        if problems:
            print("\nFAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print(
                "\nEvery new blanket handler needs either a re-raise or "
                "`# noqa: BLE001 - <why a typed exception is impossible here>`.",
                file=sys.stderr,
            )
            return 1
        print("gate: no regression against the pinned baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
