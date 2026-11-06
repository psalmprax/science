"""Scope scan: a predictive name may not return something the function did not produce.

The static-table census (``scripts/audit_static_tables.py``) answers *what does
this function return*; it prints a table lookup that way whatever the function is
called. That is the right question for an inventory and the wrong one for a gate,
because a units registry called ``get_unit_vector`` is fine and a lookup called
``predict_off_targets`` is the failure this repository exists to prevent.

This scan asks the narrower question. A function is reported only when **all**
three hold:

1. its name says it produces an answer about the world -- one of the prefixes in
   :data:`PREDICTIVE_PREFIXES`, matched at the start of the name (``predict_*``,
   ``design_*``, ``discover_*``, ``rank``, ``*_projection``...);
2. it returns without reading any parameter, so the value cannot be a function of
   the inputs; and
3. what it returns is either a literal the body does not compute (a fixed dict, a
   constant string or number) or a module/class-level container, as found by the
   census.

A function that computes from its inputs, or that raises instead of returning, is
not reported: refusing is the behaviour the repository asks for, and
``optimize_confinement`` raises on purpose.

Exit status is the check:

* ``0`` -- findings are exactly the recorded baseline;
* ``1`` -- a new finding appeared, or a recorded one disappeared, because either
  means the record below is stale;
* ``2`` -- the scan could not run.

The baseline is ``scripts/scope_scan_baseline.json``. Every entry needs a ``why``
and a matching row in ``docs/SCAFFOLD_INVENTORY.md``; a test enforces both, so a
new constant cannot be silenced by adding it to this file alone.

A recorded entry states *why the name is tolerated*, never that the return is
computed: the reason is the excuse, and the scan still knows the function is a
constant.

Status: audit-only
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BASELINE_PATH = SCRIPT_DIR / "scope_scan_baseline.json"

#: Name prefixes that promise an answer about the world. ``get_``/``_`` are
#: stripped before matching, so ``_search_targets`` and ``get_rank`` are covered.
PREDICTIVE_PREFIXES = (
    "predict",
    "predicted",
    "prediction",
    "estimate",
    "forecast",
    "project",
    "design",
    "optimize",
    "optimise",
    "discover",
    "search",
    "recommend",
    "select",
    "infer",
    "propose",
    "classify",
    "diagnose",
    "rank",
    "score",
)

#: The prefix must end at a word boundary: ``proposer_id`` and
#: ``selected_descriptor`` are properties of a proposer and a chosen slot, not
#: promises to propose a candidate or to select a feature from data.
_PREFIX = re.compile(
    r"^(?:" + "|".join(PREDICTIVE_PREFIXES) + r")(?:_|$)", re.IGNORECASE
)

#: Roots the repo check covers. ``--root``-style override keeps the scan reusable
#: on a package it does not gate yet.
DEFAULT_ROOTS = ("apedp", "apedp_dyn")

#: Directories that are vendored, generated, or not package code. A scan that
#: walked these would report other people's tables as this repository's.
SKIPPED_PARTS = frozenset(
    {
        "tests",
        "__pycache__",
        "_qualtran_src",
        "node_modules",
        "benchmark_corpus",
        "benchmark_corpus_synthesis",
        "benchmark_corpus_synthesis_50",
        "benchmark_corpus_external_60",
    }
)

_LITERAL_SCALARS = (ast.Constant,)


def is_predictive_name(name: str) -> bool:
    """True when the function name promises a prediction, ranking or design."""
    stripped = name.lstrip("_")
    if stripped.startswith("get_"):
        stripped = stripped[len("get_"):]
    return bool(_PREFIX.match(stripped))


def _is_literal(node: ast.expr) -> bool:
    """A value the function body did not compute from anything."""
    if isinstance(node, _LITERAL_SCALARS):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    if isinstance(node, ast.Dict):
        return all(
            part is None or _is_literal(part)
            for part in (*node.keys, *node.values)
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(element) for element in node.elts)
    return False


def _names_touched(node: ast.expr) -> set[str]:
    touched: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            touched.add(child.id)
        elif isinstance(child, ast.Attribute):
            touched.add(child.attr)
    return touched


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = function.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names - {"self", "cls"}


def _module_paths(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path for path in root.rglob("*.py")
        if not SKIPPED_PARTS.intersection(path.parts)
    )


def _census(
    root_path: pathlib.Path, display: str
) -> dict[str, set[tuple[str, tuple[str, ...]]]]:
    """{display-qualified module: {(function, tables)}} from the static-table census."""
    spec = importlib.util.spec_from_file_location(
        "audit_static_tables", SCRIPT_DIR / "audit_static_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    for path, entries in module.census(root_path).items():
        relative = pathlib.Path(path).resolve().relative_to(root_path).as_posix()
        found[f"{display.rstrip('/')}/{relative}"] = {
            (name, tuple(tables)) for name, tables in entries
        }
    return found


def _body_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every name the body reads or writes, at any depth."""
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _finding_bases(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    tables: set[tuple[str, tuple[str, ...]]],
) -> list[str]:
    """Why this function is a constant, or an empty list if it is not."""
    parameters = _parameters(function)
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return []  # it raises instead of answering: a refusal, not a constant

    per_return = [_names_touched(returned.value) for returned in returns]
    if any(touched & parameters for touched in per_return):
        return []  # at least one path answers from the inputs

    own_tables = {
        table for name, table_names in tables if name == function.name for table in table_names
    }
    bases: list[str] = []
    # A literal is a constant only when the body never reads a parameter at all:
    # labels chosen by branching on the inputs are control flow, not fabrication.
    if all(_is_literal(returned.value) for returned in returns) and not (
        parameters & _body_names(function)
    ):
        bases.append("literal")
    # A table lookup counts only when every path is a lookup; a sieve that
    # classifies a frame and then maps the class to a table is doing the work.
    if own_tables and all(touched & own_tables for touched in per_return):
        bases.extend(f"table:{table}" for table in sorted(own_tables))
    return sorted(set(bases))


def scan(
    root: str | pathlib.Path = "apedp", *, display: str | None = None
) -> list[dict[str, object]]:
    """Every predictive-named function that returns an uncomputed constant.

    ``display`` is the prefix recorded in the baseline (``apedp``), so a caller
    that passes an absolute path still gets the same keys the file holds.
    """
    root_path = pathlib.Path(root).resolve()
    display = display if display is not None else str(root)
    census = _census(root_path, display)
    findings: list[dict[str, object]] = []
    for path in _module_paths(root_path):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        module = path.relative_to(root_path).as_posix()
        module = f"{display.rstrip('/')}/{module}"
        tables = census.get(module, set())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not is_predictive_name(node.name):
                continue
            bases = _finding_bases(node, tables)
            if bases:
                findings.append(
                    {"module": module, "function": node.name, "bases": bases}
                )
    findings.sort(key=lambda f: (str(f["module"]), str(f["function"])))
    return findings


def _key(finding: dict[str, object]) -> tuple[str, str]:
    return str(finding["module"]), str(finding["function"])


def read_baseline(
    path: pathlib.Path = BASELINE_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (entry["module"], entry["function"]): entry  # type: ignore[index]
        for entry in data["recorded"]
    }


def compare(
    findings: list[dict[str, object]],
    baseline: dict[tuple[str, str], dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """New findings and vanished recordings, both of which mean the record drifted."""
    seen = {_key(finding) for finding in findings}
    new = [finding for finding in findings if _key(finding) not in baseline]
    stale = [
        dict(baseline[key], module=key[0], function=key[1])
        for key in sorted(baseline) if key not in seen
    ]
    return {"new": new, "stale": stale}


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    roots: tuple[str, ...] = DEFAULT_ROOTS
    if arguments and not arguments[0].startswith("-"):
        roots = (arguments.pop(0),)
    use_baseline = "--no-baseline" not in arguments

    exit_code = 0
    recorded = read_baseline() if use_baseline else {}
    for root in roots:
        findings = scan(root)
        if not use_baseline:
            print(json.dumps({"root": root, "findings": findings}, indent=2))
            continue
        prefix = root.rstrip("/") + "/"
        baseline = {key: value for key, value in recorded.items() if key[0].startswith(prefix)}
        report = compare(findings, baseline)
        print(json.dumps({"root": root, "findings": findings, **report}, indent=2))
        if report["new"] or report["stale"]:
            print(
                f"scope scan: {len(report['new'])} unrecorded, "
                f"{len(report['stale'])} stale for root {root!r}",
                file=sys.stderr,
            )
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError) as error:
        print(f"scope scan could not run: {error}", file=sys.stderr)
        raise SystemExit(2) from None
