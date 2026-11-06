"""Static-table census for the ``apedp`` package.

Finds functions that return a value read from a module- or class-level container
literal rather than computed, so a lookup table cannot quietly pass for a model.

Two exclusions keep the output honest:

* A name that is also an instance attribute (``self.errors``) or a class-body
  annotated field is a field, not a table, even if some class-level literal
  shares the name. Without this, every ``to_dict`` that returns its own fields
  is reported as a table lookup.
* The one exception is a ``ClassVar`` carrying a container literal: that is a
  table by declaration, and ``VectorizedDimensionalEngine.DIM_MAP`` is read
  through ``self`` while being exactly such a table. Requiring a bare ``ClassVar``
  annotation keeps the field exclusion from swallowing it.
* A return that touches any parameter is a computation over the inputs and is
  not a pure lookup, so it is not reported.

Output is JSON: ``{module: [[function, [tables...]], ...]}``. Sorted, so two runs
on the same tree compare equal and a document can be checked against it.

``--check`` turns the census into a gate: it compares the live tree against
``scripts/static_table_census.json`` and exits non-zero when a table appears,
disappears or changes its module. The record is the only place a table may be
added, and a new row there also needs a line in ``docs/SCAFFOLD_INVENTORY.md``
(``tests/test_scaffold_inventory.py`` enforces both). Exit codes: ``0`` recorded,
``1`` drifted, ``2`` the check could not run.

Without ``--check`` the script still prints the census for a root, which is how
the inventory was built.

Status: audit-only
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BASELINE_PATH = SCRIPT_DIR / "static_table_census.json"

_CONTAINERS = (ast.Dict, ast.List, ast.Tuple, ast.Set)


def _assigned_names(targets: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _class_body_tables(body: list[ast.stmt]) -> dict[str, int]:
    """Container literals assigned in a class body."""
    tables: dict[str, int] = {}
    for statement in body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        else:
            continue
        if value is not None and isinstance(value, _CONTAINERS):
            for name in _assigned_names([t for t in targets if isinstance(t, ast.Name)]):
                tables[name] = statement.lineno
    return tables


def _state_and_classvar_tables(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return (state names, ClassVar tables).

    A ``ClassVar`` container literal is a table even when a method reaches it
    through ``self``; every other attribute reached through ``self`` is state.
    """
    state: set[str] = set()
    classvar_tables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            state.add(node.attr)
        elif isinstance(node.ctx, ast.Store):
            state.add(node.attr)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            value = statement.value
            # An annotation is a field when it has no value or is ``field(...)``.
            is_field = value is None or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "field"
            )
            if is_field:
                state.add(statement.target.id)
            elif isinstance(value, _CONTAINERS) and "ClassVar" in ast.unparse(statement.annotation):
                classvar_tables.add(statement.target.id)
    return state, classvar_tables


def _names_touched(node: ast.expr) -> set[str]:
    touched: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            touched.add(child.id)
        elif isinstance(child, ast.Attribute):
            touched.add(child.attr)
    return touched


def census(root: str | pathlib.Path = "apedp") -> dict[str, list[list[object]]]:
    """Every function that returns a value read from a static container."""
    found: dict[str, list[list[object]]] = {}
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        module_tables: dict[str, int] = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets, value = [statement.target], statement.value
            else:
                continue
            if value is not None and isinstance(value, _CONTAINERS):
                for name in _assigned_names([t for t in targets if isinstance(t, ast.Name)]):
                    module_tables[name] = statement.lineno

        class_tables: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_tables.update(_class_body_tables(node.body))

        state, classvar_tables = _state_and_classvar_tables(tree)
        tables = {name: line for name, line in {**module_tables, **class_tables}.items()
                  if name not in state or name in classvar_tables}
        if not tables:
            continue

        entries: set[tuple[str, tuple[str, ...]]] = set()
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {a.arg for a in function.args.args + function.args.kwonlyargs}
            params -= {"self", "cls"}
            for returned in (n for n in ast.walk(function)
                             if isinstance(n, ast.Return) and n.value is not None):
                touched = _names_touched(returned.value)
                used = tuple(sorted(touched & set(tables)))
                if used and not (touched & params):
                    entries.add((function.name, used))

        if entries:
            found[str(path)] = [[name, list(used)] for name, used in sorted(entries)]
    return found


def read_recorded(path: pathlib.Path = BASELINE_PATH) -> dict[str, Any]:
    """The recorded census: roots, excluded trees, and module -> tables."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def census_sets(
    root_name: str, excluded: tuple[str, ...] | list[str] = ()
) -> dict[str, set[str]]:
    """{repo-relative module: tables returned} for one package root.

    A root outside the repository (a fixture, a trial tree) is keyed against its
    own parent, so the record can be exercised without writing into the repo.
    """
    found: dict[str, set[str]] = {}
    root = pathlib.Path(root_name)
    if not root.is_absolute():
        root = REPO_ROOT / root
    repo_root = REPO_ROOT.resolve()
    root_resolved = root.resolve()
    base = repo_root if root_resolved.is_relative_to(repo_root) else root_resolved.parent
    for path, entries in census(root).items():
        relative = pathlib.Path(path).resolve().relative_to(base).as_posix()
        if any(relative.startswith(prefix) for prefix in excluded):
            continue
        found[relative] = {table for _, tables in entries for table in tables}
    return found


def recorded_sets(baseline: dict[str, Any], root_name: str) -> dict[str, set[str]]:
    prefix = root_name.rstrip("/") + "/"
    recorded: dict[str, list[str]] = baseline["recorded"]
    return {
        module: set(tables)
        for module, tables in recorded.items()
        if module.startswith(prefix)
    }


def drift(
    found: dict[str, set[str]], recorded: dict[str, set[str]]
) -> dict[str, Any]:
    """What changed, split so a message can name the offending table."""
    shared = sorted(set(found) & set(recorded))
    return {
        "unrecorded_modules": sorted(set(found) - set(recorded)),
        "vanished_modules": sorted(set(recorded) - set(found)),
        "unrecorded_tables": {
            module: sorted(found[module] - recorded[module])
            for module in shared if found[module] - recorded[module]
        },
        "vanished_tables": {
            module: sorted(recorded[module] - found[module])
            for module in shared if recorded[module] - found[module]
        },
    }


def check(baseline_path: pathlib.Path = BASELINE_PATH) -> int:
    """Compare the live census against the record. ``0`` recorded, ``1`` drifted."""
    baseline = read_recorded(baseline_path)
    excluded = tuple(baseline.get("excluded", ()))
    report: dict[str, Any] = {}
    failed = False
    for root_name in baseline["roots"]:
        found = census_sets(root_name, excluded)
        recorded = recorded_sets(baseline, root_name)
        report[root_name] = {
            "modules": len(found),
            "tables": sum(len(tables) for tables in found.values()),
            **drift(found, recorded),
        }
        failed = failed or any(
            report[root_name][key]
            for key in ("unrecorded_modules", "vanished_modules",
                        "unrecorded_tables", "vanished_tables")
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        print(
            "static-table census: the tree does not match "
            f"{baseline_path.name}. Record the table (and its row in "
            "docs/SCAFFOLD_INVENTORY.md) or remove it.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if "--check" in arguments:
        arguments.remove("--check")
        baseline = BASELINE_PATH
        if "--baseline" in arguments:
            index = arguments.index("--baseline")
            baseline = pathlib.Path(arguments[index + 1])
            del arguments[index:index + 2]
        return check(baseline)
    root = (arguments or ["apedp"])[0]
    print(json.dumps(census(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
