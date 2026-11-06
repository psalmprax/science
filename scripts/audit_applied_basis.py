"""Audit the applied-science modules for unlabelled numerical returns.

Every function in ``apedp/applied_*.py`` answers a question whose name sounds
predictive ("design a chip", "analyse DAC technology") with a dictionary of reference
numbers. A dictionary of plausible constants is indistinguishable from a computed
result unless it says which it is, so each returned dictionary must carry a label: its
evidence basis, whether it was measured, or what it did not model.

This script walks the AST: for every function, each ``return { ... }`` literal is checked
for at least one label key. A return of a bare constant is reported separately, because a
constant return is a claim that needs reading rather than a labelled table.

Usage::

    python3 scripts/audit_applied_basis.py
    python3 scripts/audit_applied_basis.py --check   # exit 1 on any unlabelled return

Status: audit-only
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "applied_basis_report.json"
LABELS = frozenset({
    "evidence", "measured", "not_modelled", "basis", "is_computed", "predictive",
    "reference", "is_measured",
})


def _dict_keys(node: ast.Dict) -> set[str]:
    return {key.value for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def scan_module(path: pathlib.Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: list[dict] = []
    unlabelled: list[str] = []
    constants: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dict_returns = 0
        labelled = 0
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            value = child.value
            if isinstance(value, ast.Dict):
                dict_returns += 1
                if _dict_keys(value) & LABELS:
                    labelled += 1
                else:
                    unlabelled.append(f"{node.name}: return {{...}} without a basis label")
            elif isinstance(value, ast.Constant) and not isinstance(value.value, str):
                constants.append(f"{node.name}: returns the constant {value.value!r}")
        if dict_returns:
            functions.append({"function": node.name, "dict_returns": dict_returns,
                              "labelled": labelled})
    return {"module": str(path.relative_to(ROOT)), "functions": functions,
            "unlabelled": unlabelled, "constant_returns": constants}


def build() -> dict:
    modules = sorted(ROOT.glob("apedp/applied_*.py"))
    rows = [scan_module(path) for path in modules]
    return {
        "schema": "apedp.applied_basis_report/v1",
        "label_keys": sorted(LABELS),
        "modules": rows,
        "totals": {
            "modules": len(rows),
            "functions_with_dict_returns": sum(len(row["functions"]) for row in rows),
            "dict_returns": sum(item["dict_returns"]
                                for row in rows for item in row["functions"]),
            "unlabelled_returns": sum(len(row["unlabelled"]) for row in rows),
            "constant_returns": sum(len(row["constant_returns"]) for row in rows),
        },
        "verdict": (
            "A module whose returns all carry a label states its own basis. An unlabelled "
            "return is a number a reader cannot tell from a computed one."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when any returned dictionary carries no basis label")
    args = parser.parse_args(argv)
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    totals = report["totals"]
    print(f"applied modules: {totals['modules']}  labelled dict returns: "
          f"{totals['dict_returns'] - totals['unlabelled_returns']}/{totals['dict_returns']}")
    for row in report["modules"]:
        for problem in row["unlabelled"]:
            print(f"UNLABELLED {row['module']}: {problem}")
        for problem in row["constant_returns"]:
            print(f"CONSTANT   {row['module']}: {problem}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    if args.check and totals["unlabelled_returns"]:
        print("an applied return cannot be read without a basis label", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
