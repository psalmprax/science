#!/usr/bin/env python3
"""Which ``apedp_dyn`` modules no test names.

``tests/test_smoke_imports.py`` imports every module, so an import-time break is
caught. That is not the same claim as "this module is exercised": a module can
import cleanly, be wrong, and be touched by nothing. This script measures the
gap instead of leaving it to be discovered.

A module counts as referenced when some file under ``tests/`` names its dotted
path or its module name as a whole word. The rule is deliberately generous --
it measures *attention*, not coverage -- and the register it produces is a work
list with a ceiling rather than a certificate. ``pinned_ceiling`` is the number
of unreferenced modules the repository has accepted; a new module that nothing
names fails the pin until it is cited by a test or the ceiling is raised on
purpose, in review.

Usage::

    python scripts/audit_untested_modules.py            # report
    python scripts/audit_untested_modules.py --check    # gate
    python scripts/audit_untested_modules.py --write    # re-pin the register

Status: audit-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "apedp.untested-module-register/v1"

ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = "apedp_dyn"
OUT = ROOT / "benchmarks" / "untested_modules_register.json"


def module_paths(root: Path = ROOT) -> dict[str, str]:
    """Map repository-relative module path to its dotted import name."""
    base = root / MODULE_ROOT
    modules: dict[str, str] = {}
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts and all(part.isidentifier() for part in parts):
            modules[path.relative_to(root).as_posix()] = ".".join(parts)
    return modules


def test_corpus(root: Path = ROOT) -> str:
    """Every test source, concatenated: the only evidence of attention there is."""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "tests").rglob("*.py"))
    )


def is_referenced(dotted: str, corpus: str) -> bool:
    """Whether a test reaches this module: dotted path, import, or from-import.

    A bare word match is not enough. ``server``, ``referee`` and ``ledger``
    appear in prose and in unrelated identifiers, so counting them would report
    coverage that does not exist. Only a reference the interpreter would resolve
    counts: the dotted path anywhere, ``import <dotted>``, or
    ``from <package> import <stem>``.
    """
    if dotted in corpus:
        return True
    if re.search(rf"\bimport\s+{re.escape(dotted)}\b", corpus):
        return True
    package, _, stem = dotted.rpartition(".")
    if not package:
        return False
    # A from-import may continue across lines inside parentheses::
    #
    #     from apedp_dyn.model import (
    #         PhysicalModel,
    #         StateVariable,
    #     )
    #
    # The one-line form missed those, so a test that imported a module that way
    # was scored as never naming it. The first case found was this audit's own
    # coverage file, which imports nine benchmark problems exactly that way.
    for match in re.finditer(
        rf"\bfrom\s+{re.escape(package)}\s+import\s+(\([^()]*\)|[^\n]+)", corpus, re.S
    ):
        if re.search(rf"\b{re.escape(stem)}\b", match.group(1)):
            return True
    return False


def build(root: Path = ROOT) -> dict[str, object]:
    corpus = test_corpus(root)
    modules = module_paths(root)
    unreferenced = sorted(
        path for path, dotted in modules.items() if not is_referenced(dotted, corpus)
    )
    return {
        "schema": SCHEMA,
        "module_root": MODULE_ROOT,
        "basis": (
            "A module counts as referenced when a tests/ source names its dotted path, "
            "imports it, or imports its name from its package -- including a "
            "from-import that continues across lines inside parentheses. A bare word match does "
            "not count: prose and unrelated identifiers are not coverage. Import "
            "coverage from test_smoke_imports.py is separate and is not counted here, "
            "because importing a module is not testing it."
        ),
        "pinned_ceiling": len(unreferenced),
        "totals": {
            "modules": len(modules),
            "referenced": len(modules) - len(unreferenced),
            "unreferenced": len(unreferenced),
        },
        "unreferenced": unreferenced,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the tree disagrees with the pinned register")
    parser.add_argument("--write", action="store_true",
                        help="re-pin the register from the tree")
    args = parser.parse_args(argv)

    payload = build()
    totals = payload["totals"]
    assert isinstance(totals, dict)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}: {totals['unreferenced']} unreferenced "
              f"of {totals['modules']}")
        return 0

    print(f"apedp_dyn modules: {totals['modules']}; named by a test: "
          f"{totals['referenced']}; named by none: {totals['unreferenced']}")

    if args.check:
        if not OUT.is_file():
            print(f"FAILED: {OUT.relative_to(ROOT)} is missing; run --write")
            return 1
        recorded = json.loads(OUT.read_text(encoding="utf-8"))
        # Narrow the JSON values explicitly: ``build()`` is typed as a mapping of
        # objects, and a register that is not a list of paths is a malformed
        # artefact worth reporting as an empty set rather than a crash.
        fresh_obj = payload["unreferenced"]
        pinned_obj = recorded.get("unreferenced", [])
        fresh = [str(p) for p in fresh_obj] if isinstance(fresh_obj, list) else []
        pinned = [str(p) for p in pinned_obj] if isinstance(pinned_obj, list) else []
        if fresh != pinned:
            added = sorted(set(fresh) - set(pinned))
            removed = sorted(set(pinned) - set(fresh))
            print("FAILED: the register is stale", file=sys.stderr)
            for path in added:
                print(f"  - new unreferenced module: {path}", file=sys.stderr)
            for path in removed:
                print(f"  - now referenced (drop it from the register): {path}", file=sys.stderr)
            return 1
        print("gate: the register matches the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
