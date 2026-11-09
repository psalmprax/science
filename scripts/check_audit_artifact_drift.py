#!/usr/bin/env python3
"""Fail when a generated audit artefact disagrees with the tree it describes.

Each committed artefact below is a measurement, not a claim: it is produced by a
script that reads the tree. A test can assert that equality, but only for the
tests a given run selects, and a hand-edited report passes every test that does
not rebuild it. This gate rebuilds all of them in one place and diffs, so an
artefact cannot be tuned by hand past CI.

The failure this exists for: ``benchmarks/formal/lean_axiom_audit.json`` once
reported 0 theorems proved circularly while a rebuild measured 94, and the
number had been written rather than generated. Regenerating is allowed and
explicit (``--write``); editing the JSON is not.

Exit codes:
    0  every artefact matches its generator (or ``--check`` was not given)
    1  at least one artefact is stale, missing, or its generator failed
    2  usage error

Status: audit-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _lean_audit() -> dict[str, Any]:
    from scripts.audit_lean_formal_layer import build

    return build()


def _applied_basis() -> dict[str, Any]:
    from scripts.audit_applied_basis import build

    return build()


def _untested_modules() -> dict[str, Any]:
    from scripts.audit_untested_modules import build

    return build()


#: Artefacts rebuilt in-process: the committed JSON must equal the payload the
#: generator produces from the current tree.
REBUILT: dict[str, Callable[[], dict[str, Any]]] = {
    "benchmarks/formal/lean_axiom_audit.json": _lean_audit,
    "benchmarks/applied_basis_report.json": _applied_basis,
    "benchmarks/untested_modules_register.json": _untested_modules,
}

#: Artefacts whose own ``--check`` compares the tree to the record it reads.
CHECKED_BY_COMMAND: dict[str, list[str]] = {
    "scripts/static_table_census.json": [
        sys.executable,
        "scripts/audit_static_tables.py",
        "--check",
    ],
}


def _short(value: object, width: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _diff(recorded: Any, rebuilt: Any, prefix: str = "") -> list[str]:
    """Name the fields that disagree, so the message says what to regenerate."""
    if isinstance(recorded, dict) and isinstance(rebuilt, dict):
        problems: list[str] = []
        for key in sorted(set(recorded) | set(rebuilt)):
            label = f"{prefix}{key}"
            if key not in recorded:
                problems.append(f"{label}: produced by the generator, absent from the artefact")
            elif key not in rebuilt:
                problems.append(f"{label}: in the artefact, no longer produced by the generator")
            else:
                problems.extend(_diff(recorded[key], rebuilt[key], f"{label}."))
        return problems
    if isinstance(recorded, list) and isinstance(rebuilt, list):
        if len(recorded) != len(rebuilt):
            return [f"{prefix or '<root>'}: {len(recorded)} recorded entries vs "
                    f"{len(rebuilt)} rebuilt"]
        if recorded != rebuilt:
            differing = [index for index, (left, right) in enumerate(zip(recorded, rebuilt, strict=True))
                         if left != right]
            return [f"{prefix or '<root>'}: {len(differing)} entries differ "
                    f"(first at index {differing[0]})"]
        return []
    if recorded != rebuilt:
        return [f"{prefix or '<root>'}: recorded {_short(recorded)} vs rebuilt {_short(rebuilt)}"]
    return []


def drift(
    rebuilt: Mapping[str, Callable[[], dict[str, Any]]] | None = None,
    commands: Mapping[str, list[str]] | None = None,
) -> list[str]:
    """Return one human-readable problem per stale artefact; empty means clean."""
    problems: list[str] = []
    for relative, builder in (rebuilt if rebuilt is not None else REBUILT).items():
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        try:
            fresh = builder()
        except Exception as exc:  # noqa: BLE001 - a generator that raises is a stale artefact, reported not masked
            problems.append(f"{relative}: generator raised {type(exc).__name__}: {exc}")
            continue
        problems.extend(f"{relative}: {line}" for line in _diff(recorded, fresh))

    for relative, command in (commands if commands is not None else CHECKED_BY_COMMAND).items():
        try:
            result = subprocess.run(  # noqa: S603 - argv is a fixed, in-repository command
                command, cwd=ROOT, capture_output=True, text=True, check=False,
            )
        except OSError as exc:
            problems.append(f"{relative}: could not run {command[0]}: {exc}")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            problems.append(
                f"{relative}: {' '.join(command[1:])} exited {result.returncode}"
                + (f" ({detail[-1]})" if detail else "")
            )
    return problems


def write_artefacts() -> list[str]:
    """Regenerate every in-process artefact. Returns the paths written."""
    written: list[str] = []
    for relative, builder in REBUILT.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(builder(), indent=2) + "\n", encoding="utf-8")
        written.append(relative)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when any artefact disagrees with the tree")
    parser.add_argument("--write", action="store_true",
                        help="regenerate the in-process artefacts from the tree")
    args = parser.parse_args(argv)

    if args.write:
        for relative in write_artefacts():
            print(f"wrote {relative}")
        return 0

    problems = drift()
    if problems:
        print("AUDIT ARTEFACT DRIFT — regenerate, do not hand-edit:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\n  python scripts/check_audit_artifact_drift.py --write", file=sys.stderr)
        return 1 if args.check else 0
    print(f"audit artefacts match the tree: {len(REBUILT) + len(CHECKED_BY_COMMAND)} checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
