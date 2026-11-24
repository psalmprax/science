#!/usr/bin/env python3
"""Check the machine-checked Lean core in ``lean_checked/``.

Two layers, so the claim "these theorems are kernel-checked" cannot drift:

1. Static, always run, no Lean needed. No source file may contain ``sorry``,
   declare an ``axiom``, or use ``native_decide``, ``unsafe``, ``opaque`` or
   ``implemented_by`` (each is a way to have the kernel accept something it did
   not check). Every ``theorem`` in the sources must be listed in
   ``Audit.lean``, so no theorem escapes the axiom audit.

2. With Lean, run when ``lake`` is on PATH and required with
   ``--require-lean`` (CI). The project must build, and ``#print axioms`` for
   every theorem may name only Lean's own three axioms -- ``propext``,
   ``Quot.sound``, ``Classical.choice`` -- never ``sorryAx`` and never an axiom
   of the project's own.

``--write`` records the result in ``lean_checked/axiom_audit.json``; ``--check``
fails if the recorded file does not match what was just measured.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "lean_checked"
RECORD = PROJECT / "axiom_audit.json"
STANDARD_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})
FORBIDDEN = {
    "sorry": re.compile(r"\bsorry\b"),
    "axiom declaration": re.compile(r"^\s*(?:private\s+|protected\s+)?axiom\s", re.M),
    "native_decide": re.compile(r"\bnative_decide\b"),
    "unsafe": re.compile(r"^\s*(?:private\s+|protected\s+)?unsafe\s", re.M),
    "opaque": re.compile(r"^\s*(?:private\s+|protected\s+)?opaque\s", re.M),
    "implemented_by": re.compile(r"implemented_by"),
}
_THEOREM = re.compile(r"^\s*theorem\s+([A-Za-z_][\w.']*)", re.M)
_COMMENT = re.compile(r"/-.*?-/", re.S)
_LINE_COMMENT = re.compile(r"--.*$", re.M)
_AXIOM_LINE = re.compile(r"^'([^']+)' (does not depend on any axioms|depends on axioms: \[(.*)\])$")


def _code(text: str) -> str:
    """Source with comments removed, so prose may mention the forbidden words."""
    return _LINE_COMMENT.sub("", _COMMENT.sub("", text))


def source_files(project: Path = PROJECT) -> list[Path]:
    return [*sorted((project / "ApedpChecked").rglob("*.lean")), project / "ApedpChecked.lean"]


def static_check(project: Path = PROJECT) -> tuple[list[str], list[str]]:
    """Return (problems, theorem names declared in the sources)."""
    problems: list[str] = []
    declared: list[str] = []
    for path in source_files(project):
        code = _code(path.read_text(encoding="utf-8"))
        for label, pattern in FORBIDDEN.items():
            if pattern.search(code):
                problems.append(f"{path.name}: contains {label}")
        declared.extend(_THEOREM.findall(code))
    audit = _code((project / "Audit.lean").read_text(encoding="utf-8"))
    audited = {name.split(".")[-1] for name in re.findall(r"#print axioms\s+([\w.']+)", audit)}
    missing = sorted(name for name in declared if name.split(".")[-1] not in audited)
    if missing:
        problems.append(f"theorems not listed in Audit.lean: {', '.join(missing)}")
    if len(declared) != len(set(declared)):
        problems.append("duplicate theorem names across files; the audit matches by name")
    return problems, declared


def kernel_check() -> tuple[list[str], dict[str, list[str]], str]:
    """Build the project and read each theorem's axioms. Needs ``lake``."""
    problems: list[str] = []
    lean, lake = shutil.which("lean"), shutil.which("lake")
    if lean is None or lake is None:
        return ["lake/lean not found on PATH"], {}, ""
    version = subprocess.run(  # noqa: S603 - fixed argv (shutil.which result), no shell
        [lean, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    build = subprocess.run(  # noqa: S603 - fixed argv (shutil.which result), no shell
        [lake, "build"], cwd=PROJECT, capture_output=True, text=True, check=False
    )
    if build.returncode != 0:
        return [f"lake build failed:\n{build.stdout[-2000:]}{build.stderr[-2000:]}"], {}, version
    if re.search(r"\bwarning\b", build.stdout + build.stderr):
        problems.append("lake build emitted warnings")
    audit = subprocess.run(  # noqa: S603 - fixed argv (shutil.which result), no shell
        [lake, "env", "lean", "Audit.lean"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    if audit.returncode != 0:
        return [f"Audit.lean failed:\n{audit.stdout[-2000:]}{audit.stderr[-2000:]}"], {}, version
    axioms: dict[str, list[str]] = {}
    for line in audit.stdout.splitlines():
        match = _AXIOM_LINE.match(line.strip())
        if not match:
            continue
        used = [a.strip() for a in (match.group(3) or "").split(",") if a.strip()]
        axioms[match.group(1)] = sorted(used)
        foreign = sorted(set(used) - STANDARD_AXIOMS)
        if foreign:
            problems.append(f"{match.group(1)} depends on {', '.join(foreign)}")
    return problems, axioms, version


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--require-lean", action="store_true", help="fail if lake is not available")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="record the measured audit")
    mode.add_argument(
        "--check", action="store_true", help="fail if the record differs from the measurement"
    )
    args = parser.parse_args()

    problems, declared = static_check()
    print(
        f"static: {len(declared)} theorems in {len(source_files())} files, {len(problems)} problem(s)"
    )
    have_lean = shutil.which("lake") is not None and shutil.which("lean") is not None
    if not have_lean:
        if args.require_lean or args.write:
            problems.append("lake/lean not found on PATH")
        else:
            print(
                "kernel: skipped (lake not on PATH); the static layer alone does not check a proof"
            )
    else:
        kernel_problems, axioms, version = kernel_check()
        problems.extend(kernel_problems)
        if not kernel_problems and len(axioms) != len(declared):
            problems.append(
                f"audit covered {len(axioms)} theorems, sources declare {len(declared)}"
            )
        axiom_free = sorted(name for name, used in axioms.items() if not used)
        measured = {
            "lean_version": version,
            "theorems": len(axioms),
            "standard_axioms_allowed": sorted(STANDARD_AXIOMS),
            "axiom_free": axiom_free,
            "axioms_by_theorem": dict(sorted(axioms.items())),
        }
        print(
            f"kernel: {version}; {len(axioms)} theorems checked, "
            f"{len(axiom_free)} with no axioms at all"
        )
        if args.write and not problems:
            RECORD.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {RECORD.relative_to(ROOT)}")
        elif args.check:
            recorded = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.is_file() else None
            if recorded != measured:
                problems.append(
                    f"{RECORD.relative_to(ROOT)} does not match the measurement; re-run with --write"
                )
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
