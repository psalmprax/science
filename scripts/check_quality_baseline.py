"""Fail when lint or type-check debt GROWS past a pinned baseline.

Why this exists: `make lint` ran `ruff --select E9,F`, a rule set narrow enough
that the full configured ruleset -- the one the repository says it enforces --
could accumulate hundreds of findings with the gate still green, and
`make typecheck` ran mypy over the two modules listed in `pyproject.toml`, so a
green type gate said nothing about the other ~1,200 files.

`apedp/` and `tests/` are held at zero findings. The paths that are not yet
clean are pinned at their measured count here and may only move DOWN: a new
finding fails the gate, while removing an old one is always allowed. Raising a
baseline requires editing `quality_baseline.json` and recording the reason in
`docs/QUALITY_BASELINE.md`.

Usage:
    python scripts/check_quality_baseline.py              # both gates
    python scripts/check_quality_baseline.py --ruff-only
    python scripts/check_quality_baseline.py --mypy-only
    python scripts/check_quality_baseline.py --update     # re-pin to current

Status: audit-only (quality gate).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "quality_baseline.json"
_MYPY_TOTAL = re.compile(r"Found (\d+) errors? in")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # argv is built here from a fixed interpreter and repo paths, never from input.
    return subprocess.run(  # noqa: S603
        cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def count_ruff(path: str) -> int:
    proc = _run([sys.executable, "-m", "ruff", "check", path,
                 "--output-format", "json"])
    try:
        findings = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ruff produced output this script cannot parse for {path}: {exc}"
        ) from exc
    return len(findings)


def count_mypy(path: str) -> int:
    proc = _run([sys.executable, "-m", "mypy", path, "--ignore-missing-imports"])
    match = _MYPY_TOTAL.search(proc.stdout + proc.stderr)
    if match:
        return int(match.group(1))
    if proc.returncode == 0:
        return 0
    raise RuntimeError(
        f"mypy failed without a parseable total for {path}:\n"
        f"{(proc.stdout + proc.stderr).strip()[-800:]}")


def load_baseline() -> dict[str, dict[str, int]]:
    if not BASELINE_PATH.exists():
        raise RuntimeError(f"missing baseline file: {BASELINE_PATH}")
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {"ruff": data["ruff"], "mypy": data["mypy"]}


def evaluate(kind: str, baseline: dict[str, dict[str, int]],
             counter, only: str | None) -> tuple[list[tuple[str, int, int, bool]], bool]:
    rows: list[tuple[str, int, int, bool]] = []
    ok = True
    for path, limit in baseline[kind].items():
        current = counter(path)
        within = current <= limit
        ok = ok and within
        rows.append((path, current, limit, within))
    return rows, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff-only", action="store_true")
    parser.add_argument("--mypy-only", action="store_true")
    parser.add_argument("--update", action="store_true",
                        help="re-pin the baseline to the current counts")
    args = parser.parse_args(argv)
    if args.ruff_only and args.mypy_only:
        parser.error("choose at most one of --ruff-only / --mypy-only")

    baseline = load_baseline()
    kinds = []
    if not args.mypy_only:
        kinds.append(("ruff", "ruff", count_ruff))
    if not args.ruff_only:
        kinds.append(("mypy", "mypy", count_mypy))

    failed = False
    updates: dict[str, dict[str, int]] = {}
    for kind, label, counter in kinds:
        rows, ok = evaluate(kind, baseline, counter, None)
        updates[kind] = {path: current for path, current, _limit, _ in rows}
        print(f"\n{label} findings (baseline may only be lowered)")
        print(f"{'path':<16}{'now':>8}{'limit':>8}  status")
        for path, current, limit, within in rows:
            state = "ok" if within else f"REGRESSION (+{current - limit})"
            print(f"{path:<16}{current:>8}{limit:>8}  {state}")
        if not ok:
            failed = True
            print(f"{label}: FAIL — debt grew past the pinned baseline.")

    if args.update:
        current = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for kind, values in updates.items():
            current[kind] = values
        BASELINE_PATH.write_text(json.dumps(current, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"\nbaseline re-pinned: {BASELINE_PATH}")
        return 0

    if failed:
        print("\nRecord any intentional increase in docs/QUALITY_BASELINE.md "
              "before re-pinning with --update.")
        return 1
    print("\nquality baseline: all paths within their pinned limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
