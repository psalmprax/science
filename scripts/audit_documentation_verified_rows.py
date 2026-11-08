"""Execute every test file cited by a "Verified (N/N)" row in docs/DOCUMENTATION.md.

The registry states counted results ("Verified (`tests/test_x.py`, 8/8)") that no
test enforces: nothing fails when the cited file stops passing, or when the
count in the prose drifts from the file. This harness closes that gap by
actually running each cited file and comparing the measured result with the
claim.

It is a reporting instrument, not a gate: a mismatch is recorded, never
suppressed, and the exit status stays 0 so that a scheduled audit can produce a
report even when rows are stale. ``--strict`` makes mismatches exit nonzero for
use as a CI gate once the registry is clean.

Status: audit-only (documentation integrity harness).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "DOCUMENTATION.md"

# `Verified (8/8)`, `Verified (`tests/x.py`, 8/8)`, `Verified (4/4 tests)` ...
_VERIFIED = re.compile(r"Verified \(")
# Bare counted claims such as 8/8, 33/33, 45/45.
_COUNT = re.compile(r"(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])")
# Cited runnable paths.
_CITED = re.compile(r"`((?:tests|benchmarks)/[^`]+?\.py)`")
_PYTEST_SUMMARY = re.compile(
    r"(?:(\d+) failed[,\s])?.*?(?:(\d+) passed)", re.S
)


@dataclass
class RowClaim:
    line_no: int
    text: str
    counts: list[tuple[int, int]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    cites_cargo: bool = False


@dataclass
class FileResult:
    path: str
    status: str          # "pass" | "fail" | "missing" | "timeout" | "error"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    seconds: float = 0.0
    detail: str = ""


def parse_rows(doc_path: Path) -> list[RowClaim]:
    rows: list[RowClaim] = []
    for index, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
        if not _VERIFIED.search(line):
            continue
        rows.append(
            RowClaim(
                line_no=index,
                text=line,
                counts=[(int(a), int(b)) for a, b in _COUNT.findall(line)],
                files=sorted(set(_CITED.findall(line))),
                cites_cargo="cargo test" in line,
            )
        )
    return rows


def run_test_file(rel_path: str, timeout_s: int) -> FileResult:
    target = REPO_ROOT / rel_path
    if not target.exists():
        return FileResult(rel_path, "missing", detail="cited file not on disk")
    started = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - argv is fixed: interpreter, -m,
            # pytest, and a repo-relative path read from the registry.
            [sys.executable, "-m", "pytest", rel_path, "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return FileResult(rel_path, "timeout", seconds=time.monotonic() - started,
                          detail=f"exceeded {timeout_s}s")
    elapsed = time.monotonic() - started
    out = proc.stdout + proc.stderr

    def count(word: str) -> int:
        matches = re.findall(rf"(\d+) {word}", out)
        return int(matches[-1]) if matches else 0

    passed, failed, errors = count("passed"), count("failed"), count("error")
    if proc.returncode == 5 and passed == 0 and failed == 0:
        # pytest exit 5 = no tests collected. A registry row may cite a producer
        # script; that row's claim is unmeasured here, not contradicted.
        return FileResult(rel_path, "no_tests", 0, 0, 0, elapsed,
                          "pytest collected no tests (producer script)")
    status = "pass" if proc.returncode == 0 else "fail"
    detail = ""
    if status == "fail":
        tail = [ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        lines = out.strip().splitlines()
        detail = "; ".join(tail[:6]) or (lines[-1] if lines else "nonzero exit")
    return FileResult(rel_path, status, passed, failed, errors, elapsed, detail)


def judge(row: RowClaim, results: dict[str, FileResult]) -> tuple[str, str]:
    """Return (verdict, explanation) for one registry row."""
    if not row.files:
        return "NO_CITED_FILE", "row states a result but cites no runnable file"
    if row.cites_cargo:
        return "NOT_AUTOMATED", "row also cites `cargo test`, which this harness does not run"

    measured = [results[f] for f in row.files]
    if any(r.status == "missing" for r in measured):
        return "CITED_FILE_MISSING", "a cited file does not exist"
    if any(r.status == "timeout" for r in measured):
        return "TIMEOUT", "a cited file did not finish inside the harness timeout"
    if any(r.status == "skipped" for r in measured):
        return "SKIPPED_BY_AUDIT", (
            "excluded from this run by --skip (resource-heavy); its claim is "
            "therefore unmeasured here, not confirmed")
    if all(r.status == "no_tests" for r in measured):
        return "NOT_AUTOMATED", (
            "the cited path collects no tests (it is a producer script); the "
            "counted claim is unmeasured here")
    if any(r.status == "fail" for r in measured):
        return "MEASURED_FAILURE", "a cited file does not currently pass"

    total_passed = sum(r.passed for r in measured)
    if not row.counts:
        return "UNCOUNTED_CLAIM", (
            f"row asserts a qualitative result; cited file(s) pass "
            f"({total_passed} passed) but no count is claimed"
        )

    claimed = {c for c in row.counts if c[0] == c[1]}
    if not claimed:
        return "MANUAL_REVIEW", f"counts {row.counts} are not of the N/N form"
    if len(row.files) == 1:
        res = measured[0]
        # A row may quote several N/N figures: the test count and figures
        # measured INSIDE a test ("20/20 random starts", "100/100 true
        # positives"). MATCH if the measured pass count appears as a self-
        # referential N/N claim; otherwise report the drift, not a guess.
        if (res.passed, res.passed) in claimed and res.failed == 0:
            return "MATCH", (
                f"claimed {res.passed}/{res.passed}, measured {res.passed} passed")
        best = max(claimed, key=lambda c: c[0])
        return "MISMATCH", (
            f"no claimed count equals the measured pass count; largest claim "
            f"{best[0]}/{best[1]}, measured {res.passed} passed, "
            f"{res.failed} failed, {res.errors} error(s)"
        )
    # Multi-file rows: the counts may be per-file or aggregate; accept a claim
    # that equals any single file's pass count or the aggregate.
    per_file = {r.path: r.passed for r in measured}
    for value, denominator in claimed:
        if value in per_file.values() or value == total_passed:
            return "PARTIAL_MATCH", (
                f"claimed {value}/{denominator}; per-file passes {per_file}, "
                f"aggregate {total_passed} — the claim does not name which file "
                "it counts"
            )
    return "MISMATCH", (
        f"claimed {sorted(claimed)}; per-file passes {per_file}, aggregate {total_passed}"
    )


def render(rows: list[RowClaim], results: dict[str, FileResult],
           verdicts: list[tuple[str, str]], elapsed: float) -> str:
    counts: dict[str, int] = {}
    for verdict, _ in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    lines = [
        "# Documentation \"Verified (N/N)\" rows — execution audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Harness: `scripts/audit_documentation_verified_rows.py` "
        f"(audit-only; executions took {elapsed:.1f}s)",
        f"Registry: `docs/DOCUMENTATION.md` — **{len(rows)} rows**, "
        f"**{len(results)} distinct cited files**",
        "",
        "Every cited file was executed with `pytest -q`. A row is `MATCH` only when",
        "its stated count equals the measured pass count with zero failures. Rows",
        "that cite no file, cite a missing file, or cite `cargo test` are reported",
        "as unmeasurable rather than assumed correct.",
        "",
        "## Summary",
        "",
        "| Verdict | Rows |",
        "|---|---|",
    ]
    for name in sorted(counts, key=lambda k: (-counts[k], k)):
        lines.append(f"| `{name}` | {counts[name]} |")
    lines += ["", "## Per-file results", "",
              "| Cited file | Status | Passed | Failed | Errors | Seconds |",
              "|---|---|---|---|---|---|"]
    for path, res in sorted(results.items()):
        lines.append(
            f"| `{path}` | {res.status} | {res.passed} | {res.failed} | "
            f"{res.errors} | {res.seconds:.1f} |"
        )
    lines += ["", "## Per-row verdicts", "",
              "| DOCUMENTATION.md line | Verdict | Claim | Finding |",
              "|---|---|---|---|"]
    for row, (verdict, why) in zip(rows, verdicts, strict=True):
        claim = ", ".join(f"{a}/{b}" for a, b in row.counts) or "qualitative"
        files = ", ".join(f"`{f}`" for f in row.files) or "none"
        lines.append(
            f"| {row.line_no} | `{verdict}` | {claim} | {why} "
            f"({files}) |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- `MATCH` — the row's count reproduces. The claim is still only as strong",
        "  as the test it cites: a tautological assertion reproduces too.",
        "- `PARTIAL_MATCH` — the count appears in the data but the row does not say",
        "  which cited file it counts.",
        "- `MISMATCH` / `MEASURED_FAILURE` — a documented result is not what the",
        "  cited test currently does. Either the row or the test is wrong.",
        "- `NO_CITED_FILE` / `NOT_AUTOMATED` — the row cannot be checked by running",
        "  pytest. These are the rows a reader has to trust, and the ones a",
        "  registry should either cite a runnable file for or stop counting.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=str(DOC_PATH))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" /
                                            "DOCUMENTATION_VERIFIED_ROWS_AUDIT.md"))
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-file pytest timeout in seconds")
    parser.add_argument("--only", default=None,
                        help="restrict to cited paths containing this substring")
    parser.add_argument("--skip", action="append", default=[],
                        help="cited path substring to exclude; recorded as SKIPPED, never omitted silently")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any row is MISMATCH/MEASURED_FAILURE")
    parser.add_argument("--json", default=None, help="also write raw results as JSON")
    args = parser.parse_args(argv)

    rows = parse_rows(Path(args.doc))
    files: list[str] = []
    for row in rows:
        for path in row.files:
            if path not in files and (args.only is None or args.only in path):
                files.append(path)

    results: dict[str, FileResult] = {}
    # Resumable: a long registry audit outlives one process. Each completed file
    # is written to the JSON cache immediately, so a killed run loses at most the
    # file it was executing and the next invocation continues from there.
    cache: dict[str, dict[str, object]] = {}
    cache_path = Path(args.json) if args.json else None
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text()).get("files", {})
        except json.JSONDecodeError:
            cache = {}

    def _flush() -> None:
        if cache_path is None:
            return
        merged = dict(cache)
        merged.update({k: v.__dict__ for k, v in results.items()})
        cache_path.write_text(json.dumps(
            {"files": merged,
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))

    for path in files:
        if path in cache:
            results[path] = FileResult(**cache[path])  # type: ignore[arg-type]
            print(f"[audit] cached {path} ({results[path].status})",
                  file=sys.stderr, flush=True)
            continue
        if any(s in path for s in args.skip):
            results[path] = FileResult(path, "skipped",
                                       detail="excluded by --skip")
            print(f"[audit] skipped {path}", file=sys.stderr, flush=True)
            _flush()
            continue
        print(f"[audit] running {path}", file=sys.stderr, flush=True)
        results[path] = run_test_file(path, args.timeout)
        print(f"[audit]   -> {results[path].status} "
              f"({results[path].passed} passed, {results[path].failed} failed)",
              file=sys.stderr, flush=True)
        _flush()

    verdicts = [judge(row, results) for row in rows]
    # Total time the harness actually spent executing pytest, including results
    # restored from the resumable cache. Replaying from cache must not report
    # the (near-zero) wall clock of the replay as the execution cost.
    executed = sum(r.seconds for r in results.values())
    report = render(rows, results, verdicts, executed)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"[audit] report written: {args.out}", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "rows": [
                    {"line_no": r.line_no, "counts": r.counts, "files": r.files,
                     "verdict": v, "finding": w}
                    for r, (v, w) in zip(rows, verdicts, strict=True)
                ],
                "files": {k: v.__dict__ for k, v in results.items()},
            }, indent=2))

    bad = [v for v, _ in verdicts if v in ("MISMATCH", "MEASURED_FAILURE")]
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
