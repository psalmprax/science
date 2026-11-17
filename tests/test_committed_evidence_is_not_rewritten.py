"""Running the tests does not rewrite committed evidence (F-05).

Four tracked result files were replaced in place by every full run: the
five-gate programme's JSON and Markdown reports, written by the benchmark
function a test imports, and two Rust profile records, written by `cargo test`.
The numbers changed on every machine -- timestamps, wall times, the recorded
workspace path -- so the suite left the tree dirty and a reader could not tell
the recorded measurement from an incidental rerun.

Each writer now writes only when asked: `run_all_gates(write=True)` for the
Python one, `APEDP_WRITE_BENCHMARK_ARTIFACTS=1` for the two Rust ones. These
tests check the writers, statically for Rust (which may not be built here) and
by running it for Python.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

#: The four files, and who used to rewrite them.
EVIDENCE = {
    "benchmarks/five_gate_validation_results.json": "benchmarks/run_five_gate_validation.py",
    "docs/five_gate_validation_results.md": "benchmarks/run_five_gate_validation.py",
    "benchmarks/kernel_profile_results.json": "rust/apedp-kernel/tests/rime_profile.rs",
    "benchmarks/sre_tree_vs_arena_baseline.json": "rust/apedp-kernel/tests/sre_tree_vs_arena_profile.rs",
}
RUST_PROFILES = (
    "rust/apedp-kernel/tests/rime_profile.rs",
    "rust/apedp-kernel/tests/sre_tree_vs_arena_profile.rs",
)


def test_the_evidence_files_are_all_present():
    for name in EVIDENCE:
        assert (ROOT / name).is_file(), name


def test_running_the_five_gate_programme_does_not_touch_the_committed_report():
    """The default is to compute and return, not to write."""
    from run_five_gate_validation import run_all_gates

    before = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in EVIDENCE
        if "five_gate" in name
    }
    results = run_all_gates()
    assert results["all_passed"] is True
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    assert after == before, "run_all_gates() rewrote committed evidence without being asked"


def test_the_five_gate_writer_still_exists_behind_the_flag():
    import inspect

    import run_five_gate_validation as module

    signature = inspect.signature(module.run_all_gates)
    assert signature.parameters["write"].default is False
    source = inspect.getsource(module.run_all_gates)
    assert "if not write:" in source and "JSON_OUT.write_text" in source


@pytest.mark.parametrize("name", RUST_PROFILES)
def test_a_rust_profile_test_writes_only_when_asked(name: str):
    """Checked by reading the source: the Rust toolchain may not be present,
    and a test that silently skipped here would leave the defect unpinned."""
    source = (ROOT / name).read_text(encoding="utf-8")
    assert "APEDP_WRITE_BENCHMARK_ARTIFACTS" in source, name
    # Every write of the record sits inside that condition: the guard appears
    # before any `fs::write`, and the file has exactly one of each.
    assert source.count("std::fs::write") == 1, name
    guard = source.index("APEDP_WRITE_BENCHMARK_ARTIFACTS")
    assert guard < source.index("std::fs::write"), name


def test_the_committed_mutation_report_covers_the_register_it_claims():
    """A committed report is evidence about the register as it stood when it ran.

    `benchmarks/mutation_battery_report.json` is dated 2026-10-23 and CI re-runs
    the battery on every push, so the committed file is a record rather than the
    live gate. What must not drift silently is its *coverage*: a mutant added to
    `scripts/run_mutation_battery.py` without re-running leaves a report that
    claims every mutant was killed while saying nothing about the new one.
    """
    import json
    import re

    report = json.loads((ROOT / "benchmarks" / "mutation_battery_report.json").read_text())
    registered = set(re.findall(r'"id":\s*"([^"]+)"', (ROOT / "scripts" / "run_mutation_battery.py").read_text()))
    reported = {mutant["id"] for mutant in report["mutants"]}
    assert reported == registered, (
        "the committed mutation report no longer covers the register; re-run "
        "scripts/run_mutation_battery.py. Missing from the report: "
        f"{sorted(registered - reported)}; no longer registered: {sorted(reported - registered)}"
    )
    assert report["summary"]["n_mutants"] == len(registered)
    assert report["status"] == "ALL_KILLED"
