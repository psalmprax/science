"""Sandbox pins: the gap contract and the no-certificate-no-verified rule.

The sandbox (`benchmarks/openai_math_sandbox/`) mirrors the OpenAI math-release
pipeline (attempt -> claim -> independent verification -> retraction
accounting) for the subset of problems APEDP-X can machine-check. These tests
gate two things:

1. Honesty invariant: a task is only reported `verified` when a checker
   produced a certificate; unsupported entries always carry a reason.
2. Gap contract: the OpenAI-headline categories with no engine path must be
   declared in the registry, so capability gaps stay visible instead of
   silently dropping out of the report.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "openai_math_sandbox_runner", _ROOT / "benchmarks" / "openai_math_sandbox" / "run_sandbox.py"
)
sandbox = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sandbox)

HEADLINE_GAPS = {
    "LEAN-1",
    "OPEN-RH",
    "OPEN-BSD",
    "OPEN-HODGE",
    "OPEN-KAKEYA",
    "OPEN-ALGO",
    "OPEN-NS",
    "OPEN-SCALE",
}


def test_registry_integrity():
    registry = sandbox.load_registry()
    ids = [p["id"] for p in registry["problems"]]
    assert len(ids) == len(set(ids)), "duplicate problem ids"
    for p in registry["problems"]:
        assert p["name"] and p["category"], p
        if p["status"] in ("supported", "conditional"):
            assert p.get("engine") and p.get("checker"), p
        elif p["status"] == "unsupported":
            assert len(p.get("reason", "")) > 40, p
        else:
            raise AssertionError(f"unknown registry status: {p['status']}")


def test_headline_gap_categories_are_declared():
    registry = sandbox.load_registry()
    declared = {p["id"] for p in registry["problems"] if p["status"] == "unsupported"}
    assert HEADLINE_GAPS <= declared, sorted(HEADLINE_GAPS - declared)


def test_fast_run_invariants_hold():
    report = sandbox.run_all(fast=True)
    results = report["results"]
    for r in results:
        assert r["status"] in sandbox.STATUSES, r
        if r["status"] == "verified":
            assert r["certificate"], f"{r['id']} verified without certificate"
            assert r["duration_ms"] >= 0.0
        if r["status"] == "unsupported":
            assert r["reason"], f"{r['id']} unsupported without reason"
    counts = report["counts"]
    assert sum(counts.values()) == len(results)
    assert counts["verified"] + counts["failed"] == sum(
        1 for r in results if r["registry_status"] in ("supported", "conditional")
    ) - counts["unavailable"]
    registry_unsupported = sum(
        1 for p in sandbox.load_registry()["problems"] if p["status"] == "unsupported"
    )
    assert counts["unsupported"] == registry_unsupported


def test_deterministic_law_checks_verify_with_certificates():
    for pid in ("LAW-1", "LAW-2", "GA-1"):
        checker = sandbox.CHECKS[pid]
        res = checker(fast=True)
        assert res["status"] == "verified", (pid, res)
        assert res["certificate"], (pid, res)


def test_pde_conditional_semantics():
    res = sandbox.check_pde_burgers(fast=True)
    assert res["status"] in ("verified", "unavailable"), res
    if res["status"] == "unavailable":
        assert "pending" in res["reason"], res


def test_binary_checks_never_fail_due_to_missing_binary():
    for pid in ("NT-1", "NT-2"):
        res = sandbox.CHECKS[pid](fast=True)
        assert res["status"] in ("verified", "unavailable"), (pid, res)
