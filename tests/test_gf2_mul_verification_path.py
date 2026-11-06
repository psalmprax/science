"""Pins the certificate script that separates what a multiplier claim proves from what it cannot.

Status: audit-only
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/verify_gf2_multiplier_candidate.py"
RANK_SCRIPT = REPO_ROOT / "scripts/search_gf2_mul_rank.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_54_certifies_algebraically_and_stops_before_a_netlist():
    result = _run("--seed", "karatsuba-54", "--samples", "64")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "ALGEBRAIC_CERTIFICATE_ONLY" in result.stdout
    assert "reversible_netlist" in result.stdout and "MISSING" in result.stdout


def test_the_published_51_does_not_certify():
    """It ships all-zero factors: the exact screen must refuse it, with the mismatch count."""
    result = _run("--seed", "published-51", "--samples", "16")
    assert result.returncode == 1
    assert "REJECTED_BY_EXACT_SCREEN" in result.stdout


def test_a_certificate_round_trips_through_check(tmp_path):
    out = tmp_path / "cert.json"
    written = _run("--seed", "karatsuba-63", "--samples", "32", "--out", str(out))
    assert written.returncode == 0, written.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["rank"] == 63
    assert record["status"] == "ALGEBRAIC_CERTIFICATE_ONLY"
    assert record["gates"]["algebraic_all_1728_coefficients"] == "PASS"
    assert record["gates"]["embedded_benchmark_receipt"] == "MISSING"

    checked = _run("--check", "--out", str(out))
    assert checked.returncode == 0, checked.stdout + checked.stderr[-2000:]
    assert "OK" in checked.stdout


def test_the_rank_search_record_checks_out(tmp_path):
    out = tmp_path / "rank.json"
    written = subprocess.run(
        [sys.executable, str(RANK_SCRIPT), "--budget", "20", "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert written.returncode == 0, written.stderr[-3000:]
    record = json.loads(out.read_text())
    assert record["target_met"] is False
    rows = {row["seed"]: row for row in record["rows"]}
    assert rows["Karatsuba-63"]["rank_after_search"] == 54
    assert rows["Karatsuba-54-BaseComposed"]["rank_after_search"] == 54

    checked = subprocess.run(
        [sys.executable, str(RANK_SCRIPT), "--check", "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr[-2000:]
