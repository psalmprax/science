"""Regression check for the 89Q invsqrt champion rebuild.

Status: audit-only

The champion row in ``docs/invsqrt_frontier_and_portfolio_synthesis.md`` is a
compiled artifact plus a receipt. ``scripts/test_shared_pddd_circuit.py`` is the
only code path that rebuilds it, so this test pins the fields that make the claim
reproducible — the circuit hash above all — and a generous ceiling on the stage
wall-clock, so a build that silently gets ten times slower is caught.

The build needs the ``kickmix`` extension, which targets CPython 3.13, so the test
is opt-in and follows the same convention as ``tests/test_quantum_compilation.py``:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_dyn_invsqrt_champion_rebuild.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")
SCRIPT = REPO_ROOT / "scripts/test_shared_pddd_circuit.py"

# Pinned in artifacts/quantum_compilation/invsqrt/invsqrt_89q_shared_pddd_full_receipt.json
# (the artifact tree is gitignored, which is exactly why the value is pinned here).
PINNED_CIRCUIT_SHA256 = "ce04c9d54dc42e42455444022b6652a6efb0be2630aca7b485d3efd4e58fa9de"
PINNED_NUM_QUBITS = 89
PINNED_MAX_TOFFOLI = 1466
PINNED_WORST_ERROR_OVER_TOLERANCE = 0.9996227434818049
PINNED_VERIFIED_SHOTS = 25600

# Ceilings, not targets: one measured run is ~4.5 s in total. These leave room for a
# slow or contended machine while still failing on an order-of-magnitude regression.
MAX_TOTAL_SECONDS = 120.0
MAX_SINGLE_STAGE_SECONDS = 60.0


@pytest.fixture(scope="module")
def rebuild() -> dict:
    if not KICKMIX_PYTHON:
        pytest.skip("set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter with kickmix")
    if not Path(KICKMIX_PYTHON).exists():
        pytest.skip(f"APEDP_Q_KICKMIX_PYTHON does not exist: {KICKMIX_PYTHON}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [KICKMIX_PYTHON, str(SCRIPT), "--json"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    last = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert last, f"no JSON payload in output:\n{result.stdout[-4000:]}"
    payload = json.loads(last[-1])
    assert "receipt" in payload and "timings" in payload, payload
    return payload


def test_champion_rebuild_reproduces_the_pinned_circuit(rebuild: dict) -> None:
    receipt = rebuild["receipt"]
    assert receipt["status"] == "PASS"
    assert receipt["circuit_sha256"] == PINNED_CIRCUIT_SHA256
    assert receipt["num_qubits"] == PINNED_NUM_QUBITS
    assert receipt["max_toffoli_count"] == PINNED_MAX_TOFFOLI
    assert receipt["worst_error_over_tolerance"] == pytest.approx(
        PINNED_WORST_ERROR_OVER_TOLERANCE, rel=0.0, abs=1e-15
    )
    assert receipt["verified_shots"] == PINNED_VERIFIED_SHOTS


def test_champion_rebuild_score_is_in_the_pinned_band(rebuild: dict) -> None:
    """score = Q x toffoli_equivalent, and the mean Toffoli count is a sample mean.

    The pinned 157,601.0808 came from a 200,540,928-shot run; the rebuild runs 25,600
    shots, so the score is compared against a narrow band rather than an equality.
    """
    receipt = rebuild["receipt"]
    assert receipt["num_qubits"] * receipt["toffoli_equivalent"] == pytest.approx(
        receipt["score"], rel=1e-12
    )
    assert 157_590.0 < receipt["score"] < 157_615.0


def test_champion_rebuild_stage_timings_are_within_budget(rebuild: dict) -> None:
    timings = rebuild["timings"]
    assert "exact domain compile" in timings
    assert "build raw circuit" in timings
    assert "official verification (25,600 shots)" in timings

    total = rebuild["total_seconds"]
    assert total == pytest.approx(sum(timings.values()), rel=0.02), "stage timings do not add up"
    assert total < MAX_TOTAL_SECONDS, f"rebuild took {total:.1f}s (ceiling {MAX_TOTAL_SECONDS}s)"
    for stage, seconds in timings.items():
        assert seconds < MAX_SINGLE_STAGE_SECONDS, (
            f"stage '{stage}' took {seconds:.1f}s (ceiling {MAX_SINGLE_STAGE_SECONDS}s)"
        )
