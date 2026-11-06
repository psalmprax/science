"""Cross-check the four measured decoder-stage receipts against each other.

The stage receipts are produced by separate emitters (``scripts/emit_bm_iteration.py``,
``emit_chien_forney_stage.py``, ``emit_uncompute_pass.py``, ``emit_chien_variants.py``,
``bound_archive_overhead.py``). Each is a measurement; this test pins the *arithmetic that
combines them* so a stale receipt cannot quietly disagree with the note in
``docs/ATTESTED_WIDTH_AND_ACCURACY_BUDGET.md``.

It imports no engine: these are JSON artifacts, so the test runs under any interpreter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPTS = Path(__file__).resolve().parent.parent / "benchmarks" / "quantum_compilation"
ARCHIVE = 396919.33
KEY_EQUATION = 313110.0
FORNEY_PER_BLOCK = 315.0
POSITIONS = 4095
MAX_LIVE = 35


def load(name: str) -> dict:
    path = RECEIPTS / name
    if not path.exists():
        pytest.skip(f"{name} not built in this checkout")
    return json.loads(path.read_text())


def test_key_equation_stage_is_the_measured_projection() -> None:
    receipt = load("bm_iteration_toffolis.json")
    row = receipt["rows"][-1]
    assert row["degree"] == 12 and row["t"] == 35
    assert row["toffolis"] == 4473
    assert row["key_equation_toffolis"] == KEY_EQUATION


def test_forney_block_is_linear_and_priced_per_position() -> None:
    receipt = load("chien_forney_stage.json")
    rows = receipt["forney_block"]
    assert [r["toffolis_per_position"] for r in rows] == [18.0, 108.0, 315.0]
    # Eight positions cost exactly eight blocks, so the block is per-position.
    assert rows[-1]["toffolis"] == FORNEY_PER_BLOCK * 8
    # The coefficient sweep is constant multiplies: no ANDs at all.
    assert all(r["toffolis"] == 0 for r in receipt["constant_multiply_sweep"])


def test_uncompute_costs_the_same_as_the_forward_pass() -> None:
    receipt = load("uncompute_pass.json")
    for row in receipt["rows"]:
        assert row["adjoint_equals_forward"] is True
        assert row["uncompute_toffolis"] == row["key_equation_toffolis"]
    assert receipt["accounting"]["four_stage_total"] == (
        KEY_EQUATION + FORNEY_PER_BLOCK * POSITIONS + KEY_EQUATION
    )
    assert receipt["accounting"]["ratio_vs_archive"] == pytest.approx(4.8275, abs=1e-3)


def test_only_the_runtime_gated_variant_fits_the_remainder() -> None:
    receipt = load("chien_variants.json")
    variants = receipt["variants"]
    assert variants["unconditional_per_position"]["fits_remainder"] is False
    assert variants["shared_single_inversion"]["fits_remainder"] is False
    assert variants["runtime_gated_blocks"]["fits_remainder"] is True
    assert variants["runtime_gated_blocks"]["toffolis"] == FORNEY_PER_BLOCK * MAX_LIVE
    # The runtime reading lands below the archive; the static one does not.
    total = receipt["runtime_conditioned_total"]["total"]
    assert total == KEY_EQUATION + FORNEY_PER_BLOCK * MAX_LIVE
    assert total < ARCHIVE
    assert ARCHIVE / total == pytest.approx(1.2245, abs=1e-3)


def test_the_overhead_bound_closes_the_accounting() -> None:
    receipt = load("archive_overhead_bound.json")
    region = receipt["feasible_region"]
    assert region["remainder"] == pytest.approx(ARCHIVE - KEY_EQUATION, abs=1e-2)
    assert region["max_blocks_at_zero_overhead"] == pytest.approx(
        (ARCHIVE - KEY_EQUATION) / FORNEY_PER_BLOCK, abs=1e-2
    )
    assert region["overhead_left_at_worst_case_weight"] == pytest.approx(
        ARCHIVE - KEY_EQUATION - FORNEY_PER_BLOCK * MAX_LIVE, abs=1e-2
    )
    assert region["adjoint_uncompute_fits"] is False
    assert receipt["readings"]["runtime_gated_with_adjoint"]["fits"] is False
