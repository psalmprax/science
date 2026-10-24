"""Report fidelity: the benchmark report derives from its JSON, verbatim.

Two machine-enforced properties:
  1. `verdict_for_row` implements the stated rule on canonical rows (collapse,
     smooth refusal, exact recovery by bypass, undefined rho without exactness).
  2. Every cell of the committed markdown's main table equals the committed
     JSON under the stated rendering rules. A hand-edit to any number or any
     verdict breaks this test. The template is separately guarded: no hardcoded
     verdict labels may remain in the harness source.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apedp_dyn.epistemic.report_fidelity import (
    check_row_against_report,
    render_verdict_cell,
    verdict_for_row,
)

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "benchmarks" / "run_landauer_rime_vs_ga_benchmark.py"
JSON_PATH = ROOT / "benchmarks" / "landauer_benchmark_results.json"
MD_PATH = ROOT / "docs" / "landauer_conductance_staircase_rime_vs_ga_benchmark.md"


def test_verdict_rule_on_canonical_rows():
    v, why = verdict_for_row(-11530.5976, 0.7238, 149.3235)
    assert v == "REFUSED" and "collapse" in why, (v, why)
    v, why = verdict_for_row(0.9486, 0.7386, 0.3153)
    assert v == "REFUSED", (v, why)
    # Exact recovery bypass: rho1 of numerical noise must not refuse R2 == 1.
    v, why = verdict_for_row(0.9999999999999999, 0.9691, 1.57e-8)
    assert v == "PASSED" and "exact recovery by rule" in why, (v, why)
    # Transmuter: undefined rho1 with zero residuals passes by the same rule.
    v, why = verdict_for_row(1.0, float("nan"), 0.0)
    assert v == "PASSED", (v, why)
    # Undefined rho1 WITHOUT exact recovery refuses, honestly.
    v, why = verdict_for_row(0.95, float("nan"), 0.4)
    assert v == "REFUSED" and "undefined" in why, (v, why)
    # Non-numeric input refuses instead of raising.
    v, _ = verdict_for_row("bad", 0.1, 0.1)
    assert v == "REFUSED"


def _main_table_rows(md: str) -> dict[str, str]:
    wanted = {
        "ga_polynomial": "| **Genetic Algorithm",
        "unassisted_rime": "| **Unassisted RIME",
        "assisted_rime": "| **RIME with",
        "autonomous_transmuter": "| **Autonomous Epistemic",
    }
    out = {}
    for line in md.splitlines():
        for key, prefix in wanted.items():
            if line.startswith(prefix):
                out[key] = line
    return out


def test_committed_report_matches_committed_json_cell_for_cell():
    if not JSON_PATH.exists():
        pytest.skip(f"benchmark JSON absent: {JSON_PATH}")
    if not MD_PATH.exists():
        pytest.skip(f"benchmark report absent: {MD_PATH}")
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    clean = data["experiments"]["clean_ideal"]
    md_rows = _main_table_rows(MD_PATH.read_text(encoding="utf-8"))
    assert set(md_rows) == {"ga_polynomial", "unassisted_rime", "assisted_rime",
                            "autonomous_transmuter"}, sorted(md_rows)
    precisions = {
        "ga_polynomial": (("r2_train", 4), ("r2_extrapolation", 4),
                          ("rmse_extrapolation", 4), ("rho1", 4)),
        "unassisted_rime": (("r2_train", 4), ("r2_extrapolation", 4),
                            ("rmse_extrapolation", 4), ("rho1", 4)),
        "assisted_rime": (("r2_train", 6), ("r2_extrapolation", 6),
                          ("rmse_extrapolation", 6), ("rho1", 4)),
        "autonomous_transmuter": (("r2_train", 6), ("r2_extrapolation", 6),
                                  ("rmse_extrapolation", 6), ("rho1", 4)),
    }
    failures: list[str] = []
    for key, cells in precisions.items():
        problems = check_row_against_report(clean[key], md_rows[key], numeric_cells=cells)
        failures.extend(f"{key}: {p}" for p in problems)
    # GA condition-number cell is JSON-derived prose: checked explicitly.
    assert f"`{clean['ga_polynomial']['cond_number']:.2e}`" in md_rows["ga_polynomial"]
    assert not failures, "\n".join(failures)


def test_no_hardcoded_verdict_labels_in_harness():
    """Verdicts render from verdict_for_row; hardcoded labels are a regression."""
    src = HARNESS.read_text(encoding="utf-8")
    assert "**PASSED** (Exact Recovery)" not in src, "hardcoded PASS label in harness"
    assert "**REFUSED** ($" not in src, "hardcoded REFUSE label in harness"
    assert "verdict_for_row" in src, "harness does not derive verdicts from rows"


def test_rendered_verdict_cells_are_present_in_report():
    """End-to-end: the function the template calls produces the cells on disk."""
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    clean = data["experiments"]["clean_ideal"]
    md = MD_PATH.read_text(encoding="utf-8")
    for key in ("ga_polynomial", "unassisted_rime", "assisted_rime", "autonomous_transmuter"):
        row = clean[key]
        cell = render_verdict_cell(*verdict_for_row(
            row["r2_extrapolation"], row["rho1"], row["rmse_extrapolation"]))
        assert cell in md, (key, cell)
