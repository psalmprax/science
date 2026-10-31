"""Pins for tranche 3 and the cost-model sweep.

Tranche 3 changes two things at once: unseen slices (six new daily assets, four symbols at
30 minutes) and a **continuous** optimiser whose every evaluation is a recorded trial, so
the charge is an iteration count rather than a grid. The sweep re-prices every priced trial
at three fee rates from stored aggregates only -- no panel and no holdout slice is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORDS = {
    "tranche1": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger.json",
    "tranche2": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche2.json",
    "tranche3": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche3.json",
}
RECORD3 = RECORDS["tranche3"]
HOLDOUT_LEDGER = (ROOT / "benchmarks" / "quantum_compilation"
                  / "quantum_signal_holdouts.jsonl")
GRID_ALPHAS = {0.5, 1.0, 1.5, 2.0}


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_tranche_three_passed_its_own_checks():
    record = _record(RECORD3)
    assert record["schema"] == "apedp.quantum_signal_ledger.v2"
    assert record["checks"] and all(record["checks"].values()), record["checks"]


def test_the_charge_is_the_iteration_count_not_a_grid():
    record = _record(RECORD3)
    parameterised = record["parameterised"]
    assert parameterised["optimiser"] == "continuous"
    assert parameterised["grid_size"] is None
    assert parameterised["grid"] is None
    assert parameterised["evaluations_per_depth"] == 17
    ledger = parameterised["trial_ledger"]
    # The declared budget is charged even where a probe was skipped (an alpha step that
    # would have crossed zero), so the effective count is the budget and the difference
    # is disclosed rather than silently dropped.
    assert ledger["effective_trial_count"] == len(record["assets"]) * 34
    assert ledger["records"] <= ledger["effective_trial_count"]
    assert ledger["hidden_trials"] == (ledger["effective_trial_count"]
                                       - ledger["records"])
    assert record["trial_components"]["parameter_fits"] == len(record["assets"]) * 34
    assert record["n_trials"] == sum(record["trial_components"].values())
    for fit in parameterised["fits"]:
        assert fit["optimiser"] == "continuous"
        assert fit["trial_count"] <= fit["effective_trial_count"]
        assert fit["effective_trial_count"] == 34
        assert {candidate["layers"] for candidate in fit["candidates"]} == {1, 2}


def test_the_parameters_are_not_drawn_from_the_grid():
    """A grid search cannot produce 0.8 or 1.2; the continuous search does."""
    record = _record(RECORD3)
    alphas = {fit["alphas"][0] for fit in record["parameterised"]["fits"]}
    assert alphas - GRID_ALPHAS, alphas
    assert any(abs(alpha - round(alpha)) > 1e-9 for alpha in alphas - GRID_ALPHAS)


def test_the_charged_rule_rejected_a_profitable_candidate():
    record = _record(RECORD3)
    evaluated = [entry for entry in record["holdout"].values()
                 if "holdout_window" in entry]
    assert len(evaluated) == len(record["assets"])
    for entry in evaluated:
        assert entry["searched_trials"] == 34
        expected = (entry["net"] > 0.0 and entry["turnover"] > 0.0
                    and entry["sharpe"] >= entry["hurdle"])
        assert entry["passed"] is expected, entry["dataset_id"]
    profitable_rejections = [entry for entry in evaluated
                             if entry["net"] > 0.0 and entry["turnover"] > 0.0
                             and not entry["passed"]]
    assert profitable_rejections, "the hurdle never bound, so the charge is decorative"
    assert all("hurdle" in entry["decision_reason"] for entry in profitable_rejections)


def test_the_registry_only_grows():
    record = _record(RECORD3)
    assert len(record["burned_holdouts"]) >= 28
    nodes = [json.loads(line)
             for line in HOLDOUT_LEDGER.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    seen = {(exp["dataset_id"], exp["slice_identifier"])
            for node in nodes for exp in node.get("data_exposures", [])}
    for dataset_id, slice_id in record["burned_holdouts"]:
        assert (dataset_id, slice_id) in seen


def test_the_cost_sweep_reprices_every_trial_exactly():
    for name, path in RECORDS.items():
        record = _record(path)
        sweep = record["cost_sensitivity"]
        assert sweep["rates"] == [0.0005, 0.001, 0.002], name
        assert sweep["sharpe_recomputed"] is False
        assert "per-bar" in sweep["why"]
        priced = [*record["rows"], *record["baselines"]]
        gross = sum(row["gross"] for row in priced)
        turnover = sum(row["turnover"] for row in priced)
        for label, rate in (("5bps", 0.0005), ("10bps", 0.001), ("20bps", 0.002)):
            entry = sweep["sweep"][label]
            assert entry["fee_per_unit_turnover"] == rate
            assert entry["gross"] == pytest.approx(gross, abs=1e-9)
            assert entry["fees"] == pytest.approx(turnover * rate, abs=1e-9)
            assert entry["net"] == pytest.approx(entry["gross"] - entry["fees"], abs=1e-9)
            assert entry["rows"] == len(priced)
            assert entry["rows_net_positive"] == sum(
                1 for row in priced if row["gross"] - row["turnover"] * rate > 0.0)
        # Charging twice as much cannot leave the family better off.
        assert sweep["sweep"]["20bps"]["net"] <= sweep["sweep"]["5bps"]["net"]
        assert sweep["sweep"]["10bps"]["fees"] == pytest.approx(
            sweep["sweep"]["5bps"]["fees"] * 2.0, abs=1e-9)
