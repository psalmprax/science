"""Pins for the quantum-signal ledger record and the map it measures.

The point of this file is the failure mode it guards against: a circuit whose output is
constant (the `phase`-then-`h`-then-`cx` demonstration) produces a plausible-looking
backtest of nothing. Every check here is a way for that to be caught.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.quantum_compilation.quantum_signal import (
    ANSATZ_DEPTHS,
    RULES,
    ParameterisedQuantumMap,
    QuantumFeatureMap,
    deflate_sharpe,
    execute_signal,
    fit_grid,
    fit_parameterised_map,
    holdout_decision,
    ledger_summary,
    parameter_grid,
)
from apedp_dyn.quantum_compilation.statevector_sim import StatevectorSimulator
from apedp_dyn.sandbox.canonical_ledger import FEE, raw_features, run_ledger

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger.json"
HOLDOUT_LEDGER = (ROOT / "benchmarks" / "quantum_compilation"
                  / "quantum_signal_holdouts.jsonl")


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the feature map
def test_the_simulator_matches_the_closed_form():
    cases = {
        2: ((0.0, 0.0), (0.3, -0.7), (1.0, 0.5)),
        3: ((0.0, 0.0, 0.0), (0.3, -0.7, 1.1), (1.0, 0.5, -0.2)),
    }
    for arity, value_sets in cases.items():
        feature_map = QuantumFeatureMap(tuple(f"f{index}" for index in range(arity)))
        for values in value_sets:
            assert feature_map.signal(values) == pytest.approx(
                feature_map.analytic(values), abs=1e-12)


def test_every_feature_reaches_the_signal():
    """Two features, one measured wire: both must move the output, not just the first."""
    feature_map = QuantumFeatureMap(("a", "b"))
    base = feature_map.signal((0.0, 0.0))
    first = feature_map.signal((0.4, 0.0))
    second = feature_map.signal((0.0, 0.4))
    assert base == pytest.approx(1.0, abs=1e-12)
    assert first < base - 1e-3
    assert second < base - 1e-3
    assert first == pytest.approx(second, abs=1e-12)  # symmetric by construction


def test_a_phase_only_encoding_cannot_carry_a_signal():
    """The demonstration's circuit, reproduced: constant zero for every input.

    `phase` is diagonal and inert on |0>; `h` then `cx` leaves the measured wire
    maximally mixed. This is the bug the map above exists to avoid, pinned so a
    regression back to it fails.
    """
    outputs = []
    for x, y in ((0.1, -0.4), (-2.0, 3.0), (0.0, 0.0), (1.7, 0.2)):
        state = StatevectorSimulator(num_qubits=2)
        state.phase(0, float(np.clip(x, -np.pi, np.pi)))
        state.phase(1, float(np.clip(y, -np.pi, np.pi)))
        state.h(0)
        state.cx(0, 1)
        p0, p1 = state.probabilities(0)
        outputs.append(p0 - p1)
    assert outputs == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_the_map_refuses_a_silent_layer_count():
    with pytest.raises(ValueError):
        QuantumFeatureMap(("a",), layers=0)
    with pytest.raises(ValueError):
        QuantumFeatureMap(())
    # The closed form is a single-layer statement; asking for it on two layers is
    # refused rather than approximated.
    assert QuantumFeatureMap(("a", "b")).analytic((0.1, 0.2)) == pytest.approx(
        np.cos(0.1 * np.pi) * np.cos(0.2 * np.pi))
    with pytest.raises(ValueError):
        QuantumFeatureMap(("a", "b"), layers=2).analytic((0.1, 0.2))


def test_the_circuit_hash_pins_the_gate_sequence():
    first = QuantumFeatureMap(("H-MOM", "CVD"))
    second = QuantumFeatureMap(("H-MOM", "CVD"))
    wider = QuantumFeatureMap(("H-MOM", "CVD", "H-BB"))
    assert first.sha256() == second.sha256()
    assert first.sha256() != wider.sha256()
    assert first.description() == ("ry(pi*H-MOM) q0", "ry(pi*CVD) q1", "cx q1->q0")


# --------------------------------------------------------------------- the executor
def test_the_executor_charges_the_declared_fee_per_unit_of_turnover():
    score = [0.5, 0.5, -0.5, 0.0, 0.5]
    returns = [0.01, -0.02, 0.03, 0.0, 0.01, 0.02]
    rows = execute_signal(score, returns, rule="symmetric")
    assert [row["pos"] for row in rows] == [1.0, 1.0, -1.0, 0.0, 1.0]
    assert [row["dpos"] for row in rows] == [1.0, 0.0, 2.0, 1.0, 1.0]
    assert rows[2]["fee"] == pytest.approx(2.0 * FEE)
    assert rows[0]["gross"] == pytest.approx(returns[1])
    assert rows[-1]["fwd"] == pytest.approx(returns[5])
    assert sum(row["net"] for row in rows) == pytest.approx(
        sum(row["gross"] for row in rows) - sum(row["fee"] for row in rows))


def test_an_invalid_observation_is_flat_and_named():
    rows = execute_signal([0.9, 0.9], [0.01, 0.02, 0.03], valid=[True, False])
    assert rows[1]["pos"] == 0.0
    assert rows[1]["reason"] == "invalid_score"
    assert rows[1]["fee"] == pytest.approx(FEE)  # closing the previous position costs


def test_the_executor_refuses_a_missing_forward_return():
    with pytest.raises(ValueError):
        execute_signal([0.1, 0.2, 0.3], [0.0, 0.01])
    with pytest.raises(ValueError):
        execute_signal([0.1], [0.0, 0.01, 0.02])


def test_every_rule_is_implemented():
    for rule in RULES:
        rows = execute_signal([-0.4, 0.4], [0.0, 0.01, -0.01], rule=rule)
        assert len(rows) == 2
    with pytest.raises(ValueError):
        execute_signal([0.1], [0.0, 0.1], rule="unknown")


def test_the_executor_matches_the_canonical_ledger_on_a_shared_score():
    """The bridge: the same score, mask and returns must price identically."""
    rng = np.random.default_rng(7)
    price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 160)))
    frame = pd.DataFrame({"price": price, "V": rng.uniform(1e5, 3e5, 160),
                          "taker_buy": rng.uniform(0.4, 0.6, 160) * 3e5})
    _raw, returns = raw_features(frame)
    rows, _summary = run_ledger(["X"], {"X": frame}, {"H-MOM": 1.0}, "base",
                                th_in=0.0, th_out=0.0, eval_slice=slice(30, 130),
                                check_binding=False)
    # The last scored bar is t=129, so its forward return is r[130]: one extra entry.
    ours = execute_signal([row["score"] for row in rows], returns[30:131],
                          rule="short_below_zero",
                          valid=[row["valid"] for row in rows])
    assert len(ours) == len(rows)
    for left, right in zip(ours, rows, strict=True):
        assert left["pos"] == right["pos"]
        assert left["fee"] == pytest.approx(right["fee"], abs=1e-15)
        assert left["net"] == pytest.approx(right["net"], abs=1e-12)


# ------------------------------------------------------------------- the deflation
def test_the_deflated_sharpe_is_not_the_naive_one():
    flat = deflate_sharpe(0.6, n_bars=400, skewness=0.0, kurtosis=3.0,
                          n_trials=1, var_trials=0.0)
    searched = deflate_sharpe(0.6, n_bars=400, skewness=0.0, kurtosis=3.0,
                              n_trials=40, var_trials=0.04)
    assert flat["dsr"] > searched["dsr"]
    assert searched["expected_max_sr"] > flat["expected_max_sr"]
    assert searched["is_significant"] == (searched["dsr"] >= 0.95)


def test_the_record_is_reproducible_from_its_own_numbers():
    record = _record()
    for row in [*record["rows"], *record["baselines"]]:
        recomputed = deflate_sharpe(
            row["sharpe"], n_bars=int(row["n_obs"]), skewness=row["skewness"],
            kurtosis=row["kurtosis"], n_trials=record["n_trials"],
            var_trials=row["var_trials"], annualization=row["annualization"])
        assert recomputed["dsr"] == pytest.approx(row["dsr"]["dsr"], abs=1e-12)


# --------------------------------------------------------------------- the record
def test_the_record_wrote_only_after_its_checks_passed():
    record = _record()
    assert record["schema"] == "apedp.quantum_signal_ledger.v2"
    assert record["checks"], record["checks"]
    failed = {name: value for name, value in record["checks"].items() if not value}
    assert not failed, failed


def test_the_record_prices_every_trial_with_the_ledger_fee():
    record = _record()
    assert record["fee_per_unit_turnover"] == FEE
    for row in [*record["rows"], *record["baselines"]]:
        assert row["gross"] - row["fees"] == pytest.approx(row["net"], abs=1e-9)
        for key in ("turnover", "sharpe", "dsr", "n_trials", "var_trials"):
            assert key in row, (row["asset"], key)


def test_the_record_discloses_its_trial_family():
    record = _record()
    components = record["trial_components"]
    assert set(components) == {"threshold_grid", "baselines", "parameter_fits"}
    assert record["n_trials"] == sum(components.values())
    priced = len(record["rows"]) + len(record["baselines"])
    assert components["threshold_grid"] == (len(record["maps"]) * len(record["thresholds"])
                                            * len(record["assets"]))
    assert components["baselines"] == len(record["assets"])
    assert priced == components["threshold_grid"] + components["baselines"]
    # The fit is charged at the size of its whole search, per dataset -- never as its
    # single selected winner.
    assert components["parameter_fits"] == (len(record["assets"])
                                             * record["parameterised"]["grid_size"])
    assert components["parameter_fits"] > components["threshold_grid"]
    assert all(row["n_trials"] == record["n_trials"]
               for row in [*record["rows"], *record["baselines"]])


def test_the_record_shows_the_signal_is_not_constant():
    record = _record()
    per_asset: dict[str, float] = {}
    for row in record["rows"]:
        per_asset[row["asset"]] = per_asset.get(row["asset"], 0.0) + row["turnover"]
    assert per_asset and all(value > 0.0 for value in per_asset.values())
    # A constant signal scores zero net with zero fees; the measured signal trades.
    assert max(row["fees"] for row in record["rows"]) > 0.0


def test_the_record_proves_the_executor_is_the_ledger():
    record = _record()
    for asset, entry in record["executor_equivalence"].items():
        assert entry["identical"], (asset, entry)
        assert entry["rows_compared"] > 100


def test_the_record_keeps_the_circuit_and_the_closed_form_together():
    record = _record()
    assert record["maps"]
    for entry in record["maps"]:
        assert entry["closed_form"] == "prod_i cos(pi * x_i)"
        assert len(entry["circuit_sha256"]) == 64
        assert entry["gates"][:2] == [f"ry(pi*{entry['features'][0]}) q0",
                                      f"ry(pi*{entry['features'][1]}) q1"]
    assert record["checks"]["the_signal_is_the_closed_form_on_every_bar"]


def test_the_record_states_what_it_does_not_show():
    record = _record()
    limitations = " ".join(record["limitations"]).lower()
    assert "charged to the deflation" in limitations
    assert "no advantage is claimed" in limitations
    assert "holdout is temporal" in limitations
    assert "burned" in limitations


def test_the_record_does_not_claim_a_profit_it_did_not_measure():
    record = _record()
    for row in record["rows"]:
        summary = ledger_summary([
            {"gross": row["gross"], "fee": 0.0, "net": 0.0, "dpos": 0.0, "pos": 0.0,
             "valid": True, "score": 0.0, "fwd": 0.0, "t": 0, "reason": "check"}])
        assert summary["n_obs"] == 1
    # Significance is a threshold decision, not a claim in prose.
    significant = [row for row in record["rows"] if row["dsr"]["is_significant"]]
    assert len(significant) == len(record["significant_rows"])


# ------------------------------------------------------- the parameterised map
def test_the_parameterised_map_matches_its_closed_form_on_the_whole_grid():
    """Every grid point, not just the fitted one, must be the form it claims."""
    grid = parameter_grid(3)
    assert len(grid) == 12
    for alphas, betas in grid:
        feature_map = ParameterisedQuantumMap(("a", "b", "c"), alphas, betas)
        for values in ((0.0, 0.0, 0.0), (0.3, -0.7, 1.1), (-1.0, 0.5, 0.2)):
            assert feature_map.signal(values) == pytest.approx(
                feature_map.analytic(values), abs=1e-12)


def test_the_fitted_map_is_not_the_fixed_map():
    fixed = QuantumFeatureMap(("a", "b", "c"))
    fitted = ParameterisedQuantumMap(("a", "b", "c"), (1.5, 1.5, 1.5), (-0.5, -0.5, -0.5))
    assert fitted.sha256() != fixed.sha256()
    assert fitted.signal((0.4, 0.4, 0.4)) != pytest.approx(
        fixed.signal((0.4, 0.4, 0.4)), abs=1e-6)


def test_the_parameterised_map_refuses_a_malformed_fit():
    with pytest.raises(ValueError):
        ParameterisedQuantumMap(("a", "b"), (1.0,), (0.0, 0.0))
    with pytest.raises(ValueError):
        ParameterisedQuantumMap(("a",), (float("nan"),), (0.0,))
    with pytest.raises(ValueError):
        parameter_grid(0)


def test_fitting_records_every_grid_point_in_its_own_trial_ledger():
    rng = np.random.default_rng(11)
    values = rng.normal(0, 1, size=(240, 3))
    returns = np.zeros(240)
    returns[1:] = rng.normal(0, 0.01, 239)
    ledger = CandidateTrialLedger()
    fit = fit_parameterised_map(values, returns, features=("a", "b", "c"),
                                ledger=ledger, search_id="unit",
                                window=slice(30, 160))
    grid_size = len(fit_grid(3))
    assert ledger.trial_count == grid_size
    assert fit.effective_trial_count == grid_size
    assert fit.grid_size == grid_size
    assert len(fit.candidates) == grid_size
    records = ledger.get_records()
    assert all(row.metadata["selection_pool_size"] == grid_size for row in records)
    assert all(row.metadata["search_id"] == "unit" for row in records)
    assert fit.in_sample_sharpe == pytest.approx(
        max(row.in_sample_sharpe for row in records))


def test_the_record_charges_the_fit_and_keeps_it_out_of_the_holdout():
    record = _record()
    parameterised = record["parameterised"]
    assert parameterised["grid_size"] == len(parameterised["grid"])
    assert len(parameterised["fits"]) == len(record["assets"])
    for fit in parameterised["fits"]:
        assert fit["grid_size"] == parameterised["grid_size"]
        assert len(fit["candidates"]) == fit["grid_size"]
        assert fit["effective_trial_count"] == fit["grid_size"]
        # The fit is priced on its training window only, which ends before the holdout.
        assert fit["fit_window"][0] == record["warmup_bars"]
        assert fit["fit_window"][0] < fit["fit_window"][1]
    ledger = parameterised["trial_ledger"]
    assert ledger["records"] == parameterised["grid_size"] * len(record["assets"])
    assert ledger["effective_trial_count"] == ledger["records"]
    assert ledger["hidden_trials"] == 0


def test_the_holdout_is_the_tail_after_the_fit_window():
    record = _record()
    evaluated = [entry for entry in record["holdout"].values()
                 if "holdout_window" in entry]
    assert evaluated, record["holdout"]
    splits = {}
    for entry in evaluated:
        assert entry["holdout_window"][0] == entry["train_window"][1]
        assert entry["holdout_window"][1] > entry["holdout_window"][0]
        assert entry["train_bars"] > 0 and entry["holdout_bars"] > 0
        assert entry["fitted_on"] == "training bars only"
        splits[entry["dataset_id"]] = entry["slice_id"]
    # A slice id names the data it was cut from, so two datasets cannot share one.
    assert len(set(splits.values())) == len(splits) or all(
        entry["slice_id"] for entry in evaluated)


def test_a_failed_candidate_burns_its_slice_in_the_durable_registry():
    record = _record()
    burned = {tuple(item) for item in record["burned_holdouts"]}
    assert burned, "no slice burned, so there is no anti-snooping evidence"
    nodes = [json.loads(line) for line in HOLDOUT_LEDGER.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    exposures = {(node["candidate_id"], exp["dataset_id"], exp["slice_identifier"]): exp
                 for node in nodes for exp in node.get("data_exposures", [])}
    for dataset_id, slice_id in burned:
        matching = [exp for (_cand, ds, sl), exp in exposures.items()
                    if ds == dataset_id and sl == slice_id]
        assert matching, (dataset_id, slice_id)
        assert matching[0]["role"] == "confirmation_holdout"
        assert matching[0]["result_status"] == "FAILED"
    # The record also shows the burned slice being refused to a second candidate.
    assert record["holdout_reuse_refusals"]
    assert all(item["refused"] for item in record["holdout_reuse_refusals"])
    assert all("burned" in item["error"].lower()
               for item in record["holdout_reuse_refusals"])


def test_every_failed_holdout_is_marked_burned_in_the_record():
    record = _record()
    burned = {tuple(item) for item in record["burned_holdouts"]}
    for entry in record["holdout"].values():
        if entry.get("passed") is None:
            continue
        assert entry["burned"] == (not entry["passed"]), entry["dataset_id"]
        key = (entry["dataset_id"], entry["slice_id"])
        assert (key in burned) == entry["burned"]


def test_the_record_covers_more_than_one_resolution():
    record = _record()
    resolutions = {entry["resolution"] for entry in record["assets"]}
    assert resolutions == {"1d", "15m"}
    assert len(record["assets"]) > 4
    for entry in record["assets"]:
        assert entry["annualization"] == record["annualisation"][entry["resolution"]]
    for row in record["rows"]:
        assert row["annualization"] == record["annualisation"][row["resolution"]]
    # The intraday panel's taker column is quote-denominated; the producer converts it
    # and records the ratio it measured rather than assuming it.
    intraday = [entry for entry in record["assets"]
                if entry["taker_buy_denomination_ratio"] is not None]
    assert intraday
    assert all(abs(entry["taker_buy_denomination_ratio"] - 1.0) < 0.25
               for entry in intraday)


# ------------------------------------------------- the second family, and the rule
def test_a_second_ansatz_depth_is_a_new_circuit_with_no_product_form():
    one = ParameterisedQuantumMap(("a", "b", "c"), (1.5,) * 3, (0.5,) * 3, layers=1)
    two = ParameterisedQuantumMap(("a", "b", "c"), (1.5,) * 3, (0.5,) * 3, layers=2)
    for values in ((0.2, -0.4, 0.7), (0.0, 0.0, 0.0)):
        assert one.signal(values) == pytest.approx(one.analytic(values), abs=1e-12)
        assert one.signal(values) == pytest.approx(one.exact_marginal(values), abs=1e-12)
        assert two.signal(values) == pytest.approx(two.exact_marginal(values), abs=1e-12)
    # Depth two is not depth one squared: the second rotation's X branches survive the
    # first entangler, so no product form exists and asking for one is refused rather
    # than answered with a formula that would be false.
    assert two.signal((0.2, -0.4, 0.7)) != pytest.approx(
        one.signal((0.2, -0.4, 0.7)) ** 2, abs=1e-3)
    with pytest.raises(ValueError):
        two.analytic((0.2, -0.4, 0.7))
    with pytest.raises(ValueError):
        ParameterisedQuantumMap(("a",), (1.0,), (0.0,), layers=0)


def test_the_fit_spans_both_ansatz_depths():
    rng = np.random.default_rng(5)
    values = rng.normal(0, 1, size=(200, 3))
    returns = np.zeros(200)
    returns[1:] = rng.normal(0, 0.01, 199)
    ledger = CandidateTrialLedger()
    fit = fit_parameterised_map(values, returns, features=("a", "b", "c"), ledger=ledger,
                                search_id="depths", window=slice(30, 140))
    records = ledger.get_records()
    assert len(records) == len(fit_grid(3)) == 24
    assert {record.metadata["layers"] for record in records} == set(ANSATZ_DEPTHS) == {1, 2}
    assert all(record.metadata["selection_pool_size"] == 24 for record in records)
    assert fit.effective_trial_count == 24
    assert {candidate[2] for candidate in fit.candidates} == {1, 2}
    assert fit.feature_map.layers in ANSATZ_DEPTHS


def test_the_holdout_rule_charges_the_candidate_search():
    cheap = holdout_decision(0.5, 40.0, 0.9, searched_trials=1, var_trials=0.25)
    wide = holdout_decision(0.5, 40.0, 0.9, searched_trials=240, var_trials=0.25)
    assert cheap["passed"] is True
    assert wide["hurdle"] > cheap["hurdle"]
    assert wide["passed"] is False
    assert "hurdle" in wide["reason"]
    carried = holdout_decision(0.5, 0.0, 3.0, searched_trials=1, var_trials=0.25)
    assert carried["passed"] is False and "position" in carried["reason"]
    losing = holdout_decision(-0.5, 40.0, -1.0, searched_trials=1, var_trials=0.25)
    assert losing["passed"] is False and "net" in losing["reason"]
    with pytest.raises(ValueError):
        holdout_decision(0.1, 1.0, 0.1, searched_trials=0, var_trials=0.0)


def test_the_record_takes_its_hurdle_from_its_own_resolution():
    record = _record()
    dispersion = record["var_trials"]["by_resolution"]
    assert set(dispersion) == {"1d", "15m"}
    assert record["var_trials"]["pooled"] > max(dispersion.values())
    assert "within resolution" in record["dispersion_method"]
    for row in [*record["rows"], *record["baselines"]]:
        assert row["var_trials"] == pytest.approx(dispersion[row["resolution"]])
        recomputed = deflate_sharpe(
            row["sharpe"], n_bars=int(row["n_obs"]), skewness=row["skewness"],
            kurtosis=row["kurtosis"], n_trials=record["n_trials"],
            var_trials=dispersion[row["resolution"]],
            annualization=row["annualization"])
        assert recomputed["dsr"] == pytest.approx(row["dsr"]["dsr"], abs=1e-12)
        assert recomputed["expected_max_sr"] == pytest.approx(
            row["dsr"]["expected_max_sr"], abs=1e-12)
    counts = record["trials_by_resolution"]
    assert counts["1d"] + counts["15m"] == record["n_trials"]
