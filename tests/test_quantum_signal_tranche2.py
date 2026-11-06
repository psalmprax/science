"""Pins for the second quantum-signal tranche: two depths, a charged burn, split verdicts.

Tranche 1 (`quantum_signal_ledger.json`) fitted one depth and decided its slices on a
sign-and-turnover rule. Tranche 2 is a second record that could not reuse those slices:
it measures unseen assets at a new resolution, searches both ansatz depths, and charges
each family's holdout decision its own 24-point search. These pins hold it to those
claims rather than to the numbers it happens to have produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD2 = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche2.json"
HOLDOUT_LEDGER = (ROOT / "benchmarks" / "quantum_compilation"
                  / "quantum_signal_holdouts.jsonl")


def _record() -> dict:
    return json.loads(RECORD2.read_text(encoding="utf-8"))


def test_the_tranche_is_a_record_of_its_own_with_every_check_passed():
    record = _record()
    assert record["schema"] == "apedp.quantum_signal_ledger.v2"
    assert record["checks"] and all(record["checks"].values()), record["checks"]
    assert len(record["checks"]) == 14


def test_the_fit_is_charged_for_two_depths_per_dataset():
    record = _record()
    components = record["trial_components"]
    parameterised = record["parameterised"]
    assert parameterised["ansatz_depths"] == [1, 2]
    assert parameterised["grid_size"] == 24
    assert components["parameter_fits"] == len(record["assets"]) * 24
    assert record["n_trials"] == sum(components.values())
    assert record["n_trials"] == 310
    for fit in parameterised["fits"]:
        assert fit["grid_size"] == 24
        assert len(fit["candidates"]) == 24
        assert {candidate["layers"] for candidate in fit["candidates"]} == {1, 2}
        # The fit's own `closed_form` describes the map it selected. A selected depth-2
        # map has no product form, and the record must say so rather than quote one.
        if fit["layers"] == 1:
            assert fit["closed_form"] == "prod_i cos(alpha_i * x_i + beta_i)"
            assert fit["reference"] == "product closed form"
        else:
            assert fit["closed_form"] is None
            assert fit["reference"] == "dense unitary marginal (exact_marginal)"


def test_each_resolution_is_adjudicated_on_its_own_hurdle():
    record = _record()
    report = record["significance_report"]
    assert set(report) == {"1d", "1h"}
    priced = len(record["rows"]) + len(record["baselines"])
    assert sum(entry["rows"] for entry in report.values()) == priced
    for entry in report.values():
        assert entry["n_trials_charged"] == record["n_trials"]
        assert entry["hurdle"] >= 0.0
        assert entry["verdict"] == ("no row clears the floor"
                                    if entry["rows_clearing_floor"] == 0
                                    else f"{entry['rows_clearing_floor']} row(s) clear the floor")
        assert entry["best"]["dsr"] == pytest.approx(
            max(row["dsr"]["dsr"] for row in [*record["rows"], *record["baselines"]]
                if row["resolution"] == entry["best"]["dataset_id"].rsplit("@", 1)[-1]))
    # The two hurdles are different numbers: that is the reason for the split report.
    hurdles = {entry["hurdle"] for entry in report.values()}
    assert len(hurdles) == 2


def test_the_burn_is_charged_to_the_fitted_candidate_search():
    record = _record()
    entries = [entry for entry in record["holdout"].values()
               if "holdout_window" in entry]
    assert len(entries) == len(record["assets"])
    for entry in entries:
        assert entry["searched_trials"] == 24
        expected = (entry["net"] > 0.0 and entry["turnover"] > 0.0
                    and entry["sharpe"] >= entry["hurdle"])
        assert entry["passed"] is expected, entry["dataset_id"]
        assert entry["burned"] == (not entry["passed"])
    # At least one row nets and still fails: the charge, not the sign, is what rejected it.
    charged = [entry for entry in entries
               if entry["net"] > 0.0 and entry["turnover"] > 0.0 and not entry["passed"]]
    assert charged, "no candidate was rejected by the hurdle, so the rule did not bind"
    assert all("hurdle" in entry["decision_reason"] for entry in charged)


def test_every_burned_slice_is_in_the_durable_registry():
    record = _record()
    burned = {tuple(item) for item in record["burned_holdouts"]}
    assert len(burned) >= 18
    nodes = [json.loads(line)
             for line in HOLDOUT_LEDGER.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    seen = {(node["candidate_id"], exp["dataset_id"], exp["slice_identifier"])
            for node in nodes for exp in node.get("data_exposures", [])}
    for dataset_id, slice_id in burned:
        assert any(ds == dataset_id and sl == slice_id for _cand, ds, sl in seen)


def test_depth_two_is_measured_against_depth_one_at_the_same_charge():
    record = _record()
    summary = record["depth_comparison"]
    assert summary["depths"] == [1, 2]
    assert summary["charge_per_family"] == 24
    assert summary["datasets_compared"] == len(record["assets"])
    rows = summary["rows"]
    assert len(rows) == summary["datasets_compared"]
    for row in rows:
        assert row["delta"] == pytest.approx(row["depth2_net"] - row["depth1_net"])
    assert summary["depth2_beats_depth1_holdout"] == sum(1 for row in rows
                                                         if row["delta"] > 0.0)
    assert summary["mean_holdout_delta"] == pytest.approx(
        sum(row["delta"] for row in rows) / len(rows))
    for entry in record["holdout"].values():
        comparison = entry.get("depth_comparison")
        if not comparison:
            continue
        for layers in ("depth1", "depth2"):
            assert comparison[layers]["searched_trials"] == 24
            assert 0.0 <= comparison[layers]["hurdle"] < 1e6
