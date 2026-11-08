"""What the quantum-signal study could detect, and what the programme really spent.

The five tranche records report how many rows cleared the 0.95 floor. None of
them reported the smallest true Sharpe the design could have detected at that
floor -- so "no row clears 0.95" could not be told apart from "the test is too
blunt to see anything smaller than a fantasy". This record carries both numbers,
and re-prices every row against the trials the whole programme spent rather than
each tranche's own count.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD
from apedp_dyn.epistemic.trial_ledger import compute_deflated_sharpe_ratio

ROOT = Path(__file__).resolve().parent.parent
POWER = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_power.json"
TRANCHE_FILES = {
    "tranche1": "quantum_signal_ledger.json",
    "tranche2": "quantum_signal_ledger_tranche2.json",
    "tranche3": "quantum_signal_ledger_tranche3.json",
    "tranche4": "quantum_signal_ledger_tranche4.json",
    "tranche5": "quantum_signal_ledger_tranche5.json",
}


@pytest.fixture(scope="module")
def record() -> dict:
    assert POWER.is_file(), f"power record missing at {POWER}"
    return json.loads(POWER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tranches() -> dict[str, dict]:
    return {name: json.loads((POWER.parent / filename).read_text(encoding="utf-8"))
            for name, filename in TRANCHE_FILES.items()}


def _dsr(observed: float, cell: dict) -> float:
    return compute_deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_trials=int(cell["n_trials"]),
        var_trials=float(cell["var_trials"]),
        n_bars=int(cell["n_bars_median"]),
        skewness=float(cell["skewness_median"]),
        kurtosis=float(cell["kurtosis_median"]),
        annualization_factor=float(cell["annualization"]),
    )["dsr"]


def test_every_tranche_and_resolution_is_covered(record, tranches):
    assert record["floor"] == DSR_THRESHOLD == 0.95
    assert set(record["by_record"]) == set(TRANCHE_FILES)
    for name, per_resolution in record["by_record"].items():
        resolutions = {row["resolution"] for row in [*tranches[name]["rows"],
                                                    *tranches[name]["baselines"]]}
        assert set(per_resolution) == resolutions, name
        assert per_resolution, f"{name} reports no resolution"


def test_the_minimum_detectable_sharpe_is_where_the_floor_is_met(record):
    """The number is defined by the DSR, so evaluating the DSR there must land on it."""
    for name, per_resolution in record["by_record"].items():
        for resolution, cell in per_resolution.items():
            achieved = _dsr(cell["minimum_detectable_sharpe"], cell)
            assert achieved == pytest.approx(DSR_THRESHOLD, abs=1e-6), (name, resolution)


def test_power_is_monotone_and_bounded(record):
    for name, per_resolution in record["by_record"].items():
        for resolution, cell in per_resolution.items():
            curve = cell["power_curve"]
            assert len(curve) == len(record["reference_true_sharpes"])
            from itertools import pairwise

            for low, high in pairwise(curve):
                assert low["true_sharpe"] < high["true_sharpe"]
                assert low["power"] <= high["power"] + 1e-12, (name, resolution)
            for point in curve:
                assert 0.0 <= point["power"] <= 1.0


def _power_module():
    spec = importlib.util.spec_from_file_location(
        "quantum_signal_power_producer",
        ROOT / "scripts" / "measure_quantum_signal_power.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_curve_is_reproducible_from_the_recorded_inputs(record):
    """The committed curve must be regenerable from the cell's own four numbers."""
    module = _power_module()
    for per_resolution in record["by_record"].values():
        for cell in per_resolution.values():
            for point in cell["power_curve"]:
                fresh = module.power_at(
                    point["true_sharpe"],
                    n_bars=int(cell["n_bars_median"]),
                    n_trials=int(cell["n_trials"]),
                    var_trials=float(cell["var_trials"]),
                    skewness=float(cell["skewness_median"]),
                    kurtosis=float(cell["kurtosis_median"]),
                    annualization=float(cell["annualization"]))
                assert fresh["power"] == pytest.approx(point["power"], abs=1e-12)
                assert fresh["detectable_at_floor"] == pytest.approx(
                    cell["minimum_detectable_sharpe"], abs=1e-9)


def test_the_headline_reports_a_measured_power_for_a_sharpe_of_one(record):
    """The claim this record exists to make: a Sharpe of 1.0 is essentially invisible."""
    for name, cell in ((name, entry)
                       for name, per_resolution in record["headline"].items()
                       for entry in per_resolution.values()):
        assert 0.0 <= cell["power_at_sharpe_1"] < 0.5, name
        assert cell["minimum_detectable_sharpe"] > 1.0, name


def test_the_programme_charge_covers_every_tranche(record, tranches):
    programme = record["programme"]
    assert programme["total_trials"] == sum(int(entry["n_trials"])
                                            for entry in tranches.values())
    assert programme["trials_by_record"] == {name: int(entry["n_trials"])
                                             for name, entry in tranches.items()}
    assert len(programme["rows"]) == sum(len(entry["rows"]) + len(entry["baselines"])
                                         for entry in tranches.values())


def test_the_programme_hurdle_never_falls_below_the_as_run_hurdle(record):
    for row in record["programme"]["rows"]:
        assert row["hurdle_programme"] >= row["hurdle_as_run"] - 1e-9, row["dataset_id"]
        assert row["dsr_programme"] <= row["dsr_as_run"] + 1e-12, row["dataset_id"]
    assert record["programme"]["rows_that_become_significant"] == 0


def test_the_programme_still_finds_nothing_significant(record):
    assert record["programme"]["rows_clearing_floor_as_run"] == 0
    assert record["programme"]["rows_clearing_floor_programme"] == 0


def test_the_record_reproduces_the_tranche_dsr_it_reads(record, tranches):
    """A power record that disagreed with the records it read would be worthless."""
    for entry in tranches.values():
        for row in [*entry["rows"], *entry["baselines"]][:5]:
            recomputed = compute_deflated_sharpe_ratio(
                observed_sharpe=float(row["sharpe"]),
                n_trials=int(entry["n_trials"]),
                var_trials=float(entry["var_trials"]["by_resolution"][row["resolution"]]),
                n_bars=int(row["n_obs"]),
                skewness=float(row["skewness"]),
                kurtosis=float(row["kurtosis"]),
                annualization_factor=float(row["annualization"]),
            )["dsr"]
            assert recomputed == pytest.approx(float(row["dsr"]["dsr"]), abs=1e-9)


def test_every_check_the_producer_ran_passed(record):
    assert record["checks"] and all(record["checks"].values()), (
        "failed checks: " + ", ".join(n for n, ok in record["checks"].items() if not ok))


def test_the_record_states_what_it_read(record):
    assert "no panel" in record["method"]["inputs_read"]
    assert "no holdout slice" in record["method"]["inputs_read"]
    assert any("floor on the programme's multiplicity" in item
               for item in record["limitations"])
