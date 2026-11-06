"""Tranche 5: the per-bar series, and what the fee does to the Sharpe at each rate.

Tranches 1-4 stored only aggregates, so `cost_sensitivity` could re-price net and fees
(linear in turnover) but had to refuse Sharpe and DSR. Tranche 5 stores the bars behind
every priced row -- net at the canonical 5 bps plus that bar's `|dpos|` -- so the same
question can be answered for the whole distribution: what is the Sharpe, and the DSR,
under a 5, 10 or 20 bps fee?

These tests pin the storage, not just the summary: the series must be the bars the
record's own statistics came from (re-pricing at 5 bps reproduces the recorded Sharpe),
and the fee share of gross must be reported per resolution rather than pooled.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche5.json"
SERIES = (ROOT / "benchmarks" / "quantum_compilation"
          / "quantum_signal_series_tranche5.jsonl")
PRODUCER = ROOT / "scripts" / "measure_quantum_signal_ledger.py"


def _producer():
    spec = importlib.util.spec_from_file_location("quantum_signal_producer_t5", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> dict:
    assert RECORD.is_file(), f"tranche 5 record missing at {RECORD}"
    return json.loads(RECORD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def series() -> dict[str, dict]:
    assert SERIES.is_file(), f"per-bar series artifact missing at {SERIES}"
    entries = {}
    for line in SERIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            entries[item.pop("key")] = item
    return entries


def test_every_priced_row_has_its_bars_stored(record, series):
    priced = [*record["rows"], *record["baselines"]]
    assert priced, "the record prices no rows"
    assert record["series_artifact"] == str(SERIES.relative_to(ROOT))
    assert record["series_rows"] == len(series) == len(priced)
    for row in priced:
        entry = series[row["series_key"]]
        bars = len(entry["net_5bps"])
        assert bars == int(row["n_obs"]), row["dataset_id"]
        assert len(entry["fee_units"]) == bars
        # net at the canonical rate and the per-bar fee basis must account for the row
        assert sum(entry["net_5bps"]) == pytest.approx(row["net"], abs=1e-6)
        assert sum(entry["fee_units"]) == pytest.approx(row["turnover"], abs=1e-6)


def test_the_stored_bars_reproduce_the_recorded_sharpe(record, series):
    """Re-pricing at the canonical 5 bps must land on the recorded Sharpe exactly."""
    producer = _producer()
    priced = [*record["rows"], *record["baselines"]]
    for row in priced:
        net, sharpe, skew, kurtosis = producer._series_statistics(
            series[row["series_key"]], producer.FEE, producer.FEE, row["annualization"])
        assert sharpe == pytest.approx(row["sharpe"], abs=1e-9), row["dataset_id"]
        assert skew == pytest.approx(row["skewness"], abs=1e-9), row["dataset_id"]
        assert kurtosis == pytest.approx(row["kurtosis"], abs=1e-9), row["dataset_id"]
        assert len(net) == int(row["n_obs"])


def test_the_record_says_its_sharpe_was_recomputed(record):
    sweep = record["cost_sensitivity"]
    assert sweep["sharpe_recomputed"] is True
    assert sweep["series_max_sharpe_deviation"] <= 1e-9
    assert sweep["series_max_net_deviation"] <= 1e-9
    assert sweep["series_rows"] == len(record["rows"]) + len(record["baselines"])
    for name in ("5bps", "10bps", "20bps"):
        entry = sweep["sweep"][name]
        assert entry["series_bars"] > 0
        assert entry["sharpe_by_resolution"] and entry["dsr_by_resolution"]
        assert entry["rows_clearing_floor"] == sum(
            item["rows_clearing_floor"] for item in entry["dsr_by_resolution"].values())


def test_fee_share_of_gross_is_reported_per_resolution(record):
    entry = record["cost_sensitivity"]["sweep"]["5bps"]
    resolutions = set(entry["gross_by_resolution"])
    assert resolutions == {"1d", "5m"}
    for resolution in resolutions:
        gross = entry["gross_by_resolution"][resolution]
        fees = entry["fees_by_resolution"][resolution]
        assert fees > 0.0
        share = entry["fee_share_of_gross_by_resolution"][resolution]
        if gross > 0:
            assert share == pytest.approx(fees / gross, rel=1e-12)
        else:
            # A share of a negative gross is undefined; the ratio to |gross| is not.
            assert share is None
        assert entry["fees_over_absolute_gross_by_resolution"][resolution] == \
            pytest.approx(fees / abs(gross), rel=1e-12)


def test_raising_the_fee_cannot_raise_any_rows_net_or_sharpe(record, series):
    producer = _producer()
    priced = [*record["rows"], *record["baselines"]]
    for row in priced:
        previous_net = None
        previous_sharpe = None
        for rate in (0.0005, 0.001, 0.002):
            net = (np.asarray(series[row["series_key"]]["net_5bps"], float)
                   - (rate - producer.FEE)
                   * np.asarray(series[row["series_key"]]["fee_units"], float))
            total = float(net.sum())
            _n, sharpe, _s, _k = producer._series_statistics(
                series[row["series_key"]], rate, producer.FEE, row["annualization"])
            if previous_net is not None:
                assert total <= previous_net + 1e-12
                assert sharpe <= previous_sharpe + 1e-12
            previous_net, previous_sharpe = total, sharpe


def test_tranche_five_ranks_both_resolutions_separately(record):
    report = record["significance_report"]
    assert set(report) == {"1d", "5m"}
    assert report["1d"]["rows"] and report["5m"]["rows"]
    assert report["1d"]["floor"] == report["5m"]["floor"] == 0.95
    # A 5-minute Sharpe is annualised on 105,120 bars, not on 365.
    assert record["annualisation"]["5m"] == 105120.0
    assert record["annualisation"]["1d"] == 365.0
    for resolution, entry in report.items():
        assert entry["rows_clearing_floor"] == sum(
            1 for row in [*record["rows"], *record["baselines"]]
            if row["resolution"] == resolution and row["dsr"]["is_significant"])


def test_tranche_five_is_charged_and_burned(record):
    assert record["schema"] == "apedp.quantum_signal_ledger.v2"
    assert all(record["checks"].values()), (
        "failed checks: " + ", ".join(name for name, ok in record["checks"].items() if not ok))
    assert record["n_trials"] == sum(record["trial_components"].values())
    # 120 threshold rows (2 maps x 3 thresholds x 20 fold-datasets) + 20 baselines
    assert len(record["rows"]) == 120 and len(record["baselines"]) == 20
    assert record["trial_components"]["parameter_fits"] == record["n_trials"] - 140
    burned = [entry for entry in record["holdout"].values() if entry.get("passed") is False]
    assert burned and all(entry["burned"] for entry in burned)
    assert record["holdout_reuse_refusals"]
    assert all(item["refused"] for item in record["holdout_reuse_refusals"])
    # Every fold slice is consumed once, on its own test block.
    for entry in record["holdout"].values():
        if "holdout_window" in entry:
            assert entry["holdout_window"][0] == entry["train_window"][1]
            assert entry["holdout_window"][1] > entry["holdout_window"][0]


def test_a_record_without_series_still_refuses_to_reprice_sharpe(record):
    """The older tranches keep saying they cannot re-price Sharpe; that is the honest path."""
    producer = _producer()
    trimmed = {"rows": record["rows"], "baselines": record["baselines"]}
    sweep = producer.cost_sensitivity(trimmed["rows"], trimmed["baselines"])
    assert sweep["sharpe_recomputed"] is False
    assert sweep["series_rows"] == 0
    assert "not re-priced" in sweep["why"]
