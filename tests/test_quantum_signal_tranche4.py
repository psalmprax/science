"""Pins for tranche 4: walk-forward folds, per-fold burns, and the fee-share report.

Each base dataset is judged on two contiguous test blocks, each with its own training
prefix and its own slice id. A fold that fails burns its own slice, so the record shows
the dataset read across regimes without any slice being read twice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apedp_dyn.quantum_compilation.quantum_signal import walk_forward_windows

ROOT = Path(__file__).resolve().parent.parent
RECORDS = {
    "tranche1": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger.json",
    "tranche2": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche2.json",
    "tranche3": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche3.json",
    "tranche4": ROOT / "benchmarks" / "quantum_compilation" / "quantum_signal_ledger_tranche4.json",
}
RECORD4 = RECORDS["tranche4"]
HOLDOUT_LEDGER = (ROOT / "benchmarks" / "quantum_compilation"
                  / "quantum_signal_holdouts.jsonl")


def _record(path: Path = RECORD4) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_fold_windows_are_contiguous_and_expanding():
    windows = walk_forward_windows(1000, 2, 30)
    assert windows == [(30, 353, 353, 676), (30, 676, 676, 999)]
    assert all(window[1] == window[2] for window in windows)  # test starts at train end
    assert windows[0][3] == windows[1][1]  # no overlap, no gap
    assert {window[0] for window in windows} == {30}  # expanding, not sliding
    with pytest.raises(ValueError):   # 2 post-warm-up bars cannot fill 2 folds + 1 blocks
        walk_forward_windows(33, 2, 30)
    with pytest.raises(ValueError):
        walk_forward_windows(1000, 0, 30)
    # A short but feasible series still yields folds, and never a zero-width one.
    assert all(window[3] > window[2] for window in walk_forward_windows(40, 2, 30))


def test_every_dataset_is_judged_once_per_fold():
    record = _record()
    datasets = record["assets"]
    assert {entry["folds"] for entry in datasets} == {2}
    assert {entry["fold"] for entry in datasets} == {1, 2}
    base = {entry["dataset_id"].split("#fold")[0] for entry in datasets}
    assert len(datasets) == 2 * len(base)
    for entry in datasets:
        train_start, train_end = entry["train_window"]
        test_start, test_end = entry["holdout_window"]
        assert test_start == train_end
        assert test_end > test_start
        assert train_start < train_end
    # Folds of one dataset cover disjoint, adjacent blocks.
    for dataset in base:
        folds = sorted((entry for entry in datasets
                        if entry["dataset_id"].startswith(f"{dataset}#fold")),
                       key=lambda entry: entry["fold"])
        assert folds[0]["holdout_window"][1] == folds[1]["holdout_window"][0]


def test_the_charge_covers_both_folds():
    record = _record()
    components = record["trial_components"]
    assert components["threshold_grid"] == 2 * 3 * len(record["assets"])
    assert components["baselines"] == len(record["assets"])
    assert components["parameter_fits"] == len(record["assets"]) * 34
    assert record["n_trials"] == 820 == sum(components.values())
    assert record["depth_comparison"]["charge_per_family"] == 34


def test_each_fold_burns_its_own_slice():
    record = _record()
    evaluated = [entry for entry in record["holdout"].values()
                 if "holdout_window" in entry]
    assert len(evaluated) == len(record["assets"])
    slices = set()
    for entry in evaluated:
        assert "#fold" in entry["dataset_id"]
        assert entry["slice_id"].startswith("fold")
        assert entry["slice_id"] not in slices
        slices.add(entry["slice_id"])
        assert entry["passed"] == (entry["net"] > 0.0 and entry["turnover"] > 0.0
                                   and entry["sharpe"] >= entry["hurdle"])
    assert len(record["burned_holdouts"]) >= 48
    nodes = [json.loads(line)
             for line in HOLDOUT_LEDGER.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    seen = {(exp["dataset_id"], exp["slice_identifier"])
            for node in nodes for exp in node.get("data_exposures", [])}
    for dataset_id, slice_id in record["burned_holdouts"]:
        assert (dataset_id, slice_id) in seen


def test_the_fee_share_is_reported_per_resolution_at_every_rate():
    for name, path in RECORDS.items():
        record = _record(path)
        sweep = record["cost_sensitivity"]
        priced = [*record["rows"], *record["baselines"]]
        for label, rate in (("5bps", 0.0005), ("10bps", 0.001), ("20bps", 0.002)):
            shares = sweep["sweep"][label]["fee_share_of_gross_by_resolution"]
            assert set(shares) == {row["resolution"] for row in priced}, name
            fees_by_resolution = sweep["sweep"][label]["fees_by_resolution"]
            gross_by_resolution = sweep["sweep"][label]["gross_by_resolution"]
            over_abs = sweep["sweep"][label]["fees_over_absolute_gross_by_resolution"]
            assert set(fees_by_resolution) == set(shares), name
            for resolution, share in shares.items():
                family = [row for row in priced if row["resolution"] == resolution]
                gross = sum(row["gross"] for row in family)
                turnover = sum(row["turnover"] for row in family)
                assert fees_by_resolution[resolution] == pytest.approx(turnover * rate,
                                                                      abs=1e-12)
                assert gross_by_resolution[resolution] == pytest.approx(gross, abs=1e-9)
                if gross > 0:
                    assert share == pytest.approx(turnover * rate / gross, abs=1e-12)
                else:
                    assert share is None
                # The always-defined companion: the bill relative to the gross's size.
                if abs(gross) > 0:
                    assert over_abs[resolution] == pytest.approx(
                        turnover * rate / abs(gross), abs=1e-12)
        # The canonical rate is the one the ledger charges.
        assert record["fee_per_unit_turnover"] == 0.0005
