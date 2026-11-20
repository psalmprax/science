"""Pins for the 2026-11-14 follow-up to the integrity-hardening round.

Each section closes an item the 2026-11-12 round recorded as still open in
``docs/SYSTEM_FLAWS_AUDIT.md``:

  * H-02  promotion stages took their metrics as numbers from the caller;
  * H-04  a holdout slice was identified by its name, not by its content;
  * H-06/H-07  the trading sandbox forgot its trials and never registered the
    holdout it scored;
  * H-12  the market panels' universes were chosen in hindsight, and one panel
    carries forward-filled bars for assets that stopped trading.

Everything here drives the public API with generated data or with the data files
the repository ships; nothing reads source text.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic import promotion_lifecycle as pl
from apedp_dyn.epistemic.agnostic_adapter import InvariantKind
from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
    HoldoutSliceFingerprint,
    fingerprint_holdout_slice,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    EVIDENCE_DERIVED,
    EVIDENCE_REPORTED,
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
    PromotionViolationError,
    information_coefficient,
    one_sided_ic_p_value,
)
from apedp_dyn.epistemic.proof_claim import ProofClaimStatus
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.market.universe import (
    UniverseAudit,
    audit_universe,
    find_stale_runs,
    load_universe_selection,
    mask_stale_bars,
    point_in_time_membership,
)
from tests.promotion_evidence import (
    business_days,
    correlated_pair,
    evidence_backed_tracker,
    forecasting_evidence,
    lean_result,
    persistent_trial_ledger,
    strategy_returns,
)

REPO = Path(__file__).resolve().parent.parent
MARKET = REPO / "data" / "market"
HOLDOUT = DatasetRole.CONFIRMATION_HOLDOUT
SINGLE = UniverseAudit.single_instrument("SYNTHETIC")


# ---------------------------------------------------------------------------
# H-04: holdout identity by content
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(0, 1, (250, 3)), axis=0) + 100.0, business_days("2024-01-02", 250)


def _ledger(*candidates: str, **kwargs) -> FullLineageExperimentLedger:
    ledger = FullLineageExperimentLedger(**kwargs)
    for cid in candidates:
        ledger.register_candidate(cid, "proposer", f"f_{cid}")
    return ledger


def test_fingerprint_is_stable_and_serialisable(prices):
    px, dates = prices
    first = fingerprint_holdout_slice(px, index=dates)
    assert first == fingerprint_holdout_slice(px.copy(), index=dates)
    assert first.n_rows == 250 and first.n_cols == 3
    assert len(first.row_sketch) == 250 and len(first.column_sketch) == 3
    assert first.index_kind == "datetime_ns"
    assert HoldoutSliceFingerprint.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    # A different representation of the same dates is the same slice.
    as_text = fingerprint_holdout_slice(px, index=[str(d) for d in dates])
    assert as_text.content_hash == first.content_hash and as_text.row_sketch == first.row_sketch


@pytest.mark.parametrize(
    ("label", "variant"),
    [
        ("identical rows under a new dataset and slice name", lambda px, d: (("COPY", "fresh"), px, d)),
        ("second half presented as its own slice", lambda px, d: (("COPY", "h2"), px[125:], d[125:])),
        ("columns reordered", lambda px, d: (("COPY", "perm"), px[:, ::-1], d)),
        ("a single column of it", lambda px, d: (("COPY", "one"), px[:, 0], d)),
        ("rescaled, same dataset, same dates", lambda px, d: (("SPY", "scaled"), px * 3.7, d)),
    ],
)
def test_a_burned_slice_is_recognised_under_another_identity(prices, label, variant):
    px, dates = prices
    ledger = _ledger("failed", "next", max_holdout_inspections=None)
    ledger.record_data_exposure(
        "failed", "SPY", "holdout_2024", HOLDOUT, "FAILED",
        fingerprint=fingerprint_holdout_slice(px, index=dates),
    )
    (dataset_id, slice_id), data, index = variant(px, dates)
    fingerprint = fingerprint_holdout_slice(data, index=index)
    assert ledger.is_holdout_burned(dataset_id, slice_id, fingerprint), label
    with pytest.raises(HoldoutContaminationError, match="holds the same data as 'SPY:holdout_2024'"):
        ledger.record_data_exposure("next", dataset_id, slice_id, HOLDOUT, fingerprint=fingerprint)
    # Without the fingerprint the new name still looks unused: that is the gap
    # the fingerprint closes, and why a ledger can be told to require one.
    assert not ledger.is_holdout_burned(dataset_id, slice_id)


def test_genuinely_different_slices_are_not_confused(prices):
    px, dates = prices
    rng = np.random.default_rng(1)
    other = np.cumsum(rng.normal(0, 1, (250, 3)), axis=0) + 50.0
    ledger = _ledger("failed", "a", "b", max_holdout_inspections=None)
    ledger.record_data_exposure(
        "failed", "SPY", "holdout_2024", HOLDOUT, "FAILED",
        fingerprint=fingerprint_holdout_slice(px, index=dates),
    )
    # Another asset over the same dates, and the same asset over later dates.
    ledger.record_data_exposure(
        "a", "BTC", "holdout_2024", HOLDOUT, fingerprint=fingerprint_holdout_slice(other, index=dates)
    )
    ledger.record_data_exposure(
        "b", "SPY", "holdout_2025", HOLDOUT,
        fingerprint=fingerprint_holdout_slice(other, index=business_days("2025-01-02", 250)),
    )


def test_one_coincidental_column_is_not_an_overlap():
    """A shared calendar feature in two wide matrices is not the same holdout."""
    rng = np.random.default_rng(2)
    counter = np.arange(200.0)
    a = np.column_stack([counter, rng.normal(size=(200, 9))])
    b = np.column_stack([counter, rng.normal(size=(200, 9))])
    assert fingerprint_holdout_slice(a).overlap_reason(fingerprint_holdout_slice(b), same_dataset=False) is None


def test_the_inspection_budget_counts_through_a_renamed_copy(prices):
    px, dates = prices
    fingerprint = fingerprint_holdout_slice(px, index=dates)
    ledger = _ledger("winner", "next")
    ledger.record_data_exposure("winner", "SPY", "h", HOLDOUT, "PASSED", fingerprint=fingerprint)
    with pytest.raises(HoldoutContaminationError, match="other slice name"):
        ledger.record_data_exposure("next", "ZZZ", "another", HOLDOUT, fingerprint=fingerprint)


def test_a_slice_name_cannot_be_rebound_to_other_data(prices):
    px, dates = prices
    ledger = _ledger("a")
    ledger.record_data_exposure("a", "SPY", "h", HOLDOUT, fingerprint=fingerprint_holdout_slice(px, index=dates))
    with pytest.raises(HoldoutContaminationError, match="already bound to different content"):
        ledger.record_data_exposure(
            "a", "SPY", "h", HOLDOUT, fingerprint=fingerprint_holdout_slice(px + 1.0, index=dates)
        )


def test_content_identity_survives_a_reopened_ledger(tmp_path, prices):
    px, dates = prices
    path = tmp_path / "ledger.jsonl"
    fingerprint = fingerprint_holdout_slice(px, index=dates)
    first = _ledger("failed", persistence_path=path, max_holdout_inspections=None)
    first.record_data_exposure("failed", "SPY", "h", HOLDOUT, "FAILED", fingerprint=fingerprint)

    reopened = FullLineageExperimentLedger(persistence_path=path, max_holdout_inspections=None)
    reopened.register_candidate("next", "proposer", "g")
    assert reopened.get_slice_fingerprint("spy", " H ") == fingerprint
    with pytest.raises(HoldoutContaminationError, match="identical content"):
        reopened.record_data_exposure("next", "RENAMED", "brand_new", HOLDOUT, fingerprint=fingerprint)


def test_a_ledger_can_require_fingerprints():
    ledger = _ledger("a", require_holdout_fingerprint=True)
    with pytest.raises(ValueError, match="without a content fingerprint"):
        ledger.record_data_exposure("a", "SPY", "h", HOLDOUT)
    # Development data is not a holdout and needs none.
    ledger.record_data_exposure("a", "SPY", "train", DatasetRole.TRAINING_DEVELOPMENT)


def test_an_index_is_never_taken_implicitly(prices):
    """Two windows of a reset-index frame must not look like the same rows."""
    px, _dates = prices
    frame_a = pd.DataFrame(px[:100]).reset_index(drop=True)
    frame_b = pd.DataFrame(px[100:200]).reset_index(drop=True)
    assert fingerprint_holdout_slice(frame_a).index_range is None
    assert (
        fingerprint_holdout_slice(frame_a).overlap_reason(
            fingerprint_holdout_slice(frame_b), same_dataset=True
        )
        is None
    )
    with pytest.raises(ValueError, match="one value per row"):
        fingerprint_holdout_slice(px, index=np.arange(10))


# ---------------------------------------------------------------------------
# H-02: promotion stages compute their own metrics
# ---------------------------------------------------------------------------


def test_reported_numbers_are_refused_unless_the_tracker_allows_them():
    strict = PromotionTracker("cand")
    for call in (
        lambda: strict.verify_stage_1("P2_VERIFIED", True, True),
        lambda: strict.verify_stage_2(0.9, True),
        lambda: strict.verify_stage_3(0.0, 0.0),
        lambda: strict.verify_stage_4(oos_ic=0.9, oos_p_value=0.0),
        lambda: strict.verify_stage_5(9.0, 1.0, 99.0),
        lambda: strict.verify_stage_6(9.0, 0.0),
    ):
        with pytest.raises(PromotionViolationError, match="numbers supplied by the caller"):
            call()
    assert strict.stage_records == {}


def test_a_stage_passed_on_reported_numbers_says_so():
    tracker = PromotionTracker("cand", allow_reported_metrics=True)
    passed = tracker.verify_stage_1("P2_VERIFIED", True, True)
    assert passed.evidence_source == EVIDENCE_REPORTED
    assert passed.status_label == "MATHEMATICAL_CONSISTENCY_VERIFIED_ON_REPORTED_METRICS"
    assert tracker.is_evidence_backed() is False
    refused = tracker.verify_stage_2(0.0, True)
    assert refused.status_label == "PREDICTIVE_RECOVERY_REFUSED"  # a refusal needs no qualifier


def test_the_derived_path_passes_five_stages_and_is_evidence_backed(tmp_path):
    tracker = evidence_backed_tracker("cand", tmp_path, through=5)
    assert tracker.highest_stage_passed == PromotionStage.STAGE_5_ECONOMIC_VALIDITY
    assert set(tracker.evidence_sources.values()) == {EVIDENCE_DERIVED}
    assert tracker.is_evidence_backed()
    assert all(not rec.status_label.endswith("_ON_REPORTED_METRICS") for rec in tracker.stage_records.values())
    stage4 = tracker.stage_records[PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER]
    assert stage4.metrics["holdout_content_hash"]
    assert stage4.metrics["in_sample_ic"] == tracker.stage_records[
        PromotionStage.STAGE_2_PREDICTIVE_RECOVERY
    ].metrics["in_sample_ic"]


def test_one_reported_stage_makes_the_whole_promotion_not_evidence_backed(tmp_path):
    tracker = PromotionTracker("cand", allow_reported_metrics=True)
    tracker.verify_stage_1_from_lean(lean_result("cand"))
    tracker.verify_stage_2(in_sample_ic=0.2, cv_stable=True)
    assert tracker.is_evidence_backed(through=PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY)
    assert tracker.is_evidence_backed() is False


# -- stage 1 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"status": ProofClaimStatus.P1_CANDIDATE}, "not P2"),
        ({"axiom_dependent": True}, "rests on a declared axiom"),
        ({"kernel_exit_code": 1}, "kernel rejected it"),
        ({"kernel_exit_code": None}, "kernel never ran"),
        ({"proven": False}, "a theorem is unproven"),
        ({"source": "theorem t : 1 = 1 := by sorry"}, "uses sorry"),
        ({"kinds": (InvariantKind.BOUND_POSITIVITY,)}, "no dimensional theorem"),
        ({"kinds": (InvariantKind.SCALE_HOMOGENEITY,)}, "no bound theorem"),
        ({"kinds": ()}, "no theorems at all"),
    ],
)
def test_stage_1_refuses_a_lean_result_that_is_not_an_independent_proof(kwargs, why):
    tracker = PromotionTracker("cand")
    record = tracker.verify_stage_1_from_lean(lean_result("cand", **kwargs))
    assert record.passed is False, why
    assert record.evidence_source == EVIDENCE_DERIVED
    assert tracker.is_halted


def test_stage_1_refuses_a_lean_result_about_another_candidate():
    with pytest.raises(PromotionViolationError, match="not 'cand'"):
        PromotionTracker("cand").verify_stage_1_from_lean(lean_result("somebody_else"))


# -- stage 2 ---------------------------------------------------------------


def test_stage_2_computes_the_ic_and_refuses_noise():
    predictions, targets = correlated_pair(600, 0.0, seed=4)
    tracker = PromotionTracker("cand")
    tracker.verify_stage_1_from_lean(lean_result("cand"))
    record = tracker.verify_stage_2_from_predictions(predictions, targets)
    assert record.passed is False
    assert record.metrics["in_sample_ic"] == pytest.approx(information_coefficient(predictions, targets)[0])
    assert len(record.metrics["fold_ics"]) == 5


def test_stage_2_requires_stability_in_every_fold():
    """A strong average carried by one stretch of the sample is not recovery."""
    rng = np.random.default_rng(6)
    predictions = rng.normal(size=600)
    targets = rng.normal(size=600)
    targets[:150] = predictions[:150]  # perfect in the first quarter, noise after
    assert information_coefficient(predictions, targets)[0] > 0.2
    tracker = PromotionTracker("cand")
    tracker.verify_stage_1_from_lean(lean_result("cand"))
    record = tracker.verify_stage_2_from_predictions(predictions, targets)
    assert record.metrics["cv_stable"] is False
    assert record.passed is False


def test_stage_2_refuses_a_sample_too_small_to_fold():
    predictions, targets = correlated_pair(60, 0.9, seed=4)  # 12 per fold
    tracker = PromotionTracker("cand")
    tracker.verify_stage_1_from_lean(lean_result("cand"))
    assert tracker.verify_stage_2_from_predictions(predictions, targets).passed is False
    with pytest.raises(ValueError, match="at least 2"):
        PromotionTracker("x").verify_stage_2_from_predictions(predictions, targets, n_folds=1)


# -- stage 3 ---------------------------------------------------------------


def _through_stage_2(candidate_id: str = "cand") -> PromotionTracker:
    tracker = PromotionTracker(candidate_id)
    tracker.verify_stage_1_from_lean(lean_result(candidate_id))
    tracker.verify_stage_2_from_predictions(*correlated_pair(600, 0.30, seed=1))
    return tracker


def test_stage_3_must_beat_every_incumbent():
    evidence = forecasting_evidence()
    # Add an incumbent that is as good as the candidate itself.
    evidence["incumbent_forecasts"]["STRONG"] = evidence["candidate_forecast"] + 1e-9
    record = _through_stage_2().verify_stage_3_from_forecasts(**evidence)
    assert record.passed is False
    assert record.metrics["dm_hardest_incumbent"] == "STRONG"


def test_stage_3_refuses_a_relabelled_literature_factor():
    evidence = forecasting_evidence()
    evidence["candidate_signal"] = 2.0 * evidence["literature_features"]["KYLE_LAMBDA"] + 0.5
    record = _through_stage_2().verify_stage_3_from_forecasts(**evidence)
    assert record.passed is False
    assert record.metrics["max_literature_r2"] > 0.99


def test_stage_3_refuses_when_a_literature_feature_cannot_be_evaluated():
    """A constant baseline used to drop out silently and leave R-squared at zero."""
    evidence = forecasting_evidence()
    evidence["literature_features"]["AMIHUD_ILLIQUIDITY"] = np.ones(600)
    record = _through_stage_2().verify_stage_3_from_forecasts(**evidence)
    assert record.passed is False
    assert record.metrics["literature_features_evaluated"] == ["KYLE_LAMBDA"]
    with pytest.raises(ValueError, match="at least one incumbent"):
        _through_stage_2("other").verify_stage_3_from_forecasts(
            evidence["y_true"], evidence["candidate_forecast"], {}, evidence["candidate_signal"], {}
        )


# -- stage 4 ---------------------------------------------------------------


def _through_stage_3(candidate_id: str = "cand") -> PromotionTracker:
    tracker = _through_stage_2(candidate_id)
    tracker.verify_stage_3_from_forecasts(**forecasting_evidence())
    return tracker


def test_stage_4_needs_a_persistent_ledger_and_a_registered_candidate(tmp_path):
    predictions, targets = correlated_pair(400, 0.28, seed=2)
    tracker = _through_stage_3()
    with pytest.raises(PromotionViolationError, match="persistent experiment ledger"):
        tracker.verify_stage_4_from_holdout(
            predictions, targets, experiment_ledger=FullLineageExperimentLedger(),
            dataset_id="d", slice_identifier="h",
        )
    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "e.jsonl")
    with pytest.raises(PromotionPrerequisiteError, match="not registered"):
        tracker.verify_stage_4_from_holdout(
            predictions, targets, experiment_ledger=ledger, dataset_id="d", slice_identifier="h"
        )
    assert PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER not in tracker.stage_records


def test_stage_4_failure_burns_the_holdout_for_everyone(tmp_path):
    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "e.jsonl")
    for cid in ("cand", "rival"):
        ledger.register_candidate(cid, "proposer", "f")
    index = business_days("2025-01-02", 400)
    predictions, targets = correlated_pair(400, 0.0, seed=8)  # no transfer
    failed = _through_stage_3().verify_stage_4_from_holdout(
        predictions, targets, experiment_ledger=ledger, dataset_id="spy", slice_identifier="h_2025",
        index=index,
    )
    assert failed.passed is False
    assert ledger.is_holdout_burned("SPY", "h_2025")
    # The rival cannot use those observations, whatever it calls them.
    good_predictions = targets + 0.1
    with pytest.raises(HoldoutContaminationError):
        _through_stage_3("rival").verify_stage_4_from_holdout(
            good_predictions, targets, experiment_ledger=ledger, dataset_id="fresh_name",
            slice_identifier="never_seen", index=index,
        )


def test_stage_4_holdout_is_spent_even_when_the_first_candidate_passed(tmp_path):
    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "e.jsonl")
    for cid in ("cand", "rival"):
        ledger.register_candidate(cid, "proposer", "f")
    predictions, targets = correlated_pair(400, 0.28, seed=2)
    kwargs = {"experiment_ledger": ledger, "dataset_id": "spy", "slice_identifier": "h"}
    assert _through_stage_3().verify_stage_4_from_holdout(predictions, targets, **kwargs).passed
    with pytest.raises(HoldoutContaminationError, match="inspection budget"):
        _through_stage_3("rival").verify_stage_4_from_holdout(predictions, targets, **kwargs)


def test_overlapping_targets_do_not_count_as_independent_observations():
    assert one_sided_ic_p_value(0.10, 400) < 0.05 < one_sided_ic_p_value(0.10, 400 // 21)
    assert one_sided_ic_p_value(float("nan"), 400) == 1.0
    assert one_sided_ic_p_value(-0.3, 400) > 0.99


# -- stage 5 ---------------------------------------------------------------


def test_stage_5_refuses_in_memory_and_empty_trial_ledgers(tmp_path):
    gross, turnover, index = strategy_returns()
    tracker = evidence_backed_tracker("cand", tmp_path, through=4)
    with pytest.raises(PromotionViolationError, match="persistent trial ledger"):
        tracker.verify_stage_5_from_returns(
            gross, turnover, index, trial_ledger=CandidateTrialLedger(), universe=SINGLE
        )
    with pytest.raises(PromotionViolationError, match="holds no trials"):
        tracker.verify_stage_5_from_returns(
            gross, turnover, index,
            trial_ledger=CandidateTrialLedger(persistence_path=tmp_path / "empty.jsonl"), universe=SINGLE,
        )


def test_stage_5_charges_fees_and_will_not_lower_them(tmp_path):
    gross, turnover, index = strategy_returns()
    ledger = persistent_trial_ledger(tmp_path / "t.jsonl")
    tracker = evidence_backed_tracker("cand", tmp_path, through=4)
    with pytest.raises(PromotionViolationError, match="looser than the preregistered floor"):
        tracker.verify_stage_5_from_returns(
            gross, turnover, index, trial_ledger=ledger, universe=SINGLE, fee_bps=0.0
        )
    record = tracker.verify_stage_5_from_returns(gross, turnover, index, trial_ledger=ledger, universe=SINGLE)
    net = gross - turnover * 3.0 / 1e4
    periods = record.metrics["periods_per_year"]
    assert 250 < periods < 265  # read off the business-day timestamps, not supplied
    assert record.metrics["net_sharpe"] == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(periods))
    assert record.metrics["breakeven_fee_bps"] == pytest.approx(gross.sum() / turnover.sum() * 1e4)


def test_stage_5_deflates_by_the_journaled_search(tmp_path):
    """The best of a long noise search fails; the identical returns pass a short one."""
    rng = np.random.default_rng(12)
    index = business_days("2021-01-04", 1000)
    best, best_sharpe = None, -np.inf
    long_search = CandidateTrialLedger(persistence_path=tmp_path / "long.jsonl")
    for i in range(300):
        returns = rng.normal(0.0, 0.01, 1000)
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        long_search.record_trial(f"t{i}", "noise", "f", "TACTICAL", sharpe, returns=returns)
        if sharpe > best_sharpe:
            best, best_sharpe = returns, sharpe
    turnover = np.zeros(1000)
    turnover[0] = 1.0
    assert best is not None
    deflated = evidence_backed_tracker("cand", tmp_path / "a", through=4).verify_stage_5_from_returns(
        best, turnover, index, trial_ledger=long_search, universe=SINGLE
    )
    assert deflated.passed is False
    assert deflated.metrics["dsr_n_trials"] == 300
    assert deflated.metrics["dsr_metric"] < 0.95


@pytest.mark.parametrize("panel_file", ["multi_asset_90_5yr_orderflow.csv", "crypto_top15_orderflow.csv"])
def test_stage_5_refuses_returns_earned_on_a_hindsight_universe(tmp_path, panel_file):
    panel = pd.read_csv(MARKET / panel_file)
    selection = load_universe_selection(MARKET / "PROVENANCE.json", panel_file)
    assert selection is not None
    universe = audit_universe(
        panel, selected_as_of=selection["selected_as_of"], date_column=panel.columns[0]
    )
    gross, turnover, index = strategy_returns()
    record = evidence_backed_tracker("cand", tmp_path, through=4).verify_stage_5_from_returns(
        gross, turnover, index, trial_ledger=persistent_trial_ledger(tmp_path / "t.jsonl"), universe=universe
    )
    # The same returns pass on a single instrument (see the derived-path test).
    assert record.metrics["net_sharpe"] > 1.0 and record.metrics["dsr_metric"] >= 0.95
    assert record.passed is False
    assert "universe does not support an economic claim" in record.reason
    assert record.metrics["universe_fit_for_economic_claims"] is False


# -- stage 6 ---------------------------------------------------------------


def _freeze_clock(monkeypatch, moment: datetime.datetime) -> None:
    monkeypatch.setattr(pl, "_utc_now", lambda: moment)


def test_stage_6_only_counts_returns_realised_after_stage_5(tmp_path, monkeypatch):
    frozen = datetime.datetime(2026, 1, 5, 12, 0, tzinfo=datetime.timezone.utc)
    _freeze_clock(monkeypatch, frozen)
    tracker = evidence_backed_tracker("cand", tmp_path, through=5)
    assert tracker.stage_records[PromotionStage.STAGE_5_ECONOMIC_VALIDITY].recorded_at == frozen.isoformat()

    rng = np.random.default_rng(9)
    returns = rng.normal(0.002, 0.008, 120)
    _freeze_clock(monkeypatch, frozen + datetime.timedelta(days=300))
    # A backtest dressed up as live performance: dated before the freeze.
    with pytest.raises(PromotionViolationError, match="not prospective"):
        tracker.verify_stage_6_from_returns(returns, business_days("2025-06-02", 120))
    # Performance that has not happened yet.
    with pytest.raises(PromotionViolationError, match="lie in the future"):
        tracker.verify_stage_6_from_returns(returns, business_days("2026-09-01", 120))
    assert PromotionStage.STAGE_6_REAL_MARKET_PROMOTION not in tracker.stage_records

    index = business_days("2026-01-06", 120)
    record = tracker.verify_stage_6_from_returns(returns, index)
    equity = np.cumprod(1.0 + returns)
    drawdown = float(np.max(1.0 - equity / np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]))
    assert record.passed is True
    assert record.status_label == "REAL_MARKET_PROMOTION_GRANTED"
    assert record.metrics["max_drawdown"] == pytest.approx(drawdown)
    assert tracker.is_evidence_backed()


def test_stage_6_refuses_a_losing_or_too_short_record(tmp_path, monkeypatch):
    frozen = datetime.datetime(2026, 1, 5, tzinfo=datetime.timezone.utc)
    _freeze_clock(monkeypatch, frozen)
    tracker = evidence_backed_tracker("cand", tmp_path, through=5)
    _freeze_clock(monkeypatch, frozen + datetime.timedelta(days=300))
    short = tracker.verify_stage_6_from_returns(np.full(10, 0.01), business_days("2026-01-06", 10))
    assert short.passed is False  # ten observations do not make a Sharpe ratio
    with pytest.raises(ValueError, match="datetime index"):
        evidence_backed_tracker("other", tmp_path / "o", through=5).verify_stage_6_from_returns(
            np.full(30, 0.01), np.arange(30)
        )


def test_evidence_source_and_timestamp_survive_the_journal(tmp_path):
    journal = tmp_path / "promotion.jsonl"
    evidence_backed_tracker("cand", tmp_path, through=3, state_path=journal)
    resumed = PromotionTracker("cand", state_path=journal)
    assert resumed.highest_stage_passed == PromotionStage.STAGE_3_UNIQUE_IDENTIFICATION
    assert resumed.is_evidence_backed()
    assert all(rec.recorded_at for rec in resumed.stage_records.values())


# ---------------------------------------------------------------------------
# H-06 / H-07: the trading sandbox journals its search and registers its holdout
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox():
    from apedp_dyn.sandbox import trading_signals_sandbox as module

    return module


@pytest.fixture(scope="module")
def passing_run(sandbox):
    """A (panel, seed) whose GA winner passes the holdout, found once per module."""
    panel = sandbox.load_real_btc_orderflow(600)
    for seed in range(40, 70):
        result = sandbox.run_epistemic_trading_adjudication(
            panel=panel, dataset_name="btc_orderflow", pop_size=20, generations=4, seed=seed
        )
        if result["holdout_registration"]["outcome"] == "PASSED":
            return panel, seed, result
    pytest.skip("no passing seed in range for this data")


def _persistent(tmp_path):
    return {
        "trial_ledger": CandidateTrialLedger(persistence_path=tmp_path / "trials.jsonl"),
        "experiment_ledger": FullLineageExperimentLedger(persistence_path=tmp_path / "experiments.jsonl"),
    }


def _run(sandbox, panel, seed, name="btc_orderflow", **ledgers):
    return sandbox.run_epistemic_trading_adjudication(
        panel=panel, dataset_name=name, pop_size=20, generations=4, seed=seed, **ledgers
    )


def test_default_run_discloses_that_nothing_persists(passing_run):
    _panel, _seed, result = passing_run
    assert result["trial_multiplicity"]["ledger_persistent"] is False
    assert result["holdout_registration"]["ledger_persistent"] is False
    assert result["holdout_registration"]["content_hash"]
    assert result["promotion_lifecycle"]["stage_1_evidence_source"] == EVIDENCE_DERIVED
    assert result["promotion_lifecycle"]["stage_1_status"] == "MATHEMATICAL_CONSISTENCY_REFUSED"


def test_persistent_ledgers_do_not_change_the_verdict(sandbox, passing_run, tmp_path):
    panel, seed, baseline = passing_run
    persisted = _run(sandbox, panel, seed, **_persistent(tmp_path))
    for key in ("verdict", "justified_tier", "metrics", "best_model", "market_extrapolation_ladder"):
        assert persisted[key] == baseline[key], key
    assert persisted["trial_multiplicity"]["ledger_persistent"] is True
    assert persisted["holdout_registration"]["ledger_persistent"] is True


def test_another_seed_cannot_score_the_same_holdout(sandbox, passing_run, tmp_path):
    """Re-running with new seeds until the holdout looks good is refused."""
    panel, seed, _baseline = passing_run
    first = _run(sandbox, panel, seed, **_persistent(tmp_path))
    with pytest.raises(HoldoutContaminationError, match="inspection budget"):
        _run(sandbox, panel, seed + 1, **_persistent(tmp_path))
    # The refused seed's search still happened, and is charged.
    trials = CandidateTrialLedger(persistence_path=tmp_path / "trials.jsonl")
    assert trials.trial_count == 2 * first["trial_multiplicity"]["recorded_trials"]
    assert len({rec.metadata["search_fingerprint"] for rec in trials.get_records()}) == 2


def test_renaming_the_dataset_does_not_free_the_holdout(sandbox, passing_run, tmp_path):
    panel, seed, _baseline = passing_run
    _run(sandbox, panel, seed, **_persistent(tmp_path))
    with pytest.raises(HoldoutContaminationError, match="hold the same data"):
        _run(sandbox, panel, seed + 2, name="a_completely_new_name", **_persistent(tmp_path))


def test_replaying_the_same_search_is_not_charged_twice(sandbox, passing_run, tmp_path):
    panel, seed, _baseline = passing_run
    first = _run(sandbox, panel, seed, **_persistent(tmp_path))
    replay = _run(sandbox, panel, seed, **_persistent(tmp_path))
    assert replay["trial_multiplicity"]["search_replayed_from_ledger"] is True
    assert replay["trial_multiplicity"]["recorded_trials"] == first["trial_multiplicity"]["recorded_trials"]
    assert (
        replay["trial_multiplicity"]["dsr_selection_adjusted"]
        == first["trial_multiplicity"]["dsr_selection_adjusted"]
    )
    assert replay["holdout_registration"]["ledger_candidate_id"] == first["holdout_registration"]["ledger_candidate_id"]


def test_a_failed_candidate_burns_the_sandbox_holdout(sandbox, tmp_path):
    panel = sandbox.make_synthetic_dataset(600, seed=42)
    failed = _run(sandbox, panel, 42, name="synthetic", **_persistent(tmp_path))
    if failed["holdout_registration"]["outcome"] != "FAILED":
        pytest.skip("this configuration no longer fails its holdout")
    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "experiments.jsonl")
    assert ledger.is_holdout_burned("synthetic", failed["holdout_registration"]["slice_identifier"])
    with pytest.raises(HoldoutContaminationError, match="previously burned"):
        _run(sandbox, panel, 43, name="synthetic", **_persistent(tmp_path))


def test_env_flag_opens_the_deployment_ledgers(sandbox, passing_run, tmp_path, monkeypatch):
    panel, seed, _baseline = passing_run
    monkeypatch.setenv("APEDP_STORAGE_DIR", str(tmp_path))
    monkeypatch.delenv("APEDP_LEDGER_PATH", raising=False)
    monkeypatch.delenv("APEDP_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv(sandbox.PERSISTENT_LEDGERS_ENV, "1")
    result = _run(sandbox, panel, seed)
    assert result["trial_multiplicity"]["ledger_persistent"] is True
    assert result["holdout_registration"]["ledger_persistent"] is True
    assert (tmp_path / "trial_ledger_trading_btc_orderflow.jsonl").is_file()
    assert (tmp_path / "experiment_ledger.jsonl").is_file()
    with pytest.raises(HoldoutContaminationError):
        _run(sandbox, panel, seed + 1)


# ---------------------------------------------------------------------------
# H-12: universe selection and forward-filled delistings
# ---------------------------------------------------------------------------


def _toy_panel() -> pd.DataFrame:
    """Four assets, 200 bars; DEAD stops trading at bar 120 and is forward-filled."""
    rng = np.random.default_rng(3)
    dates = [str(d) for d in business_days("2022-01-03", 200)]
    frame = pd.DataFrame({"observation_date": dates})
    for i, asset in enumerate(("BIG", "MID", "SMALL", "DEAD")):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        volume = rng.uniform(0.9, 1.1, 200) * (1000.0 / (i + 1))
        if asset == "DEAD":
            volume *= 50.0  # the largest name while it lived
            close[120:] = close[119]
            volume[120:] = volume[119]
        frame[f"{asset}_CLOSE"] = close
        frame[f"{asset}_VOL"] = volume
    return frame


def test_forward_filled_tail_is_found_and_masked():
    panel = _toy_panel()
    runs = find_stale_runs(panel)
    assert [(run.asset, run.bars, run.reaches_end) for run in runs] == [("DEAD", 80, True)]
    assert runs[0].last_real_bar == panel["observation_date"].iloc[119]
    masked = mask_stale_bars(panel)
    assert masked["DEAD_CLOSE"].iloc[120:].isna().all() and masked["DEAD_VOL"].iloc[120:].isna().all()
    assert masked["DEAD_CLOSE"].iloc[:120].notna().all()
    assert masked[["BIG_CLOSE", "MID_CLOSE", "SMALL_CLOSE"]].notna().all().all()
    assert find_stale_runs(masked) == []


def test_a_pegged_price_with_live_volume_is_not_stale():
    panel = _toy_panel()
    panel["BIG_CLOSE"] = 1.0  # a stable quote: price repeats, volume does not
    assert [run.asset for run in find_stale_runs(panel)] == ["DEAD"]


def test_audit_separates_hindsight_from_a_list_fixed_in_advance():
    panel = mask_stale_bars(_toy_panel())
    hindsight = audit_universe(panel, selected_as_of="2026-10-08")
    assert hindsight.hindsight_selected is True and not hindsight.fit_for_economic_claims
    assert hindsight.assets_that_stop_trading == ("DEAD",)
    in_advance = audit_universe(panel, selected_as_of="2021-12-31")
    assert in_advance.hindsight_selected is False and in_advance.fit_for_economic_claims
    unknown = audit_universe(panel, selected_as_of=None)
    assert unknown.hindsight_selected is None and not unknown.fit_for_economic_claims
    # Forward-filled bars disqualify a universe however it was selected.
    assert not audit_universe(_toy_panel(), selected_as_of="2021-12-31").fit_for_economic_claims
    assert UniverseAudit.single_instrument("BTC").fit_for_economic_claims


def test_point_in_time_membership_uses_only_the_past():
    panel = _toy_panel()
    membership = point_in_time_membership(panel, top_n=2, lookback_bars=20, rebalance_every=10)
    assert not membership.iloc[:20].any().any()  # no universe before a full lookback
    assert membership.iloc[20:120]["DEAD"].all()  # the biggest name while it traded
    assert not membership.iloc[120:]["DEAD"].any()  # dropped on the bar it stops
    assert (membership.iloc[20:].sum(axis=1) <= 2).all()
    # Changing the future does not change the past's membership.
    altered = panel.copy()
    altered.loc[150:, "SMALL_VOL"] *= 1e6
    again = point_in_time_membership(altered, top_n=2, lookback_bars=20, rebalance_every=10)
    assert again.iloc[:150].equals(membership.iloc[:150])
    assert again.iloc[-1]["SMALL"] and not membership.iloc[-1]["SMALL"]


def test_shipped_90_asset_panel_carries_five_forward_filled_delistings():
    """Measured on the file in the repository; update this when the panel is rebuilt."""
    panel = pd.read_csv(MARKET / "multi_asset_90_5yr_orderflow.csv")
    found = {run.asset: (run.last_real_bar, run.bars) for run in find_stale_runs(panel)}
    assert found == {
        "MATIC": ("2024-09-10", 328),
        "MKR": ("2025-09-15", 75),
        "EOS": ("2025-05-23", 152),
        "XMR": ("2024-02-20", 468),
        "FTM": ("2025-01-13", 243),
    }
    recorded = json.loads((MARKET / "PROVENANCE.json").read_text())["multi_asset_90_5yr_orderflow.csv"]
    defect = recorded["known_defects"]["forward_filled_delistings"]
    assert {a["asset"]: (a["last_real_bar"], a["bars"]) for a in defect["assets"]} == found
    assert defect["fabricated_asset_days"] == sum(bars for _bar, bars in found.values()) == 1266


@pytest.mark.parametrize(
    "panel_file",
    [
        "crypto_top15.csv",
        "crypto_top15_orderflow.csv",
        "equity_top15.csv",
        "multi_asset_30_orderflow.csv",
        "multi_asset_60_orderflow.csv",
        "multi_asset_90_5yr_orderflow.csv",
    ],
)
def test_every_multi_asset_panel_is_recorded_as_hindsight_selected(panel_file):
    selection = load_universe_selection(MARKET / "PROVENANCE.json", panel_file)
    assert selection is not None, panel_file
    assert selection["hindsight_selected"] is True
    assert selection["fit_for_economic_claims"] is False
    first_bar = str(pd.read_csv(MARKET / panel_file, usecols=[0], nrows=1).iloc[0, 0])
    assert selection["selected_as_of"] > first_bar[:10]


def test_universe_builder_tool_writes_a_masked_panel_and_membership(tmp_path):
    import subprocess
    import sys

    panel_path = tmp_path / "panel.csv"
    _toy_panel().to_csv(panel_path, index=False)
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable, str(REPO / "tools" / "build_point_in_time_universe.py"),
            "--panel", str(panel_path), "--out", str(out), "--pool-fixed-as-of", "2021-12-31",
            "--top-n", "2", "--lookback-bars", "20", "--rebalance-every", "10",
        ],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((out / "universe_audit.json").read_text())
    assert [run["asset"] for run in report["forward_filled_runs_removed"]] == ["DEAD"]
    assert report["candidate_pool_audit"]["fit_for_economic_claims"] is True
    assert pd.read_csv(out / "panel_masked.csv")["DEAD_CLOSE"].iloc[120:].isna().all()
    membership = pd.read_csv(out / "membership.csv", index_col=0)
    assert membership.shape == (200, 4) and int(membership.iloc[-1].sum()) == 2


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def _tracked(*paths: str) -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        pytest.skip("not a git checkout")
    return proc.stdout.split()


def test_removed_residue_stays_out_of_the_index():
    """A link into one developer's virtualenv, an empty file, and a shadowed module."""
    assert _tracked("apedp_kernel", "Crash", "apedp/discovery.py") == []


def test_discovery_package_is_what_the_removed_shim_pointed_at():
    import apedp.discovery as discovery

    assert not (REPO / "apedp" / "discovery.py").exists()
    assert (REPO / "apedp" / "discovery" / "__init__.py").is_file()
    assert hasattr(discovery, "MultiVariableSRE")
