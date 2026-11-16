"""The referee runs by itself: thresholds, state machine, ledgers, universe audit.

Every import in this file is the referee core, numpy or pandas
(``tests/referee_core_manifest.py``). CI's ``referee-core`` job runs it in an
environment with numpy, scipy, pandas and pytest and nothing else -- no
torch, no SymPy, no native kernel, and the repository itself not installed --
so a pass there is the constructive proof that a promotion decision does not
depend on anything the referee is supposed to be judging.

These tests are not a second copy of the integrity suites
(``tests/test_integrity_*.py``). They walk the main paths once each, enough to
show the whole decision surface is reachable without the rest of the tree.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.conf import canonical_spec
from apedp_dyn.epistemic import promotion_lifecycle as pl
from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
    LedgerIntegrityError,
    fingerprint_holdout_slice,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
    PromotionViolationError,
)
from apedp_dyn.epistemic.proof_claim import ProofClaimStatus
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.market.universe import UniverseAudit, audit_universe
from tests.promotion_evidence import (
    business_days,
    correlated_pair,
    evidence_backed_tracker,
    lean_result,
    persistent_trial_ledger,
    strategy_returns,
)

UTC = datetime.timezone.utc


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, moment: datetime.datetime) -> None:
    monkeypatch.setattr(pl, "_utc_now", lambda: moment)


def test_the_spec_on_disk_is_the_pinned_spec() -> None:
    """The thresholds every stage reads are the preregistered ones."""
    spec = canonical_spec.load_and_verify_canonical_spec(enforce_hash=True)  # raises on drift
    assert isinstance(spec, dict) and spec
    assert len(canonical_spec.PINNED_SPEC_SHA256) == 64
    assert 0.0 < canonical_spec.DIEBOLD_MARIANO_P_VALUE_THRESHOLD < 1.0


def test_a_candidate_reaches_stage_six_on_derived_evidence_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = datetime.datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    _freeze_clock(monkeypatch, frozen)
    tracker = evidence_backed_tracker("cand", tmp_path, through=5)
    assert int(tracker.highest_stage_passed) == 5

    _freeze_clock(monkeypatch, frozen + datetime.timedelta(days=300))
    live = np.random.default_rng(9).normal(0.002, 0.008, 120)
    record = tracker.verify_stage_6_from_returns(live, business_days("2026-01-06", 120))

    assert record.passed is True
    assert tracker.highest_stage_passed is PromotionStage.STAGE_6_REAL_MARKET_PROMOTION
    assert tracker.is_evidence_backed()
    assert not tracker.is_halted
    tracker.assert_consistent()


def test_reported_numbers_are_refused_and_stages_cannot_be_skipped(tmp_path) -> None:
    tracker = PromotionTracker("cand")
    with pytest.raises(PromotionViolationError):
        tracker.verify_stage_2(in_sample_ic=0.9, cv_stable=True)  # a number, not evidence
    with pytest.raises(PromotionPrerequisiteError):
        tracker.verify_stage_2_from_predictions(*correlated_pair(600, 0.30, seed=1))
    assert int(tracker.highest_stage_passed) == 0


def test_stage_one_judges_the_kernel_not_the_claim() -> None:
    """A result that says P2 but did not come out of the kernel clean is not P2."""
    for defect in (
        {"kernel_exit_code": 1},
        {"kernel_exit_code": None},
        {"axiom_dependent": True},
        {"source": "theorem t : 1 = 1 := by sorry"},
        {"status": ProofClaimStatus.P1_CANDIDATE},
    ):
        tracker = PromotionTracker("cand")
        record = tracker.verify_stage_1_from_lean(lean_result("cand", **defect))
        assert record.passed is False, defect
        assert tracker.is_halted, defect
    with pytest.raises(PromotionViolationError):
        PromotionTracker("cand").verify_stage_1_from_lean(lean_result("someone_else"))


def test_noise_fails_stage_two_and_the_failure_is_final() -> None:
    tracker = PromotionTracker("cand")
    tracker.verify_stage_1_from_lean(lean_result("cand"))
    rng = np.random.default_rng(0)
    record = tracker.verify_stage_2_from_predictions(rng.normal(size=600), rng.normal(size=600))
    assert record.passed is False
    assert tracker.is_halted
    with pytest.raises(PromotionPrerequisiteError):
        tracker.verify_stage_2_from_predictions(*correlated_pair(600, 0.30, seed=1))


def test_the_holdout_is_claimed_by_content_once_and_burned_on_failure(tmp_path) -> None:
    ledger = FullLineageExperimentLedger(persistence_path=tmp_path / "experiments.jsonl")
    for name in ("first", "second", "third"):
        ledger.register_candidate(name, "proposer", f"f_{name}")
    rows = np.random.default_rng(4).normal(size=(250, 3))
    index = business_days("2025-01-02", 250)
    fingerprint = fingerprint_holdout_slice(rows, index=index)

    ledger.record_data_exposure(
        "first", "SPY", "2025", DatasetRole.CONFIRMATION_HOLDOUT, fingerprint=fingerprint
    )
    # The same rows under another name are the same holdout.
    with pytest.raises(HoldoutContaminationError):
        ledger.record_data_exposure(
            "second",
            "SPY_RENAMED",
            "fresh",
            DatasetRole.CONFIRMATION_HOLDOUT,
            fingerprint=fingerprint_holdout_slice(rows.copy(), index=index),
        )
    # A failure on the slice burns it for everyone, the failing candidate included.
    other = fingerprint_holdout_slice(np.random.default_rng(5).normal(size=(250, 3)), index=index)
    ledger.record_data_exposure(
        "third",
        "QQQ",
        "2025",
        DatasetRole.CONFIRMATION_HOLDOUT,
        fingerprint=other,
        outcome_status="FAILED",
    )
    assert ledger.is_holdout_burned("QQQ", "2025")
    with pytest.raises(HoldoutContaminationError):
        ledger.record_data_exposure("third", "QQQ", "2025", DatasetRole.CONFIRMATION_HOLDOUT)

    # A second process sees the same state: it is in the journal, not in memory.
    reopened = FullLineageExperimentLedger(persistence_path=tmp_path / "experiments.jsonl")
    assert reopened.is_holdout_burned("qqq", " 2025 ")  # and names are normalised
    with pytest.raises(HoldoutContaminationError):
        reopened.record_data_exposure("second", "SPY", "2025", DatasetRole.CONFIRMATION_HOLDOUT)


def test_an_edited_ledger_is_refused_on_load(tmp_path) -> None:
    path = tmp_path / "experiments.jsonl"
    ledger = FullLineageExperimentLedger(persistence_path=path)
    ledger.register_candidate("cand", "proposer", "f(x)")
    ledger.record_data_exposure("cand", "SPY", "2025", DatasetRole.CONFIRMATION_HOLDOUT)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    entry = json.loads(lines[0])
    entry["formula_repr" if "formula_repr" in entry else next(iter(entry))] = "tampered"
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        FullLineageExperimentLedger(persistence_path=path)


def test_the_trial_count_is_journaled_and_deflates_the_sharpe(tmp_path) -> None:
    path = tmp_path / "trials.jsonl"
    ledger = persistent_trial_ledger(path, n_trials=40)
    assert ledger.is_persistent
    reopened = CandidateTrialLedger(persistence_path=path)
    assert reopened.effective_trial_count == ledger.effective_trial_count >= 40
    few = persistent_trial_ledger(tmp_path / "few.jsonl", n_trials=2)
    gross, _turnover, _index = strategy_returns()
    sharpe = float(np.mean(gross) / np.std(gross, ddof=1) * np.sqrt(252))
    many_dsr = reopened.dsr_with_selection_adjustment(sharpe, n_bars=len(gross))
    few_dsr = few.dsr_with_selection_adjustment(sharpe, n_bars=len(gross))
    assert many_dsr["ledger_persistent"] is True
    assert many_dsr["dsr"] <= few_dsr["dsr"]  # more searching, less credit


def test_stage_five_refuses_a_universe_chosen_with_hindsight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_clock(monkeypatch, datetime.datetime(2026, 1, 5, tzinfo=UTC))
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({"observation_date": pd.date_range("2024-01-02", periods=60, freq="B")})
    for asset in ("AAA", "BBB"):
        # A price path, not independent draws: commit 101b9c3 made the audit
        # flag single-bar moves of 4x as likely unadjusted splits, and
        # independent lognormal levels move that much every few bars. The
        # hindsight check is what this test is about, so the prices should give
        # the audit nothing else to object to.
        panel[f"{asset}_CLOSE"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=60)))
        panel[f"{asset}_VOL"] = rng.lognormal(size=60)

    in_advance = audit_universe(panel, selected_as_of="2023-12-29")
    hindsight = audit_universe(panel, selected_as_of="2025-12-31")
    assert isinstance(hindsight, UniverseAudit)
    assert in_advance.fit_for_economic_claims
    assert hindsight.hindsight_selected is True and not hindsight.fit_for_economic_claims

    gross, turnover, index = strategy_returns()
    tracker = evidence_backed_tracker("cand", tmp_path, through=4)
    record = tracker.verify_stage_5_from_returns(
        gross,
        turnover,
        index,
        trial_ledger=persistent_trial_ledger(tmp_path / "trials.jsonl"),
        universe=hindsight,
    )
    # The same returns pass on a single instrument (the walk above); what is
    # refused here is the universe they were earned on.
    assert record.passed is False
    assert record.metrics["universe_fit_for_economic_claims"] is False
    assert int(tracker.highest_stage_passed) == 4
