"""apedp-referee runs by itself: thresholds, state machine, ledgers, universe audit.

Every import in this file is ``apedp_referee``, numpy, pandas or the evidence
helpers beside it. CI's ``referee-core`` job installs the package from its own
directory into an environment that has numpy, scipy, pandas and pytest and
nothing else, then runs these tests from outside the checkout: no torch, no
SymPy, no native kernel, no ``apedp_dyn``. A pass there is the constructive
proof that a promotion decision does not depend on anything the referee is
supposed to be judging. ``test_this_is_the_package_under_test`` makes sure the
copy being tested is the one that was installed when that is what the job
says it is testing.

These tests are not a second copy of the repository's integrity suites
(``tests/test_integrity_*.py``). They walk the main paths once each, enough to
show the whole decision surface is reachable from the package alone.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from referee_evidence import (
    business_days,
    correlated_pair,
    evidence_backed_tracker,
    lean_result,
    persistent_trial_ledger,
    strategy_returns,
)

from apedp_referee import canonical_spec
from apedp_referee import promotion_lifecycle as pl
from apedp_referee.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
    LedgerIntegrityError,
    fingerprint_holdout_slice,
)
from apedp_referee.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
    PromotionViolationError,
)
from apedp_referee.proof_claim import ProofClaimStatus
from apedp_referee.trial_ledger import CandidateTrialLedger
from apedp_referee.universe import UniverseAudit, audit_universe

UTC = datetime.timezone.utc


def test_this_is_the_package_under_test() -> None:
    """When CI says it is testing an installed copy, it is, and nothing else came along.

    ``APEDP_REFEREE_EXPECT_INSTALLED=1`` is set by the job that installs the
    package into a clean environment. There the module must come from
    site-packages, not from a checkout on ``sys.path``, and the rest of the
    repository must not be importable at all.
    """
    import importlib.util
    import os
    import sysconfig
    from pathlib import Path

    import apedp_referee

    if os.environ.get("APEDP_REFEREE_EXPECT_INSTALLED") != "1":
        pytest.skip("only meaningful where the job installed the package alone")
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    assert Path(apedp_referee.__file__).resolve().is_relative_to(site), apedp_referee.__file__
    for name in ("apedp_dyn", "apedp", "apedp_kernel", "torch", "sympy"):
        assert importlib.util.find_spec(name) is None, f"{name} is importable here"


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, moment: datetime.datetime) -> None:
    monkeypatch.setattr(pl, "_utc_now", lambda: moment)


def test_the_spec_on_disk_is_the_pinned_spec() -> None:
    """The thresholds every stage reads are the preregistered ones."""
    spec = canonical_spec.load_and_verify_canonical_spec(enforce_hash=True)  # raises on drift
    assert isinstance(spec, dict) and spec
    assert len(canonical_spec.PINNED_SPEC_SHA256) == 64
    assert 0.0 < canonical_spec.DIEBOLD_MARIANO_P_VALUE_THRESHOLD < 1.0


def test_a_candidate_reaches_stage_six_on_derived_evidence_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_reported_numbers_are_refused_and_stages_cannot_be_skipped(tmp_path: Path) -> None:
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


def test_the_holdout_is_claimed_by_content_once_and_burned_on_failure(tmp_path: Path) -> None:
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


def test_an_edited_ledger_is_refused_on_load(tmp_path: Path) -> None:
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


def test_the_trial_count_is_journaled_and_deflates_the_sharpe(tmp_path: Path) -> None:
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


# ---------------------------------------------------------------------------
# Three faults an outside audit found on 27 November 2026, fixed on the owner's
# word of the 28th: a strategy that never trades could not pass Stage 5, a
# trial's Sharpe ratio carried no units, and Stage 1 read a Lean source by
# searching it for one word.
# ---------------------------------------------------------------------------


def _fit_universe() -> UniverseAudit:
    """A universe chosen before its evidence starts: fit for an economic claim."""
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({"observation_date": pd.date_range("2024-01-02", periods=60, freq="B")})
    for asset in ("AAA", "BBB"):
        panel[f"{asset}_CLOSE"] = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=60)))
        panel[f"{asset}_VOL"] = rng.lognormal(size=60)
    got = audit_universe(panel, selected_as_of="2023-12-29")
    assert got.fit_for_economic_claims, got.reasons
    return got


def test_a_strategy_that_never_trades_is_not_refused_for_its_infinite_break_even_fee(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_clock(monkeypatch, datetime.datetime(2026, 1, 5, tzinfo=UTC))
    gross, turnover, index = strategy_returns()
    still = np.zeros_like(turnover)
    record = evidence_backed_tracker("cand", tmp_path, through=4).verify_stage_5_from_returns(
        gross, still, index, trial_ledger=persistent_trial_ledger(tmp_path / "t.jsonl"), universe=_fit_universe())
    assert record.metrics["total_turnover"] == 0.0 and record.metrics["fee_test_moot_no_turnover"] is True
    assert record.metrics["breakeven_fee_bps"] == float("inf")
    # with nothing traded the net return is the gross one, and the other two tests decide
    traded = evidence_backed_tracker("cand2", tmp_path / "b", through=4).verify_stage_5_from_returns(
        gross, turnover, index, trial_ledger=persistent_trial_ledger(tmp_path / "t2.jsonl"), universe=_fit_universe())
    assert traded.passed, traded.reason
    assert record.passed, record.reason
    assert record.metrics["net_sharpe"] > traded.metrics["net_sharpe"]


def test_no_turnover_and_no_profit_still_fails_and_a_reported_infinite_fee_is_still_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_clock(monkeypatch, datetime.datetime(2026, 1, 5, tzinfo=UTC))
    gross, turnover, index = strategy_returns()
    losing = evidence_backed_tracker("cand", tmp_path, through=4).verify_stage_5_from_returns(
        -np.abs(gross), np.zeros_like(turnover), index, trial_ledger=persistent_trial_ledger(tmp_path / "t.jsonl"), universe=_fit_universe())
    assert not losing.passed and losing.metrics["fee_test_moot_no_turnover"] is False
    reported = evidence_backed_tracker("cand3", tmp_path / "c", through=4)
    refused = reported._decide_stage_5(2.0, 0.99, float("inf"), fee_is_moot=True)      # reported evidence may not say so
    assert not refused.passed
    assert not evidence_backed_tracker("cand4", tmp_path / "d", through=4)._decide_stage_5(2.0, 0.99, float("inf")).passed


def test_a_trial_s_sharpe_ratio_states_its_units_and_mixed_units_are_brought_together(tmp_path: Path) -> None:
    daily = [0.09, 0.10, 0.11, 0.12]                              # Sharpe ratios per period
    annual = [value * np.sqrt(252.0) for value in daily]
    stated_daily, stated_annual, unstated = (CandidateTrialLedger(persistence_path=tmp_path / f"{name}.jsonl") for name in "abc")
    for i, value in enumerate(daily):
        stated_daily.record_trial(f"t{i}", "p", "f", "TACTICAL", value, n_bars=504, annualization_factor=1.0)
        stated_annual.record_trial(f"t{i}", "p", "f", "TACTICAL", annual[i], n_bars=504, annualization_factor=252.0)
        unstated.record_trial(f"t{i}", "p", "f", "TACTICAL", value, n_bars=504)
    from_daily, from_annual, from_unstated = (
        ledger.dsr_with_selection_adjustment(observed_sharpe=2.0, n_bars=504, annualization_factor=252.0)
        for ledger in (stated_daily, stated_annual, unstated))
    # the same search, written down in two units, deflates the same observed ratio by the same amount
    assert from_daily["expected_max_sr"] == pytest.approx(from_annual["expected_max_sr"]) and from_daily["dsr"] == pytest.approx(from_annual["dsr"])
    assert from_daily["trial_sharpe_units"] == {"stated_and_the_same": 0, "stated_and_converted": 4, "unstated": 0}
    assert from_annual["trial_sharpe_units"] == {"stated_and_the_same": 4, "stated_and_converted": 0, "unstated": 0}
    # a record that states nothing is used as it is, as every record was before, and is counted
    assert from_unstated["trial_sharpe_units"] == {"stated_and_the_same": 0, "stated_and_converted": 0, "unstated": 4}
    assert from_unstated["expected_max_sr"] < from_daily["expected_max_sr"]
    assert stated_daily.get_sharpe_distribution() == unstated.get_sharpe_distribution()          # asked for no units: as recorded
    reopened = CandidateTrialLedger(persistence_path=tmp_path / "a.jsonl")
    assert [r.annualization_factor for r in reopened.get_records()] == [1.0] * 4
    old = tmp_path / "old.jsonl"
    old.write_text("\n".join(json.dumps({k: v for k, v in json.loads(line).items() if k != "annualization_factor"})
                             for line in (tmp_path / "a.jsonl").read_text().splitlines()) + "\n")
    assert [r.annualization_factor for r in CandidateTrialLedger(persistence_path=old).get_records()] == [None] * 4
    for bad in (0.0, -252.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="annualization_factor"):
            unstated.record_trial("x", "p", "f", "TACTICAL", 1.0, annualization_factor=bad)


_REFUSED_SOURCES = [
    ("", "no Lean source"),
    ("   \n", "no Lean source"),
    ("-- theorem t : x = x := rfl", "no Lean source"),
    ("/- theorem t : x = x := rfl -/", "no Lean source"),
    ("theorem t (x : Nat) : x = x := by sorry", "unproven"),
    ("theorem t (x : Nat) : x = x := by admit", "unproven"),
    ("theorem t (x : Nat) : x = x := sorryAx _", "unproven"),
    ("theorem t : 2 + 2 = 4 := by native_decide", "compiled code"),
    ("theorem t : 2 + 2 = 4 := Lean.ofReduceBool _ _ rfl", "compiled code"),
    ("theorem t (x : Nat) : x = x := by\n  /- nested /- comment -/ still a comment -/\n  sorry", "unproven"),
]


def test_stage_1_reads_a_lean_source_outside_its_comments_and_for_more_than_one_word() -> None:
    from apedp_referee.proof_evidence import lean_source_objections

    for source, objection in _REFUSED_SOURCES:
        assert any(objection in found for found in lean_source_objections(source)), (source, lean_source_objections(source))
        record = PromotionTracker("cand").verify_stage_1_from_lean(lean_result("cand", source=source))
        assert not record.passed and record.metrics["lean_source_objections"], source


_ACCEPTED_SOURCES = [
    "theorem scale (x : Nat) : x = x := rfl",
    "-- no sorry here, and no admit\ntheorem t (x : Nat) : x = x := rfl",
    "/- an earlier draft ended in sorry -/\ntheorem t (x : Nat) : x = x := rfl",
    'def note : String := "sorry -- not a comment, not a proof"\ntheorem t (x : Nat) : x = x := rfl',
    "theorem sorry_free (x : Nat) : x = x := rfl\ntheorem admitted' (x : Nat) : x = x := rfl",
    "def quote : Char := '\"'\ntheorem t (x : Nat) : x = x := rfl -- sorry",
    "axiom R : Type\ntheorem t (x : Nat) : x = x := rfl",
]


def test_a_word_in_a_comment_a_string_or_a_longer_name_is_not_an_objection() -> None:
    from apedp_referee.proof_evidence import lean_source_objections

    for source in _ACCEPTED_SOURCES:
        assert lean_source_objections(source) == [], source
        assert PromotionTracker("cand").verify_stage_1_from_lean(lean_result("cand", source=source)).passed, source
