"""Pins for the 2026-11-12 integrity-hardening round.

Each test reproduces a bypass that worked against the tree before this round
and asserts it is now refused. They are behavioural: every one drives the public
API the way a caller would, none reads source text.

Covered:
  * promotion gates could be passed by handing them looser thresholds, by
    writing to the tracker's flags, by re-running a stage, or by building a new
    tracker after a halt;
  * a confirmation holdout could be inspected by any number of candidates, be
    reused under a respelled name, and be un-burned by deleting a ledger line;
  * a Deflated Sharpe Ratio from an in-memory ledger forgot earlier searches;
  * the formula parser evaluated unbounded numeric towers while parsing;
  * the keyless UI API answered non-loopback clients.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
    LedgerIntegrityError,
    normalize_holdout_key,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionStage,
    PromotionTracker,
    PromotionViolationError,
)
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger

HOLDOUT = DatasetRole.CONFIRMATION_HOLDOUT


def _passed_through_stage(tracker: PromotionTracker, stage: int) -> PromotionTracker:
    """Advance a tracker with honest, canonical-threshold inputs."""
    if stage >= 1:
        tracker.verify_stage_1("P2_VERIFIED", dimensions_consistent=True, bounded=True)
    if stage >= 2:
        tracker.verify_stage_2(in_sample_ic=0.08, cv_stable=True)
    if stage >= 3:
        tracker.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.30)
    if stage >= 4:
        tracker.verify_stage_4(oos_ic=0.06, oos_p_value=0.01)
    if stage >= 5:
        tracker.verify_stage_5(net_sharpe=1.6, dsr_metric=0.99, breakeven_fee_bps=8.0)
    return tracker


# ---------------------------------------------------------------------------
# Promotion lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "call"),
    [
        (1, lambda t: t.verify_stage_2(0.0001, True, min_ic=0.0)),
        (2, lambda t: t.verify_stage_3(0.9, 0.3, dm_alpha=1.0)),
        (2, lambda t: t.verify_stage_3(0.001, 0.99, max_r2_ceiling=1.0)),
        (3, lambda t: t.verify_stage_4(oos_ic=1e-6, oos_p_value=0.01, min_oos_ic=0.0)),
        (3, lambda t: t.verify_stage_4(oos_ic=0.06, oos_p_value=0.9, alpha=1.0)),
        (3, lambda t: t.verify_stage_4(oos_ic=0.02, oos_p_value=0.01, min_retention=0.0)),
        (4, lambda t: t.verify_stage_5(-3.0, 0.99, 8.0, min_sharpe_hurdle=-10.0)),
        (4, lambda t: t.verify_stage_5(1.6, 0.0, 8.0, min_dsr_hurdle=0.0)),
        (4, lambda t: t.verify_stage_5(1.6, 0.99, 0.0, min_fee_bps=0.0)),
        (5, lambda t: t.verify_stage_6(-5.0, 0.1, prospective_hurdle=-10.0)),
        (5, lambda t: t.verify_stage_6(1.0, 0.9, max_dd_ceiling=1.0)),
        (1, lambda t: t.verify_stage_2(0.08, True, min_ic=float("nan"))),
    ],
)
def test_a_looser_threshold_is_refused_not_applied(stage, call):
    tracker = _passed_through_stage(PromotionTracker("cand"), stage)
    with pytest.raises(PromotionViolationError, match="looser than the preregistered"):
        call(tracker)
    # The refused call left no record behind: the stage is still open.
    assert tracker.highest_stage_passed == PromotionStage(stage)


def test_a_stricter_threshold_is_still_allowed():
    tracker = _passed_through_stage(PromotionTracker("cand"), 1)
    rec = tracker.verify_stage_2(in_sample_ic=0.03, cv_stable=True, min_ic=0.05)
    assert rec.passed is False  # 0.03 clears the canonical 0.02 but not the stricter 0.05


def test_junk_candidate_cannot_reach_real_market_promotion():
    """The exact walk that reached REAL_MARKET_PROMOTION_GRANTED before this round."""
    tracker = PromotionTracker("junk")
    tracker.verify_stage_1("P2_VERIFIED", True, True)
    with pytest.raises(PromotionViolationError):
        tracker.verify_stage_2(0.0001, True, min_ic=0.0)
    rec = tracker.verify_stage_2(0.0001, True)
    assert rec.passed is False
    assert tracker.is_halted
    assert tracker.highest_stage_passed == PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY


def test_writing_to_the_flags_does_not_unhalt_a_candidate():
    tracker = PromotionTracker("fail")
    tracker.verify_stage_1("P1_CANDIDATE", dimensions_consistent=False, bounded=True)
    assert tracker.is_halted
    tracker.is_halted = False
    tracker.highest_stage_passed = PromotionStage.STAGE_5_ECONOMIC_VALIDITY
    with pytest.raises(PromotionViolationError):
        tracker.verify_stage_6(prospective_sharpe=5.0, max_drawdown=0.01)


def test_claiming_a_stage_without_its_records_is_refused():
    tracker = PromotionTracker("skip")
    tracker.highest_stage_passed = PromotionStage.STAGE_5_ECONOMIC_VALIDITY
    with pytest.raises(PromotionViolationError, match="without a passing record"):
        tracker.verify_stage_6(prospective_sharpe=5.0, max_drawdown=0.01)


def test_a_stage_cannot_be_rerun_with_different_inputs():
    tracker = _passed_through_stage(PromotionTracker("rerun"), 2)
    with pytest.raises(PromotionPrerequisiteError, match="already on record"):
        tracker.verify_stage_2(in_sample_ic=0.02, cv_stable=True)
    with pytest.raises(PromotionPrerequisiteError, match="already on record"):
        tracker.verify_stage_1("P2_VERIFIED", True, True)


def test_a_halted_candidate_cannot_restart_at_stage_1():
    tracker = PromotionTracker("restart")
    tracker.verify_stage_1("P1_CANDIDATE", False, True)
    with pytest.raises(PromotionPrerequisiteError, match="HALTED"):
        tracker.verify_stage_1("P2_VERIFIED", True, True)


def test_a_halt_survives_a_new_tracker_when_journaled(tmp_path):
    journal = tmp_path / "promotion.jsonl"
    first = _passed_through_stage(PromotionTracker("cand", state_path=journal), 1)
    first.verify_stage_2(in_sample_ic=0.001, cv_stable=True)
    assert first.is_halted

    resumed = PromotionTracker("cand", state_path=journal)
    assert resumed.is_halted
    assert resumed.highest_stage_passed == PromotionStage.STAGE_1_MATHEMATICAL_CONSISTENCY
    with pytest.raises(PromotionPrerequisiteError, match="HALTED"):
        resumed.verify_stage_2(in_sample_ic=0.5, cv_stable=True)

    # Another candidate sharing the journal is unaffected.
    other = PromotionTracker("other", state_path=journal)
    assert not other.is_halted
    assert other.highest_stage_passed == PromotionStage.UNEVALUATED


def test_a_damaged_promotion_journal_is_refused(tmp_path):
    journal = tmp_path / "promotion.jsonl"
    journal.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(PromotionViolationError, match="unreadable"):
        PromotionTracker("cand", state_path=journal)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio: derived from a persistent ledger, not reported
# ---------------------------------------------------------------------------


def _noise_search(ledger: CandidateTrialLedger, n_trials: int = 400, n_bars: int = 1000):
    rng = np.random.default_rng(7)
    best_sharpe, best_returns = -np.inf, None
    for i in range(n_trials):
        returns = rng.normal(0.0, 0.01, n_bars)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
        ledger.record_trial(f"t{i}", "noise", "f", "TACTICAL", sharpe, returns=returns)
        if sharpe > best_sharpe:
            best_sharpe, best_returns = sharpe, returns
    return best_sharpe, best_returns


def test_stage_5_from_ledger_refuses_an_in_memory_ledger():
    tracker = _passed_through_stage(PromotionTracker("cand"), 4)
    ledger = CandidateTrialLedger()
    ledger.record_trial("w", "p", "f", "TACTICAL", 1.7, n_bars=1000)
    assert ledger.is_persistent is False
    with pytest.raises(PromotionViolationError, match="persistent trial ledger"):
        tracker.verify_stage_5_from_ledger(
            net_sharpe=1.7, trial_ledger=ledger, n_bars=1000, breakeven_fee_bps=8.0,
            annualization_factor=252.0,
        )


def test_stage_5_from_ledger_charges_the_whole_search(tmp_path):
    """Best-of-many noise does not pass, and a reopened ledger remembers the search."""
    path = tmp_path / "trials.jsonl"
    best_sharpe, _ = _noise_search(CandidateTrialLedger(persistence_path=path))

    reopened = CandidateTrialLedger(persistence_path=path)  # a later process
    assert reopened.is_persistent
    assert reopened.trial_count == 400

    tracker = _passed_through_stage(PromotionTracker("cand"), 4)
    rec = tracker.verify_stage_5_from_ledger(
        net_sharpe=best_sharpe, trial_ledger=reopened, n_bars=1000, breakeven_fee_bps=8.0,
        annualization_factor=252.0,
    )
    assert rec.passed is False
    assert rec.metrics["dsr_source"] == "trial_ledger"
    assert rec.metrics["dsr_n_trials"] == 400
    assert rec.metrics["dsr_metric"] < 0.95


def test_dsr_payload_discloses_ledger_persistence(tmp_path):
    ephemeral = CandidateTrialLedger()
    ephemeral.record_trial("a", "p", "f", "TACTICAL", 1.0, n_bars=500)
    assert ephemeral.dsr_with_selection_adjustment(1.0, 500)["ledger_persistent"] is False
    durable = CandidateTrialLedger(persistence_path=tmp_path / "t.jsonl")
    durable.record_trial("a", "p", "f", "TACTICAL", 1.0, n_bars=500)
    assert durable.dsr_with_selection_adjustment(1.0, 500)["ledger_persistent"] is True


# ---------------------------------------------------------------------------
# Confirmation-holdout guard
# ---------------------------------------------------------------------------


def _ledger_with(*candidates: str, **kwargs) -> FullLineageExperimentLedger:
    ledger = FullLineageExperimentLedger(**kwargs)
    for cid in candidates:
        ledger.register_candidate(cid, "proposer", f"formula_{cid}")
    return ledger


def test_a_second_candidate_cannot_inspect_a_holdout_nobody_failed():
    """Before: fifty candidates inspected one slice and nothing was ever burned."""
    ledger = _ledger_with("c0", "c1")
    ledger.record_data_exposure("c0", "SPY", "holdout_2024", HOLDOUT)  # IN_PROGRESS
    with pytest.raises(HoldoutContaminationError, match="inspection budget is 1"):
        ledger.record_data_exposure("c1", "SPY", "holdout_2024", HOLDOUT)
    assert ledger.get_holdout_inspection_counts() == {("spy", "holdout_2024"): 1}


def test_a_passed_holdout_is_spent_too():
    ledger = _ledger_with("winner", "next")
    ledger.record_data_exposure("winner", "SPY", "h", HOLDOUT, outcome_status="PASSED")
    with pytest.raises(HoldoutContaminationError):
        ledger.record_data_exposure("next", "SPY", "h", HOLDOUT)


def test_the_same_candidate_may_record_its_own_slice_again():
    ledger = _ledger_with("c0")
    ledger.record_data_exposure("c0", "SPY", "h", HOLDOUT)
    ledger.record_data_exposure("c0", "SPY", "h", HOLDOUT, outcome_status="PASSED")
    assert ledger.get_holdout_inspection_counts() == {("spy", "h"): 1}


def test_development_slices_are_not_budgeted():
    ledger = _ledger_with("c0", "c1")
    for cid in ("c0", "c1"):
        ledger.record_data_exposure(cid, "SPY", "train", DatasetRole.TRAINING_DEVELOPMENT)
    assert ledger.get_holdout_inspection_counts() == {}


def test_the_budget_can_be_raised_or_disabled_explicitly():
    ledger = _ledger_with("a", "b", "c", max_holdout_inspections=2)
    ledger.record_data_exposure("a", "SPY", "h", HOLDOUT)
    ledger.record_data_exposure("b", "SPY", "h", HOLDOUT)
    with pytest.raises(HoldoutContaminationError, match="budget is 2"):
        ledger.record_data_exposure("c", "SPY", "h", HOLDOUT)

    unlimited = _ledger_with("a", "b", "c", max_holdout_inspections=None)
    for cid in ("a", "b", "c"):
        unlimited.record_data_exposure(cid, "SPY", "h", HOLDOUT)

    with pytest.raises(ValueError, match="max_holdout_inspections"):
        FullLineageExperimentLedger(max_holdout_inspections=0)


@pytest.mark.parametrize(
    ("dataset_id", "slice_id"),
    [
        ("SPY", "holdout_2024 "),
        ("spy", "holdout_2024"),
        (" SPY", "HOLDOUT_2024"),
        ("SPY", "holdout_２０２４"),  # full-width digits (NFKC)
    ],
)
def test_a_respelled_slice_is_the_same_burned_slice(dataset_id, slice_id):
    ledger = _ledger_with("c0", "c_new", max_holdout_inspections=None)
    ledger.record_data_exposure("c0", "SPY", "holdout_2024", HOLDOUT, outcome_status="FAILED")
    assert normalize_holdout_key(dataset_id, slice_id) == ("spy", "holdout_2024")
    assert ledger.is_holdout_burned(dataset_id, slice_id)
    with pytest.raises(HoldoutContaminationError, match="previously burned"):
        ledger.record_data_exposure("c_new", dataset_id, slice_id, HOLDOUT)
    # The registry still reports the slice under the name it was burned with.
    assert ledger.get_burned_holdouts() == {("SPY", "holdout_2024"): "c0"}


def test_rejection_burns_under_the_normalised_identity():
    ledger = _ledger_with("c0", "c1")
    ledger.record_data_exposure("c0", "SPY", "h", HOLDOUT)
    ledger.update_lifecycle_state("c0", CandidateLifecycleState.CONFIRMATION_REJECTED)
    assert ledger.is_holdout_burned("spy", " H ")


def test_budget_and_burn_survive_a_reopened_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = _ledger_with("c0", persistence_path=path)
    first.record_data_exposure("c0", "SPY", "h", HOLDOUT)

    reopened = FullLineageExperimentLedger(persistence_path=path)  # a later process
    reopened.register_candidate("c1", "proposer", "g")
    with pytest.raises(HoldoutContaminationError, match="inspection budget"):
        reopened.record_data_exposure("c1", "SPY", "h", HOLDOUT)

    first.update_lifecycle_state("c0", CandidateLifecycleState.CONFIRMATION_REJECTED)
    again = FullLineageExperimentLedger(persistence_path=path)
    assert again.is_holdout_burned("SPY", "h")


def test_two_live_instances_share_one_budget_through_the_file(tmp_path):
    """The budget is enforced under the file lock, not from stale memory."""
    path = tmp_path / "ledger.jsonl"
    a = _ledger_with("ca", persistence_path=path)
    b = FullLineageExperimentLedger(persistence_path=path)
    b.register_candidate("cb", "proposer", "g")
    a.record_data_exposure("ca", "SPY", "h", HOLDOUT)
    with pytest.raises(HoldoutContaminationError):
        b.record_data_exposure("cb", "SPY", "h", HOLDOUT)
    # The refused exposure was rolled back, not journaled.
    assert b.get_candidate("cb").data_exposures == []


# ---------------------------------------------------------------------------
# Ledger tamper evidence
# ---------------------------------------------------------------------------


def _burned_ledger_file(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    ledger = _ledger_with("a", "b", persistence_path=path)
    ledger.record_data_exposure("a", "SPY", "h1", HOLDOUT, outcome_status="FAILED")
    ledger.register_candidate("c", "proposer", "h")
    return path


def test_every_entry_is_chained_and_the_head_is_recorded(tmp_path):
    path = _burned_ledger_file(tmp_path)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[0]["prev_hash"] == "GENESIS"
    for prev, cur in itertools.pairwise(entries):
        assert cur["prev_hash"] == prev["entry_hash"]
    ledger = FullLineageExperimentLedger(persistence_path=path)
    assert ledger.head_hash == entries[-1]["entry_hash"]
    head = json.loads(path.with_name(path.name + ".head").read_text())
    assert head == {"size_bytes": path.stat().st_size, "head_hash": entries[-1]["entry_hash"]}


def test_deleting_the_burn_line_is_detected(tmp_path):
    """Before: dropping one JSONL line silently made the burned slice reusable."""
    path = _burned_ledger_file(tmp_path)
    lines = path.read_text().splitlines()
    burn_lines = [i for i, line in enumerate(lines) if '"FAILED"' in line]
    kept = [line for i, line in enumerate(lines) if i not in burn_lines]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        FullLineageExperimentLedger(persistence_path=path)


def test_truncating_the_tail_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = _ledger_with("a", persistence_path=path)
    ledger.record_data_exposure("a", "SPY", "h1", HOLDOUT, outcome_status="FAILED")
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")  # the burn was the last line
    with pytest.raises(LedgerIntegrityError, match="truncated or replaced"):
        FullLineageExperimentLedger(persistence_path=path)


def test_editing_an_entry_is_detected(tmp_path):
    path = _burned_ledger_file(tmp_path)
    path.write_text(path.read_text().replace('"FAILED"', '"PASSED"'), encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="hash chain breaks"):
        FullLineageExperimentLedger(persistence_path=path)


def test_a_corrupt_line_fails_closed_on_load_and_on_exposure(tmp_path):
    path = _burned_ledger_file(tmp_path)
    live = FullLineageExperimentLedger(persistence_path=path)
    live.register_candidate("z", "proposer", "q")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{torn line\n")
    with pytest.raises(LedgerIntegrityError):
        FullLineageExperimentLedger(persistence_path=path)
    # Before: the exposure-time refresh skipped unreadable lines and carried on.
    with pytest.raises(LedgerIntegrityError):
        live.record_data_exposure("z", "QQQ", "fresh", HOLDOUT)


def test_a_pre_chain_ledger_still_loads_and_is_extended(tmp_path):
    """Ledgers written before the chain existed carry no hashes; they stay readable."""
    path = tmp_path / "ledger.jsonl"
    legacy = {
        "candidate_id": "old", "proposer_id": "p", "formula": "x", "version": 1,
        "parent_candidate_id": None, "feature_set": [], "parameter_choices": {},
        "attempted_variants": [], "state": "confirmation_rejected", "rejection_reason": None,
        "data_exposures": [{
            "candidate_id": "old", "dataset_id": "SPY", "slice_identifier": "h_old",
            "role": "confirmation_holdout", "inspected_at": "2026-10-01T00:00:00+00:00",
            "result_status": "FAILED",
        }],
        "metrics": {}, "created_at": "", "updated_at": "",
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    ledger = FullLineageExperimentLedger(persistence_path=path)
    assert ledger.is_holdout_burned("SPY", "h_old")
    ledger.register_candidate("new", "p", "y")

    reopened = FullLineageExperimentLedger(persistence_path=path)
    assert {n.candidate_id for n in reopened.get_all_candidates()} == {"old", "new"}
    # The first chained entry is bound to the legacy line it follows.
    path.write_text(
        path.read_text().replace('"h_old"', '"h_renamed"', 1), encoding="utf-8"
    )
    with pytest.raises(LedgerIntegrityError):
        FullLineageExperimentLedger(persistence_path=path)


def test_head_marker_can_be_disabled_for_a_rewound_fixture(tmp_path):
    """A committed fixture ledger is rewound by git; the chain still guards it."""
    path = tmp_path / "ledger.jsonl"
    ledger = _ledger_with("a", "b", persistence_path=path, head_marker=False)
    assert not path.with_name(path.name + ".head").exists()
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")  # rewound by one entry
    reopened = FullLineageExperimentLedger(persistence_path=path, head_marker=False)
    assert [n.candidate_id for n in reopened.get_all_candidates()] == ["a"]
    path.write_text(path.read_text().replace("formula_a", "formula_x"), encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="hash chain breaks"):
        FullLineageExperimentLedger(persistence_path=path, head_marker=False)
    assert ledger.max_holdout_inspections == 1


def test_an_in_memory_ledger_has_no_head():
    assert FullLineageExperimentLedger().head_hash is None


# ---------------------------------------------------------------------------
# Formula parser: bounded evaluation
# ---------------------------------------------------------------------------

_HOSTILE = [
    "9**9**9**9",
    "9^9^9^9",
    "zeta(10**7)",
    "gamma(10**8)",
    "x**(10**9) + 1",
    "9**(9**(9 + x - x))",  # the exponent only becomes numeric once evaluated
    "9**abs(-9**9)",
    "9**sqrt(9**18)",
    "(10**8)!",
    "((10**999)**999)**999",
    "besselj(0, 10**9)",
    "2**max(1, 9**9)",
]

_LEGITIMATE = [
    ("0.5*m*v**2", "0.5*m*v**2"),
    ("G*m1*m2/r**2", "G*m1*m2/r**2"),
    ("1e-15*x + 6.02e23", "1.0e-15*x + 6.02e+23"),
    ("10**23*k", "100000000000000000000000*k"),
    ("sin(5000*t) + log(1000000)", "sin(5000*t) + log(1000000)"),
    ("min(a, b)*ln(x)", "log(x)*Min(a, b)"),
    ("(x + 1)**100", "(x + 1)**100"),
    ("zeta(2) + gamma(5)", "pi**2/6 + 24"),
    ("x^2 + y^2", "x**2 + y**2"),
]


def _parse_in_child(module_name: str, formula: str, queue) -> None:
    import importlib

    parse = importlib.import_module(module_name).safe_parse_expr
    try:
        queue.put(("accepted", str(parse(formula))))
    except ValueError as exc:
        queue.put(("rejected", type(exc).__name__))


@pytest.mark.parametrize("module_name", ["apedp.safety", "apedp_dyn.safety"])
@pytest.mark.parametrize("formula", _HOSTILE)
def test_unbounded_numeric_evaluation_is_refused_quickly(module_name, formula):
    """Run in a child process: before this round these inputs never returned."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_parse_in_child, args=(module_name, formula, queue))
    started = time.monotonic()
    child.start()
    child.join(timeout=60)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail(f"{module_name}.safe_parse_expr({formula!r}) did not return")
    assert queue.get(timeout=5) == ("rejected", "FormulaResourceError")
    assert time.monotonic() - started < 60


@pytest.mark.parametrize("module_name", ["apedp.safety", "apedp_dyn.safety"])
@pytest.mark.parametrize(("formula", "expected"), _LEGITIMATE)
def test_ordinary_formulas_parse_exactly_as_before(module_name, formula, expected):
    import importlib

    assert str(importlib.import_module(module_name).safe_parse_expr(formula)) == expected


@pytest.mark.parametrize("module_name", ["apedp.safety", "apedp_dyn.safety"])
def test_the_resource_error_is_a_value_error_for_existing_handlers(module_name):
    import importlib

    module = importlib.import_module(module_name)
    assert issubclass(module.FormulaResourceError, ValueError)
    with pytest.raises(ValueError, match="exceeds the limit"):
        module.safe_parse_pde("Derivative(u(x, t), t) - 9**9**9**9")
    # A PDE residual with ordinary orders still parses.
    assert module.safe_parse_pde("Derivative(u(x, t), t) - c*Derivative(u(x, t), x, 2)") is not None


# ---------------------------------------------------------------------------
# UI API: keyless mode serves loopback only
# ---------------------------------------------------------------------------


def _request(peer: str | None, headers: dict[str, str] | None = None):
    client = None if peer is None else SimpleNamespace(host=peer)
    return SimpleNamespace(client=client, headers=headers or {})


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "testclient"])
def test_keyless_ui_api_serves_loopback(monkeypatch, peer):
    pytest.importorskip("fastapi")
    from apedp_dyn.ui.helpers import _require_api_key

    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    assert _require_api_key(_request(peer)) is None


@pytest.mark.parametrize("peer", ["10.0.0.7", "203.0.113.9", "", None])
def test_keyless_ui_api_refuses_non_loopback(monkeypatch, peer):
    fastapi = pytest.importorskip("fastapi")
    from apedp_dyn.ui.helpers import _require_api_key

    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    with pytest.raises(fastapi.HTTPException) as excinfo:
        _require_api_key(_request(peer))
    assert excinfo.value.status_code == 401


def test_a_configured_key_still_admits_remote_clients(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from apedp_dyn.ui.helpers import _require_api_key

    monkeypatch.setenv("APEDP_API_KEY", "s3cret")
    assert _require_api_key(_request("203.0.113.9", {"x-api-key": "s3cret"})) is None
    with pytest.raises(fastapi.HTTPException):
        _require_api_key(_request("203.0.113.9", {"x-api-key": "wrong"}))


# ---------------------------------------------------------------------------
# Governance gate: runs with nothing installed
# ---------------------------------------------------------------------------


def test_governance_drift_gate_needs_only_the_standard_library():
    """The CI job installs nothing; the script must not import the package."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "check_governance_manifest_drift.py"
    # -I -S: isolated mode without site-packages, i.e. a bare interpreter.
    proc = subprocess.run(
        [sys.executable, "-I", "-S", str(script)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert "ModuleNotFoundError" not in proc.stderr
    assert proc.returncode == 0, proc.stderr
    assert "sealed:" in proc.stdout
    assert "disk:" in proc.stdout
