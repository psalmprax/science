"""The Python referees agree with the Lean models, on every small case.

``lean_checked/`` proves properties of two *models*: the promotion state
machine (``Promotion.lean``) and the confirmation-holdout guard
(``Holdout.lean``). The Lean kernel checks the theorems; it says nothing about
whether ``PromotionTracker`` and ``FullLineageExperimentLedger`` behave like the
models. Without that link, "machine-checked" describes a text beside the code.

This file is the link, and it is a test, not a proof. Each model's transition
function is transcribed into Python below, next to the Lean it copies, and run
side by side with the real implementation over *every* operation sequence up to
a small length: after each operation the two must agree on whether it was
refused and on the resulting state. Bounded exhaustive agreement is the claim;
sequences longer than the bound, slice identity by content, persistence and
concurrency are outside it. ``Dimensions.lean`` has no counterpart here.

If a Lean definition is edited, ``test_the_transcriptions_track_the_lean_source``
fails until the transcription is re-read against it.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Any

import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.promotion_lifecycle import (
    PromotionPrerequisiteError,
    PromotionTracker,
)

LEAN = Path(__file__).resolve().parents[1] / "lean_checked" / "ApedpChecked"

# ---------------------------------------------------------------------------
# Promotion.lean
# ---------------------------------------------------------------------------
#   def step (t : Tracker) (stage : Nat) (ok : Bool) : Option Tracker :=
#     if t.halted then none
#     else if stage ≠ t.highest + 1 then none
#     else if ok then some ⟨stage, false⟩
#     else some ⟨t.highest, true⟩

ModelTracker = tuple[int, bool]  # (highest, halted)


def model_step(t: ModelTracker, stage: int, ok: bool) -> ModelTracker | None:
    highest, halted = t
    if halted:
        return None
    if stage != highest + 1:
        return None
    if ok:
        return (stage, False)
    return (highest, True)


# One passing and one failing evaluation per stage, on the reported-metrics
# entry points: the state machine is what is under test here, not the metrics.
_PASS = {
    1: lambda t: t.verify_stage_1("P2_VERIFIED", dimensions_consistent=True, bounded=True),
    2: lambda t: t.verify_stage_2(in_sample_ic=0.08, cv_stable=True),
    3: lambda t: t.verify_stage_3(dm_p_value=0.001, max_literature_r2=0.30),
    4: lambda t: t.verify_stage_4(oos_ic=0.06, oos_p_value=0.01),
    5: lambda t: t.verify_stage_5(net_sharpe=1.6, dsr_metric=0.99, breakeven_fee_bps=8.0),
    6: lambda t: t.verify_stage_6(prospective_sharpe=5.0, max_drawdown=0.01),
}
_FAIL = {
    1: lambda t: t.verify_stage_1("P1_CANDIDATE", dimensions_consistent=False, bounded=True),
    2: lambda t: t.verify_stage_2(in_sample_ic=0.0, cv_stable=False),
    3: lambda t: t.verify_stage_3(dm_p_value=0.9, max_literature_r2=0.30),
    4: lambda t: t.verify_stage_4(oos_ic=-0.1, oos_p_value=0.9),
    5: lambda t: t.verify_stage_5(net_sharpe=-3.0, dsr_metric=0.0, breakeven_fee_bps=0.0),
    6: lambda t: t.verify_stage_6(prospective_sharpe=-5.0, max_drawdown=0.9),
}
STAGES = (1, 2, 3, 4, 5, 6)


def _real_state(tracker: PromotionTracker) -> ModelTracker:
    return (int(tracker.highest_stage_passed), bool(tracker.is_halted))


def _run_promotion(sequence: tuple[tuple[int, bool], ...]) -> None:
    tracker = PromotionTracker("conformance", allow_reported_metrics=True)
    model: ModelTracker = (0, False)
    for position, (stage, ok) in enumerate(sequence):
        expected = model_step(model, stage, ok)
        before = _real_state(tracker)
        try:
            record = (_PASS if ok else _FAIL)[stage](tracker)
        except PromotionPrerequisiteError:
            assert expected is None, f"{sequence}: step {position} refused, the model grants it"
            assert _real_state(tracker) == before, f"{sequence}: a refusal changed the state"
            continue
        assert expected is not None, f"{sequence}: step {position} granted, the model refuses it"
        assert record.passed is ok, f"{sequence}: the canned stage-{stage} input did not {ok=}"
        model = expected
        assert _real_state(tracker) == model, f"{sequence}: states differ after step {position}"


def test_promotion_tracker_agrees_with_the_lean_model_on_every_short_sequence() -> None:
    """All 12 + 144 + 1,728 sequences of up to three (stage, outcome) steps."""
    steps = [(stage, ok) for stage in STAGES for ok in (True, False)]
    checked = 0
    for length in (1, 2, 3):
        for sequence in itertools.product(steps, repeat=length):
            _run_promotion(sequence)
            checked += 1
    assert checked == 12 + 144 + 1728


def test_promotion_tracker_agrees_with_the_lean_model_along_the_whole_ladder() -> None:
    """The full pass, a failure at each depth, and every single deviation from
    the pass: what the short sequences cannot reach (stages 4 to 6)."""
    ladder = tuple((stage, True) for stage in STAGES)
    _run_promotion(ladder)
    for depth in STAGES:
        failed_here = (*ladder[: depth - 1], (depth, False))
        _run_promotion(failed_here)
        for stage, ok in itertools.product(STAGES, (True, False)):
            _run_promotion((*failed_here, (stage, ok)))  # nothing follows a failure
            _run_promotion((*ladder[: depth - 1], (stage, ok), *ladder[depth - 1 :]))


# ---------------------------------------------------------------------------
# Holdout.lean
# ---------------------------------------------------------------------------
#   def claim (budget : Nat) (s : Slice) (c : Nat) : Option Slice :=
#     if s.burned then none
#     else if c ∈ s.inspectors then some s
#     else if s.inspectors.length < budget then some { s with inspectors := c :: s.inspectors }
#     else none
#
#   def burn (s : Slice) : Slice := { s with burned := true }

ModelSlice = tuple[tuple[int, ...], bool]  # (inspectors, burned)


def model_claim(budget: int, s: ModelSlice, c: int) -> ModelSlice | None:
    inspectors, burned = s
    if burned:
        return None
    if c in inspectors:
        return s
    if len(inspectors) < budget:
        return ((c, *inspectors), burned)
    return None


def model_burn(s: ModelSlice) -> ModelSlice:
    return (s[0], True)


CANDIDATES = (0, 1, 2)
HOLDOUT = DatasetRole.CONFIRMATION_HOLDOUT


def _real_slice(ledger: FullLineageExperimentLedger) -> tuple[frozenset[int], bool]:
    inspectors = ledger._holdout_inspectors.get(("spy", "h"), [])
    return (frozenset(int(c[1:]) for c in inspectors), ledger.is_holdout_burned("SPY", "h"))


def _run_holdout(budget: int, sequence: tuple[tuple[str, int], ...]) -> None:
    ledger = FullLineageExperimentLedger(max_holdout_inspections=budget)
    for c in CANDIDATES:
        ledger.register_candidate(f"c{c}", "proposer", f"formula_{c}")
    model: ModelSlice = ((), False)
    for position, (op, c) in enumerate(sequence):
        # "inspect": c scores on the slice. "fail": c scores on it and fails,
        # which in the model is a claim followed, if granted, by a burn.
        granted = model_claim(budget, model, c)
        before = _real_slice(ledger)
        try:
            ledger.record_data_exposure(
                f"c{c}", "SPY", "h", HOLDOUT, outcome_status="FAILED" if op == "fail" else "PASSED"
            )
        except HoldoutContaminationError:
            assert granted is None, f"{budget=} {sequence}: step {position} refused, model grants"
            assert _real_slice(ledger) == before, f"{sequence}: a refusal changed the state"
            continue
        assert granted is not None, f"{budget=} {sequence}: step {position} granted, model refuses"
        model = model_burn(granted) if op == "fail" else granted
        assert _real_slice(ledger) == (frozenset(model[0]), model[1]), (
            f"{budget=} {sequence}: states differ after step {position}"
        )
        assert len(model[0]) <= budget  # Slice.budget_is_respected, observed


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_holdout_guard_agrees_with_the_lean_model_on_every_short_sequence(budget: int) -> None:
    """All 6 + 36 + 216 + 1,296 sequences of up to four operations by three
    candidates on one slice, under each inspection budget."""
    operations = [(op, c) for op in ("inspect", "fail") for c in CANDIDATES]
    checked = 0
    for length in (1, 2, 3, 4):
        for sequence in itertools.product(operations, repeat=length):
            _run_holdout(budget, sequence)
            checked += 1
    assert checked == 6 + 36 + 216 + 1296


def test_a_rejected_candidate_burns_the_slices_it_inspected() -> None:
    """The ledger's second route to `burn`: a terminal lifecycle state."""
    from apedp_dyn.epistemic.experiment_ledger import CandidateLifecycleState

    ledger = FullLineageExperimentLedger(max_holdout_inspections=2)
    for c in CANDIDATES:
        ledger.register_candidate(f"c{c}", "proposer", f"formula_{c}")
    ledger.record_data_exposure("c0", "SPY", "h", HOLDOUT)
    model = model_claim(2, ((), False), 0)
    assert model is not None
    ledger.update_lifecycle_state("c0", CandidateLifecycleState.CONFIRMATION_REJECTED)
    model = model_burn(model)
    assert _real_slice(ledger) == (frozenset(model[0]), True)
    for c in CANDIDATES:  # Slice.nothing_after_a_burn
        assert model_claim(2, model, c) is None
        with pytest.raises(HoldoutContaminationError):
            ledger.record_data_exposure(f"c{c}", "SPY", "h", HOLDOUT)


# ---------------------------------------------------------------------------
# A check that cannot fail is not a check: seed defects in the real referees.
# ---------------------------------------------------------------------------

_reachable = PromotionTracker._assert_stage_reachable
_available = FullLineageExperimentLedger._assert_holdout_available


def _stages_may_be_skipped(self: PromotionTracker, target: Any) -> None:
    if self.is_halted or target in self.stage_records:
        raise PromotionPrerequisiteError("refused")


def _a_failed_stage_may_be_rerun(self: PromotionTracker, target: Any) -> None:
    if target in self.stage_records and self.stage_records[target].passed:
        raise PromotionPrerequisiteError("refused")
    if int(self.highest_stage_passed) < int(target) - 1:
        raise PromotionPrerequisiteError("refused")


def _the_halt_flag_is_ignored(self: PromotionTracker, target: Any) -> None:
    halted, self.is_halted = self.is_halted, False
    try:
        _reachable(self, target)
    finally:
        self.is_halted = halted


def _budget_off_by_one(self: Any, key: Any, candidate: str, fingerprint: Any = None) -> None:
    budget = self._max_holdout_inspections
    self._max_holdout_inspections = budget + 1
    try:
        _available(self, key, candidate, fingerprint)
    finally:
        self._max_holdout_inspections = budget


def _the_burner_may_return(self: Any, key: Any, candidate: str, fingerprint: Any = None) -> None:
    burned_under = self._burned_index.get(("spy", "h"))
    if burned_under is not None and self._burned_holdouts[burned_under] == candidate:
        return
    _available(self, key, candidate, fingerprint)


def _promotion_sweep() -> None:
    test_promotion_tracker_agrees_with_the_lean_model_on_every_short_sequence()


def _holdout_sweep() -> None:
    for budget in (1, 2):
        test_holdout_guard_agrees_with_the_lean_model_on_every_short_sequence(budget)


@pytest.mark.parametrize(
    ("owner", "attribute", "defect", "sweep"),
    [
        (PromotionTracker, "_assert_stage_reachable", _stages_may_be_skipped, _promotion_sweep),
        (
            PromotionTracker,
            "_assert_stage_reachable",
            _a_failed_stage_may_be_rerun,
            _promotion_sweep,
        ),
        (FullLineageExperimentLedger, "_burn", lambda self, key, cand: None, _holdout_sweep),
        (
            FullLineageExperimentLedger,
            "_assert_holdout_available",
            _budget_off_by_one,
            _holdout_sweep,
        ),
        (
            FullLineageExperimentLedger,
            "_assert_holdout_available",
            _the_burner_may_return,
            _holdout_sweep,
        ),
    ],
    ids=[
        "stage-skipped",
        "failed-stage-rerun",
        "burn-forgotten",
        "budget-off-by-one",
        "burner-returns",
    ],
)
def test_a_seeded_defect_in_the_real_referee_is_caught(
    monkeypatch: pytest.MonkeyPatch, owner: type, attribute: str, defect: Any, sweep: Any
) -> None:
    monkeypatch.setattr(owner, attribute, defect)
    with pytest.raises(AssertionError, match=r"model|states differ"):
        sweep()


def test_the_halt_flag_alone_is_redundant_in_the_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorded so nobody mistakes it for coverage. In the Lean model `halted`
    is what stops a failed stage being re-run. In the Python the same refusal
    also follows from two other rules -- a stage is evaluated once, and only
    after the previous one passed -- so ignoring the flag changes no outcome.
    The property holds twice over; the conformance sweep cannot tell which
    rule delivered it, and `failed-stage-rerun` above is the mutant that
    removes both."""
    monkeypatch.setattr(PromotionTracker, "_assert_stage_reachable", _the_halt_flag_is_ignored)
    _promotion_sweep()


# ---------------------------------------------------------------------------
# The transcriptions above are copies; this notices when the original moves.
# ---------------------------------------------------------------------------


def _definition(source: str, name: str) -> str:
    match = re.search(rf"^def {name}\b.*?(?=^\S|\Z)", source, re.M | re.S)
    assert match, f"definition {name} not found"
    return " ".join(match.group(0).split())


TRANSCRIBED = {
    ("Promotion.lean", "step"): "08049fa5aaf7",
    ("Holdout.lean", "claim"): "06fa28ca9a38",
    ("Holdout.lean", "burn"): "f29f414ba392",
}


def _digest(file: str, name: str) -> str:
    text = _definition((LEAN / file).read_text(encoding="utf-8"), name)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def test_the_transcriptions_track_the_lean_source() -> None:
    measured = {key: _digest(*key) for key in TRANSCRIBED}
    assert measured == TRANSCRIBED, (
        "a Lean definition transcribed in this file changed; re-read the Python copy "
        "against it, then update TRANSCRIBED"
    )
