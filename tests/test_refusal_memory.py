"""The refusal-memory discovery loop (apedp_dyn/discovery_l2/refusal_memory.py).

Most tests drive the loop with a scripted runner, so they test the loop's
discipline (which data each round sees, what is charged, what the memory may
read) rather than the engine. One test runs the real engines end to end.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import refusal_memory as rm
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_referee.trial_ledger import CandidateTrialLedger

ROOT = Path(__file__).resolve().parents[1]
DIMS = {"a": Dimension(m=1), "b": Dimension(l=1)}
TARGET = Dimension(m=1, l=1, t=-2)
PROMOTE, REFUSE = "unique_support", "multi_hypothesis"


class Scripted:
    """A runner that promotes on the named rungs and records what each call was given."""

    def __init__(self, promote_on: set[str]):
        self.promote_on = promote_on
        self.calls: list[dict] = []

    def __call__(self, rung, X, Y, dimensions, target_dimension, *, noise_sigma, seed, X_holdout=None, Y_holdout=None):
        self.calls.append({"rung": rung.name, "Y": np.asarray(Y).copy(),
                           "Y_holdout": None if Y_holdout is None else np.asarray(Y_holdout).copy()})
        status = PROMOTE if rung.name in self.promote_on else REFUSE
        return rm.RoundOutcome(status, "decisive_evidential_support" if status == PROMOTE else "hypothesis_space_limited",
                               "a" if status == PROMOTE else None, 10)


def _data(n: int = 40):
    # Y is the row number, so every array a round receives says which rows it holds.
    return {"a": np.arange(n, dtype=float) + 1, "b": np.ones(n)}, np.arange(n, dtype=float)


def _loop(ledger, runner, domain="d1", dataset_id="x"):
    X, Y = _data()
    return rm.run_discovery_loop(X, Y, DIMS, TARGET, domain=domain, dataset_id=dataset_id, ledger=ledger, runner=runner)


def test_rounds_never_see_the_confirmation_half():
    runner = Scripted({"superposition"})
    result = _loop(CandidateTrialLedger(), runner)
    *rounds, final = runner.calls
    assert [c["rung"] for c in rounds] == ["strict", "unit_coefficients", "superposition"]
    assert final["rung"] == "superposition" and final["Y_holdout"] is not None
    validation, confirmation = set(final["Y"]), set(final["Y_holdout"])
    assert not validation & confirmation and validation | confirmation == set(range(40))
    for call in rounds:
        assert call["Y_holdout"] is None and set(call["Y"]) == validation
    assert result.confirmed and result.chosen_rung == "superposition"


def test_every_round_and_the_confirmation_are_charged_as_trials():
    ledger = CandidateTrialLedger()
    _loop(ledger, Scripted({"unit_coefficients"}))
    records = ledger.get_records()
    assert [r.metadata["stage"] for r in records] == ["proposer_validation", "proposer_validation", "confirmation"]
    assert len({r.metadata["search_id"] for r in records}) == 1
    assert ledger.effective_trial_count == 30  # three calls, ten expressions each
    assert all(r.proposer_id == rm.PROPOSER_ID and r.strategy_archetype == "equation_discovery" for r in records)


def test_nothing_is_confirmed_when_no_rung_promotes():
    runner = Scripted(set())
    result = _loop(CandidateTrialLedger(), runner)
    assert result.chosen_rung is None and result.confirmation is None and not result.confirmed
    assert all(c["Y_holdout"] is None for c in runner.calls)


def test_the_confirmation_half_is_spent_once(tmp_path):
    path = tmp_path / "trials.jsonl"
    _loop(CandidateTrialLedger(path), Scripted({"strict"}))
    with pytest.raises(rm.ConfirmationAlreadySpent):
        _loop(CandidateTrialLedger(path), Scripted({"strict"}))  # a new process, same ledger file
    _loop(CandidateTrialLedger(path), Scripted({"strict"}), dataset_id="y")  # other data are fine
    _loop(CandidateTrialLedger(path), Scripted({"strict"}), domain="d2")  # and other domains


def test_a_refusal_that_was_spent_on_a_failed_confirmation_still_spends_it():
    ledger = CandidateTrialLedger()

    class PassesValidationOnly(Scripted):
        def __call__(self, *args, **kwargs):
            outcome = super().__call__(*args, **kwargs)
            return rm.RoundOutcome(REFUSE, "hypothesis_space_limited", None, 10) if kwargs.get("Y_holdout") is not None else outcome

    result = _loop(ledger, PassesValidationOnly({"strict"}))
    assert result.chosen_rung == "strict" and not result.confirmed
    with pytest.raises(rm.ConfirmationAlreadySpent):
        _loop(ledger, Scripted({"strict"}))


def test_the_domain_learns_to_skip_a_rung_only_after_two_datasets():
    ledger = CandidateTrialLedger()
    _loop(ledger, Scripted({"unit_coefficients"}), dataset_id="x1")
    assert rm.RefusalMemory(ledger, "d1").rungs_to_try("x2") == [0, 1, 2]
    _loop(ledger, Scripted({"unit_coefficients"}), dataset_id="x2")
    assert rm.RefusalMemory(ledger, "d1").rungs_to_try("x3") == [1, 2]
    runner = Scripted({"unit_coefficients"})
    _loop(ledger, runner, dataset_id="x3")
    assert runner.calls[0]["rung"] == "unit_coefficients"
    # Another domain learns nothing from d1.
    assert rm.RefusalMemory(ledger, "d2").rungs_to_try("x9") == [0, 1, 2]


def test_a_rung_that_promoted_anywhere_is_not_skipped():
    ledger = CandidateTrialLedger()
    _loop(ledger, Scripted({"strict"}), dataset_id="x0")  # strict worked here
    _loop(ledger, Scripted({"unit_coefficients"}), dataset_id="x1")
    _loop(ledger, Scripted({"unit_coefficients"}), dataset_id="x2")
    assert rm.RefusalMemory(ledger, "d1").rungs_to_try("x3") == [0, 1, 2]


def test_a_refusal_with_no_higher_promotion_is_not_a_reason_to_skip():
    ledger = CandidateTrialLedger()
    _loop(ledger, Scripted(set()), dataset_id="x1")
    _loop(ledger, Scripted(set()), dataset_id="x2")
    assert rm.RefusalMemory(ledger, "d1").rungs_to_try("x3") == [0, 1, 2]


def test_the_last_rung_is_never_skipped():
    ledger = CandidateTrialLedger()
    for name in ("x1", "x2"):
        for rung in rm.LADDER:
            ledger.record_trial(f"{name}-{rung.name}", rm.PROPOSER_ID, "", "equation_discovery", 0.0,
                                status=REFUSE, metadata={"domain": "d1", "dataset_id": name, "rung": rung.name,
                                                         "stage": "proposer_validation", "status": REFUSE})
        ledger.record_trial(f"{name}-later", rm.PROPOSER_ID, "", "equation_discovery", 0.0, status=PROMOTE,
                            metadata={"domain": "d1", "dataset_id": name, "rung": "a_rung_from_a_later_ladder",
                                      "stage": "proposer_validation", "status": PROMOTE})
    assert rm.RefusalMemory(ledger, "d1").rungs_to_try("x3")[-1] == len(rm.LADDER) - 1


def test_the_memory_reads_verdicts_and_nothing_else():
    """Holdout errors, selection statistics and fitted values in a record do not
    reach the policy: two ledgers that differ only in those decide the same."""
    def ledger_with(extra: dict, stat: float, sharpe: float):
        ledger = CandidateTrialLedger()
        for name in ("x1", "x2"):
            for rung, status in (("strict", REFUSE), ("unit_coefficients", PROMOTE)):
                ledger.record_trial(f"{name}-{rung}", rm.PROPOSER_ID, "theta*a", "equation_discovery", sharpe,
                                    status=status, selection_statistic=stat,
                                    metadata={"domain": "d1", "dataset_id": name, "rung": rung,
                                              "stage": "proposer_validation", "status": status, **extra})
        return ledger

    plain = ledger_with({}, 0.0, 0.0)
    loaded = ledger_with({"holdout_rmse": 1e-9, "confirmation_rmse": 0.0, "param_values": {"theta": 9.81}},
                         -1e9, 99.0)
    assert rm.RefusalMemory(plain, "d1").rungs_to_try("x3") == rm.RefusalMemory(loaded, "d1").rungs_to_try("x3")
    for view in rm.RefusalMemory(loaded, "d1")._views():
        assert set(view) <= rm.READABLE_FIELDS


def test_the_module_imports_no_holdout_ledger():
    tree = ast.parse((ROOT / "apedp_dyn/discovery_l2/refusal_memory.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("experiment_ledger" in (m or "") for m in imported), imported


def test_the_split_is_seeded_and_disjoint():
    X, Y = _data(41)
    xa, ya, _xb, yb = rm.split_development(X, Y, seed=7)
    again = rm.split_development(X, Y, seed=7)
    assert np.array_equal(ya, again[1]) and np.array_equal(yb, again[3])
    assert len(ya) == 20 and len(yb) == 21 and not set(ya) & set(yb)
    assert np.array_equal(xa["a"], ya + 1)  # rows stay aligned across variables


def test_end_to_end_a_law_with_a_dimensional_constant_climbs_one_rung():
    """weight = g * mass: the strict grammar has no expression with the
    dimension of force, so it is refused; the next rung lets the coefficient
    carry the units and is promoted, then confirmed on the unseen half."""
    rng = np.random.default_rng(5)
    n = 200
    X = {"mass": rng.uniform(1, 10, n), "span": rng.uniform(1, 10, n)}
    y = 9.81 * X["mass"] * (1 + 0.01 * rng.standard_normal(n))
    ledger = CandidateTrialLedger()
    result = rm.run_discovery_loop(X, y, {"mass": Dimension(m=1), "span": Dimension(l=1)}, TARGET,
                                   domain="mechanics", dataset_id="weight", ledger=ledger,
                                   noise_sigma=0.01 * float(np.mean(y)))
    assert [r["rung"] for r in result.rounds] == ["strict", "unit_coefficients"]
    assert result.rounds[0]["status"] == "no_valid_candidate"
    assert result.confirmed and result.confirmation["promoted"] == "mass"
    assert ledger.trial_count == 3
    assert ledger.effective_trial_count == sum(max(1, r["generated"]) for r in [*result.rounds, result.confirmation])
