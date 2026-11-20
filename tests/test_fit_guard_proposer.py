"""A candidate the path cannot fit is dropped before the path sees it (audit M-24)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.proposers import fit_guard
from apedp_dyn.proposers.fit_guard import FitGuardProposer, fit_problem
from apedp_dyn.proposers.portfolio import FixedFormulasProposer

RNG = np.random.default_rng(3)
A = RNG.uniform(0.5, 2.0, 120)
B = RNG.uniform(0.5, 2.0, 120)
FRAME = pd.DataFrame({"a": A, "b": B, "target": 3.0 * A * B + 1.0})


def test_good_candidates_pass_unchanged_and_in_order():
    guard = FitGuardProposer(FixedFormulasProposer(["a*b", "a + b", "sin(a)"]))
    assert [p.formula for p in guard.fit_and_propose(FRAME)] == ["a*b", "a + b", "sin(a)"]
    assert guard.dropped == [] and guard.last_run["fit_guard_dropped"] == 0
    assert guard.proposer_id == "fixed_formulas"


def test_a_candidate_that_is_not_finite_on_the_rows_is_dropped_with_the_reason():
    """``log(a - 5)`` is undefined on these rows; unguarded, the path's fit raises the recorded LinAlgError on it."""
    guard = FitGuardProposer(FixedFormulasProposer(["a*b", "log(a - 5)", "sqrt(b - 9)"]))
    assert [p.formula for p in guard.fit_and_propose(FRAME)] == ["a*b"]
    assert [f for f, _ in guard.dropped] == ["log(a - 5)", "sqrt(b - 9)"]
    assert guard.dropped[0][1] == "fit raised LinAlgError (relative noise)"


def test_a_fit_that_raises_drops_the_candidate_instead_of_ending_the_search(monkeypatch):
    """The recorded failure: lstsq's SVD did not converge inside the path's own fit."""
    real = cr.fitted_candidate

    def failing(expr, names, dev_X, dev_y, noise="relative"):
        if "exp" in str(expr):
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
        return real(expr, names, dev_X, dev_y, noise)

    monkeypatch.setattr(fit_guard.cr, "fitted_candidate", failing)
    guard = FitGuardProposer(FixedFormulasProposer(["exp(a)", "a*b"]))
    assert [p.formula for p in guard.fit_and_propose(FRAME)] == ["a*b"]
    assert guard.dropped == [("exp(a)", "fit raised LinAlgError (relative noise)")]


def test_an_unparseable_formula_is_reported_as_such():
    assert fit_problem("a +* b", ["a", "b"], FRAME[["a", "b"]].to_numpy(), FRAME["target"].to_numpy()).startswith("not parseable")
    assert fit_problem("a*b", ["a", "b"], FRAME[["a", "b"]].to_numpy(), FRAME["target"].to_numpy()) == ""


def test_the_wrapped_stages_own_failures_stay_visible():
    inner = FixedFormulasProposer(["a*b"])
    inner.failures = ["member 1: boom"]
    inner.last_run = {"members": 4}
    guard = FitGuardProposer(inner)
    guard.fit_and_propose(FRAME)
    assert guard.failures == ["member 1: boom"] and guard.last_run == {"members": 4, "fit_guard_dropped": 0}
