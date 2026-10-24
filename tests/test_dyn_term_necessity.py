"""Regression suite for the E3b term-necessity / near-tie contract (doc SS66-67).

Fast OLS-based pins (no engine runs): the engine's E3b behavior is recorded in
docs/TRADING_EQUATION_FORENSICS.md; this suite pins the guard semantics so a
future UNIQUE claim cannot pass with decorative terms or hidden ties.
"""

import itertools

import numpy as np
import numpy.linalg as la
import pytest

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.epistemic.term_necessity import (
    NecessityVerdict,
    TermAblation,
    ablate_terms,
    apply_ood_upgrade,
    near_tie_class,
    necessity_contract_check,
    ood_necessity,
    ood_necessity_check,
)
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

rng = np.random.default_rng(31)
N = 2500
X = rng.uniform(-2, 2, N)
E = rng.normal(0, 0.1, N)
HOLDOUT = rng.random(N) < 0.3


def _verdicts(terms, y):
    return {a.term: a.verdict for a in ablate_terms(terms, y, HOLDOUT)}


def test_A_necessary_term_retained():
    y = X**2 + 0.5 * X**3 + E
    v = _verdicts({"x2": X**2, "x3": X**3}, y)
    assert v["x3"] is NecessityVerdict.RETAIN
    assert v["x2"] is NecessityVerdict.RETAIN


def test_B_decorative_term_removed():
    y = X**2 + E
    v = _verdicts({"x2": X**2, "debris": np.sqrt(np.abs(X))}, y)
    assert v["debris"] is NecessityVerdict.REMOVE
    assert v["x2"] is NecessityVerdict.RETAIN


def test_C_tiny_effect_borderline():
    y = X**2 + 0.01 * X**3 + E
    v = _verdicts({"x2": X**2, "x3": X**3}, y)
    assert v["x3"] is NecessityVerdict.BORDERLINE


def _ols_rmse(col, y):
    A = np.column_stack([np.ones(N), col])
    w, *_ = la.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((y - A @ w) ** 2)))


def test_D_collinear_pair_tied():
    y = X**2 + E
    cls = near_tie_class([_ols_rmse(X**2, y), _ols_rmse((2 * X) ** 2, y)], tol=1e-9)
    assert cls == [[0, 1]]


def _ols_fn(cols):
    def fit(Xtr, ytr):
        w, *_ = la.lstsq(np.column_stack([np.ones(len(Xtr))] + [Xtr[:, c] for c in cols]), ytr, rcond=None)
        return lambda Xo: np.column_stack([np.ones(len(Xo))] + [Xo[:, c] for c in cols]) @ w

    return fit


def test_E_ood_necessity():
    x = rng.uniform(-2, 2, 4000)
    y = x**2 + 0.3 * x**3 + rng.normal(0, 0.1, 4000)
    tr, ood = np.abs(x) < 1.0, np.abs(x) >= 1.0
    Xo = np.column_stack([x**2, x**3])
    r_without, r_with = ood_necessity(_ols_fn([0]), _ols_fn([0, 1]), Xo[tr], y[tr], Xo[ood], y[ood])
    assert r_with < 0.2 * r_without  # ~12x in E3b-E


def test_F_unstable_contract_fails():
    t = np.arange(N)
    y = X**2 + np.where(t < N // 2, 0.5, -0.5) * X**3 + E
    abls = ablate_terms({"x2": X**2, "x3": X**3}, y, HOLDOUT)
    base = float(np.sqrt(np.mean((y[HOLDOUT] - y[HOLDOUT].mean()) ** 2)))
    assert base > 1.0  # stationary forms miss by ~15x noise floor
    check = necessity_contract_check(abls, top_class_size=1)
    assert not check.passed or any(a.verdict is not NecessityVerdict.RETAIN for a in abls)


def test_ood_upgrade_small_holdout_large_ood():
    # E3b-E pattern: negligible in-domain, 12x OOD -> RETAIN (one-way ratchet)
    base = TermAblation(term="x3", delta_rmse=1e-4, rel_delta=1e-3,
                        coef_full=0.01, verdict=NecessityVerdict.BORDERLINE)
    up = apply_ood_upgrade(base, rmse_without_ood=1.29, rmse_with_ood=0.11)
    assert up.verdict is NecessityVerdict.RETAIN
    flat = apply_ood_upgrade(base, rmse_without_ood=0.10, rmse_with_ood=0.10)
    assert flat.verdict is NecessityVerdict.BORDERLINE
    assert ood_necessity_check(1.0, 1.0) is NecessityVerdict.BORDERLINE  # never demotes


def test_epsilon_ladder_monotone():
    # sensitivity sweep pin: guard response tracks effect size, never false-RETAINs zero
    y0 = X**2 + E
    v0 = {a.term: a.verdict for a in ablate_terms({"x2": X**2, "x3": X**3}, y0, HOLDOUT)}
    assert v0["x3"] is NecessityVerdict.REMOVE
    y1 = X**2 + 1.0 * X**3 + E
    v1 = {a.term: a.verdict for a in ablate_terms({"x2": X**2, "x3": X**3}, y1, HOLDOUT)}
    assert v1["x3"] is NecessityVerdict.RETAIN


def _mk(cid, terms, resid):
    c = CandidateDynamics(lhs="y", terms=terms, candidate_id=cid)
    return ScoredCandidate(
        c,
        {
            "gate4_status": {
                "passed": True,
                "holdout_residuals": list(resid),
                "holdout_rmse": 1.0,
                "holdout_obs_rms": 2.0,
            }
        },
        train_loss=1.0,
        holdout_rmse=1.0,
    )


def test_tie_invariance_identical_predictions():
    resid = rng.normal(0, 1, 200)
    base = [_mk("H1", (Term("x2", "a0"),), resid), _mk("H2", (Term("x2", "a0"),), resid), _mk("H3", (Term("x2", "a0"),), resid)]
    verdicts = {str(getattr(adjudicate(list(p), bootstrap=False, data_origin="sim"), "status", None)) for p in itertools.permutations(base)}
    assert len(verdicts) == 1


def test_representation_invariance():
    resid = rng.normal(0, 1, 200)
    simple = (Term("x2", "a0"),)
    decorated = (Term("x2", "a0"), Term("junk", "a1"), Term("junk2", "a2"))
    r1 = adjudicate([_mk("H_simple", simple, resid), _mk("H_decorated", decorated, resid)], bootstrap=False, data_origin="sim")
    r2 = adjudicate([_mk("Z_other", decorated, resid), _mk("A_other", simple, resid)], bootstrap=False, data_origin="sim")
    assert str(getattr(r1, "status", r1)) == str(getattr(r2, "status", r2))


def test_unique_contract_requires_retain_and_singleton():
    y = X**2 + E
    abls = ablate_terms({"x2": X**2, "debris": np.sqrt(np.abs(X))}, y, HOLDOUT)
    bad = necessity_contract_check(abls, top_class_size=1)
    assert not bad.passed and any("debris" in r for r in bad.reasons)
    tied = necessity_contract_check([a for a in abls if a.term == "x2"], top_class_size=3)
    assert not tied.passed and any("3 != 1" in r for r in tied.reasons)
    y2 = X**2 + 0.5 * X**3 + E
    good = necessity_contract_check(ablate_terms({"x2": X**2, "x3": X**3}, y2, HOLDOUT), top_class_size=1)
    assert good.passed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
