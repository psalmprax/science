"""The refined path on vector fields: a polynomial library as a candidate, and nothing reported without a search stage."""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.ode import refined_ode
from apedp_dyn.ode.window_rows import WindowRows, window_rows

T = np.linspace(0.0, 10.0, 10_000)


def _duffing(_t: float, x: np.ndarray) -> list[float]:
    return [x[1], -0.3 * x[1] - 1.2 * x[0] - 0.8 * x[0] ** 3]


def _rows(init: list[float], noise: float = 0.0, seed: int = 0) -> WindowRows:
    clean = solve_ivp(_duffing, (0.0, 10.0), init, t_eval=T, method="DOP853", rtol=1e-12, atol=1e-14).y.T
    rng = np.random.default_rng(seed)
    return window_rows(T, clean + noise * clean.std(axis=0) * rng.standard_normal(clean.shape))


def _dev_and_pool(noise: float = 0.0) -> tuple[WindowRows, WindowRows]:
    a, b, pool = _rows([1.5, 0.0], noise, 0), _rows([-0.7, 1.1], noise, 1), _rows([0.8, 0.4], noise, 2)
    dev = WindowRows(np.vstack([a.states, b.states]), np.vstack([a.rates, b.rates]), np.concatenate([a.times, b.times]),
                     a.degree, np.maximum(a.fit_rms, b.fit_rms), np.maximum(a.rate_sd, b.rate_sd))
    return dev, pool


class _Scripted(ProposerBase):
    """Stands in for PySR: a search stage that proposes fixed formulas."""

    def __init__(self, formulas):
        self.formulas = formulas

    @property
    def proposer_id(self) -> str:
        return "scripted"

    def fit_and_propose(self, train_data, target_col=None, guesses=None, **kwargs):
        return tuple(CandidateProposal(proposer_id="scripted", candidate_id=f"s{i}", formula=f, strategy_archetype=None,
                                       declared_scale_invariant=False) for i, f in enumerate(self.formulas))


def _same(law: str, truth: str) -> bool:
    x0, x1 = sp.symbols("x0 x1")
    found, wanted = sp.sympify(law, locals={"x0": x0, "x1": x1}), sp.sympify(truth, locals={"x0": x0, "x1": x1})
    terms = lambda e: {t.as_coeff_Mul()[1]: float(t.as_coeff_Mul()[0]) for t in sp.Add.make_args(sp.expand(e))}
    a, b = terms(found), terms(wanted)
    return set(a) == set(b) and all(abs(a[k] - b[k]) < 0.02 * max(1.0, abs(b[k])) for k in b)


def test_the_polynomial_library_finds_the_duffing_field_and_a_search_stage_makes_it_reportable():
    dev, pool = _dev_and_pool()
    results = refined_ode.discover_system(dev, pool, proposer_factory=lambda: _Scripted(["x0 + x1", "x0*x1", "sin(x0)"]))
    assert results[0].law and _same(results[0].law, "x1")
    assert results[1].law and _same(results[1].law, "-0.3*x1 - 1.2*x0 - 0.8*x0**3")
    assert all(r.tentative_law is None for r in results)


def test_without_a_search_stage_the_law_is_tentative():
    """Audit M-19: at 1 % noise the first passes alone 'confirmed' wrong laws on the development systems."""
    dev, pool = _dev_and_pool()
    result = refined_ode.discover_component(dev, pool, 1)
    assert result.law is None and result.tentative_law
    assert _same(result.tentative_law, "-0.3*x1 - 1.2*x0 - 0.8*x0**3")


def test_the_reported_law_carries_no_term_the_data_do_not_notice():
    """Audit M-15: the frozen ODE route reported -x0**3 plus terms of size 1e-8 that explode outside the data."""
    dev, pool = _dev_and_pool()
    bloated = "-0.3*x1 - 1.2*x0 - 0.8*x0**3 + 1e-9*exp(exp(x0))"
    result = refined_ode.discover_component(dev, pool, 1, proposer=_Scripted([bloated, "x0*x1"]))
    law = result.law or result.tentative_law
    assert law and "exp" not in law
    assert _same(law, "-0.3*x1 - 1.2*x0 - 0.8*x0**3")
