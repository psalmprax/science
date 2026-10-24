"""Grammar-reachability pins: saturating rational laws (Langmuir / Hill).

These pin the *reachability* half of the claim: the proposer's emitted set must
contain the textbook saturating form with a fitted half-saturation constant, and
must recover its parameters, while a power law must not be crowned as one. The
native rational-pair enumerator cannot express these laws (its constants are
DAG leaves, i.e. integers), which is why the family lives in representation 8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer


def _langmuir(p, q_max, k):
    return q_max * k * p / (1.0 + k * p)


def _propose(p, q):
    return RimeAdProposer(target_col="q").fit_and_propose(
        pd.DataFrame({"p": np.asarray(p, float), "q": np.asarray(q, float)}))


def _by_id(props, candidate_id):
    return next((c for c in props if c.candidate_id == candidate_id), None)


def test_langmuir_is_reachable_and_parameters_recover_exactly():
    p = np.linspace(0.1, 2.5, 11)
    q = _langmuir(p, 40.0, 1.6)  # c = 1/K = 0.625
    cand = _by_id(_propose(p, q), "saturating_rational")
    assert cand is not None, "grammar cannot emit the Langmuir family"
    assert abs(cand.parameters["q_max"] - 40.0) < 1e-6
    assert abs(cand.parameters["half_saturation"] - 0.625) < 1e-6
    assert abs(cand.parameters["offset"]) < 1e-6
    # The emitted score is the closed form, on points it never saw.
    p_new = pd.DataFrame({"p": np.linspace(0.05, 5.0, 37)})
    np.testing.assert_allclose(
        np.asarray(cand.compute_score(p_new), float).ravel(),
        _langmuir(p_new["p"].to_numpy(float), 40.0, 1.6),
        rtol=1e-9, atol=1e-9,
    )


def test_langmuir_survives_observation_noise():
    rng = np.random.default_rng(7)
    p = np.linspace(0.1, 3.0, 24)
    q = _langmuir(p, 12.0, 0.9) * (1.0 + 0.02 * rng.standard_normal(p.size))
    cand = _by_id(_propose(p, q), "saturating_rational")
    assert cand is not None
    assert cand.metadata["r2"] > 0.99
    assert abs(cand.parameters["q_max"] - 12.0) / 12.0 < 0.05
    # q_max and c are correlated under noise (moving either shifts the knee);
    # 2% observation noise moves c by ~8% while the fitted law stays R2 > 0.99.
    assert abs(cand.parameters["half_saturation"] - 1.0 / 0.9) / (1.0 / 0.9) < 0.10


def test_cooperative_hill_form_is_reachable():
    p = np.linspace(0.05, 6.0, 40)
    q = 3.5 * p ** 2 / (2.0 ** 2 + p ** 2) + 0.4
    cand = _by_id(_propose(p, q), "hill_rational")
    assert cand is not None, "cooperative Hill n=2 is outside the emitted set"
    assert abs(cand.parameters["half_saturation"] - 2.0) < 0.05
    assert abs(cand.parameters["offset"] - 0.4) < 0.02


def test_power_law_is_not_claimed_as_a_saturating_law():
    """A concave-but-unbounded law must not be laundered into a plateau."""
    p = np.geomspace(0.01, 100.0, 60)
    q = np.sqrt(p)
    props = _propose(p, q)
    for cid in ("saturating_rational", "hill_rational"):
        cand = _by_id(props, cid)
        if cand is not None:
            assert cand.metadata["r2"] < 0.99, (
                f"{cid} falsely fits a power law at R2={cand.metadata['r2']}")


def test_constant_target_emits_no_saturating_candidate():
    p = np.linspace(0.1, 2.0, 12)
    props = _propose(p, np.full_like(p, 3.0))
    assert _by_id(props, "saturating_rational") is None
    assert _by_id(props, "hill_rational") is None


def test_emitted_formula_is_parser_evaluable():
    p = np.linspace(0.1, 2.5, 11)
    q = _langmuir(p, 40.0, 1.6)
    cand = _by_id(_propose(p, q), "saturating_rational")
    assert cand is not None
    grid = np.linspace(0.2, 4.0, 21)
    parsed = RustKernelBridge().evaluate_expression(cand.formula, ["p"], [grid])
    scored = np.asarray(cand.compute_score(pd.DataFrame({"p": grid})), float).ravel()
    np.testing.assert_allclose(parsed, scored, rtol=1e-5, atol=1e-5)


def test_saturating_candidate_stays_in_the_emitted_set_at_pin_budget():
    """BIC may rank a flexible rival first; reachability must not depend on it."""
    p = np.linspace(0.1, 2.5, 11)
    q = _langmuir(p, 40.0, 1.6)
    props = _propose(p, q)
    ids = [c.candidate_id for c in props]
    assert "saturating_rational" in ids
    assert len(props) >= 2
