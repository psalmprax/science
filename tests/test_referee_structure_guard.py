"""Guard 3 of the referee: a promoted law must leave no structure in its holdout residuals over the input space."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.epistemic.adjudicate import Adjudication, EpistemicStatus
from apedp_dyn.epistemic.structure_guard import FLAG, guard_promotion
from apedp_dyn.epistemic.structure_guard import promotion_structure as _promotion_structure

RNG = np.random.default_rng(0)
X = RNG.uniform(1.0, 5.0, (400, 3))


def test_noise_passes_and_a_missing_term_does_not():
    noise = 0.05 * RNG.standard_normal(400)
    ok, why = _promotion_structure(noise, X)
    assert ok and "permutation p=" in why
    # a law that leaves out a smooth term: accurate on average, structured over the inputs, white in row order
    missing = noise + 0.2 * np.sin(X[:, 0])
    lag1 = abs(np.corrcoef(missing[:-1], missing[1:])[0, 1])
    assert lag1 < 0.2                                   # Guard 2 would see nothing: the rows have no order
    refused, reason = _promotion_structure(missing, X)
    assert not refused and "neighbour R^2=" in reason


def test_the_guard_is_deterministic_and_fails_closed():
    noise = 0.05 * np.random.default_rng(1).standard_normal(400)
    assert _promotion_structure(noise, X) == _promotion_structure(noise, X)
    assert _promotion_structure(None, X)[0] is False and _promotion_structure(noise, None)[0] is False
    assert _promotion_structure(noise[:100], X)[0] is False            # misaligned evidence
    assert _promotion_structure(noise[:10], X[:10])[0] is False        # too few rows
    assert _promotion_structure(np.zeros(400), X) == (True, "residuals are exactly zero: exact fit, no error left to test")
    assert _promotion_structure(np.full(400, 0.3), X)[0] is False      # a fixed offset is not an exact fit


def test_false_refusals_of_a_true_law_stay_near_the_preregistered_level():
    rng = np.random.default_rng(2)
    refused = sum(not _promotion_structure(rng.standard_normal(200), rng.uniform(0, 1, (200, 2)), permutations=199)[0]
                  for _ in range(150))
    assert refused <= 6                                  # the level is 1 %; six in 150 would be four times that
    assert DIEBOLD_MARIANO_P_VALUE_THRESHOLD == pytest.approx(0.01)


def _engine_result(structure_rule: bool):
    rng = np.random.default_rng(3)
    a, b = rng.uniform(1.0, 5.0, 600), rng.uniform(1.0, 5.0, 600)
    # no expression of the template engine's depth-1 grammar equals this; it must not promote the least bad one
    y = a * np.exp(-0.37 * b) + 0.3 * np.sin(2.0 * a)
    engine = L2DiscoveryEngine(operators=("+", "-", "*", "/"), max_depth=1, max_complexity=2)
    return engine.discover(X={"a": a, "b": b}, Y=y, dimensions={"a": Dimension(), "b": Dimension()},
                           target_dimension=Dimension(), seed=0, structure_rule=structure_rule)


def test_the_engine_does_not_promote_an_inadequate_law_when_the_guard_is_on():
    with_guard = _engine_result(True).adjudication
    assert with_guard.status is not EpistemicStatus.UNIQUE_SUPPORT and with_guard.promoted is None
    without = _engine_result(False).adjudication
    if without.status is EpistemicStatus.UNIQUE_SUPPORT:          # the behaviour the guard exists to stop
        assert "STRUCTURED_RESIDUALS_IN_INPUT_SPACE" in with_guard.diagnostic_flags


def test_the_guard_is_off_unless_asked_for():
    import inspect

    signature = inspect.signature(L2DiscoveryEngine.discover)
    assert signature.parameters["structure_rule"].default is False


def _pair_result(structure_rule: bool):
    rng = np.random.default_rng(5)
    a, b, c = (rng.uniform(1.0, 5.0, 800) for _ in range(3))
    # a*b beats every other product or quotient the grammar holds by a wide margin, and still misses sqrt(c)
    y = a * b * np.sqrt(c)
    engine = L2DiscoveryEngine(operators=("*", "/"), max_depth=2, max_complexity=3)
    dims = {"a": Dimension(), "b": Dimension(), "c": Dimension()}
    return engine.discover(X={"a": a, "b": b, "c": c}, Y=y, dimensions=dims, target_dimension=Dimension(), seed=0,
                           structure_rule=structure_rule)


def test_beating_one_rival_by_a_margin_is_not_enough_either():
    """The pair path: the better of two inadequate candidates is not promoted when the guard is on."""
    without = _pair_result(False).adjudication
    assert without.status is EpistemicStatus.UNIQUE_SUPPORT and without.rival is not None, without.reason
    with_guard = _pair_result(True).adjudication
    assert with_guard.status is EpistemicStatus.INSUFFICIENT_EVIDENCE and with_guard.promoted is None
    assert "STRUCTURED_RESIDUALS_IN_INPUT_SPACE" in with_guard.diagnostic_flags
    assert "better of two inadequate candidates" in with_guard.reason


class _Scored:
    """The two things the guard reads from a promoted candidate: its name and what gate 4 recorded."""

    def __init__(self, residuals, obs_rms):
        self.candidate = type("C", (), {"candidate_id": "a*b"})()
        self.provenance = {"gate4_status": {"holdout_residuals": residuals, "holdout_obs_rms": obs_rms}}


def test_the_guard_acts_on_the_verdict_so_no_route_to_a_promotion_escapes_it():
    """It is applied to what ``adjudicate`` returned, whichever of its exits that came from (audit M-10)."""
    truth = X[:, 0] * X[:, 1]
    missing_term = 0.4 * X[:, 2] ** 2
    obs_rms = float(np.sqrt(np.mean(truth**2)))
    for rival in (None, object()):                       # the sole survivor, and the winner of a pair
        promoted = Adjudication(EpistemicStatus.UNIQUE_SUPPORT, promoted=_Scored(missing_term, obs_rms), rival=rival,
                                best_rmse=1.25, reason="as adjudicated", diagnostic_flags=("EARLIER",))
        refused = guard_promotion(promoted, X)
        assert refused.status is EpistemicStatus.INSUFFICIENT_EVIDENCE and refused.promoted is None
        assert refused.diagnostic_flags == (FLAG, "EARLIER") and refused.best_rmse == 1.25
        assert "as adjudicated" in refused.reason                    # the original verdict stays readable
    noise = Adjudication(EpistemicStatus.UNIQUE_SUPPORT, promoted=_Scored(RNG.normal(0, 0.1, 400), obs_rms))
    assert guard_promotion(noise, X) is noise
    for status in EpistemicStatus:                        # a verdict that promotes nothing is never touched
        if status is not EpistemicStatus.UNIQUE_SUPPORT:
            verdict = Adjudication(status)
            assert guard_promotion(verdict, None) is verdict


def test_a_promotion_without_the_evidence_to_test_it_is_refused():
    promoted = Adjudication(EpistemicStatus.UNIQUE_SUPPORT, promoted=_Scored(None, 1.0))
    assert guard_promotion(promoted, X).status is EpistemicStatus.INSUFFICIENT_EVIDENCE
    promoted = Adjudication(EpistemicStatus.UNIQUE_SUPPORT, promoted=_Scored(RNG.normal(0, 0.1, 400), 1.0))
    assert guard_promotion(promoted, None).status is EpistemicStatus.INSUFFICIENT_EVIDENCE


def test_the_referee_file_itself_is_unchanged_by_the_guard():
    """The guard wraps ``adjudicate``; a copy of it inside would need the freeze record re-minted."""
    import sys

    source = Path(sys.modules["apedp_dyn.epistemic.adjudicate"].__file__).read_text(encoding="utf-8")
    assert "structure_rule" not in source and "structure_guard" not in source


def test_rounding_error_on_noise_free_data_is_an_exact_fit_not_a_missing_term():
    """Development found it: at noise 0 the guard refused 4 of 7 correct laws for the structure of their rounding error."""
    a, b = X[:, 0], X[:, 1]
    truth = a * b
    rounding = truth * 3e-16                      # a coefficient one unit in the last place off: follows the prediction
    obs_rms = float(np.sqrt(np.mean(truth**2)))
    refused, _ = _promotion_structure(rounding, X)             # without the scale of the observations it cannot know
    assert not refused
    ok, why = _promotion_structure(rounding, X, obs_rms=obs_rms)
    assert ok and "exact fit up to rounding" in why
    # the allowance is for rounding only: an error of one part in a million with the same shape is still refused
    ok, _ = _promotion_structure(truth * 1e-6, X, obs_rms=obs_rms)
    assert not ok


def test_the_engine_promotes_the_true_law_on_noise_free_data_with_the_guard_on():
    rng = np.random.default_rng(11)
    a, b, c = (rng.uniform(1.0, 5.0, 600) for _ in range(3))
    dims = {"a": Dimension(), "b": Dimension(), "c": Dimension()}
    engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"), max_depth=2, max_complexity=5)
    adj = engine.discover(X={"a": a, "b": b, "c": c}, Y=6.2832 * a * b / c, dimensions=dims, target_dimension=Dimension(),
                          seed=0, structure_rule=True).adjudication
    assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj.reason
    assert adj.promoted.candidate.metadata["canonical_expr"] in ("a*b/c", "(a*b)/c", "a*(b/c)"), adj.promoted.candidate.metadata


def test_structure_under_heavier_noise_is_seen_at_a_larger_neighbourhood():
    """Development: with eight neighbours alone a wrong law with test R² 0.990 passed at 10 % noise (p = 0.17)."""
    rng = np.random.default_rng(5)
    inputs = rng.uniform(1.0, 5.0, (500, 4))
    smooth = 0.1 * (inputs[:, 0] - 3.0) * (inputs[:, 3] - 3.0)            # a missing term, smooth over the inputs
    noisy = smooth + rng.normal(0.0, 3.0 * np.std(smooth), 500)            # under noise three times its size
    assert _promotion_structure(noisy, inputs, scales=(8,))[0]             # eight neighbours alone do not see it
    ok, why = _promotion_structure(noisy, inputs)
    assert not ok and "k=64" in why, why


def test_the_level_holds_with_several_neighbourhood_sizes():
    """Taking the largest of five statistics must not refuse a true law five times as often."""
    rng = np.random.default_rng(22)
    refused = 0
    for _ in range(200):
        inputs = rng.uniform(1.0, 5.0, (160, 3))
        refused += not _promotion_structure(rng.normal(0.0, 1.0, 160), inputs, permutations=199)[0]
    assert refused <= 7, refused          # the level is 0.01; 7 of 200 is the 99.9th percentile of that


def test_neighbourhoods_larger_than_a_quarter_of_the_sample_are_left_out():
    rng = np.random.default_rng(23)
    inputs = rng.uniform(1.0, 5.0, (40, 2))
    _, why = _promotion_structure(rng.normal(0.0, 1.0, 40), inputs)
    assert "k=[4, 8]" in why
    assert _promotion_structure(rng.normal(0.0, 1.0, 12), inputs[:12])[0] is False
