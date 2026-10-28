"""The audited metric placeholders: each now measures, or refuses.

`EnsembleSearcher.get_diversity_score` returned 0.8, `DeepEnsemble.predict`
reported a zero aleatoric term, and `PhysicsConstraintEngine.dimensional_analysis_score`
returned 0.5 or 1.0 depending only on whether a target had been set. None of the
three looked at the thing it claimed to measure. These tests pin the replacement
contract: a real number when there is data to measure, ``ValueError`` when there is
not.
"""
from __future__ import annotations

import pytest

from apedp.advanced_search import Dimensions, PhysicsConstraintEngine
from apedp.ensemble_search import EnsembleSearcher


def _python_engine() -> PhysicsConstraintEngine:
    engine = PhysicsConstraintEngine()
    engine._rust = None  # exercise the Python formula regardless of the kernel
    return engine


# -- ensemble diversity ----------------------------------------------------

def test_ensemble_diversity_refuses_a_plan_with_no_solutions():
    frame = None
    search = EnsembleSearcher()
    plan = search.search_ensemble(frame, ["x"], "target", budget_total=30)
    with pytest.raises(ValueError, match="at least two reported solutions"):
        search.get_diversity_score(plan["results"])


def test_ensemble_diversity_is_the_distinct_fraction():
    search = EnsembleSearcher()

    identical = {"a": {"status": "ok", "expression": "m*x"},
                 "b": {"status": "ok", "expression": "m*x"}}
    divergent = {"a": {"best": {"expression": "m*x"}},
                 "b": {"equations": ["g*t**2"]}}
    partial = {"a": "m*x", "b": "m*x", "c": "k*x"}

    assert search.get_diversity_score(identical) == 0.0
    assert search.get_diversity_score(divergent) == 1.0
    assert search.get_diversity_score(partial) == pytest.approx(0.5)


# -- dimensional analysis score -------------------------------------------

def test_dimensional_score_refuses_without_terms():
    engine = PhysicsConstraintEngine()
    engine.set_target(Dimensions(mass=1))
    with pytest.raises(ValueError, match="no terms"):
        engine.dimensional_analysis_score()
    with pytest.raises(ValueError, match="no terms"):
        engine.dimensional_analysis_score([])


def test_dimensional_score_refuses_without_a_target():
    engine = PhysicsConstraintEngine()
    with pytest.raises(ValueError, match="target"):
        engine.dimensional_analysis_score([Dimensions(mass=1)])


@pytest.mark.parametrize("use_rust", [False, True])
def test_dimensional_score_is_the_matching_fraction(use_rust):
    engine = PhysicsConstraintEngine()
    if not use_rust:
        engine._rust = None
    engine.set_target(Dimensions(mass=1, time=-2))
    terms = [Dimensions(mass=1, time=-2), Dimensions(mass=1, time=-2),
             Dimensions(mass=1)]
    assert engine.dimensional_analysis_score(terms) == pytest.approx(2 / 3)


@pytest.mark.parametrize("use_rust", [False, True])
def test_dimensional_consistency_compares_every_exponent(use_rust):
    engine = PhysicsConstraintEngine()
    if not use_rust:
        engine._rust = None
    engine.set_target(Dimensions(mass=1))
    # A mismatch outside the mass/length/time triple used to pass in the Python path.
    assert not engine.check_dimensional_consistency(Dimensions(mass=1, temperature=1))
    assert engine.check_dimensional_consistency(Dimensions(mass=1))


# -- aleatoric uncertainty -------------------------------------------------

def test_deep_ensemble_does_not_report_unmeasured_aleatoric():
    import jax

    from apedp.neural_discovery.architectures import MLPResNet, init_model
    from apedp.neural_discovery.config import NeuralDiscoveryConfig
    from apedp.neural_discovery.uncertainty import DeepEnsemble

    config = NeuralDiscoveryConfig(
        architecture="mlp_resnet", hidden_dims=(8,), num_layers=1, ensemble_size=2)
    def model_fn():
        return MLPResNet(hidden_dims=(8,), num_layers=1, output_dim=1)
    ensemble = DeepEnsemble(model_fn, config)

    x = jax.numpy.ones((4, 3))
    key = jax.random.PRNGKey(0)
    for index in range(2):
        ensemble.models.append(model_fn())
        ensemble.params_list.append(
            init_model(ensemble.models[-1], x.shape, jax.random.fold_in(key, index), config))

    output = ensemble.predict(x)
    assert output.aleatoric is None, "a deterministic regressor must not report measured noise"
    assert output.epistemic is not None
