"""Tests for Tri-Hypothesis End-to-End Pipeline Integration in DiscoveryPipeline."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.decision.stage_c import EpistemicState, ModelClass
from apedp.neural_discovery.config import NeuralDiscoveryConfig
from apedp.pipeline import DiscoveryPipeline


@pytest.fixture
def fast_neural_config():
    return NeuralDiscoveryConfig(
        max_epochs=20,
        uncertainty_method="dropout",
        num_layers=2,
        hidden_dims=(32, 32),
        dropout_rate=0.05,
        learning_rate=1e-2,
    )


def test_pipeline_tri_hypothesis_symbolic_dominance(tmp_path, fast_neural_config):
    """When a clean quadratic law generated the data, symbolic candidate matches
    neural ceiling, has low complexity, and is uniquely supported."""
    rng = np.random.default_rng(42)
    x = np.linspace(1.0, 5.0, 50)
    # Ground truth: y = 2.0*x**2 + 1.0
    y = 2.0 * (x**2) + 1.0 + rng.normal(0, 0.05, size=len(x))

    data = {"x": x, "y": y}
    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "tri_results"))

    # Supply explicit hypotheses: candidate quadratic vs an underfitting linear
    res = pipeline.run_scientific_investigation(
        data=data,
        target_column="y",
        hypotheses_to_evaluate=[
            {"candidate_id": "quad_sym", "expression": "2.0*x**2 + 1.0", "complexity": 3.0},
            {"candidate_id": "linear_sym", "expression": "12.0*x - 10.0", "complexity": 2.0},
        ],
        neural_config=fast_neural_config,
        include_neural=True,
        include_hybrid=False,
    )

    assert res["status"] == "COMPLETED"
    decision = res["decision"]
    assert decision["winner"] == "quad_sym"
    assert decision["state"] == EpistemicState.UNIQUE_SUPPORT.value
    assert len(res["candidates"]) >= 2
    
    # Check candidate profiles
    cand_map = {c["candidate_id"]: c for c in res["candidates"]}
    assert "neural_adversary" in cand_map
    assert "quad_sym" in cand_map
    assert cand_map["neural_adversary"]["model_class"] == ModelClass.NEURAL.value
    assert cand_map["quad_sym"]["model_class"] == ModelClass.SYMBOLIC.value


def test_pipeline_tri_hypothesis_multi_hypothesis_and_active_discrimination(tmp_path, fast_neural_config):
    """When two admissible hypotheses are observationally close in the training regime,
    Stage C emits MULTI_HYPOTHESIS and computes active discriminating measurement x*."""
    # Narrow regime x in [0.1, 0.5]:
    # y is constructed as the midpoint between H1 and H2 so their holdout errors are identical
    x = np.linspace(0.1, 0.5, 40)
    y = 10.0 + 1.975 * x + 0.05 * (x**2)

    data = {"x": x, "y": y}
    pipeline = DiscoveryPipeline(output_dir=str(tmp_path / "multi_results"))

    res = pipeline.run_tri_hypothesis(
        data=data,
        target_column="y",
        hypotheses_to_evaluate=[
            {"candidate_id": "h1_linear", "expression": "10.0 + 2.0*x", "complexity": 2.0},
            {"candidate_id": "h2_quad", "expression": "10.0 + 1.95*x + 0.1*x**2", "complexity": 3.0},
        ],
        holdout_margin_ratio=1.20,
        include_neural=False,
        include_hybrid=False,
        enable_active_discrimination=True,
    )

    decision = res["decision"]
    assert decision["state"] == EpistemicState.MULTI_HYPOTHESIS.value
    assert decision["discrimination_required"] is True
    assert set(decision["supported_hypotheses"]) == {"h1_linear", "h2_quad"}

    # Active discrimination check
    ahd = res["active_discrimination"]
    assert ahd is not None
    assert "primary_recommendation" in ahd
    primary = ahd["primary_recommendation"]
    assert primary is not None
    x_star = primary["coordinates"]["x"]
    # The optimal measurement should push towards the domain boundary where H1 and H2 diverge most
    assert 0.1 <= x_star <= 0.5
    assert "bayesian_eig" in ahd
    assert "maximal_divergence" in ahd
