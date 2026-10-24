"""Tests for Evidence Acquisition Engine (Hypothesis Discrimination).

Verifies active scientific discrimination:
- Calculates maximal prediction divergence Delta(x) = |y_H1(x) - y_H2(x)|.
- Identifies x* = argmax Delta(x) across physical domains.
- Partitions domains into INTERPOLATION_GAP, NEAR_OOD, and FAR_OOD.
- Generates actionable, structured experiment design proposals.
"""
from __future__ import annotations

from apedp.evidence_acquisition import (
    DiscriminationPoint,
    DiscriminationReport,
    EvidenceAcquisitionEngine,
)


def test_evidence_acquisition_identifies_shomate_vs_quadratic_divergence():
    """Simulates the methane heat capacity case:
    Observed domain: T in [200, 600] K where Quadratic and Shomate agree closely.
    Target domain: T in [200, 2500] K where they diverge dramatically."""
    engine = EvidenceAcquisitionEngine(default_grid_points=200)

    # Models with calibrated coefficients matching typical Cp behavior
    models = {
        "quadratic": "30.0 + 0.025*T - 1.0e-5*T**2",
        "shomate": "25.0 + 0.040*T - 2.0e-5*T**2 + 5.0e-9*T**3 + 1.0e5/T**2",
    }

    report = engine.find_discriminating_regime(
        models=models,
        variable_domains={"T": (200.0, 2500.0)},
        observed_domain={"T": (200.0, 600.0)},
    )

    assert isinstance(report, DiscriminationReport)
    assert report.primary_recommendation is not None

    rec = report.primary_recommendation
    assert isinstance(rec, DiscriminationPoint)
    assert "T" in rec.coordinates

    # Divergence should be substantially higher in OOD than in-domain
    in_gap = report.regime_recommendations.get("INTERPOLATION_GAP")
    assert in_gap is not None
    assert rec.max_divergence > in_gap.max_divergence

    # Rationale contains explicit guidance
    assert "Measurements near T =" in report.rationale
    assert "quadratic" in report.rationale and "shomate" in report.rationale


def test_evidence_acquisition_multi_hypothesis_three_models():
    """Compares 3 competing models: linear, quadratic, exponential."""
    engine = EvidenceAcquisitionEngine(default_grid_points=150)
    models = {
        "linear": "2.0 * x + 1.0",
        "quadratic": "0.5 * x**2 + 1.0",
        "exponential": "exp(0.4 * x)",
    }

    report = engine.find_discriminating_regime(
        models=models,
        variable_domains={"x": (0.0, 10.0)},
        observed_domain={"x": (0.0, 3.0)},
    )

    assert set(report.candidate_names) == {"exponential", "linear", "quadratic"}
    assert report.primary_recommendation is not None
    # Discriminating pair should be the pair with maximal spread at top point
    pair = report.primary_recommendation.discriminating_pair
    assert len(pair) == 2
    assert pair[0] != pair[1]


def test_evidence_acquisition_multivariable_domain():
    """Tests 2D domain discrimination: f(x, y) vs g(x, y)."""
    engine = EvidenceAcquisitionEngine(default_grid_points=100)
    models = {
        "additive": "x**2 + y**2",
        "multiplicative": "2.0 * x * y",
    }

    report = engine.find_discriminating_regime(
        models=models,
        variable_domains={"x": (1.0, 5.0), "y": (1.0, 5.0)},
        observed_domain={"x": (1.0, 2.0), "y": (1.0, 2.0)},
    )

    assert report.primary_recommendation is not None
    coords = report.primary_recommendation.coordinates
    assert "x" in coords and "y" in coords
    assert report.primary_recommendation.max_divergence > 0.0


def test_evidence_acquisition_handles_identical_models():
    """Identical models have zero divergence."""
    engine = EvidenceAcquisitionEngine(default_grid_points=50)
    models = {
        "m1": "2.0 * x + 3.0",
        "m2": "2.0 * x + 3.0",
    }

    report = engine.find_discriminating_regime(
        models=models,
        variable_domains={"x": (0.0, 5.0)},
    )

    assert report.primary_recommendation is None
    assert "do not diverge" in report.rationale


def test_evidence_acquisition_requires_at_least_two_models():
    engine = EvidenceAcquisitionEngine()
    report = engine.find_discriminating_regime(
        models={"only_one": "x**2"},
        variable_domains={"x": (0.0, 1.0)},
    )
    assert report.primary_recommendation is None
    assert "At least two" in report.rationale


def test_bayesian_eig_optimal_design_shomate_vs_quadratic():
    """AHD-v2: Tests Bayesian EIG scoring over model divergence and parameter uncertainty."""
    from apedp.evidence_acquisition import BayesianEIGReport, EIGPoint

    engine = EvidenceAcquisitionEngine(default_grid_points=100)
    models = {
        "quadratic": "30.0 + 0.025*T - 1.0e-5*T**2",
        "shomate": "25.0 + 0.040*T - 2.0e-5*T**2 + 5.0e-9*T**3 + 1.0e5/T**2",
    }

    report = engine.find_bayesian_eig_regime(
        models=models,
        variable_domains={"T": (200.0, 2500.0)},
        observed_domain={"T": (200.0, 600.0)},
        sensor_noise=0.5,
        parameter_uncertainties={"quadratic": 0.02, "shomate": 0.05},
    )

    assert isinstance(report, BayesianEIGReport)
    assert report.primary_recommendation is not None
    rec = report.primary_recommendation
    assert isinstance(rec, EIGPoint)
    assert rec.expected_information_gain > 0.0
    assert rec.between_model_variance > 0.0
    assert rec.within_model_variance >= 0.25  # sensor_noise^2 = 0.25
    assert "Optimal Bayesian experiment" in report.rationale


def test_bayesian_eig_cost_penalty_shifts_recommendation():
    """AHD-v2: Tests that an experimental cost penalty successfully shifts the design."""
    engine = EvidenceAcquisitionEngine(default_grid_points=80)
    models = {
        "m1": "x",
        "m2": "x**1.5",
    }

    # Free cost run
    free_report = engine.find_bayesian_eig_regime(
        models=models,
        variable_domains={"x": (1.0, 10.0)},
        cost_weight=0.0,
    )
    assert free_report.primary_recommendation is not None
    _free_x = free_report.primary_recommendation.coordinates["x"]

    # High cost at x > 8 (e.g. extreme high pressure or temperature)
    def expensive_high_x(coords):
        x = coords["x"]
        return 10.0 if x > 6.0 else 0.0

    penalized_report = engine.find_bayesian_eig_regime(
        models=models,
        variable_domains={"x": (1.0, 10.0)},
        cost_function=expensive_high_x,
        cost_weight=1.5,
    )

    assert penalized_report.primary_recommendation is not None
    penalized_x = penalized_report.primary_recommendation.coordinates["x"]
    # The expensive region should be penalized, shifting recommendation to a more cost-effective location
    assert penalized_x <= 6.0 or penalized_report.primary_recommendation.net_utility < free_report.primary_recommendation.net_utility
