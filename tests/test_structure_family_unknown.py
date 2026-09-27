"""Structure Family Unknown Benchmark: Blind Recovery Across Distinct Mathematical Classes.

Evaluates whether APEDP's AutonomousModelSynthesizer can autonomously identify,
compose, and estimate the governing physical law from observations alone,
without being given any structural formula templates, hints, or prompt priors.

Tests five fundamental physical classes:
1. Pure Power Law: y = a * x^p
2. Logarithmic Boundary Layer: y = a * log(1 + b * x^c)
3. Exponential Relaxation: y = a * exp(-b * x)
4. Saturation Rational: y = (a * x^p) / (1 + b * x^q)
5. Composite Envelope: y = (a * x^p) / (1 + b * x^q) * exp(-c * x)
"""
# RECONSTRUCTION NOTE: this file was deleted from the worktree by a concurrent
# session and restored here from bytecode forensics
# (tests/__pycache__/test_structure_family_unknown.cpython-312-pytest-9.1.1.pyc):
# every seed, range, generating law, assertion threshold, family expectation,
# and the xfail marker below was recovered from compiled constants. Assertion
# ORDER within a test is conventional (len -> val -> family -> expression ->
# params -> epistemic) since bytecode does not preserve it; all assertions are
# behavior-equivalent under any order. If any backup of the original surfaces,
# diff against it.
from __future__ import annotations

import numpy as np
import pytest

from apedp.decision.stage_c import EpistemicState
from apedp.model_synthesis import AutonomousModelSynthesizer


@pytest.fixture
def synthesizer() -> AutonomousModelSynthesizer:
    return AutonomousModelSynthesizer()


def test_blind_recovery_pure_power_law(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of pure scaling law: y = 2.5 * x^1.5."""
    rng = np.random.default_rng(101)
    x_tr = np.linspace(0.5, 4.0, 40)
    y_tr = 2.5 * (x_tr ** 1.5) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 3.8, 20)
    y_val = 2.5 * (x_val ** 1.5)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family == "power_law"
    assert "x**1.5" in top.expression_str or "x**1.50" in top.expression_str
    # Parameter A ~ 2.5
    assert abs(top.parameters.get("A", 0.0) - 2.5) < 0.1
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_logarithmic_boundary_layer(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of logarithmic boundary transport: y = 3.0 * log(1 + 2.0 * x)."""
    rng = np.random.default_rng(202)
    x_tr = np.linspace(0.5, 5.0, 40)
    y_tr = 3.0 * np.log(1.0 + 2.0 * x_tr) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 4.8, 20)
    y_val = 3.0 * np.log(1.0 + 2.0 * x_val)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family in {"logarithmic_boundary_layer", "rational_transition"}
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_exponential_relaxation(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of exponential dissipation: y = 4.0 * exp(-0.5 * x)."""
    rng = np.random.default_rng(303)
    x_tr = np.linspace(0.5, 5.0, 40)
    y_tr = 4.0 * np.exp(-0.5 * x_tr) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 4.8, 20)
    y_val = 4.0 * np.exp(-0.5 * x_val)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert "exp" in top.expression_str.lower()
    assert top.structural_family in {"exponential_decay", "rational_exponential_envelope"}
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_saturation_rational(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of Michaelis-Menten / Hill saturation: y = (3.0 * x^1.5) / (1 + 1.2 * x^1.5)."""
    rng = np.random.default_rng(404)
    x_tr = np.linspace(0.5, 4.5, 40)
    y_tr = 3.0 * (x_tr ** 1.5) / (1.0 + 1.2 * (x_tr ** 1.5)) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 4.3, 20)
    y_val = 3.0 * (x_val ** 1.5) / (1.0 + 1.2 * (x_val ** 1.5))

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family == "rational_transition"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_composite_envelope(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of rational-exponential transition: y = (2.45 * x^1.5 / (1 + 0.85 * x^2)) * exp(-0.3 * x)."""
    rng = np.random.default_rng(505)
    x_tr = np.linspace(0.5, 5.0, 45)
    y_tr = (
        2.45 * (x_tr ** 1.5) / (1.0 + 0.85 * (x_tr ** 2.0)) * np.exp(-0.3 * x_tr)
        + rng.normal(0, 0.015, len(x_tr))
    )
    x_val = np.linspace(0.6, 4.8, 25)
    y_val = 2.45 * (x_val ** 1.5) / (1.0 + 0.85 * (x_val ** 2.0)) * np.exp(-0.3 * x_val)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family == "rational_exponential_envelope"
    assert "exp" in top.expression_str.lower()
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_power_law_unseen_exponent(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of unseen power law: y = 3.0 * x^1.7."""
    rng = np.random.default_rng(606)
    x_tr = np.linspace(0.5, 4.0, 40)
    y_tr = 3.0 * (x_tr ** 1.7) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 3.8, 20)
    y_val = 3.0 * (x_val ** 1.7)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family == "power_law"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_exponential_unseen_rate(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of unseen exponential rate: y = 4.0 * exp(-0.37 * x)."""
    rng = np.random.default_rng(707)
    x_tr = np.linspace(0.5, 5.0, 40)
    y_tr = 4.0 * np.exp(-0.37 * x_tr) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 4.8, 20)
    y_val = 4.0 * np.exp(-0.37 * x_val)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert "exp" in top.expression_str.lower()
    assert top.structural_family in {"exponential_decay", "rational_exponential_envelope"}
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


# Unknown-exponent blind 808 exposes a real engine limit: off-grid rational
# exponents resolve to a neighbouring envelope family at overlapping validation
# MSE (36% outside the 15% parsimony-tiebreak margin). Marked xfail to document
# the boundary without breaking the suite. (606 power-law off-grid exponent was
# fixed by Stage D + parsimony tiebreak; 909 damped off-grid frequency by Stage D;
# both promoted to strict tests.)
@pytest.mark.xfail(reason="off-grid rational exponents gain spurious exp envelope; needs continuous refinement")
def test_blind_recovery_rational_unseen_exponents(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of unseen rational: y = (2.0 * x^1.3) / (1 + 0.7 * x^2.1)."""
    rng = np.random.default_rng(808)
    x_tr = np.linspace(0.5, 4.5, 40)
    y_tr = 2.0 * (x_tr ** 1.3) / (1.0 + 0.7 * (x_tr ** 2.1)) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.6, 4.3, 20)
    y_val = 2.0 * (x_val ** 1.3) / (1.0 + 0.7 * (x_val ** 2.1))

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert top.structural_family == "rational_transition"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}


def test_blind_recovery_damped_oscillation_unseen(synthesizer: AutonomousModelSynthesizer):
    """Blind recovery of unseen damped oscillation: y = 2.0 * exp(-0.2 * x) * cos(2.7 * x)."""
    rng = np.random.default_rng(909)
    x_tr = np.linspace(0.0, 6.0, 60)
    y_tr = 2.0 * np.exp(-0.2 * x_tr) * np.cos(2.7 * x_tr) + rng.normal(0, 0.015, len(x_tr))
    x_val = np.linspace(0.05, 5.95, 30)
    y_val = 2.0 * np.exp(-0.2 * x_val) * np.cos(2.7 * x_val)

    models = synthesizer.synthesize(x_tr, y_tr, x_val, y_val)
    assert len(models) > 0, "No models synthesized"
    top = models[0]

    assert top.val_mse < 1e-4, f"Validation MSE too high: {top.val_mse}"
    assert "cos" in top.expression_str.lower()
    assert top.structural_family == "damped_oscillation"
    assert top.epistemic_state in {EpistemicState.UNIQUE_SUPPORT, EpistemicState.MULTI_HYPOTHESIS}
