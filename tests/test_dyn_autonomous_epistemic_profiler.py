"""Tests for Autonomous Epistemic Profiler, Coordinate Transmuter, and Frontier Quantum Equations.

Verifies:
1. Automated topological diagnosis on Brus, Landauer, Schwinger, and Planck.
2. Exact parameter recovery via Coordinate Transmutation without heuristic search.
3. Discovery on Frontier Quantum Equations:
   - Landau-Zener Non-Adiabatic Transition Probability (P = exp(-alpha / v))
   - BCS Superconducting Weak-Coupling Gap (Delta = Delta_0 * exp(-1 / g))
   - Integer Quantum Hall Effect Conductance (sigma_xy = G_0 * floor(1 / B))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.autonomous_profiler import profile_dataset
from apedp_dyn.epistemic.coordinate_transmuter import transmute_and_propose


# ============================================================================
# Section 1: Benchmark Physical Regimes Verification
# ============================================================================

def test_brus_quantum_dot_power_law_profiling():
    """Verify autonomous detection of negative power law y ~ R^(-2)."""
    R = np.linspace(1.5, 8.0, 50)
    E_brus = 4.5 / (R**2)
    profile = profile_dataset(R, E_brus)

    assert profile.is_pure_power_law is True
    assert abs(profile.power_law_exponent - (-2.0)) < 0.05
    assert "inv_sq" in profile.recommended_grammar

    # Coordinate transmutation
    res = transmute_and_propose(R, E_brus, profile, feature_name="R")
    assert res.success is True
    assert res.r2 > 0.9999
    assert abs(res.parameters["prefactor"] - 4.5) < 1e-3
    assert abs(res.parameters["power"] - (-2.0)) < 1e-3


def test_landauer_staircase_quantization_profiling():
    """Verify autonomous detection of discrete plateau staircase topology."""
    w = np.linspace(0.1, 4.9, 100)
    G = 2.0 * np.floor(w)
    profile = profile_dataset(w, G)

    assert profile.is_quantized_staircase is True
    assert profile.flat_derivative_fraction > 0.80
    assert len(profile.discrete_levels) >= 4
    assert "floor" in profile.recommended_grammar

    # Coordinate transmutation
    res = transmute_and_propose(w, G, profile, feature_name="w")
    assert res.success is True
    assert res.r2 > 0.99
    assert abs(res.parameters["quantum_step"] - 2.0) < 0.1


def test_schwinger_tunneling_essential_singularity_profiling():
    """Verify autonomous detection of non-perturbative tunneling barrier."""
    E = np.linspace(0.15, 1.2, 50)
    Gamma = 10.0 * (E**2) * np.exp(-1.0 / E)
    profile = profile_dataset(E, Gamma)

    assert profile.dynamic_range > 1000.0
    assert profile.recommended_loss == "log_mse"
    assert profile.is_arrhenius_tunneling is True
    assert profile.recommended_coordinate_transmutation == "log_linear_singular"

    # Coordinate transmutation: recovers C=10.0, k=2.0, c=1.0 to machine precision
    res = transmute_and_propose(E, Gamma, profile, feature_name="E")
    assert res.success is True
    assert res.r2 > 0.999999
    assert abs(res.parameters["prefactor"] - 10.0) < 1e-4
    assert abs(res.parameters["power"] - 2.0) < 1e-4
    assert abs(res.parameters["barrier"] - 1.0) < 1e-4


def test_planck_dual_asymptotic_crossover_profiling():
    """Verify autonomous detection of Rayleigh-Jeans vs Wien boundary crossover."""
    nu = np.linspace(0.05, 10.0, 100)
    u = (nu**3) / (np.exp(nu) - 1.0)
    profile = profile_dataset(nu, u)

    assert profile.is_dual_asymptotic is True
    assert profile.low_x_exponent > 1.2  # Rayleigh-Jeans limit ~ nu^2
    assert profile.high_x_exponent < -2.0  # Exponential decay in Wien limit
    assert any(tok in profile.recommended_grammar for tok in ("bose", "planck"))


# ============================================================================
# Section 2: Frontier Quantum Equations
# ============================================================================

def test_frontier_equation_1_landau_zener_transition():
    """Landau-Zener transition probability: P(v) = exp(-alpha / v).

    Essential singularity at v -> 0 (adiabatic limit P -> 0).
    Sudden quench limit v -> inf (P -> 1).
    """
    v = np.linspace(0.12, 4.0, 60)
    alpha_true = 0.75
    P_lz = np.exp(-alpha_true / v)

    profile = profile_dataset(v, P_lz)
    assert profile.is_arrhenius_tunneling is True
    assert profile.recommended_coordinate_transmutation == "log_linear_singular"

    # Autonomous discovery via coordinate transmutation
    res = transmute_and_propose(v, P_lz, profile, feature_name="v")
    assert res.success is True
    assert res.r2 > 0.999999
    assert abs(res.parameters["prefactor"] - 1.0) < 1e-4
    assert abs(res.parameters["barrier"] - alpha_true) < 1e-4
    assert res.parameters["power"] == 0.0

    # Scorer verification on DataFrame
    df_test = pd.DataFrame({"v": np.array([0.5, 1.0, 2.0])})
    pred = res.candidate.compute_score(df_test)
    expected = np.exp(-alpha_true / df_test["v"].to_numpy())
    np.testing.assert_allclose(pred, expected, rtol=1e-5)


def test_frontier_equation_2_bcs_superconducting_energy_gap():
    """BCS Superconducting Weak-Coupling Gap: Delta(g) = Delta_0 * exp(-1 / g).

    Non-perturbative essential singularity in electron-phonon coupling g = N(0)*V.
    """
    g = np.linspace(0.15, 0.75, 50)
    Delta_0_true = 3.52  # 2 * hbar * omega_D
    c_coupling_true = 1.0
    Delta_bcs = Delta_0_true * np.exp(-c_coupling_true / g)

    profile = profile_dataset(g, Delta_bcs)
    assert profile.dynamic_range > 100.0
    assert profile.recommended_loss == "log_mse"
    assert profile.is_arrhenius_tunneling is True

    # Autonomous discovery via coordinate transmutation
    res = transmute_and_propose(g, Delta_bcs, profile, feature_name="g")
    assert res.success is True
    assert res.r2 > 0.999999
    assert abs(res.parameters["prefactor"] - Delta_0_true) < 1e-3
    assert abs(res.parameters["barrier"] - c_coupling_true) < 1e-3
    assert res.parameters["power"] == 0.0


def test_frontier_equation_3_integer_quantum_hall_staircase():
    """Integer Quantum Hall Effect (IQHE) Hall Conductance: sigma_xy = G_0 * floor(1 / B).

    Quantized plateaus in inverse magnetic field x = 1/B.
    """
    inv_B = np.linspace(0.9, 5.8, 120)
    G_0_true = 1.0  # e^2 / h in atomic units
    sigma_xy = G_0_true * np.floor(inv_B)

    profile = profile_dataset(inv_B, sigma_xy)
    assert profile.is_quantized_staircase is True
    assert profile.flat_derivative_fraction > 0.80
    assert len(profile.discrete_levels) >= 5
    assert "floor" in profile.recommended_grammar

    # Autonomous discovery via coordinate transmutation
    res = transmute_and_propose(inv_B, sigma_xy, profile, feature_name="inv_B")
    assert res.success is True
    assert res.r2 > 0.99
    assert abs(res.parameters["quantum_step"] - G_0_true) < 0.1
