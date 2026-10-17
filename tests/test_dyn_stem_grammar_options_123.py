"""Comprehensive Verification and Regression Suite for STEM Grammar Options 1, 2, and 3.

Validates:
- Option 1: Elementary STEM scalar primitives in RIME kernel (sin, cos, safe_exp, safe_log, diff).
- Option 2: Modular domain discovery engines (HamiltonianProposer, SolenoidalFluidProposer, CompartmentalSirProposer).
- Option 3: Typed physical dimensional homogeneity engine (SI units, Buckingham-Pi constraints, dimensional gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.epistemic.dimensional_homogeneity import (
    ACCELERATION,
    DIMENSIONLESS,
    ENERGY,
    FORCE,
    FREQUENCY,
    LENGTH,
    MASS,
    TIME,
    VELOCITY,
    DimensionalHomogeneityGate,
    validate_expression_dimensions,
)
from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    StrategyArchetype,
)
from apedp_dyn.proposers.rime_proposer import (
    RimeProposer,
    _apply_transform,
    is_rime_available,
)
from apedp_dyn.proposers.stem_proposers import (
    CompartmentalSirProposer,
    HamiltonianProposer,
    SolenoidalFluidProposer,
)

# ==============================================================================
# OPTION 1: RIME Native STEM Primitives
# ==============================================================================

def test_option_1_stem_transforms_domain_safety():
    """Verify that sin, cos, exp, log, diff transforms are finite and NaN-preserving."""
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0, np.nan])

    for kind in ("sin", "cos", "exp", "log", "diff"):
        out = _apply_transform(kind, x)
        assert np.isnan(out[-1]), f"{kind} must preserve NaN at boundary"
        if kind != "diff":
            assert np.all(np.isfinite(out[:-1])), f"{kind} must be finite on domain"

    # Extreme value clamping for exp (prevent overflow)
    extreme = np.array([1000.0, -1000.0])
    out_exp = _apply_transform("exp", extreme)
    assert np.all(np.isfinite(out_exp)), "exp must be clamped to prevent inf"
    assert out_exp[0] == pytest.approx(np.exp(60.0)), "Positive extreme clamped to 60"
    assert out_exp[1] == pytest.approx(np.exp(-60.0)), "Negative extreme clamped to -60"

    # Discrete difference
    t_seq = np.array([10.0, 12.0, 15.0, 19.0, np.nan])
    diff_out = _apply_transform("diff", t_seq)
    assert np.isnan(diff_out[0]), "First difference index 0 must be NaN warmup"
    assert np.array_equal(diff_out[1:4], np.array([2.0, 3.0, 4.0])), "Diff computes x_t - x_{t-1}"


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel required")
def test_option_1_rime_proposer_stem_ops_wiring():
    """Verify RimeProposer accepts stem_ops=True and executes search."""

    t = np.linspace(0, 4 * np.pi, 200)
    # y = sin(t) planted
    df = pd.DataFrame({"t": t, "y": np.sin(t)})

    proposer = RimeProposer(
        feature_cols=["t"],
        target_col="y",
        max_depth=1,
        budget=50,
        rel_penalty=0.0,
        stem_ops=True,
    )
    proposals = proposer.fit_and_propose(df)
    # Proposer runs cleanly without error and returns proposals list
    assert isinstance(proposals, list)


@pytest.mark.skipif(not is_rime_available(), reason="native RIME kernel required")
def test_option_1_stem_ops_off_by_default():
    """Verify that stem_ops is False by default to preserve v0.1 frozen grammar."""
    proposer = RimeProposer(feature_cols=["x"], target_col="y")
    assert proposer._stem_ops is False


# ==============================================================================
# OPTION 2: Modular Multi-Engine STEM Proposers
# ==============================================================================

def test_option_2_hamiltonian_proposer_harmonic_oscillator():
    """Test HamiltonianProposer recovers energy-conserving harmonic oscillator."""
    t = np.linspace(0, 10, 500)
    omega = 2.0
    # True harmonic oscillator: q(t) = cos(omega*t), p(t) = -omega*sin(omega*t)
    # H = 0.5*p^2 + 0.5*omega^2*q^2 = 0.5*omega^2 (constant)
    q = np.cos(omega * t)
    p = -omega * np.sin(omega * t)
    df = pd.DataFrame({"q": q, "p": p})

    proposer = HamiltonianProposer(dt=t[1] - t[0], max_energy_drift=0.02)
    proposals = proposer.fit_and_propose(df)

    assert len(proposals) == 1, "Must emit exactly one conservative Hamiltonian proposal"
    prop = proposals[0]
    assert prop.strategy_archetype == StrategyArchetype.CONTINUOUS_FACTOR
    assert prop.metadata["symplectic_invariance"] is True
    assert prop.parameters["energy_drift"] < 0.02

    # Score function on unseen data
    test_df = pd.DataFrame({"q": np.array([1.0, 0.0, -1.0])})
    pred = prop.compute_score(test_df)
    assert len(pred) == 3
    # Acceleration should be -omega^2 * q ~ -4.0 * q
    assert pred[0] < -3.0
    assert pred[2] > 3.0


def test_option_2_hamiltonian_proposer_rejects_dissipative_system():
    """Verify HamiltonianProposer rejects damped/dissipative systems where energy leaks."""
    t = np.linspace(0, 10, 500)
    # Damped oscillator: amplitude decays exponentially
    decay = np.exp(-0.5 * t)
    q = decay * np.cos(2.0 * t)
    p = decay * (-2.0 * np.sin(2.0 * t) - 0.5 * np.cos(2.0 * t))
    df = pd.DataFrame({"q": q, "p": p})

    proposer = HamiltonianProposer(dt=t[1] - t[0], max_energy_drift=0.05)
    proposals = proposer.fit_and_propose(df)

    # Must abstain / emit empty list because energy is not conserved
    assert len(proposals) == 0, "Dissipative system must be rejected by energy drift gate"


def test_option_2_solenoidal_fluid_proposer():
    """Test SolenoidalFluidProposer recovers divergence-free velocity field."""
    # Stagnation point flow: u = x, v = -y => div u = 1 - 1 = 0
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    u = x
    v = -y
    df = pd.DataFrame({"x": x, "y": y, "u": u, "v": v})

    proposer = SolenoidalFluidProposer(polynomial_order=2)
    proposals = proposer.fit_and_propose(df)

    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.metadata["divergence_free"] is True
    assert prop.metadata["domain"] == "incompressible_fluid_mechanics"

    # Out-of-sample prediction
    test_df = pd.DataFrame({"x": np.array([0.5, -0.5]), "y": np.array([0.2, -0.2])})
    pred_u = prop.compute_score(test_df)
    assert pred_u[0] == pytest.approx(0.5, abs=1e-3)
    assert pred_u[1] == pytest.approx(-0.5, abs=1e-3)


def test_option_2_compartmental_sir_proposer():
    """Test CompartmentalSirProposer discovers SIR parameters with population conservation."""
    dt = 0.5
    N = 1000.0
    beta = 0.3
    gamma = 0.1

    # Simulate 50 steps of SIR
    S_list, I_list, R_list = [990.0], [10.0], [0.0]
    for _ in range(50):
        s_curr, i_curr, r_curr = S_list[-1], I_list[-1], R_list[-1]
        ds = -beta * s_curr * i_curr / N * dt
        dr = gamma * i_curr * dt
        di = -ds - dr
        S_list.append(s_curr + ds)
        I_list.append(i_curr + di)
        R_list.append(r_curr + dr)

    df = pd.DataFrame({"S": S_list, "I": I_list, "R": R_list})
    proposer = CompartmentalSirProposer(dt=dt, max_pop_drift=0.01)
    proposals = proposer.fit_and_propose(df)

    assert len(proposals) == 1
    prop = proposals[0]
    assert prop.metadata["domain"] == "mathematical_epidemiology"
    assert prop.parameters["R0"] == pytest.approx(beta / gamma, rel=0.15)
    assert prop.metadata["conserved_total_population"] == pytest.approx(N, abs=1.0)


# ==============================================================================
# OPTION 3: Typed Physical Dimensional Homogeneity Engine
# ==============================================================================

def test_option_3_valid_physical_equations():
    """Test that dimensionally valid STEM equations pass with correct inferred dimensions."""
    # 1. Kinetic energy: 0.5 * m * v^2
    var_dims = {"m": MASS, "v": VELOCITY}
    valid, msg, dim = validate_expression_dimensions("0.5 * m * v**2", var_dims, target_dimension=ENERGY)
    assert valid, f"Kinetic energy must be valid: {msg}"
    assert dim == ENERGY

    # 2. Newton's second law: m * a
    var_dims_f = {"m": MASS, "a": ACCELERATION}
    valid, msg, dim = validate_expression_dimensions("m * a", var_dims_f, target_dimension=FORCE)
    assert valid, f"Force must be valid: {msg}"
    assert dim == FORCE

    # 3. Wave equation: A * sin(omega * t)
    var_dims_w = {"A": LENGTH, "omega": FREQUENCY, "t": TIME}
    valid, msg, dim = validate_expression_dimensions("A * sin(omega * t)", var_dims_w, target_dimension=LENGTH)
    assert valid, f"Wave formula must be valid: {msg}"
    assert dim == LENGTH

    # 4. Exponential decay: exp(-lambda * t)
    var_dims_decay = {"lam": FREQUENCY, "t": TIME}
    valid, msg, dim = validate_expression_dimensions("exp(-lam * t)", var_dims_decay, target_dimension=DIMENSIONLESS)
    assert valid, f"Decay factor must be dimensionless: {msg}"
    assert dim == DIMENSIONLESS


def test_option_3_unphysical_equations_rejected():
    """Test that dimensionally inconsistent equations are strictly rejected."""
    # 1. Adding meters to seconds (x + t)
    var_dims = {"x": LENGTH, "t": TIME}
    valid, msg, _ = validate_expression_dimensions("x + t", var_dims)
    assert not valid
    assert "Dimensional mismatch in addition" in msg

    # 2. Transcendental function of dimensionful variable: sin(mass)
    var_dims_m = {"m": MASS}
    valid, msg, _ = validate_expression_dimensions("sin(m)", var_dims_m)
    assert not valid
    assert "requires dimensionless argument" in msg

    # 3. Target dimension mismatch: calculating velocity when target is force
    var_dims_v = {"v": VELOCITY}
    valid, msg, _ = validate_expression_dimensions("v", var_dims_v, target_dimension=FORCE)
    assert not valid
    assert "Target dimension mismatch" in msg


def test_option_3_dimensional_gate_on_proposal():
    """Test DimensionalHomogeneityGate adjudicating CandidateProposal instances."""
    gate = DimensionalHomogeneityGate(
        variable_dimensions={"x": LENGTH, "v": VELOCITY, "t": TIME},
        target_dimension=LENGTH,
    )

    # Valid candidate: x + v * t (both terms have dimension Length)
    good_prop = CandidateProposal(
        proposer_id="test",
        candidate_id="P1",
        formula="x + v * t",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=False,
    )
    passed, reason, dim = gate.adjudicate(good_prop)
    assert passed
    assert dim == LENGTH

    # Invalid candidate: x + v (Length + Velocity)
    bad_prop = CandidateProposal(
        proposer_id="test",
        candidate_id="P2",
        formula="x + v",
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=False,
    )
    passed, reason, _ = gate.adjudicate(bad_prop)
    assert not passed
    assert "Dimensional mismatch" in reason
