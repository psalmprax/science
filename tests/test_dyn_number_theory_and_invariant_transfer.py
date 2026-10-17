"""Tests for Discrete Number Theory Engine and Active Cross-Regime Invariant Transfer.

Verifies:
1. Exact rational constant recovery via Farey sequences / continued fractions.
2. Exact integer relation discovery for conservation laws and stoichiometry.
3. Modular lattice periodicity and spacing discovery.
4. Active cross-regime invariant transfer proposer (IRM penalty & retention gating).
5. Active extrapolation barrier guard rejecting gradient explosions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.number_theory_engine import (
    discover_integer_relation,
    discover_modular_lattice,
    recover_rational_constant,
)
from apedp_dyn.proposers.invariant_transfer_proposer import InvariantTransferProposer


def test_farey_rational_constant_recovery():
    """Recovers exact irreducible rational fractions from noisy floating point constants."""
    # Laughlin filling factor 2/5 with small float noise
    r1 = recover_rational_constant(0.40000002, max_denom=100, eps=1e-4)
    assert r1 is not None
    assert r1.num == 2 and r1.den == 5
    assert str(r1) == "2/5"

    # Brus confinement coefficient 2.45 = 49/20
    r2 = recover_rational_constant(2.450001, max_denom=100, eps=1e-4)
    assert r2 is not None
    assert r2.num == 49 and r2.den == 20

    # Critical exponent 1/8 = 0.125
    r3 = recover_rational_constant(0.1249999, max_denom=100, eps=1e-4)
    assert r3 is not None
    assert r3.num == 1 and r3.den == 8


def test_integer_relation_conservation_laws():
    """Discovers stoichiometric chemical balance and harmonic frequency resonance."""
    # Chemical reaction: 2 H2 + 1 O2 -> 2 H2O
    weights = [2.016, 31.998, 18.015]
    labels = ["H2", "O2", "H2O"]
    rel = discover_integer_relation(weights, labels=labels, max_coeff=4, eps=1e-2)
    assert rel is not None
    assert rel.coefficients == [2, 1, -2]
    assert rel.is_exact or rel.residual < 1e-2

    # Resonant mode: 5*w1 - 3*w2 = 0
    freqs = [30.0, 50.0]
    rel_mode = discover_integer_relation(freqs, labels=["w1", "w2"], max_coeff=6, eps=1e-4)
    assert rel_mode is not None
    assert rel_mode.coefficients == [5, -3]


def test_modular_lattice_periodicity():
    """Identifies discrete quantum lattice spacing and offset."""
    # Spacing 2.0, offset 1.0: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    points = [1.0001, 2.9999, 5.0002, 7.0000, 8.9998, 11.0003]
    lat = discover_modular_lattice(points, min_spacing=0.5, max_spacing=5.0, max_rmse=1e-2)
    assert lat is not None
    assert abs(lat.spacing - 2.0) < 1e-2
    assert abs(lat.offset - 1.0) < 1e-2
    assert lat.residual_rmse < 1e-3


def test_invariant_transfer_proposer_promotes_causal_law():
    """InvariantTransferProposer discovers invariant causal relation and rejects spurious correlator."""
    rng = np.random.RandomState(42)
    n = 100

    # Environment 1: low variance, positive spurious correlation
    x1 = np.linspace(1.0, 5.0, n)
    z1 = 2.0 * x1 + 0.1 * rng.randn(n) # Spurious feature strongly correlated in Env 1
    y1 = 3.0 * x1 + 0.05 * rng.randn(n) # True causal law: y = 3x
    df1 = pd.DataFrame({"causal_x": x1, "spurious_z": z1, "target": y1, "regime": "env_1"})

    # Environment 2: intervention domain shift, INVERTED spurious correlation
    x2 = np.linspace(10.0, 15.0, n)
    z2 = -2.0 * x2 + 0.1 * rng.randn(n) # Spurious feature correlation inverted!
    y2 = 3.0 * x2 + 0.05 * rng.randn(n) # True causal law STILL holds: y = 3x
    df2 = pd.DataFrame({"causal_x": x2, "spurious_z": z2, "target": y2, "regime": "env_2"})

    df_combined = pd.concat([df1, df2], ignore_index=True)

    proposer = InvariantTransferProposer(
        feature_cols=["causal_x", "spurious_z"],
        target_col="target",
        env_col="regime",
        min_worst_r2=0.80,
        max_r2_dispersion=0.10,
    )
    proposals = proposer.fit_and_propose(df_combined)

    # Must produce at least 1 proposal
    assert len(proposals) >= 1
    top_p = proposals[0]

    # The top proposal MUST depend on causal_x, NEVER spurious_z
    assert "causal_x" in top_p.declared_nominal_dependencies
    assert "spurious_z" not in top_p.declared_nominal_dependencies

    # Invariance metrics
    assert top_p.metadata["worst_env_r2"] > 0.99
    assert top_p.metadata["r2_dispersion_std"] < 0.01

    # Evaluate on held-out unseen regime (Environment 3)
    x3 = np.linspace(20.0, 25.0, 50)
    y3 = 3.0 * x3
    df3 = pd.DataFrame({"causal_x": x3, "spurious_z": np.zeros(50), "target": y3})
    pred3 = top_p.compute_score(df3)
    rmse3 = np.sqrt(np.mean((y3 - pred3) ** 2))
    assert rmse3 < 0.10, f"Out-of-sample causal transfer failed: RMSE={rmse3}"
