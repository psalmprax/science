"""Tests for Next-Concept Modeling (NCM) Structural Proposer.

Verifies:
1. NCMStructuralProposer conforms to ProposerBase and emits valid CandidateProposal objects.
2. Macroscopic archetypes fit synthetic physical trajectories (damped oscillator, advection-diffusion, saturation, cyclic).
3. Parameter and metadata validity (R2 in [0, 1], complexity metrics).
4. Robustness against short or degenerate dataframes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.ncm_structural_proposer import (
    MacroConceptArchetype,
    NCMStructuralProposer,
)


@pytest.fixture
def synthetic_physics_df() -> pd.DataFrame:
    """Generate synthetic physical series with harmonic, transport, and saturation components."""
    np.random.seed(42)
    n = 200
    t = np.linspace(0, 10, n)
    x1 = np.sin(1.5 * t) + 0.05 * np.random.randn(n)
    x2 = np.cos(1.5 * t) + 0.05 * np.random.randn(n)
    x3 = np.exp(-0.1 * t) * np.sin(2.0 * t)

    # Target is combination of damped oscillation and saturation
    y = 0.6 * x1 / (1.0 + 0.2 * np.abs(x1)) - 0.3 * np.gradient(x1) + 0.02 * np.random.randn(n)

    return pd.DataFrame({"feat_pos": x1, "feat_vel": x2, "feat_damp": x3, "target_flux": y})


def test_ncm_proposer_conformance(synthetic_physics_df):
    proposer = NCMStructuralProposer()
    assert isinstance(proposer, ProposerBase)
    assert proposer.proposer_id == "ncm_structural_proposer_v1"

    proposals = proposer.fit_and_propose(
        synthetic_physics_df,
        feature_cols=["feat_pos", "feat_vel", "feat_damp"],
        target_col="target_flux",
    )

    assert len(proposals) == 4  # All 4 archetypes

    archetype_set = {p.metadata["archetype"] for p in proposals}
    assert MacroConceptArchetype.CONSERVED_DAMPED_OSCILLATOR.value in archetype_set
    assert MacroConceptArchetype.BOUNDED_RATIONAL_SATURATION.value in archetype_set
    assert MacroConceptArchetype.DIFFUSION_ADVECTION_CONTINUUM.value in archetype_set
    assert MacroConceptArchetype.CYCLIC_HARMONIC_REGIME.value in archetype_set

    for p in proposals:
        assert isinstance(p, CandidateProposal)
        assert p.candidate_id.startswith("H-NCM-")
        assert p.proposer_id == "ncm_structural_proposer_v1"
        assert len(p.formula) > 5
        assert p.parameters["r2"] > 0.50, f"Archetype {p.metadata['archetype']} failed to achieve positive correlation"
        assert p.metadata["tier"] == "next_concept_modeling"
        assert p.metadata["complexity"] >= 2


def test_ncm_selective_archetypes(synthetic_physics_df):
    proposer = NCMStructuralProposer(
        archetypes=[MacroConceptArchetype.BOUNDED_RATIONAL_SATURATION]
    )
    proposals = proposer.fit_and_propose(
        synthetic_physics_df,
        feature_cols=["feat_pos"],
        target_col="target_flux",
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert "RationalHill" in p.formula
    assert p.parameters["r2"] > 0.80


def test_ncm_parameter_recovery_damped_oscillator():
    """Verify exact parameter recovery on planted damped harmonic oscillator dynamics."""
    t = np.linspace(0, 10, 500)
    omega = 2.0
    gamma = 0.5
    x = np.exp(-gamma * t / 2) * np.cos(omega * t)
    dx = np.gradient(x)
    # Planted linear relation: y = k * x + gamma * dx
    planted_k = 3.5
    planted_gamma = -0.7
    y = planted_k * x + planted_gamma * dx

    df = pd.DataFrame({"pos": x, "target": y})
    proposer = NCMStructuralProposer(
        archetypes=[MacroConceptArchetype.CONSERVED_DAMPED_OSCILLATOR],
        regularization_alpha=1e-6,
    )
    proposals = proposer.fit_and_propose(df, feature_cols=["pos"], target_col="target")
    assert len(proposals) == 1
    p = proposals[0]

    # Non-tautological physics recovery assertions:
    assert pytest.approx(p.parameters["omega_sq"], rel=1e-3) == planted_k
    assert pytest.approx(p.parameters["gamma"], rel=1e-3) == planted_gamma
    assert p.parameters["r2"] > 0.999
    assert p.parameters["active_terms"] == 2


def test_ncm_sparsity_threshold_prunes_inactive_features():
    """Verify that sparsity thresholding actively zeros out irrelevant noise features."""
    np.random.seed(42)
    n = 300
    x_signal = np.linspace(-2, 2, n)
    x_noise1 = np.random.randn(n) * 0.05
    x_noise2 = np.random.randn(n) * 0.05
    denom = 1.0 + 0.1 * np.mean(np.abs(np.column_stack([x_signal, x_noise1, x_noise2])), axis=1)
    y = (2.0 * x_signal) / denom

    df = pd.DataFrame({"sig": x_signal, "noise1": x_noise1, "noise2": x_noise2, "target": y})
    proposer = NCMStructuralProposer(
        archetypes=[MacroConceptArchetype.BOUNDED_RATIONAL_SATURATION],
        regularization_alpha=1e-3,
        sparsity_threshold=0.05,
    )
    proposals = proposer.fit_and_propose(df, feature_cols=["sig", "noise1", "noise2"], target_col="target")
    assert len(proposals) == 1
    p = proposals[0]

    coefs = p.parameters["coefs"]
    assert len(coefs) == 3
    # Signal feature recovered
    assert pytest.approx(coefs[0], rel=1e-2) == 2.0
    # Inactive noise features must be actively pruned to zero
    assert coefs[1] == 0.0
    assert coefs[2] == 0.0
    assert p.parameters["active_terms"] == 1
    assert p.parameters["r2"] > 0.99


def test_ncm_handles_short_data():
    df_short = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    proposer = NCMStructuralProposer()
    proposals = proposer.fit_and_propose(df_short, feature_cols=["x"], target_col="y")
    assert proposals == []

