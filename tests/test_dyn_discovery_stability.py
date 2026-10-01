"""Tests for Discovery Stability and Identifiability Matrix.

Verifies:
  1. Structural Recurrence and Distribution across multi-seed discovery.
  2. Shannon Structural Entropy (low = deterministic discovery, high = unstable).
  3. Parameter Dispersion (CV |std/mean|).
  4. Variance Split: decomposing transfer error into fit-seed variance vs. scoring-seed variance.
  5. End-to-end composition of the Identifiability & Discovery Stability Matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.epistemic.discovery_stability import (
    coefficient_cv,
    compose_identifiability_stability_matrix,
    stability_report,
    structural_distribution,
    structural_entropy,
    variance_split,
)
from apedp_dyn.epistemic.transfer_scope import TransferScope


class _FakeCand:
    def __init__(self, cid: str):
        self.candidate_id = cid


class _FakePromoted:
    def __init__(self, cid: str):
        self.candidate = _FakeCand(cid)


class _FakeAdj:
    def __init__(self, **kw):
        self.status = kw.get("status", EpistemicStatus.MULTI_HYPOTHESIS)
        self.epistemic_attribution = kw.get(
            "attribution", "intrinsic_observational_nonidentifiability"
        )
        self.promoted = kw.get("promoted", None)
        self.best_rmse = kw.get("best_rmse", 10.71)
        self.second_rmse = kw.get("second_rmse", 14.36)
        self.delta_rmse = kw.get("delta_rmse", -3.65)
        self.tau_epi = kw.get("tau_epi", 11.9)
        self.reason = kw.get("reason", "rival within tau")


class TestDiscoveryStability:
    """Rigorous tests for discovery stability and variance decomposition."""

    def test_structural_distribution(self):
        """Verifies empirical frequency of discovered structures across seeds."""
        seeds_res = ["hp_single", "hp_single", "hp_single", "linear_grain"]
        dist = structural_distribution(seeds_res)
        assert dist["hp_single"] == pytest.approx(0.75)
        assert dist["linear_grain"] == pytest.approx(0.25)

    def test_structural_entropy_deterministic_is_zero(self):
        """A deterministic discovery process (100% recurrence) has exactly zero entropy."""
        deterministic = ["hp_single"] * 10
        ent = structural_entropy(deterministic)
        assert ent == pytest.approx(0.0)

    def test_structural_entropy_uniform_is_maximal(self):
        """Uniform distribution across K structures has maximum entropy ln(K)."""
        uniform = ["H1", "H2", "H3", "H4"]
        ent = structural_entropy(uniform)
        assert ent == pytest.approx(np.log(4.0))

    def test_coefficient_cv_dispersion(self):
        """Verifies parameter coefficient of variation."""
        # Clean parameter: mean=100.0, std=1.0 -> CV = 0.01 (low)
        clean_vals = [99.0, 100.0, 101.0]
        cv_clean = coefficient_cv(clean_vals)
        assert cv_clean is not None
        assert cv_clean < 0.05

        # Unstable parameter: mean=10.0, std=8.0 -> CV = 0.8 (high)
        unstable_vals = [2.0, 10.0, 18.0]
        cv_unstable = coefficient_cv(unstable_vals)
        assert cv_unstable is not None
        assert cv_unstable > 0.5

        # Zero mean returns None
        assert coefficient_cv([0.0, 0.0, 0.0]) is None

    def test_variance_split_decomposition(self):
        """Decomposes transfer RMSE variance into fit-seed vs scoring-seed components."""
        # Construct synthetic transfer matrix (5 fit seeds x 5 score seeds)
        # where fit variation dominates scoring variation:
        rng = np.random.default_rng(42)
        fit_effects = np.array([5.0, 10.0, 25.0, 4.0, 30.0])[:, None]
        score_noise = rng.normal(0, 0.1, size=(5, 5))
        matrix = fit_effects + score_noise

        v_split = variance_split(matrix)
        assert v_split["total"] > 0
        assert v_split["fit_seed"] > v_split["scoring_seed"]
        # Fit variance share should be > 90%
        assert v_split["fit_share"] > 0.90

    def test_stability_report_generation(self):
        """Verifies full stability report generation."""
        structs = ["piecewise_hp"] * 9 + ["constant"]
        params = {"sigma0": [200.0, 201.0, 199.0, 202.0], "k_hp": [250.0, 350.0, 150.0, 400.0]}
        rep = stability_report(structs, param_samples=params)
        assert rep["n_seeds"] == 10
        assert rep["distribution"]["piecewise_hp"] == pytest.approx(0.9)
        assert rep["entropy_nats"] > 0.0
        assert rep["parameter_cv"]["sigma0"] < rep["parameter_cv"]["k_hp"]

    def test_identifiability_stability_matrix_composition(self):
        """Verifies composite Identifiability & Discovery Stability Matrix schema."""
        structs = ["piecewise_hp"] * 10
        params = {"sigma0": [200.0, 201.0], "k_hp": [300.0, 310.0]}
        rep = stability_report(structs, param_samples=params)

        search_scope = {"type": "L2_OPEN", "max_complexity": 6}
        adj = _FakeAdj(status=EpistemicStatus.UNIQUE_SUPPORT, promoted=_FakePromoted("piecewise_hp"))
        scope = TransferScope(
            candidate_id="piecewise_hp",
            rival_id="constant",
            interp_rmse_best=3.56,
            interp_rmse_rival=10.53,
            extra_rmse_best=4.10,
            extra_rmse_rival=11.02,
        )

        matrix = compose_identifiability_stability_matrix(
            search_scope=search_scope,
            stability_rep=rep,
            adjudication=adj,
            transfer_scope=scope,
        )

        assert matrix["search_scope"]["type"] == "L2_OPEN"
        assert matrix["discovery"]["structural_recurrence"] == pytest.approx(1.0)
        assert matrix["discovery"]["structural_entropy"] == 0.0
        assert matrix["identification"]["status"] == "unique_support"
        assert matrix["transfer"]["status"] == "cross_domain_supported"
        assert matrix["promotion"]["status"] == "PROMOTED"
        assert matrix["promotion"]["promoted_candidate"] == "piecewise_hp"

    def test_run_discovery_stability_audit_on_physical_system(self):
        """Verifies multi-seed audit on physical law discovery (harmonic force)."""
        from apedp_dyn.discovery_l2 import Dimension, L2DiscoveryEngine, run_discovery_stability_audit

        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 5.0, size=60)
        # Hooke potential U = 50 * x^2 with nominal 2% noise
        y_clean = 50.0 * (x**2)
        y = y_clean + rng.normal(0, 0.02 * np.std(y_clean), size=len(x))

        engine = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"), max_depth=1, max_complexity=2)
        audit = run_discovery_stability_audit(
            engine=engine,
            X={"x": x},
            Y=y,
            dimensions={"x": Dimension(l=1)},
            target_dimension=Dimension(l=2),
            seeds=(1, 2, 3, 4, 5),
            train_ratio=0.6,
        )

        assert audit.n_seeds == 5
        assert audit.matrix["discovery"]["structural_recurrence"] == pytest.approx(1.0)
        assert audit.matrix["discovery"]["structural_entropy"] == 0.0
        assert audit.matrix["identification"]["status"] == "unique_support"
        assert audit.matrix["promotion"]["status"] == "PROMOTED"


