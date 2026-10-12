"""Tests for APEDP Level 1 Constrained Materials Discovery Benchmark.

Verifies:
  1. M1 Single Mechanism: Hall-Petch grain size strengthening is uniquely supported.
  2. M2 Competing Mechanisms: Combined grain size + precipitate hardening is uniquely supported.
  3. M3 Mechanism Overlap: Microstructural Zener pinning makes mechanisms observationally
     unidentifiable; APEDP's epistemic layer properly abstains (MULTI_HYPOTHESIS), proving
     that point-estimate winners do NOT masquerade as uniquely supported models.
  4. Provenance & Taxonomy: Clean threading of DataOrigin and BenchmarkLevel.
  5. Honesty: No mock or random losses; real OLS regression on train, real holdout residuals.
"""
from __future__ import annotations

from apedp_dyn.epistemic.adjudicate import (
    BenchmarkLevel,
    DataOrigin,
    EpistemicStatus,
)
from apedp_dyn.sandbox.materials_discovery_l1 import (
    fit_l1_materials_candidates,
    generate_m1_data,
    run_l1_materials_benchmark,
)


class TestMaterialsDiscoveryL1:
    def test_m1_single_mechanism_recovers_hall_petch(self):
        """M1: Single Hall-Petch mechanism is uniquely supported over competing single terms."""
        adj, _ = run_l1_materials_benchmark("M1", noise_sigma=0.02, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "hp_single"
        assert adj.benchmark_level == BenchmarkLevel.L1_CONSTRAINED_DISCOVERY
        assert adj.data_origin == DataOrigin.SIMULATION
        assert adj.best_rmse is not None and adj.best_rmse < 10.0

    def test_m2_competing_mechanisms_recovers_dual_mechanism(self):
        """M2: Independent d and Phi variations cleanly separate dual mechanism from rivals."""
        adj, _ = run_l1_materials_benchmark("M2", noise_sigma=0.02, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "dual_hp_precip"
        assert adj.second_rmse is not None
        assert adj.delta_rmse is not None
        assert adj.delta_rmse > adj.tau_epi

    def test_m3_mechanism_overlap_abstains_with_multi_hypothesis(self):
        """M3: Zener pinning coupling creates collinearity; APEDP abstains with MULTI_HYPOTHESIS."""
        adj, candidates = run_l1_materials_benchmark("M3", noise_sigma=0.03, bootstrap=True)
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        # Verify point winner had lowest RMSE but did not earn UNIQUE_SUPPORT
        ranked = sorted(candidates, key=lambda c: c.holdout_rmse)
        assert ranked[0].candidate.candidate_id in ("dual_hp_precip", "hp_single")
        # Reason reworded with the prespecified-resolution form; assert the
        # claim itself rather than the retired "structures ... under available
        # interventions" phrasing that no longer exists in adjudicate().
        assert "observationally indistinguishable" in adj.reason

    def test_candidates_have_no_mock_losses(self):
        """Every candidate is fitted with real least-squares and real holdout residuals."""
        features, y = generate_m1_data(60, 0.02, seed=123)
        candidates = fit_l1_materials_candidates(features, y, n_train=30, noise_sigma=0.02)
        assert len(candidates) >= 5
        for c in candidates:
            assert c.train_loss is not None and c.train_loss >= 0.0
            assert c.holdout_rmse is not None and c.holdout_rmse >= 0.0
            g4 = c.provenance.get("gate4_status", {})
            assert g4.get("passed") is True
            assert len(g4.get("holdout_residuals", [])) == 30
            assert len(g4.get("holdout_sq_error", [])) == 30

    def test_provenance_and_benchmark_level_threading(self):
        """Custom data_origin and benchmark_level thread cleanly through candidates to Adjudication."""
        adj, candidates = run_l1_materials_benchmark(
            "M1",
            data_origin=DataOrigin.EXPERIMENTAL,
            bootstrap=False,
        )
        assert adj.data_origin == DataOrigin.EXPERIMENTAL
        assert adj.benchmark_level == BenchmarkLevel.L1_CONSTRAINED_DISCOVERY
        for c in candidates:
            assert c.provenance["data_origin"] == DataOrigin.EXPERIMENTAL
            assert c.provenance["benchmark_level"] == BenchmarkLevel.L1_CONSTRAINED_DISCOVERY

    def test_rmse_scales_monotonically_with_noise(self):
        """M1 true Hall-Petch model holdout RMSE scales monotonically with observation noise."""
        rmses = []
        for noise in (0.01, 0.03, 0.06):
            adj, _ = run_l1_materials_benchmark("M1", noise_sigma=noise, bootstrap=False)
            rmses.append(adj.best_rmse)
        assert rmses[0] < rmses[1] < rmses[2], f"RMSE did not scale monotonically: {rmses}"
