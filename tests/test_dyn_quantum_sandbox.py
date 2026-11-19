"""Tests for APEDP-DYN Quantum Vacuum & String-Breaking Sandbox (Duke Trapped-Ion Simulator).

Verifies the honesty and rigor standards:
1. Genuine parameter fitting (least squares on train split, holdout RMSE, real residual tracking, no mock losses)
2. Monotonic error scaling with noise
3. Proper hypothesis discrimination / abstention (recovering Schwinger mechanism under low noise,
   abstaining under subthreshold, asymptotic, or high-noise regimes)
4. Accurate data_origin threading ("simulation" and "experimental")
5. Trapped-ion spatiotemporal quench dynamics across supercritical (string-breaking),
   subcritical (confined meson), and zero-tension (ballistic) regimes.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.sandbox.quantum_vacuum_sandbox import (
    QuantumVacuumConfig,
    fit_quantum_pair_candidates,
    generate_schwinger_pair_production_data,
    run_quantum_vacuum_discovery,
    run_quench_dynamics_discovery,
    simulate_string_breaking_quench,
)


class TestQuantumVacuumSandboxHonesty:
    def test_low_noise_recovers_schwinger_mechanism(self):
        """At 1% noise, the non-perturbative Schwinger mechanism is uniquely supported."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.01,
            E_min=0.6,
            E_max=5.0,
            E_c=1.5,
            bootstrap=True,
            data_origin="simulation",
        )
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "schwinger_nonperturbative"
        assert adj.data_origin == "simulation"
        assert adj.best_rmse is not None and adj.best_rmse < 0.15

    def test_experimental_data_origin_threading(self):
        """Native data_origin='experimental' threads cleanly through candidate provenance to adjudication."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.01,
            data_origin="experimental",
            bootstrap=False,
        )
        assert adj.data_origin == "experimental"
        assert adj.promoted is not None
        assert adj.promoted.provenance["data_origin"] == "experimental"

    def test_rmse_scales_monotonically_with_noise(self):
        """Observed holdout RMSE of the true Schwinger candidate must scale monotonically with noise."""
        rmses = []
        for noise in (0.01, 0.03, 0.10):
            adj, _, _ = run_quantum_vacuum_discovery(noise_sigma=noise, bootstrap=False)
            rmses.append(adj.best_rmse)
        assert rmses[0] < rmses[1] < rmses[2], f"RMSE did not scale monotonically: {rmses}"

    def test_subthreshold_regime_abstains_at_the_null_floor(self):
        """Below pair creation threshold (E << Ec) the constant is the floor.

        At 5% noise the four saturated residuals still separate by a hair
        (Δ/τ ≈ 1.09) and F14's re-anchoring correctly picks the holdout-best
        head — which in this window is `thermal_constant`, the null model
        itself, carrying RESIDUAL_STRUCTURE_UNRESOLVED. The referee must not
        promote on that: a promotion has to beat the constant null by more than
        τ, and a null cannot beat itself, so the margin here is exactly zero.

        Reasoned re-mint. F14 (commit 68a3dadb) removed a manufactured refusal
        by re-anchoring the pair on the holdout-best head; this arm inherits
        that anchor and adds the null floor, because the promotion F14 produced
        rested on the null winning its own comparison. Neither verdict is a
        referee artifact: whether the four models are physically
        distinguishable in this field window is a *physics* claim to
        re-establish with a larger holdout or lower noise."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.05,
            E_min=0.05,
            E_max=0.25,
            E_c=1.5,
            bootstrap=True,
        )
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None
        # The holdout-best candidate IS the null model: the null margin is 0.
        assert adj.ranking[0].candidate.candidate_id == "thermal_constant"
        assert adj.delta_rmse == 0.0
        assert adj.tau_epi is not None and adj.tau_epi > 0.0
        assert "NULL_MODEL_NOT_BEATEN" in adj.diagnostic_flags


    def test_asymptotic_saturation_regime_abstains(self):
        """In deep asymptotic field window (E >> Ec), curvature is negligible and linear surrogate
        competes with Schwinger within resolution floor; adjudicator abstains."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.03,
            E_min=2.5,
            E_max=5.0,
            E_c=1.5,
            bootstrap=True,
        )
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

    def test_high_noise_regime_abstains(self):
        """Under large measurement noise (40%), noise overwhelms hypothesis discrimination;
        adjudicator abstains."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.40,
            bootstrap=True,
        )
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

    def test_candidates_have_no_mock_losses(self):
        """Ensure all candidate hypotheses are genuinely fitted with least squares, real holdout RMSE,
        and real per-timepoint squared error series (no random/mock loss generation)."""
        E, gamma_obs, _ = generate_schwinger_pair_production_data(n_points=40, noise_sigma=0.02)
        candidates = fit_quantum_pair_candidates(E, gamma_obs, n_train=20, noise_sigma=0.02)

        assert len(candidates) == 4
        candidate_ids = {c.candidate.candidate_id for c in candidates}
        assert candidate_ids == {
            "schwinger_nonperturbative",
            "perturbative_power_law",
            "classical_linear",
            "thermal_constant",
        }

        for sc in candidates:
            assert sc.train_loss is not None and np.isfinite(sc.train_loss) and sc.train_loss > 0.0
            assert sc.holdout_rmse is not None and np.isfinite(sc.holdout_rmse) and sc.holdout_rmse > 0.0
            g4 = sc.provenance["gate4_status"]
            assert g4["passed"] is True
            assert len(g4["holdout_residuals"]) == 20
            assert len(g4["holdout_sq_error"]) == 20
            # Verify residuals match holdout RMSE exactly
            computed_rmse = float(np.sqrt(np.mean(np.array(g4["holdout_residuals"])**2)))
            assert abs(computed_rmse - sc.holdout_rmse) < 1e-9


class TestTrappedIonQuenchDynamicsHonesty:
    def test_supercritical_string_breaking_regime(self):
        """When string tension exceeds critical field (E=2.5 > Ec=1.5), string breaking decay
        into particle pairs is uniquely supported."""
        cfg = QuantumVacuumConfig(
            string_tension_hz=2.5,
            critical_field_Ec=1.5,
            duration=10.0,
            noise_sigma=0.02,
            data_origin="simulation",
        )
        adj, _, _ = run_quench_dynamics_discovery(cfg, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "decay_string_breaking"
        assert adj.data_origin == "simulation"

    def test_subcritical_confined_meson_regime(self):
        """When string tension is subcritical (E=0.5 < Ec=1.5), charges are confined and undergo
        coherent Bloch oscillations without string breaking; coherent oscillation model is uniquely supported."""
        cfg = QuantumVacuumConfig(
            string_tension_hz=0.5,
            critical_field_Ec=1.5,
            duration=10.0,
            noise_sigma=0.02,
            data_origin="simulation",
        )
        adj, _, _ = run_quench_dynamics_discovery(cfg, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "coherent_oscillations"

    def test_zero_tension_ballistic_regime(self):
        """Under zero string tension (E=0.0), charges spread freely and ballistically;
        ballistic linear model is uniquely supported."""
        cfg = QuantumVacuumConfig(
            string_tension_hz=0.0,
            duration=10.0,
            noise_sigma=0.02,
            data_origin="simulation",
        )
        adj, _, _ = run_quench_dynamics_discovery(cfg, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "ballistic_linear"

    def test_spatiotemporal_edge_driven_signature(self):
        """Trapped-ion simulator reproduces the edge-driven mechanism: pairs form near the
        string ends (sites 3 and 9) before propagating to the center (site 6)."""
        cfg = QuantumVacuumConfig(
            n_ions=13,
            string_length=7,
            string_tension_hz=2.5,
            critical_field_Ec=1.5,
            duration=5.0,
            dt=0.1,
        )
        states, _ = simulate_string_breaking_quench(cfg)
        mid_idx = len(states) // 2
        mid_state = states[mid_idx]

        # Site 3 is edge of string, site 6 is center of string
        # Edge sites polarize towards +1 (vacuum pair screening) faster than center site
        sigma_edge = mid_state.site_polarization[3]
        sigma_center = mid_state.site_polarization[6]
        assert sigma_edge > sigma_center, (
            f"Edge-driven breaking violated: sigma_edge={sigma_edge} should exceed sigma_center={sigma_center}"
        )
