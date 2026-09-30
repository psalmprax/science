"""Tests for APEDP-DYN sandbox demos with fitted models (honesty verification).

Verifies that all sandbox demos:
1. Use real model fitting (least squares / OLS on train split, RMSE on holdout split, real residual tracking)
2. Exhibit monotonic error scaling with noise
3. Correctly promote the true underlying relation under low noise and abstain (MULTI_HYPOTHESIS) when evidence is ambiguous or sideways
4. Accurately thread data_origin through adjudication
"""
from __future__ import annotations

from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_dyn.sandbox.apedp_discovery import run_apedp_discovery
from apedp_dyn.sandbox.finance_demo import generate_finance_data, run_finance_discovery
from apedp_dyn.sandbox.hall_pecht_demo import run_hall_pecht_discovery
from apedp_dyn.sandbox.plasma_simulator import generate_synthetic_dataset
from apedp_dyn.sandbox.quantum_vacuum_sandbox import (
    QuantumVacuumConfig,
    run_quantum_vacuum_discovery,
    run_quench_dynamics_discovery,
)
from apedp_dyn.sandbox.trading_engine import detect_regime, generate_price_series


class TestHallPetchSandboxHonesty:
    def test_low_noise_recovers_ground_truth(self):
        """At 1% noise, d^(-1/2) Hall-Petch scaling is uniquely supported."""
        adj, _, _ = run_hall_pecht_discovery(noise_sigma=0.01, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "hp_true"
        assert adj.data_origin == "simulation"

    def test_rmse_scales_monotonically_with_noise(self):
        """Observed holdout RMSE must scale monotonically with injected noise."""
        rmses = []
        for noise in (0.01, 0.03, 0.10):
            adj, _, _ = run_hall_pecht_discovery(noise_sigma=noise, bootstrap=False)
            rmses.append(adj.best_rmse)
        assert rmses[0] < rmses[1] < rmses[2], f"RMSE did not scale monotonically: {rmses}"


class TestFinanceSandboxHonesty:
    def test_bull_regime_recovers_momentum(self):
        """In a bull regime (+5% drift), AR(1) momentum is uniquely supported."""
        prices = generate_finance_data(n_points=100, momentum=0.05, noise_sigma=0.02, regime="bull")
        adj = run_finance_discovery(prices, noise_sigma=0.02, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "momentum_ar1"
        assert adj.data_origin == "simulation"

    def test_bear_regime_recovers_momentum(self):
        """In a bear regime (-5% drift), AR(1) momentum is uniquely supported."""
        prices = generate_finance_data(n_points=100, momentum=0.05, noise_sigma=0.02, regime="bear")
        adj = run_finance_discovery(prices, noise_sigma=0.02, bootstrap=True)
        assert adj.status == EpistemicStatus.UNIQUE_SUPPORT
        assert adj.promoted is not None
        assert adj.promoted.candidate.candidate_id == "momentum_ar1"

    def test_sideways_regime_abstains(self):
        """In a sideways regime (0% drift), constant price and AR(1) are indistinguishable; adjudicator abstains."""
        prices = generate_finance_data(n_points=100, momentum=0.00, noise_sigma=0.02, regime="sideways")
        adj = run_finance_discovery(prices, noise_sigma=0.02, bootstrap=True)
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

    def test_real_observational_data_origin_threading(self):
        """FRED market series or passive observations thread data_origin='real_observational'."""
        prices = generate_finance_data(n_points=100, momentum=0.05, noise_sigma=0.02, regime="bull")
        adj = run_finance_discovery(prices, noise_sigma=0.02, bootstrap=True, data_origin="real_observational")
        assert adj.data_origin == "real_observational"
        assert adj.promoted.provenance["data_origin"] == "real_observational"


class TestPlasmaSandboxHonesty:
    def test_plasma_error_scales_monotonically(self):
        """Fusion power scaling candidate RMSE scales monotonically with observation noise."""
        datasets = generate_synthetic_dataset(n_conditions=15, noise_levels=(0.03, 0.05, 0.10))
        rmses = []
        for noise in (0.03, 0.05, 0.10):
            obs = datasets[f"noise_{noise}"]["observations"]
            adj = run_apedp_discovery(obs, noise_sigma=noise, bootstrap=False)
            rmses.append(adj.best_rmse)
        assert rmses[0] < rmses[1] < rmses[2], f"Plasma RMSE did not scale monotonically: {rmses}"


class TestTradingEngineSandboxHonesty:
    def test_trading_engine_regime_discrimination(self):
        """Trading engine detects momentum in bull and abstains with uncertain in sideways."""
        bull_prices = generate_price_series(n_points=80, momentum=0.05, noise_sigma=0.02, regime="bull", seed=42)
        res_bull = detect_regime(bull_prices, noise_sigma=0.02)
        assert res_bull["regime"] == "bull"
        assert res_bull["promoted"] == "mom_h1"

        side_prices = generate_price_series(n_points=80, momentum=0.00, noise_sigma=0.02, regime="sideways", seed=42)
        res_side = detect_regime(side_prices, noise_sigma=0.02)
        assert res_side["regime"] in ("sideways", "uncertain")


class TestQuantumVacuumSandboxHonesty:
    def test_low_noise_recovers_schwinger_mechanism(self):
        """At 1% noise, non-perturbative Schwinger mechanism is uniquely supported."""
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

    def test_rmse_scales_monotonically_with_noise(self):
        """Holdout RMSE of the true Schwinger candidate must scale monotonically with noise."""
        rmses = []
        for noise in (0.01, 0.03, 0.10):
            adj, _, _ = run_quantum_vacuum_discovery(noise_sigma=noise, bootstrap=False)
            rmses.append(adj.best_rmse)
        assert rmses[0] < rmses[1] < rmses[2], f"RMSE did not scale monotonically: {rmses}"

    def test_subthreshold_regime_abstains(self):
        """Below pair creation threshold (E << Ec), pair production is exponentially suppressed;
        models are observationally indistinguishable from vacuum stability; adjudicator abstains."""
        adj, _, _ = run_quantum_vacuum_discovery(
            noise_sigma=0.05,
            E_min=0.05,
            E_max=0.25,
            E_c=1.5,
            bootstrap=True,
        )
        assert adj.status == EpistemicStatus.MULTI_HYPOTHESIS
        assert adj.promoted is None

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

