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
