"""Unit test for real market L2 open discovery sweep."""
from __future__ import annotations

from apedp_dyn.sandbox.real_market_l2_sweep import (
    WINDOWS,
    execute_window_discovery,
)


class TestRealMarketL2Sweep:
    """Verifies the real market L2 open discovery sweep."""

    def test_single_window_sp500_calm_sweep(self):
        """Window W2 (2019 calm bull market): sole adequate candidate promotes under audit.

        Audit trail (2026-10-19): the grammar now yields exactly ONE adequate
        candidate for this window, ``vol - 1/ma_dev**(3/2)`` (complexity 4,
        pool-of-one: ranking and Pareto set each contain a single entry). The
        earlier pin expected a refusal because a reorder twin of the then-winner
        (``ma_dev**(-2)+sqrt(ma_dev)``) sat within tau; that twin is no longer
        generated, so the sole-survivor contract applies by design: absolute
        holdout fidelity (holdout RMSE 4.65e-3 vs holdout noise sigma 8.0e-3,
        fidelity margin 0.35 <= 1) plus whiteness (rho1 ~ -0.04) are required
        before promotion, and both hold.

        The apparent ~130x train/holdout error gap is a units artifact:
        ``train_loss`` is MSE (3.58e-5 -> RMSE ~ 6.0e-3) while
        ``holdout_error`` is already RMSE (4.65e-3). In RMSE terms the survivor
        generalizes (train 6.0e-3, holdout 4.7e-3), matching the historically
        accepted candidate's train/holdout profile.
        """
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False)
        assert rec.window_id == "W2_2019"
        assert rec.asset == "sp500"
        # Promotion under the hardened sole-survivor contract (pool of one).
        assert rec.epistemic_verdict == "unique_support"
        assert rec.epistemic_attribution == "decisive_evidential_support"
        assert "no structurally distinct rival survived scoring" in rec.full_reason
        # Anti-spurious guards: the survivor must actually be adequate,
        # otherwise this test would be endorsing "promoted because nothing fit".
        assert rec.out_of_grammar is False
        assert rec.holdout_error > 0
        assert rec.fidelity_margin is not None and rec.fidelity_margin <= 1.0
        assert rec.whiteness_passed is True
        assert rec.boundary_artifact is False
        assert "ma_dev" in rec.discovered_expression

    def test_single_window_nasdaq_covid_sweep(self):
        """Window W3 (2020 COVID shock on Nasdaq) recovers interaction term with UNIQUE_SUPPORT."""
        w3 = WINDOWS[2]  # W3_COVID
        rec = execute_window_discovery("nasdaq100", w3, bootstrap=False)
        assert rec.window_id == "W3_COVID"
        assert rec.asset == "nasdaq100"
        assert rec.epistemic_verdict == "unique_support"
        assert "vol" in rec.discovered_expression
        assert rec.out_of_grammar is False

    def test_multiterm_market_sweep_calm_refuses_spurious_models(self):
        """MultiTermL2Engine on W2_2019 calm market abstains with INSUFFICIENT_EVIDENCE."""
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False, multiterm=True)
        assert rec.window_id == "W2_2019"
        assert rec.epistemic_verdict == "insufficient_evidence"

    def test_multiterm_market_sweep_covid_promotes_parsimonious(self):
        """MultiTermL2Engine on W3_COVID selects parsimonious interaction term under Pareto pruning."""
        w3 = WINDOWS[2]  # W3_COVID
        rec = execute_window_discovery("nasdaq100", w3, bootstrap=False, multiterm=True)
        assert rec.window_id == "W3_COVID"
        assert rec.epistemic_verdict == "unique_support"
        assert "vol" in rec.discovered_expression

