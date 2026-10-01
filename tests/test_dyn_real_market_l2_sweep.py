"""Unit test for real market L2 open discovery sweep."""
from __future__ import annotations

from apedp_dyn.sandbox.real_market_l2_sweep import (
    WINDOWS,
    execute_window_discovery,
)


class TestRealMarketL2Sweep:
    """Verifies the real market L2 open discovery sweep."""

    def test_single_window_sp500_calm_sweep(self):
        """Window W2 (2019 calm bull market) discovers without crash, yields INSUFFICIENT_EVIDENCE."""
        w2 = WINDOWS[1]  # W2_2019
        rec = execute_window_discovery("sp500", w2, bootstrap=False)
        assert rec.window_id == "W2_2019"
        assert rec.asset == "sp500"
        assert rec.epistemic_verdict == "insufficient_evidence"
        assert rec.out_of_grammar is True

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

