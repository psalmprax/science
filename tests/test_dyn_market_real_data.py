"""Tests for Real Market Data Benchmark Suite (S&P 500 & Nasdaq 100).

Verifies APEDP's epistemic sovereignty and refusal of false promotion on genuine historical markets:
  1. Factor Zoo Overfitting Immunity on Returns:
     Refuses promotion of any candidate factor on daily returns, allocating 0.0% capital.
  2. Epistemic Sizing & Attribution on Market Regimes.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.adjudicate import EpistemicAttribution, EpistemicStatus
from apedp_dyn.sandbox.market_real_data_benchmark import (
    evaluate_real_market_price_collinearity,
    evaluate_real_market_return_discovery,
    load_market_series,
)


class TestRealMarketDataBenchmark:
    """Verifies epistemic behavior on real historical market data."""

    def test_load_market_series_validity(self):
        """Historical market series load successfully with >= 1400 trading days."""
        dates_sp, px_sp = load_market_series("sp500")
        assert len(dates_sp) >= 1400
        assert len(px_sp) == len(dates_sp)
        assert dates_sp[0] == "2018-01-02"
        assert px_sp[0] > 2000.0

        dates_nq, px_nq = load_market_series("nasdaq100")
        assert len(dates_nq) >= 1400
        assert len(px_nq) == len(dates_nq)
        assert px_nq[0] > 5000.0

    def test_factor_zoo_immunity_on_2019_calm_sp500(self):
        """On 2019 calm bull returns, open factor discovery abstains (0% capital allocated)."""
        rep = evaluate_real_market_return_discovery(
            asset="sp500",
            start_date="2019-01-02",
            end_date="2019-12-31",
            bootstrap=False,
        )
        assert rep.adjudication_status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert rep.epistemic_attribution in (
            EpistemicAttribution.OUT_OF_GRAMMAR_OR_MISSPECIFIED.value,
            EpistemicAttribution.MEASUREMENT_NOISE_SWAMPED.value,
        )
        assert rep.position_size == 0.0
        assert rep.promoted_expression == ""

    def test_factor_zoo_immunity_on_2020_covid_nasdaq(self):
        """During COVID crash/rebound, open factor discovery refuses spurious overfit."""
        rep = evaluate_real_market_return_discovery(
            asset="nasdaq100",
            start_date="2020-01-02",
            end_date="2020-06-30",
            bootstrap=False,
        )
        assert rep.adjudication_status == EpistemicStatus.INSUFFICIENT_EVIDENCE
        assert rep.position_size == 0.0

    def test_factor_zoo_immunity_on_2022_bear_market(self):
        """During 2022 bear market, returns remain unpromoted on both S&P 500 and Nasdaq."""
        for asset in ("sp500", "nasdaq100"):
            rep = evaluate_real_market_return_discovery(
                asset=asset,  # type: ignore[arg-type]
                start_date="2022-01-03",
                end_date="2022-12-30",
                bootstrap=False,
            )
            assert rep.adjudication_status == EpistemicStatus.INSUFFICIENT_EVIDENCE
            assert rep.position_size == 0.0

    def test_price_level_collinearity_and_adjudication(self):
        """On price levels, competing trend/lag models are adjudicated with bootstrap discipline."""
        rep = evaluate_real_market_price_collinearity(
            asset="sp500",
            start_date="2021-01-04",
            end_date="2022-12-30",
            bootstrap=True,
            seed=42,
        )
        # Verify valid adjudication status
        assert rep.adjudication_status in (
            EpistemicStatus.UNIQUE_SUPPORT,
            EpistemicStatus.MULTI_HYPOTHESIS,
        )
        if rep.adjudication_status == EpistemicStatus.MULTI_HYPOTHESIS:
            assert rep.position_size <= 0.5
        else:
            assert rep.promoted_expression == "AR1_lag"

    def test_missing_date_range_raises_key_error(self):
        """Querying dates outside available history raises KeyError."""
        with pytest.raises(KeyError):
            evaluate_real_market_return_discovery(
                asset="sp500",
                start_date="1999-01-01",
                end_date="1999-12-31",
            )
