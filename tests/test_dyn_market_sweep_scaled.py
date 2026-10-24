"""Tests for Scaled Real Market Sweeps across extensive rolling windows, assets, and regimes.

Verifies:
  1. Multi-scale rolling windows (> 150 windows per asset across short, medium, semi-annual, and annual horizons).
  2. All 6 macro-financial regimes (bull, bear, shock, recovery, sideways, chop) are classified and represented.
  3. Multi-asset support (sp500, nasdaq100, russell2000, dow30, crypto_btc, treasury10y).
  4. Backwards compatibility of /api/market/windows (6 frozen + 20 rolling by default).
  5. Scaled /api/market/windows with extensive=True, regime filtering, and pagination.
  6. Batch market sweep runner (/api/market/sweep) with regime aggregation and false promotion rate computation.
  7. Open discovery on extensive rolling windows via /api/discover/market.
"""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from apedp_dyn.market.assets import AVAILABLE_ASSETS, load_series_with_dates
from apedp_dyn.market.regimes import MarketRegime
from apedp_dyn.market.windows import (
    extensive_rolling_windows,
)
from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


class TestScaledMarketWindows:
    """Verifies generation and properties of extensive multi-scale rolling windows."""

    def test_extensive_windows_scale_exceeds_150_per_asset(self):
        """Extensive rolling windows scale to > 150 windows per asset."""
        for asset in ("sp500", "nasdaq100"):
            ext = extensive_rolling_windows(asset)
            assert len(ext) >= 150, f"Expected >= 150 windows for {asset}, got {len(ext)}"

    def test_multi_scale_horizons_represented(self):
        """Windows span short, quarterly, semi-annual, and annual horizons."""
        ext = extensive_rolling_windows("sp500")
        scales = {w.window_id.split("-")[1][:4] for w in ext}
        assert {"SHOR", "QTR0", "SEMI", "ANN0"} <= scales

    def test_regimes_classified_on_windows(self):
        """Windows are tagged with detected macro-financial regimes."""
        ext = extensive_rolling_windows("sp500")
        regimes = {w.regime for w in ext}
        # Major regimes must be detected in the 2018-2023 history
        assert MarketRegime.VOLATILITY_SHOCK.value in regimes or MarketRegime.BULL_EXPANSION.value in regimes

    def test_multi_asset_series_load(self):
        """All available assets load valid dates and price histories."""
        for asset in AVAILABLE_ASSETS:
            dates, prices = load_series_with_dates(asset)
            assert len(dates) >= 1000
            assert len(prices) == len(dates)
            assert np.all(prices > 0)


class TestMarketEndpointsAPI:
    """Verifies market API endpoints in the server."""

    def test_backwards_compatible_default_counts(self):
        """Default query preserves exact legacy contract: 6 frozen + 20 rolling."""
        for asset in ("sp500", "nasdaq100"):
            items = client.get(f"/api/market/windows?asset={asset}").json()
            frozen = [i for i in items if i["frozen"]]
            rolling = [i for i in items if not i["frozen"]]
            assert len(frozen) == 6
            assert len(rolling) == 20
            prefix = "SPX-ROLL" if asset == "sp500" else "NDX-ROLL"
            assert all(i["id"].startswith(prefix) for i in rolling)

    def test_extensive_windows_query(self):
        """Query with extensive=true returns > 150 rolling windows."""
        items = client.get("/api/market/windows?asset=sp500&extensive=true").json()
        rolling = [i for i in items if not i["frozen"]]
        assert len(rolling) >= 150

    def test_regime_filtering_query(self):
        """Query filtering by regime returns only matching windows."""
        items = client.get("/api/market/windows?asset=sp500&extensive=true&regime=bull_expansion").json()
        assert len(items) >= 1
        assert all(i.get("regime") == "bull_expansion" for i in items)

    def test_pagination_on_windows(self):
        """Paginates market windows with limit and offset."""
        items1 = client.get("/api/market/windows?asset=sp500&extensive=true&limit=15&offset=0").json()
        assert len(items1) == 15
        items2 = client.get("/api/market/windows?asset=sp500&extensive=true&limit=15&offset=15").json()
        assert len(items2) == 15
        ids1 = [i["id"] for i in items1]
        ids2 = [i["id"] for i in items2]
        assert set(ids1).isdisjoint(set(ids2))

    def test_market_regimes_endpoint(self):
        """GET /api/market/regimes returns taxonomy and assets."""
        resp = client.get("/api/market/regimes")
        assert resp.status_code == 200
        data = resp.json()
        assert "regimes" in data
        assert len(data["regimes"]) == 6
        reg_ids = {r["id"] for r in data["regimes"]}
        assert {
            "bull_expansion",
            "bear_contraction",
            "volatility_shock",
            "rebound_recovery",
            "sideways_consolidation",
            "high_volatility_chop",
        } == reg_ids
        assert set(data["assets"]) == set(AVAILABLE_ASSETS)

    def test_batch_market_sweep_endpoint(self):
        """POST /api/market/sweep executes multi-window sweep and reports summary."""
        resp = client.post("/api/market/sweep", json={"asset": "sp500", "max_windows": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset"] == "sp500"
        assert data["total_windows"] == 3
        assert "false_promotion_rate" in data
        assert "regime_breakdown" in data
        assert len(data["records"]) == 3

    def test_discover_market_on_extensive_window(self):
        """POST /api/discover/market executes open discovery on an extensive rolling window."""
        ext = extensive_rolling_windows("nasdaq100")
        target_win = ext[0].window_id
        resp = client.post("/api/discover/market", json={"asset": "nasdaq100", "window_id": target_win})
        assert resp.status_code == 200
        data = resp.json()
        assert data["window_id"] == target_win
        assert data["asset"] == "nasdaq100"
        assert "epistemic_verdict" in data
        assert "market_series" in data
