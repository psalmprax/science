"""Regression tests for market data provenance and universe integrity hardening."""
from __future__ import annotations

import pandas as pd
import pytest

import apedp_dyn.market.sweep as market_sweep
import apedp_dyn.sandbox.real_market_l2_sweep as l2_sweep
from apedp_dyn.epistemic.adjudicate import DataOrigin
from apedp_dyn.market.assets import (
    AVAILABLE_ASSETS,
    REAL_OBSERVATIONAL_ASSETS,
    SYNTHETIC_PROXY_ASSETS,
    asset_data_origin,
)
from apedp_dyn.market.universe import audit_universe, find_stale_runs, mask_stale_bars


def test_asset_data_origin_partitions_available_assets():
    assert set(REAL_OBSERVATIONAL_ASSETS) | set(SYNTHETIC_PROXY_ASSETS) == set(AVAILABLE_ASSETS)
    assert set(REAL_OBSERVATIONAL_ASSETS).isdisjoint(SYNTHETIC_PROXY_ASSETS)
    for asset in REAL_OBSERVATIONAL_ASSETS:
        assert asset_data_origin(asset) == DataOrigin.REAL_OBSERVATIONAL
    for asset in SYNTHETIC_PROXY_ASSETS:
        assert asset_data_origin(asset) == DataOrigin.SYNTHETIC
    with pytest.raises(ValueError, match="Unknown asset"):
        asset_data_origin("unknown_index")


def test_market_sweep_passes_honest_data_origin(monkeypatch):
    seen_origins: list[tuple[str, bool, DataOrigin]] = []
    dates = [f"2020-01-{day:02d}" for day in range(1, 31)] + [
        f"2020-02-{day:02d}" for day in range(1, 21)
    ]

    def fake_load(asset: str):
        import numpy as np

        prices = 100.0 + np.linspace(0.0, 5.0, len(dates))
        return dates, prices

    def fake_discover(self, *, data_origin, **kwargs):
        seen_origins.append(("single", False, data_origin))

        class _Adj:
            status = type("S", (), {"value": "insufficient_evidence"})()
            attribution = type("A", (), {"value": "insufficient_power"})()
            promoted = None
            reason = "stubbed"

        return type("R", (), {"adjudication": _Adj(), "pareto_candidates": (), "all_evaluated": ()})()

    def fake_discover_multiterm(self, *, data_origin, **kwargs):
        seen_origins.append(("multi", True, data_origin))

        class _Adj:
            status = type("S", (), {"value": "insufficient_evidence"})()
            attribution = type("A", (), {"value": "insufficient_power"})()
            promoted = None
            reason = "stubbed"

        return type("R", (), {"adjudication": _Adj(), "pareto_candidates": (), "all_evaluated": ()})()

    monkeypatch.setattr(market_sweep, "load_series_with_dates", fake_load)
    monkeypatch.setattr(market_sweep.L2DiscoveryEngine, "discover", fake_discover)
    monkeypatch.setattr(market_sweep.MultiTermL2Engine, "discover_multiterm", fake_discover_multiterm)

    w = market_sweep.WindowSpec(
        window_id="W_TEST",
        train_start=dates[0],
        train_end=dates[25],
        holdout_start=dates[26],
        holdout_end=dates[-1],
        regime_description="test",
    )

    market_sweep.execute_window_discovery("sp500", w, bootstrap=False, multiterm=False)
    market_sweep.execute_window_discovery("russell2000", w, bootstrap=False, multiterm=False)
    market_sweep.execute_window_discovery("crypto_btc", w, bootstrap=False, multiterm=True)

    assert seen_origins == [
        ("single", False, DataOrigin.REAL_OBSERVATIONAL),
        ("single", False, DataOrigin.SYNTHETIC),
        ("multi", True, DataOrigin.SYNTHETIC),
    ]


def test_real_market_l2_sweep_rejects_unsupported_asset():
    with pytest.raises(ValueError, match="Unsupported real-market asset"):
        l2_sweep.load_series_with_dates("russell2000")  # type: ignore[arg-type]


def test_find_stale_runs_validation_and_near_equal_fill():
    panel = pd.DataFrame({
        "observation_date": [f"2021-01-{d:02d}" for d in range(1, 9)],
        "AAA_CLOSE": [10.0, 10.0, 10.0 + 1e-13, 10.0, 10.0 - 1e-13, 10.0, 10.0, 11.0],
        "AAA_VOL": [100.0, 100.0, 100.0, 100.0 + 1e-11, 100.0, 100.0, 100.0, 150.0],
        "NAN_CLOSE": [10.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 11.0, 12.0],
        "NAN_VOL": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    })
    with pytest.raises(ValueError, match="min_bars"):
        find_stale_runs(panel, min_bars=0)
    runs = find_stale_runs(panel, min_bars=5)
    assert [r.asset for r in runs] == ["AAA"]
    assert runs[0].bars == 6
    masked = mask_stale_bars(panel, min_bars=5)
    assert masked["AAA_CLOSE"].isna().sum() == 6
    assert pd.notna(masked["AAA_CLOSE"].iloc[0])
    assert pd.notna(masked["AAA_CLOSE"].iloc[-1])


def test_audit_universe_flags_non_positive_splits_and_timestamp_hindsight():
    panel = pd.DataFrame({
        "observation_date": [
            "2021-01-04T09:30:00",
            "2021-01-04T10:30:00",
            "2021-01-04T11:30:00",
        ],
        "NEG_CLOSE": [10.0, 0.0, 11.0],
        "NEG_VOL": [100.0, 105.0, 110.0],
        "SPLIT_CLOSE": [100.0, 25.0, 26.0],
        "SPLIT_VOL": [200.0, 210.0, 220.0],
        "OK_CLOSE": [50.0, 51.0, 52.0],
        "OK_VOL": [300.0, 310.0, 320.0],
    })
    audit = audit_universe(panel, selected_as_of="2021-01-04T10:00:00")
    assert audit.hindsight_selected is True
    assert audit.non_positive_price_assets == ("NEG",)
    assert audit.unadjusted_split_assets == ("SPLIT",)
    assert audit.fit_for_economic_claims is False
    assert any("non-positive" in reason for reason in audit.reasons)
    assert any("unadjusted splits" in reason for reason in audit.reasons)

    clean = panel[["observation_date", "OK_CLOSE", "OK_VOL"]]
    clean_audit = audit_universe(clean, selected_as_of="2021-01-04T09:30:00")
    assert clean_audit.hindsight_selected is False
    assert clean_audit.non_positive_price_assets == ()
    assert clean_audit.unadjusted_split_assets == ()
    assert clean_audit.fit_for_economic_claims is True
