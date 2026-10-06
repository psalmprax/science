"""Tests for combined 15-asset crypto grammar discovery sandbox."""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.sandbox.crypto_grammar import (
    load_crypto_dataset,
    run_crypto_experiment,
)
from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


def test_crypto_dataset_loading():
    """Verify data/market/crypto_top15.csv contains 15 assets + composite across 3 years."""
    ds = load_crypto_dataset()
    dates = ds["dates"]
    assets = ds["assets"]
    symbols = ds["symbols"]

    assert len(symbols) == 15
    assert "BTC" in symbols and "ETH" in symbols and "SOL" in symbols
    assert len(dates) >= 1000  # 3 full years: 2021-2023
    assert dates[0] == "2021-01-01"
    assert dates[-1] == "2023-12-31"

    # Verify all 15 assets have non-empty, finite data
    for s in symbols:
        assert s in assets
        assert len(assets[s]) == len(dates)
        assert not any(assets[s] <= 0)

    # Verify composite basket exists
    assert "COMPOSITE_TOP15" in assets
    assert len(assets["COMPOSITE_TOP15"]) == len(dates)


def test_crypto_grammar_experiment_composite():
    """Verify mechanical grammar discovery on the 15-asset composite basket."""
    adj, _prob, series_data = run_crypto_experiment(target_asset="COMPOSITE_TOP15")
    assert adj.status.value in ("multi_hypothesis", "unique_support", "insufficient_evidence")
    assert len(series_data["actual"]) > 10
    assert series_data["target"] == "COMPOSITE_TOP15"


def test_crypto_sandbox_api_endpoint():
    """Verify POST /api/sandbox/run with crypto_multi_grammar returns standard schema."""
    r = client.post("/api/sandbox/run", json={"sandbox_id": "crypto_multi_grammar"})
    assert r.status_code == 200
    data = r.json()

    assert data["sandbox_id"] == "crypto_multi_grammar"
    assert data["verdict"] in ("multi_hypothesis", "unique_support")
    assert "tau_epi" in data
    assert "delta_rmse" in data
    assert "pareto_candidates" in data
    assert len(data["pareto_candidates"]) > 0

    # Ensure Candidate Leaderboard has populated metrics (not missing/None)
    lead = data["pareto_candidates"][0]
    assert "formula" in lead and len(lead["formula"]) > 0
    assert "complexity" in lead and lead["complexity"] >= 1
    assert "rmse" in lead and lead["rmse"] is not None
    assert "r2" in lead
    assert "rho1" in lead

    # Verify extra metadata
    extra = data.get("extra", {})
    assert extra.get("total_assets") == 15
    assert "COMPOSITE_TOP15" in extra.get("available_assets", [])


def test_prospective_tournament_execution():
    """Verify prospective validation tournament executes across frozen 3-tier grammar library."""
    from apedp_dyn.sandbox.prospective_regime_tournament import run_prospective_tournament
    res = run_prospective_tournament("COMPOSITE_TOP15")
    assert res["target_asset"] == "COMPOSITE_TOP15"
    assert len(res["models"]) >= 10
    assert res["verdict"] in ("no_valid_candidate", "multi_hypothesis", "unique_support")
    assert res["tolerance_2023"] > 0

