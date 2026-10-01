"""Tests for APEDP Dyn Web Studio and API Endpoints.

Verifies:
  1. GET /: Serves dark-mode HTML dashboard.
  2. GET /api/status: Returns online status, physical systems, and market windows.
  3. POST /api/discover/physics:
     - Large angle pendulum recovers UNIQUE_SUPPORT with sin(theta).
     - Small angle pendulum recovers MULTI_HYPOTHESIS within tau_epi.
     - Multi-term damped oscillator discovery.
  4. POST /api/discover/market:
     - W3_COVID returns UNIQUE_SUPPORT for volatility interaction.
     - W2_2019 returns INSUFFICIENT_EVIDENCE (refuses spurious models).
  5. GET /api/challenge/ladder: Returns 10-rung False Certainty Ladder scorecard.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apedp_dyn.ui.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPEDPStudioAPI:
    """Rigorous tests for the APEDP Dyn Web Studio endpoints."""

    def test_dashboard_html_served(self, client):
        """Verifies GET / serves the interactive visual studio HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "APEDP" in resp.text
        assert "Autonomous Epistemic Discovery" in resp.text
        assert "Plotly" in resp.text or "plotly" in resp.text
        assert "katex" in resp.text

    def test_favicon_served(self, client):
        """Verifies GET /favicon.ico returns 200 SVG to prevent 404 console errors."""
        resp = client.get("/favicon.ico")
        assert resp.status_code == 200
        assert "image/svg+xml" in resp.headers.get("content-type", "")
        assert "Ψ" in resp.text


    def test_status_endpoint(self, client):
        """Verifies GET /api/status reports online system metadata."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["benchmark_level"] == "L2_OPEN_DISCOVERY"
        assert len(data["physics_systems"]) >= 4
        assert len(data["market_windows"]) >= 6

    def test_physics_discovery_large_angle_promotes(self, client):
        """Large angle pendulum discovers sin(theta) with UNIQUE_SUPPORT."""
        resp = client.post("/api/discover/physics", json={
            "system": "pendulum",
            "theta0": 2.0,
            "noise": 0.01,
            "multiterm": False,
            "n_samples": 80,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "unique_support"
        assert "sin" in data["promoted_equation"]
        assert len(data["pareto_candidates"]) >= 1
        assert "plot_series" in data
        assert len(data["plot_series"]["t"]) == 80

    def test_physics_discovery_small_angle_abstains(self, client):
        """Small angle pendulum (quasi-linear limit) returns MULTI_HYPOTHESIS within tau_epi."""
        resp = client.post("/api/discover/physics", json={
            "system": "pendulum",
            "theta0": 0.4,
            "noise": 0.01,
            "multiterm": False,
            "n_samples": 80,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "multi_hypothesis"
        cand_ids = [c["id"] for c in data["pareto_candidates"]]
        assert "theta" in cand_ids
        assert "sin(theta)" in cand_ids

    def test_market_discovery_covid_promotes(self, client):
        """W3_COVID shock on Nasdaq 100 promotes volatility interaction."""
        resp = client.post("/api/discover/market", json={
            "asset": "nasdaq100",
            "window_id": "W3_COVID",
            "multiterm": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["epistemic_verdict"] == "unique_support"
        assert "vol" in data["discovered_expression"]
        assert len(data["market_series"]["prices"]) > 0

    def test_market_discovery_calm_refuses_spurious_models(self, client):
        """W2_2019 calm bull market returns INSUFFICIENT_EVIDENCE."""
        resp = client.post("/api/discover/market", json={
            "asset": "sp500",
            "window_id": "W2_2019",
            "multiterm": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["epistemic_verdict"] == "insufficient_evidence"

    def test_challenge_ladder_endpoint(self, client):
        """GET /api/challenge/ladder returns all 10 verified rungs."""
        resp = client.get("/api/challenge/ladder")
        assert resp.status_code == 200
        ladder = resp.json()
        assert len(ladder) == 10
        for rung in ladder:
            assert rung["passed"] is True
            assert rung["status"] in ("unique_support", "multi_hypothesis", "insufficient_evidence")

    def test_stability_audit_endpoint(self, client):
        """POST /api/stability/audit returns multi-seed stability matrix."""
        resp = client.post("/api/stability/audit", json={
            "problem": "harmonic",
            "n_seeds": 3,
            "noise": 0.02,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_seeds"] == 3
        assert "matrix" in data
        assert "discovery" in data["matrix"]
        assert "identification" in data["matrix"]
        assert data["matrix"]["discovery"]["structural_recurrence"] == 1.0

    def test_trading_np93_benchmark_endpoint(self, client):
        """GET /api/benchmarks/trading-np93 returns frozen 93-window benchmark."""
        resp = client.get("/api/benchmarks/trading-np93")
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark"] == "TRADING-NP93"
        assert data["windows"] == 93
        assert data["unique"] == 0
        assert data["multi"] == 93
        assert data["promotion"] == 0
        assert data["false_promotion_rate"] == 0.0

    def test_nano_ds10_benchmark_endpoint(self, client):
        """GET /api/benchmarks/nano-ds10 returns frozen nanoscale stability scorecard."""
        resp = client.get("/api/benchmarks/nano-ds10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark"] == "NANO-DS10"
        assert data["fit_seeds"] == 10
        assert data["winner_recurrence"] == {"piecewise_hp_inv": 1.0}
        assert data["entropy_nats"] == 0.0
        assert data["k_hp_cv"] == pytest.approx(0.045, abs=0.005)
        assert data["adjudication_status"] == "multi_hypothesis"

    def test_duffing_forensic_invariant_and_dashboard_consistency(self, client):
        """Enforces the forensic binding invariant between promoted record and leaderboard table.

        Guarantees:
          1. summary.promoted_candidate_id == adjudication.promoted_candidate_id
          2. summary.holdout_rmse == leaderboard[promoted_id].holdout_rmse
          3. summary.gate4_r2 == leaderboard[promoted_id].gate4_r2
          4. summary.whiteness == leaderboard[promoted_id].whiteness
          5. delta_rmse == nearest_eligible_rival_rmse - promoted_rmse
        """
        resp = client.post("/api/discover/physics", json={
            "system": "duffing",
            "noise": 0.05,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["verdict"] == "unique_support"
        promoted = data["promoted_candidate"]
        assert promoted is not None
        assert promoted["id"] == "theta_1*(x) + theta_2*(v) + theta_3*((x)^(3))"
        assert promoted["holdout_rmse"] == pytest.approx(0.0380, abs=0.005)
        assert promoted["r2"] >= 0.99
        assert promoted["residual_rho1"] == pytest.approx(-0.049, abs=0.02)

        # Match leaderboard row
        prow = next(c for c in data["pareto_candidates"] if c["id"] == promoted["id"])
        assert prow["is_promoted"] is True
        assert prow["rmse"] == promoted["holdout_rmse"]
        assert prow["r2"] == pytest.approx(promoted["r2"], abs=1e-5)
        assert prow["rho1"] == pytest.approx(promoted["residual_rho1"], abs=1e-5)
        assert prow["is_white"] is True

        # Check nearest eligible rival and delta_rmse precision
        rival = next(c for c in data["pareto_candidates"] if c["id"] != promoted["id"] and c["complexity"] == 4 and "(x)^(3)" in c["id"])
        assert promoted["delta_rmse"] == pytest.approx(rival["rmse"] - promoted["holdout_rmse"], abs=1e-4)

    def test_registry_problem_discovery_endpoint(self, client):
        """Verifies generic L2 discovery across registry problem modules with full visual schema."""
        resp = client.post("/api/discover/registry", json={"system": "coupled"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["system"] == "coupled"
        assert data["verdict"] == "multi_hypothesis"
        assert "pareto_candidates" in data
        assert len(data["pareto_candidates"]) >= 1
        assert "plot_series" in data
        assert len(data["plot_series"]["t"]) == len(data["plot_series"]["actual"])
        assert "tau_epi" in data
        assert data["tau_epi"] > 0
        assert "reason" in data



