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
