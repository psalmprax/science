"""Physics registry: every problem module addressable, demo four untouched."""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


class TestRegistryCatalog:
    def test_lists_all_problem_modules(self):
        items = client.get("/api/physics/systems").json()
        ids = {it["id"] for it in items}
        assert {"pendulum", "duffing", "coupled", "lotka_volterra",
                "van_der_pol", "logistic"} <= ids
        assert "base" not in ids

    def test_entries_carry_equations(self):
        items = client.get("/api/physics/systems").json()
        by_id = {it["id"]: it for it in items}
        assert by_id["duffing"]["equations"], "truth equations must be exposed"


class TestRegistryRunner:
    def test_unknown_module_404(self):
        r = client.post("/api/discover/registry", json={"system": "nope"})
        assert r.status_code == 404

    def test_logistic_shape(self):
        d = client.post("/api/discover/registry",
                        json={"system": "logistic"}).json()
        assert d["system"] == "logistic"
        assert d["verdict"] in ("unique_support", "multi_hypothesis",
                                "insufficient_evidence", "no_valid_candidate")

    def test_demo_endpoint_still_accepts_literal(self):
        r = client.post("/api/discover/physics",
                        json={"system": "harmonic", "noise": 0.01})
        assert r.status_code == 200
