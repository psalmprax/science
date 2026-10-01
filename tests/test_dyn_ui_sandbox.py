"""Sandbox gallery endpoints: registry, runners, unknown-id handling."""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


class TestSandboxRegistry:
    def test_list_ids(self):
        r = client.get("/api/sandbox/list")
        assert r.status_code == 200
        ids = {it["id"] for it in r.json()}
        assert {"nano_m4", "sir_break", "quantum_quench"} <= ids

    def test_unknown_id_404(self):
        r = client.post("/api/sandbox/run", json={"sandbox_id": "nope"})
        assert r.status_code == 404


class TestSandboxRuns:
    def test_nano_m4_shape(self):
        r = client.post("/api/sandbox/run", json={"sandbox_id": "nano_m4"})
        assert r.status_code == 200
        d = r.json()
        assert d["verdict"] == "multi_hypothesis"
        assert d["promoted"] is None
        assert len(d["candidates"]) == 5
        assert d["extra"]["dc_hat_nm"] == 15.0
        assert len(d["series"]["x"]) == 60

    def test_sir_break_shape(self):
        r = client.post("/api/sandbox/run", json={"sandbox_id": "sir_break"})
        assert r.status_code == 200
        d = r.json()
        assert d["verdict"] in ("multi_hypothesis", "unique_support",
                                "insufficient_evidence")
        assert len(d["series"]["x"]) > 100
