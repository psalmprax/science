"""Sandbox gallery endpoints: registry, runners, unknown-id handling."""
from __future__ import annotations

import pytest
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


class TestSandboxGalleryExtended:
    def test_catalog_has_ten(self):
        r = client.get("/api/sandbox/list")
        assert r.status_code == 200
        assert len(r.json()) == 10

    def test_hall_petch_promotes(self):
        d = client.post("/api/sandbox/run", json={"sandbox_id": "hall_petch"}).json()
        assert d["verdict"] == "unique_support"
        assert d["promoted"] == "hp_true"
        assert len(d["series"]["x"]) == 50

    def test_creep_abstains(self):
        d = client.post("/api/sandbox/run", json={"sandbox_id": "creep_law"}).json()
        assert d["verdict"] == "multi_hypothesis"
        assert d["promoted"] is None
        assert len(d["candidates"]) == 4

    def test_game_abstains(self):
        d = client.post("/api/sandbox/run", json={"sandbox_id": "game_form"}).json()
        assert d["verdict"] == "multi_hypothesis"
        assert len(d["series"]["x"]) == 201

    def test_lynx_real_data(self):
        d = client.post("/api/sandbox/run", json={"sandbox_id": "lynx_cycle"}).json()
        assert d["verdict"] == "multi_hypothesis"
        assert len(d["series"]["x"]) == 114

    def test_qec_abstains(self):
        d = client.post("/api/sandbox/run", json={"sandbox_id": "qec_threshold"}).json()
        assert d["verdict"] == "multi_hypothesis"
        assert len(d["series"]["x"]) == 60


class TestExternal60Endpoints:
    def test_blocks_manifest(self):
        r = client.get("/api/benchmark/external60/blocks")
        assert r.status_code == 200
        blocks = {b["block"]: b["n_problems"] for b in r.json()}
        assert set(blocks) == {"A", "C", "E", "O", "P", "R", "X"}
        assert sum(blocks.values()) == 60

    def test_unknown_block_400(self):
        r = client.post("/api/benchmark/external60/run", json={"blocks": ["Z"]})
        assert r.status_code == 400

    @pytest.mark.slow
    def test_run_block_p(self):
        from pathlib import Path as _P
        r = client.post("/api/benchmark/external60/run", json={"blocks": ["P"]})
        assert r.status_code == 200
        d = r.json()
        assert d["blocks"][0]["n"] == 10
        # frozen round untouched: results live only under scratch
        scratch = _P(d.get("scratch_dir", ""))
        assert (scratch / "apedp_P.json").exists()
        assert "frozen" not in str(scratch)


class TestMarketWindowsRegistry:
    def test_counts_per_asset(self):
        c = TestClient(app, raise_server_exceptions=False)
        for asset in ("sp500", "nasdaq100"):
            items = c.get(f"/api/market/windows?asset={asset}").json()
            frozen = [i for i in items if i["frozen"]]
            rolling = [i for i in items if not i["frozen"]]
            assert len(frozen) == 6
            assert len(rolling) == 20
            assert all(i["id"].startswith("SPX-ROLL" if asset == "sp500" else "NDX-ROLL")
                       for i in rolling)

    def test_bad_asset_400(self):
        c = TestClient(app, raise_server_exceptions=False)
        assert c.get("/api/market/windows?asset=xyz").status_code == 400
