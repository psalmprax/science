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
    def test_catalog_has_items(self):
        r = client.get("/api/sandbox/list")
        assert r.status_code == 200
        assert len(r.json()) >= 11
        assert any(it["id"] == "crypto_multi_grammar" for it in r.json())

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
        """Lynx cycle sandbox: the verdict agrees with its own margin.

        The fixture dependency was remediated on 2026-10-19: the canonical
        MacLulich series (R datasets::lynx, 1821-1934, 114 annual counts) is now
        vendored in-repo at `data/market/lynx_maclulich.csv` with provenance in
        `data/market/PROVENANCE.json`, and `load_lynx` reads it by default
        instead of `/tmp/lynx.csv`. The series is therefore tracked and the
        response contract is pinned against the real data: whatever verdict the
        referee reaches, the promotion flag, the margin and the reason must
        agree, and a boundary promotion must be labelled as one.
        """
        d = client.post("/api/sandbox/run", json={"sandbox_id": "lynx_cycle"}).json()

        assert d["verdict"] in (
            "unique_support", "multi_hypothesis", "insufficient_evidence")
        # 114 annual points, 1821-1934, whatever their values.
        assert len(d["series"]["x"]) == 114

        promoted = d["verdict"] == "unique_support"
        assert (d["promoted"] is not None) is promoted
        if promoted:
            assert d["delta_rmse"] > d["tau_epi"]
        elif d["delta_rmse"] is not None and d["tau_epi"] is not None:
            assert d["delta_rmse"] <= d["tau_epi"]
            assert "within τ" in d["reason"] or "straddles" in d["reason"]

        # Diagnostic labelling is a contract, not a data reading: a boundary
        # promotion must be reported as one, and the flag must agree.
        if d["promotion_confidence"] == "low_confidence":
            assert "WEAK_PROMOTION_MARGIN" in d["diagnostic_flags"]

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


class TestCustomUploadEndpoint:
    def test_list_contains_custom_upload(self):
        r = client.get("/api/sandbox/list")
        assert r.status_code == 200
        assert any(it["id"] == "custom_upload" for it in r.json())

    def test_custom_upload_run_endpoint(self):
        csv_sample = "t,x,y\n0.0,1.0,2.0\n0.1,1.1,2.2\n0.2,1.2,2.4\n0.3,1.3,2.6\n0.4,1.4,2.8\n0.5,1.5,3.0\n0.6,1.6,3.2\n0.7,1.7,3.4\n0.8,1.8,3.6\n0.9,1.9,3.8\n1.0,2.0,4.0\n1.1,2.1,4.2\n"
        r = client.post("/api/sandbox/run", json={
            "sandbox_id": "custom_upload",
            "csv_text": csv_sample,
            "target_column": "y",
            "mode": "l2_open",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["sandbox_id"] == "custom_upload"
        assert "verdict" in d
        assert "series" in d
        assert "candidates" in d
        assert d["extra"]["target_variable"] == "y"

    def test_dedicated_upload_endpoint(self):
        csv_sample = "date,price\n2023-01-01,100.0\n2023-01-02,101.5\n2023-01-03,102.1\n2023-01-04,101.8\n2023-01-05,103.0\n2023-01-06,104.2\n2023-01-07,103.9\n2023-01-08,105.1\n2023-01-09,106.0\n2023-01-10,105.5\n2023-01-11,107.2\n2023-01-12,108.0\n2023-01-13,107.5\n2023-01-14,109.1\n2023-01-15,110.0\n"
        r = client.post("/api/sandbox/upload", json={
            "csv_text": csv_sample,
            "target_column": "price",
            "mode": "dynamical_rate",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["sandbox_id"] == "custom_upload"
        assert "verdict" in d
        assert "series" in d

