"""Physics registry: every problem module addressable, demo four untouched."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine
from apedp_dyn.problems import mass_spring_damper as _MSD
from apedp_dyn.problems.base import finalize
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


class TestDriveFeaturesOffered:
    """Forced systems must offer the drive as evidence (MSD: 230x RMSE gap)."""

    @pytest.mark.slow
    def test_msd_recovers_drive_bearing_best(self):
        prob = finalize(_MSD.spec())
        Y = np.asarray(prob.train["truth_y"])
        t = np.asarray(prob.train["truth_t"])
        U = np.array([np.atleast_1d(np.asarray(prob.train["u_of_t"](ti), dtype=float))[0]
                      for ti in t])
        feats = {"x": Y[:, 0], "v": Y[:, 1], "u": U}
        eng = L2DiscoveryEngine(operators=("+", "-", "*", "/", "pow"),
                                max_depth=2, max_complexity=5)
        res = eng.discover(X=feats, Y=np.gradient(Y[:, 1], t),
                           dimensions={k: Dimension() for k in feats},
                           target_dimension=Dimension(), train_ratio=0.6,
                           bootstrap=False, seed=42)
        cands = list(getattr(res, "candidates", None)
                     or getattr(res, "pareto_candidates", None) or [])

        def expr(c):
            try:
                return (c.provenance.get("canonical_expr")
                        or c.candidate.metadata.get("canonical_expr")
                        or c.candidate.candidate_id)
            except (AttributeError, KeyError, TypeError):
                return "?"
        ok = sorted([(c.holdout_rmse, str(expr(c)))
                     for c in cands
                     if c.holdout_rmse is not None and np.isfinite(c.holdout_rmse)])
        assert ok[0][0] < 0.01
        assert "u" in ok[0][1]
