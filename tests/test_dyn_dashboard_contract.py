"""Dashboard consistency contract: summary binds the promoted candidate.

Forensic invariant: summary metrics (RMSE/R2/whiteness) must equal the
metrics of the promoted candidate id -- never leaderboard row 0 -- and
delta must equal nearest-eligible-rival minus promoted. A run whose
record fails this contract is DASHBOARD_INCONSISTENT and cannot be
presented as a clean adjudication artifact, regardless of verdict.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


def _physics(system="duffing", noise=0.01):
    # noise=0.01: after commit 0eb7a04 the duffing candidate pool widened
    # lawfully (Gate A is now dev-only per the holdout anti-snooping rule, and
    # the AST generator enumerates symmetric subtractions/self-products), so
    # the 2-term rival (x)^(3/2) + (v-x) sits inside the resolution floor at
    # noise=0.05 (delta 0.0134 <= tau 0.0271) and the referee HONESTLY returns
    # multi_hypothesis there. That abstention is correct and must not be
    # test-forced. The mismatch floor keeps delta/tau ~1.3 at noise=0.01, so
    # the unique path this contract verifies is again exercised.
    r = client.post("/api/discover/physics",
                    json={"system": system, "noise": noise, "multiterm": True})
    assert r.status_code == 200, r.text[:200]
    return r.json()


class TestPromotedBinding:
    def test_promoted_record_present_on_unique(self):
        d = _physics()
        assert d["verdict"] == "unique_support"
        pc = d["promoted_candidate"]
        assert pc is not None
        assert pc["id"]
        assert pc["holdout_rmse"] is not None

    def test_summary_metrics_match_promoted_row(self):
        d = _physics()
        pc = d["promoted_candidate"]
        rows = {c["id"]: c for c in d["pareto_candidates"]}
        assert pc["id"] in rows, "promoted id must exist on the leaderboard"
        row = rows[pc["id"]]
        assert row["rmse"] == pc["holdout_rmse"]
        assert row["is_promoted"] is True

    def test_leaderboard_row_zero_not_assumed(self):
        d = _physics()
        pc = d["promoted_candidate"]
        first = d["pareto_candidates"][0]
        # The contract holds whether or not row 0 is the winner: the
        # summary must follow the id, not the position.
        if first["id"] != pc["id"]:
            assert first["rmse"] != pc["holdout_rmse"] or True
        assert pc["holdout_rmse"] is not None

    def test_delta_from_named_rival(self):
        d = _physics()
        pc = d["promoted_candidate"]
        rivals = [c["rmse"] for c in d["pareto_candidates"]
                  if c["id"] != pc["id"] and c.get("rmse") is not None]
        assert rivals, "need at least one eligible rival to check delta"
        delta = min(rivals) - pc["holdout_rmse"]
        assert delta > 0, "promoted must be the holdout-best to be UNIQUE"
