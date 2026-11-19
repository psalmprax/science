"""Engine competition: every applicable engine runs on the same data and each
result is returned unranked. The referee still decides promotion.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.orchestration.competition import (
    COMPETED,
    ENGINE_IDS,
    FAILED,
    NOT_APPLICABLE,
    competition_for_upload,
    run_competition,
)


def _linear_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    y = -0.5 * x + 0.01 * rng.normal(size=n)
    return {"x": x}, y


def test_order_book_and_sparse_microstructure_run_on_price_and_volume():
    rng = np.random.default_rng(2)
    n = 120
    price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    V = rng.uniform(1.0, 5.0, n)
    X = {"price": price, "V": V}
    by_engine = {r.engine: r for r in run_competition(X, price, train_ratio=0.6)}
    for engine in ("sparse_microstructure", "order_book_resiliency"):
        assert by_engine[engine].status == COMPETED, by_engine[engine].reason


def test_hamiltonian_and_solenoidal_run_when_columns_present():
    rng = np.random.default_rng(3)
    n = 120
    q = np.cos(np.linspace(0, 8, n))
    p = -np.sin(np.linspace(0, 8, n))
    X = {"q": q, "p": p}
    by_engine = {r.engine: r for r in run_competition(X, q, train_ratio=0.6)}
    assert by_engine["hamiltonian"].status != NOT_APPLICABLE
    X_fluid = {"x": q, "y": p, "u": rng.normal(size=n), "v": rng.normal(size=n)}
    by_fluid = {r.engine: r for r in run_competition(X_fluid, q, train_ratio=0.6)}
    assert by_fluid["solenoidal_fluid"].status != NOT_APPLICABLE


def test_named_column_proposer_runs_when_columns_present():
    rng = np.random.default_rng(1)
    n = 80
    S = 1000.0 - np.cumsum(rng.uniform(0, 5, n))
    infected = np.full(n, 10.0)
    recovered = 1000.0 - S - infected
    X = {"S": S, "I": infected, "R": recovered}
    results = run_competition(X, infected, train_ratio=0.6)
    by_engine = {r.engine: r for r in results}
    assert by_engine["compartmental_sir"].status == COMPETED
    assert by_engine["sindy"].status in {COMPETED, FAILED}


def test_every_engine_gets_one_record_in_order():
    X, Y = _linear_data()
    results = run_competition(X, Y)
    assert [r.engine for r in results] == list(ENGINE_IDS)
    assert all(r.status in {COMPETED, FAILED, NOT_APPLICABLE} for r in results)


def test_all_proposers_are_registered_and_named_column_ones_explain_why():
    import apedp_dyn.orchestration.competition as comp

    for engine in (
        "rime", "sindy", "basis_regression", "reservoir", "distributed_lag",
        "regime_switching", "ncm_structural", "rime_ad",
        "invariant_transfer", "sparse_microstructure",
        "order_book_resiliency", "hamiltonian", "solenoidal_fluid",
        "compartmental_sir",
    ):
        assert engine in ENGINE_IDS

    X, Y = _linear_data()
    by_engine = {r.engine: r for r in run_competition(X, Y)}
    for engine in comp._NAMED_COLUMN_PROPOSERS:
        assert by_engine[engine].status == NOT_APPLICABLE
        assert "named columns" in by_engine[engine].reason
    assert by_engine["sindy"].status == COMPETED


def test_not_applicable_engine_is_recorded_with_reason_and_not_run():
    X, Y = _linear_data()
    results = run_competition(
        X, Y, not_applicable={"l2_multiterm": "needs more columns"}
    )
    by_engine = {r.engine: r for r in results}
    skipped = by_engine["l2_multiterm"]
    assert skipped.status == NOT_APPLICABLE
    assert skipped.reason == "needs more columns"
    assert skipped.candidate_count == 0
    assert skipped.promoted_equation is None


def test_competed_engine_reports_referee_status_not_a_self_promotion():
    X, Y = _linear_data()
    results = run_competition(X, Y, not_applicable={"l2_multiterm": "skip"})
    single = {r.engine: r for r in results}["l2_single_term"]
    assert single.status == COMPETED
    # The referee's status name is passed through verbatim.
    assert single.adjudication_status != ""


def test_one_engine_failure_does_not_abort_the_others(monkeypatch):
    import apedp_dyn.orchestration.competition as comp

    def boom(*_a, **_k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(comp._RUNNERS, "l2_single_term", boom)
    X, Y = _linear_data()
    results = run_competition(X, Y, not_applicable={"l2_multiterm": "skip"})
    by_engine = {r.engine: r for r in results}
    assert by_engine["l2_single_term"].status == FAILED
    assert "synthetic failure" in by_engine["l2_single_term"].reason


def test_upload_competition_uses_csv_columns():
    csv = "x,y\n" + "\n".join(
        f"{x:.3f},{-0.5 * x:.4f}" for x in np.linspace(0.0, 6.0, 60)
    )
    records = competition_for_upload(csv, target_col="y")
    assert [r["engine"] for r in records] == list(ENGINE_IDS)
    assert all(r["status"] in {COMPETED, FAILED, NOT_APPLICABLE} for r in records)
    # Anything that did not compete must say why.
    assert all(r["reason"] for r in records if r["status"] != COMPETED)
    assert {r["engine"] for r in records if r["status"] == COMPETED} >= {"l2_single_term"}


def test_physics_and_market_routes_return_competition_array(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("APEDP_API_KEY", "test-key")
    from apedp_dyn.market.windows import WINDOWS
    from apedp_dyn.ui.server import app

    client = fastapi_testclient.TestClient(app)
    headers = {"x-api-key": "test-key"}
    phys = client.post(
        "/api/discover/physics",
        json={"system": "pendulum", "noise": 0.01, "n_samples": 60,
              "theta0": 1.0, "multiterm": False},
        headers=headers,
    )
    assert phys.status_code == 200
    assert [r["engine"] for r in phys.json()["competition"]] == list(ENGINE_IDS)

    market = client.post(
        "/api/discover/market",
        json={"asset": "sp500", "window_id": WINDOWS[0].window_id,
              "multiterm": False},
        headers=headers,
    )
    assert market.status_code == 200
    assert [r["engine"] for r in market.json()["competition"]] == list(ENGINE_IDS)


def test_upload_passes_named_columns_that_are_not_features():
    # q/p are not the target nor features here, so they must still reach the
    # named-column proposer.
    n = 120
    t = np.linspace(0, 8, n)
    csv = "t,q,p,y\n" + "\n".join(
        f"{ti:.4f},{np.cos(ti):.6f},{-np.sin(ti):.6f},{np.cos(ti):.6f}" for ti in t
    )
    records = {r["engine"]: r for r in competition_for_upload(csv, target_col="y")}
    assert records["hamiltonian"]["status"] != NOT_APPLICABLE
    assert "named columns" not in records["hamiltonian"]["reason"]


def test_upload_route_returns_competition_array(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("APEDP_API_KEY", "test-key")
    from apedp_dyn.ui.server import app

    csv = "x,y\n" + "\n".join(
        f"{x:.3f},{-0.5 * x + 0.1 * x * x:.4f}" for x in np.linspace(0.0, 6.0, 60)
    )
    client = fastapi_testclient.TestClient(app)
    resp = client.post(
        "/api/sandbox/upload",
        json={"sandbox_id": "custom_upload", "csv_text": csv, "target_column": "y"},
        headers={"x-api-key": "test-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "verdict" in body
    engines = [r["engine"] for r in body["competition"]]
    assert engines == list(ENGINE_IDS)
