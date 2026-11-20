"""The UI shows what the measured discovery routes conclude, beside the template engine's verdict.

Until 2026-11-20 the UI called the template engines only. The main search path
is the default route since PREREG-MAIN-SEARCH; these tests pin that the
endpoints carry its reading, that return-like data gets the series route and
not the main path alone (audit M-16), and that an unchallenged law is never
shown as a confirmed one (audit M-14).
"""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from apedp_dyn.discovery_l2 import route
from apedp_dyn.ui import route_endpoint
from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


def _kinetic(n: int = 240, seed: int = 0) -> tuple[dict[str, list[float]], list[float]]:
    rng = np.random.default_rng(seed)
    m, v = rng.uniform(0.5, 5.0, n), rng.uniform(1.0, 10.0, n)
    return {"m": m.tolist(), "v": v.tolist()}, (0.5 * m * v**2 + rng.normal(0.0, 0.05, n)).tolist()


def test_the_reading_runs_the_default_route_and_finds_the_law():
    columns, target = _kinetic()
    reading = route_endpoint.route_reading(columns, target)
    assert reading["route"] == route.DEFAULT_ROUTE == "main_search"
    assert reading["action"] in ("confirmed", "equivalent_class", "unchallenged")
    assert "m*v**2" in reading["law_as_confirmed"]
    assert reading["search_stage"].startswith("off")
    assert reading["complete"] is False          # first pass only: a partial reading, and the record says so
    assert reading["evidence"].endswith("RESULT-MAIN-SEARCH-2026-11.md")


def test_an_unchallenged_law_is_tentative_and_not_reported():
    columns, target = _kinetic()
    reading = route_endpoint.route_reading(columns, target)
    # exactly one of the two, and they follow from the action alone
    assert reading["reported"] == (reading["action"] in ("confirmed", "equivalent_class"))
    assert reading["tentative"] == (reading["action"] == "unchallenged")
    assert not (reading["reported"] and reading["tentative"])


def test_bad_data_gives_a_record_with_the_reason_and_no_law():
    for columns, target in (({"x": [1.0, 2.0, 3.0]}, [1.0, 2.0, 3.0]),          # too few rows
                            ({"x": [1.0] * 40}, [1.0] * 39),                     # misaligned
                            ({}, [1.0] * 40)):                                   # no inputs
        reading = route_endpoint.route_reading(columns, target)
        assert reading["action"] == "not_run" and reading["law"] is None and reading["reported"] is False
        assert reading["error"]


def test_the_series_reading_reports_nothing_in_pure_noise():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((600, 3))
    reading = route_endpoint.series_reading({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2]}, rng.standard_normal(600))
    assert reading["route"] == "series_route"
    assert reading["reported"] is False and reading["law"] is None
    assert reading["evidence"].endswith("RESULT-MARKET-PLANTED-2026-11.md")


def test_the_series_reading_reports_a_law_that_forecasts_the_later_rows():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(900)
    reading = route_endpoint.series_reading({"x": x}, 0.5 * x + rng.standard_normal(900))
    assert reading["action"] == "predictive_law" and reading["reported"] is True
    assert reading["forecast_test"]["passed"] is True
    assert reading["rows_development"] == 540 and reading["rows_later"] == 360


def test_the_status_page_names_the_default_route_and_the_retired_engines():
    r = client.get("/api/status")
    assert r.status_code == 200, r.text[:200]
    discovery = r.json()["discovery"]
    assert discovery["default_route"] == "main_search"
    assert {entry["name"] for entry in discovery["routes"]} >= {"main_search", "ode", "series_route", "l2"}
    assert "gagp" in discovery["retired_engines"]
    for entry in discovery["routes"]:
        assert entry["evidence"].startswith("preregistration/RESULT-")
    assert "/api/discover/route" in r.json()["api_endpoints"]


def test_the_physics_endpoint_carries_the_route_reading():
    r = client.post("/api/discover/physics", json={"system": "kinetic", "noise": 0.01, "multiterm": False})
    assert r.status_code == 200, r.text[:200]
    reading = r.json()["route_reading"]
    assert reading["route"] == "main_search" and reading["action"] != "not_run", reading
    assert "m*v**2" in (reading["law_as_confirmed"] or "")


def test_the_route_endpoint_takes_a_table_of_the_callers_own():
    columns, target = _kinetic()
    r = client.post("/api/discover/route", json={"columns": columns, "target": target})
    assert r.status_code == 200, r.text[:200]
    assert "m*v**2" in r.json()["law_as_confirmed"]
    ordered = client.post("/api/discover/route", json={"columns": columns, "target": target, "time_ordered": True})
    assert ordered.status_code == 200 and ordered.json()["route"] == "series_route"


def test_the_route_endpoint_refuses_a_bad_body_without_raising():
    for body in ({}, {"columns": {"x": [1, 2]}}, {"columns": [1, 2], "target": [1, 2]},
                 {"columns": {f"x{i}": [1.0] for i in range(13)}, "target": [1.0]}):
        r = client.post("/api/discover/route", json=body)
        assert r.status_code == 200, r.text[:200]
        assert r.json()["action"] == "not_run" and r.json()["error"]


def test_a_search_budget_without_pysr_is_reported_and_does_not_fail(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pysr(name, *args, **kwargs):
        if name == "pysr" or name.startswith("pysr."):
            raise ImportError("no PySR here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pysr)
    columns, target = _kinetic()
    reading = route_endpoint.route_reading(columns, target, search_seconds=5.0)
    assert reading["search_stage"].startswith("unavailable") and reading["complete"] is False
    assert "m*v**2" in reading["law_as_confirmed"]


def test_the_search_budget_is_bounded():
    assert route_endpoint._search_budget(1e9) == route_endpoint.MAX_SEARCH_SECONDS
    assert route_endpoint._search_budget(-5.0) == 0.0
    assert route_endpoint._search_budget(None) == 0.0


def test_the_status_says_which_endpoints_still_run_the_template_engine_alone():
    endpoints = route_endpoint.status_block()["endpoints"]
    assert "template engine and referee only" in endpoints["/api/discover/registry"]
    assert "route_reading" in endpoints["/api/discover/physics"]


def test_the_page_renders_the_reading_without_parsing_it_as_html():
    page = client.get("/").text
    start = page.index("function renderRouteReading")
    body = page[start:page.index("function renderSandboxResults")]
    assert "innerHTML" not in body
    assert "renderRouteReading('promoted-equation-box', data.route_reading)" in page
    assert "renderRouteReading('market-equation-box', data.series_reading)" in page
    assert 'id="search-stage-toggle"' in page and 'id="discovery-status-strip"' in page
    strip = page[page.index("async function loadDiscoveryStatus"):page.index("loadDiscoveryStatus();")]
    assert "innerHTML" not in strip


def test_the_market_endpoint_carries_the_series_reading_beside_the_template_verdict():
    r = client.post("/api/discover/market", json={"asset": "nasdaq100", "window_id": "W3_COVID"})
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    reading = body["series_reading"]
    assert reading["route"] == "series_route" and reading["action"] != "not_run", reading
    assert reading["rows_development"] + reading["rows_later"] > 100
    # a law is shown as reported only if it passed the forecast test on the later rows
    assert reading["reported"] == bool(reading["forecast_test"]["passed"])
    assert "epistemic_verdict" in body and "competition" in body          # what the endpoint returned before is still there


def test_the_server_file_did_not_grow_past_its_ceiling():
    """The endpoint lives in route_endpoint.py because ui/server.py is at its size ceiling."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ceiling = json.loads((root / "benchmarks" / "exception_hygiene_baseline.json").read_text())["size_budget"]["apedp_dyn/ui/server.py"]
    assert len((root / "apedp_dyn" / "ui" / "server.py").read_text(encoding="utf-8").splitlines()) <= ceiling
