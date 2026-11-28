"""What a request may ask of the page's server, and what it is told when something fails (outside audit, 27 Nov 2026).

Four findings of that audit, each fixed outside the files the v1.18 freeze record pins: without a key the server
trusted any loopback peer, which behind a proxy on the same machine is everybody; the audit log was appended without
a lock; the registry route passed three search budgets and an operator list to the engine as they came; and its error
replies carried the exception's own text.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from apedp_dyn.ui import helpers, registry_endpoint


def _request(peer: str | None, headers: dict[str, str] | None = None) -> Any:
    return SimpleNamespace(client=None if peer is None else SimpleNamespace(host=peer), headers=headers or {})


@pytest.mark.parametrize("header", ["x-forwarded-for", "forwarded", "x-real-ip", "x-forwarded-host", "via"])
def test_without_a_key_a_request_that_came_through_a_proxy_is_not_served(monkeypatch: pytest.MonkeyPatch, header: str) -> None:
    monkeypatch.delenv("APEDP_API_KEY", raising=False)
    assert helpers._require_api_key(_request("127.0.0.1")) is None
    with pytest.raises(fastapi.HTTPException) as refused:
        helpers._require_api_key(_request("127.0.0.1", {header: "203.0.113.9"}))
    assert refused.value.status_code == 401 and "proxy" in refused.value.detail


def test_with_a_key_a_proxied_request_is_served_on_the_key_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEDP_API_KEY", "s3cret")
    assert helpers._require_api_key(_request("127.0.0.1", {"x-forwarded-for": "203.0.113.9", "x-api-key": "s3cret"})) is None
    with pytest.raises(fastapi.HTTPException):
        helpers._require_api_key(_request("127.0.0.1", {"x-forwarded-for": "203.0.113.9", "x-api-key": "wrong"}))


def test_the_audit_log_keeps_whole_lines_when_many_threads_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "audit.jsonl"
    monkeypatch.setattr(helpers, "_AUDIT_LOG", log)
    filler = "x" * 20_000                                    # longer than one pipe buffer, so a torn write would show

    def write(worker: int) -> None:
        for index in range(25):
            helpers._audit("test/endpoint", {"worker": worker, "index": index, "filler": filler})

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 200 and all(r["filler"] == filler and r["endpoint"] == "test/endpoint" for r in records)
    assert {(r["worker"], r["index"]) for r in records} == {(w, i) for w in range(8) for i in range(25)}


def test_the_registry_route_s_search_settings_have_defaults_and_ceilings() -> None:
    assert registry_endpoint.search_request({}) == {
        "noise": 0.0, "seed": 42, "max_terms": 3, "max_depth": 2, "max_term_complexity": 4,
        "operators": ("+", "-", "*", "/", "pow", "sin", "cos", "abs")}
    heaviest = registry_endpoint.search_request({"max_terms": 4, "max_depth": 3, "max_term_complexity": 8, "noise": 0.05,
                                                 "seed": 7, "operators": ["+", "*", "+", "sin"]})
    assert (heaviest["max_terms"], heaviest["max_depth"], heaviest["max_term_complexity"]) == (4, 3, 8)
    assert heaviest["operators"] == ("+", "*", "sin") and heaviest["seed"] == 7 and heaviest["noise"] == 0.05


@pytest.mark.parametrize("setting", [
    {"max_terms": 5}, {"max_terms": 0}, {"max_terms": 10 ** 9}, {"max_terms": "many"}, {"max_terms": 2.5}, {"max_terms": True},
    {"max_depth": 4}, {"max_depth": -1}, {"max_term_complexity": 9}, {"max_term_complexity": None},
    {"noise": -0.1}, {"noise": float("nan")}, {"noise": float("inf")}, {"noise": 11}, {"noise": "loud"},
    {"seed": -1}, {"seed": 2 ** 32}, {"seed": "x"},
    {"operators": []}, {"operators": "sin"}, {"operators": ["sin", "__import__"]}, {"operators": ["+"] * 40},
    {"operators": [["+"]]}, {"operators": {"+": 1}},
], ids=lambda s: f"{next(iter(s))}={str(next(iter(s.values())))[:14]}")
def test_a_setting_outside_its_limits_is_refused_and_named(setting: dict[str, Any]) -> None:
    with pytest.raises(fastapi.HTTPException) as refused:
        registry_endpoint.search_request(setting)
    assert refused.value.status_code == 422 and next(iter(setting)) in refused.value.detail


def _a_registered_problem() -> str:
    from apedp_dyn.ui import server

    return sorted(server._problem_modules())[0]


def test_a_request_over_a_ceiling_is_refused_before_any_work_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
    from apedp_dyn.problems import base

    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("work was started for a request that is over a ceiling")

    monkeypatch.setattr(base, "finalize", never)
    with pytest.raises(fastapi.HTTPException) as refused:
        registry_endpoint.run_registry_discovery({"system": _a_registered_problem(), "multiterm": True, "max_depth": 50})
    assert refused.value.status_code == 422 and "max_depth" in refused.value.detail


def test_a_failure_inside_the_engine_is_reported_without_the_exception_s_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from apedp_dyn.discovery_l2 import l2_engine
    from apedp_dyn.ui import server

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(server, "_audit", lambda endpoint, payload: written.append({"endpoint": endpoint, **payload}))

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("could not open /srv/secret/place/keys.json")

    monkeypatch.setattr(l2_engine.L2DiscoveryEngine, "discover", broken)
    name = _a_registered_problem()
    with pytest.raises(fastapi.HTTPException) as failed:
        registry_endpoint.run_registry_discovery({"system": name})
    assert failed.value.status_code == 500
    assert "secret" not in failed.value.detail and "keys.json" not in failed.value.detail
    assert "RuntimeError" in failed.value.detail and name in failed.value.detail
    assert written and "/srv/secret/place/keys.json" in written[-1]["reason"]         # the reason is kept, server-side


def test_an_unknown_problem_s_name_is_not_echoed_back() -> None:
    with pytest.raises(fastapi.HTTPException) as missing:
        registry_endpoint.run_registry_discovery({"system": "<script>alert(1)</script>"})
    assert missing.value.status_code == 404 and "script" not in missing.value.detail


# ---------------------------------------------------------------------------
# The layer in front of the routes (apedp_dyn/ui/request_limits.py)
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from apedp_dyn.ui import request_limits


def _small_app() -> tuple[FastAPI, list[Any]]:
    app, received = FastAPI(), []
    request_limits.install(app)

    @app.post("/api/stability/audit")
    def audit(body: dict[str, Any]) -> dict[str, Any]:
        received.append(body)
        return {"got": body}

    @app.post("/api/market/sweep")
    def sweep(body: dict[str, Any]) -> dict[str, Any]:
        try:
            raise OSError("cannot read /srv/data/prices/secret.parquet")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Market sweep failed: {e}") from e

    @app.post("/api/own/words")
    def own(body: dict[str, Any]) -> dict[str, Any]:
        raise HTTPException(status_code=500, detail="Cannot reconstruct plot series for term 'x' (available states: ['v'])")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"ok": True}

    return app, received


def test_the_page_s_server_has_the_limits_installed() -> None:
    from apedp_dyn.ui import server

    assert any(layer.cls is request_limits.RequestLimits for layer in server.app.user_middleware)
    assert server.app.exception_handlers[request_limits.StarletteHTTPException] is request_limits.server_error_without_the_cause


def test_a_body_passes_through_the_layer_unchanged() -> None:
    app, received = _small_app()
    client = TestClient(app)
    sent = {"n_seeds": 5, "noise": 0.02, "text": "π and   and a long tail " + "z" * 5000}
    reply = client.post("/api/stability/audit", json=sent)
    assert reply.status_code == 200 and reply.json() == {"got": sent} and received == [sent]
    assert client.get("/api/status").json() == {"ok": True}


@pytest.mark.parametrize("n_seeds, status", [(1, 200), (50, 200), (51, 422), (0, 422), (-3, 422), (10 ** 9, 422),
                                             (2.5, 422), ("many", 422), (True, 422), (None, 422)])
def test_the_length_of_the_stability_loop_is_bounded_before_the_route_runs(n_seeds: Any, status: int) -> None:
    app, received = _small_app()
    reply = TestClient(app).post("/api/stability/audit", json={"n_seeds": n_seeds})
    assert reply.status_code == status
    assert (len(received) == 1) is (status == 200)
    if status == 422:
        assert "n_seeds" in reply.json()["detail"]


def test_the_real_stability_route_refuses_a_loop_of_a_thousand_million_runs() -> None:
    from apedp_dyn.ui import server

    reply = TestClient(server.app).post("/api/stability/audit", json={"problem": "harmonic", "n_seeds": 10 ** 9})
    assert reply.status_code == 422 and "n_seeds must be between 1 and 50" in reply.json()["detail"]
    reply = TestClient(server.app).post("/api/market/sweep", json={"asset": "SPY", "max_windows": -4})
    assert reply.status_code == 422 and "max_windows" in reply.json()["detail"]


def test_a_body_over_the_limit_is_refused_whether_declared_or_streamed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_limits, "MAX_BODY_BYTES", 2000)
    app, received = _small_app()
    client = TestClient(app)
    big = {"n_seeds": 2, "pad": "p" * 5000}
    assert client.post("/api/stability/audit", json=big).status_code == 413
    assert client.post("/api/stability/audit", content=(chunk for chunk in [b'{"n_seeds": 2, "pad": "', b"p" * 5000, b'"}']),
                       headers={"content-type": "application/json"}).status_code == 413
    assert client.post("/api/stability/audit", content=b"{}", headers={"content-length": "-5"}).status_code == 400
    assert received == []
    assert client.post("/api/stability/audit", json={"n_seeds": 2}).status_code == 200


def test_a_5xx_reply_loses_the_text_of_the_exception_that_caused_it_and_keeps_its_own_words() -> None:
    app, _ = _small_app()
    client = TestClient(app)
    failed = client.post("/api/market/sweep", json={"asset": "SPY"})
    assert failed.status_code == 500
    assert failed.json()["detail"] == f"Market sweep failed: {request_limits.REASON_KEPT}"
    own = client.post("/api/own/words", json={})
    assert own.status_code == 500 and own.json()["detail"].startswith("Cannot reconstruct plot series for term 'x'")


def test_no_more_than_the_allowed_number_of_working_requests_run_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEDP_UI_MAX_IN_FLIGHT", "2")
    app, _ = _small_app()
    layer = request_limits.RequestLimits(app)
    layer.in_flight = 2                                    # two are being worked on

    import asyncio

    sent: list[Any] = []

    async def receive() -> Any:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: Any) -> None:
        sent.append(message)

    scope = {"type": "http", "path": "/api/stability/audit", "raw_path": b"/api/stability/audit", "method": "POST",
             "headers": [], "client": ("127.0.0.1", 1), "query_string": b"", "http_version": "1.1", "scheme": "http",
             "server": ("testserver", 80), "root_path": "", "asgi": {"version": "3.0"}}
    asyncio.run(layer(scope, receive, send))
    assert sent[0]["status"] == 429 and (b"retry-after", b"5") in sent[0]["headers"] and layer.in_flight == 2
    sent.clear()
    asyncio.run(layer({**scope, "method": "GET", "path": "/api/status", "raw_path": b"/api/status"}, receive, send))
    assert sent[0]["status"] == 200                        # reading is not held up by the work in progress


def test_a_request_counts_as_in_flight_while_its_body_is_still_arriving(monkeypatch: pytest.MonkeyPatch) -> None:
    # The count used to rise only after the body had been read, so any number of slow requests passed the check
    # together and all went on to do work (found by an outside review of 28 November 2026).
    import asyncio

    monkeypatch.setenv("APEDP_UI_MAX_IN_FLIGHT", "2")
    app, received = _small_app()
    layer = request_limits.RequestLimits(app)
    scope = {"type": "http", "path": "/api/stability/audit", "raw_path": b"/api/stability/audit", "method": "POST",
             "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1), "query_string": b"",
             "http_version": "1.1", "scheme": "http", "server": ("testserver", 80), "root_path": "", "asgi": {"version": "3.0"}}

    async def three_slow_requests() -> tuple[list[int], int, int]:
        body_may_arrive = asyncio.Event()
        statuses: list[int] = []

        async def receive() -> Any:
            await body_may_arrive.wait()
            return {"type": "http.request", "body": b'{"n_seeds": 3}', "more_body": False}

        async def send(message: Any) -> None:
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        tasks = [asyncio.ensure_future(layer(dict(scope), receive, send)) for _ in range(3)]
        await asyncio.sleep(0.05)                          # every request has passed the check; no body has arrived
        while_waiting, counted = list(statuses), layer.in_flight
        body_may_arrive.set()
        await asyncio.gather(*tasks)
        return while_waiting, counted, layer.in_flight

    while_waiting, counted, afterwards = asyncio.run(three_slow_requests())
    assert while_waiting == [429] and counted == 2          # the third was turned away before any body was read
    assert afterwards == 0 and len(received) == 2

    async def one(body: bytes) -> int:
        seen: list[int] = []

        async def receive() -> Any:
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: Any) -> None:
            if message["type"] == "http.response.start":
                seen.append(message["status"])

        await layer(dict(scope), receive, send)
        return seen[0]

    assert asyncio.run(one(b'{"n_seeds": 1000000000}')) == 422 and layer.in_flight == 0      # refused by the layer: not left counted

    async def broken(scope_: Any, receive_: Any, send_: Any) -> None:
        raise RuntimeError("the route failed")

    failing = request_limits.RequestLimits(broken)
    with pytest.raises(RuntimeError):
        asyncio.run(failing(dict(scope), lambda: asyncio.sleep(0, {"type": "http.request", "body": b"{}", "more_body": False}),
                            lambda message: asyncio.sleep(0)))
    assert failing.in_flight == 0                           # nor when the route itself fails


def test_a_caller_from_elsewhere_is_limited_per_minute_and_a_local_one_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APEDP_UI_RATE_LIMIT_PER_MINUTE", "3")
    layer = request_limits.RequestLimits(_small_app()[0])
    remote = {"type": "http", "path": "/api/status", "method": "GET", "headers": [], "client": ("203.0.113.9", 1)}
    assert [layer._refused_before_reading(remote) is None for _ in range(5)] == [True, True, True, False, False]
    refused = layer._refused_before_reading(remote)
    assert refused is not None and refused.status_code == 429 and int(refused.headers["retry-after"]) >= 1
    local = {**remote, "client": ("127.0.0.1", 1)}
    assert all(layer._refused_before_reading(local) is None for _ in range(50))
    other = {**remote, "client": ("198.51.100.4", 1)}
    assert layer._refused_before_reading(other) is None    # one caller's count is not another's


def test_only_old_scratch_directories_this_server_could_have_made_are_removed(tmp_path: Path) -> None:
    import os

    day = 24 * 3600.0
    old = tmp_path / "ext60_old00001"
    old.mkdir()
    (old / "apedp_P.json").write_text("{}")
    fresh = tmp_path / "ext60_fresh001"
    fresh.mkdir()
    (fresh / "apedp_P.json").write_text("{}")
    foreign = tmp_path / "ext60_foreign1"                  # the name fits, the contents are somebody else's
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    nested = tmp_path / "ext60_nested01"
    (nested / "apedp_dir.json").mkdir(parents=True)
    unrelated = tmp_path / "other_old"
    unrelated.mkdir()
    long_ago = 1_000_000_000.0
    for path in (old, foreign, nested, unrelated):
        os.utime(path, (long_ago, long_ago))
    removed = request_limits.remove_stale_scratch(tmp_path, now=long_ago + 2 * day)
    assert removed == ["ext60_old00001"]
    assert not old.exists() and fresh.exists() and foreign.exists() and nested.exists() and unrelated.exists()
    assert request_limits.remove_stale_scratch(tmp_path / "does_not_exist") == []
