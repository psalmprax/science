"""The route in force for tables since 2026-11-23: the default route, and the last checks only on request."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apedp_dyn.discovery_l2 import default_route, table_route, vouched_route

ROOT = Path(__file__).resolve().parents[1]
FAR = "0.52*x0 + 2.2"


def _rows(n: int = 700):
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 5.0, n)
    clean = 0.8 * x + 1.9 / (x + 0.6)
    return {"x0": x}, clean + 0.01 * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(n), rng.uniform(1.0, 5.0, (300, 1))


def _record(law: str) -> dict:
    path = SimpleNamespace(contrast=SimpleNamespace(equivalent=[]))
    return {"route": "refined", "action": "confirmed", "law": law, "law_as_confirmed": law, "reported": True,
            "tentative_law": None, "tentative_reason": None, "refined": SimpleNamespace(path=path)}


def test_by_default_the_record_is_the_default_routes_and_says_why_no_checks_ran(monkeypatch) -> None:
    monkeypatch.delenv(table_route.ENVIRONMENT_VARIABLE, raising=False)
    X, y, pool = _rows()
    out = table_route.with_checks_if_requested(_record(FAR), X, y, pool)
    assert out["law"] == FAR and out["reported"] is True                      # a wrong law is not stopped: as before 22 Nov
    assert out["vouching"]["applied"] is False and out["vouching"]["in_force"] is False
    assert table_route.DECISION in out["vouching"]["why_not"] and "APEDP_TABLE_VOUCHING=on" in out["vouching"]["why_not"]
    for value in ("", "off", "yes", "1"):                                     # only "on" asks for them
        monkeypatch.setenv(table_route.ENVIRONMENT_VARIABLE, value)
        assert not table_route.checks_requested()


def test_on_request_the_checks_run_exactly_as_the_frozen_module_runs_them(monkeypatch) -> None:
    monkeypatch.setenv(table_route.ENVIRONMENT_VARIABLE, "ON")
    X, y, pool = _rows()
    mine = table_route.with_checks_if_requested(_record(FAR), X, y, pool)
    theirs = vouched_route.checked(_record(FAR), X, y, pool)
    assert {k: v for k, v in mine.items() if k != "refined"} == {k: v for k, v in theirs.items() if k != "refined"}
    assert mine["law"] is None and mine["vouching"]["applied"] is True


def test_the_whole_route_and_what_it_stands_on(monkeypatch) -> None:
    monkeypatch.delenv(table_route.ENVIRONMENT_VARIABLE, raising=False)
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (300, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))
    y = -4.0 * X[:, 0] - 0.2 * X[:, 1]
    columns = {"a": X[:, 0], "b": X[:, 1]}
    out = table_route.discover_law(columns, y, pool, lambda rows: -4.0 * pool[rows, 0] - 0.2 * pool[rows, 1])
    plain = default_route.discover_law(columns, y, pool, lambda rows: -4.0 * pool[rows, 0] - 0.2 * pool[rows, 1])
    assert set(out) == set(plain) | {"vouching"} and out["law_as_confirmed"] == plain["law_as_confirmed"]
    # the decision stands on a recorded result whose third expectation missed and whose second held
    summary = json.loads((ROOT / "preregistration" / "RESULT-VOUCHED-SRBENCH-2026-11" / "results.json").read_text())["summary"]
    assert summary["expectations"]["2_holds"] and not summary["expectations"]["3_holds"]
    digest = json.loads((ROOT / "preregistration" / "DIGEST-VOUCHED-SRBENCH.json").read_text())["sha256"]
    for rel in ("apedp_dyn/discovery_l2/vouched_route.py", "apedp_dyn/discovery_l2/vouching.py"):
        assert digest[rel] == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()    # not edited to take them out
