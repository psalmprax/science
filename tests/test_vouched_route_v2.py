"""The route in force for tables since 2026-11-23: the default route, with the second version of the last checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apedp_dyn.discovery_l2 import default_route, vouched_route, vouched_route_v2

ROOT = Path(__file__).resolve().parents[1]
FAR = "0.52*x0 + 2.2"
LAW = "0.8*x0 + 1.9/(x0 + 0.6)"


def _rows(n: int = 700):
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 5.0, n)
    clean = 0.8 * x + 1.9 / (x + 0.6)
    return {"x0": x}, clean + 0.01 * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(n), rng.uniform(1.0, 5.0, (300, 1))


def _record(law: str) -> dict:
    path = SimpleNamespace(contrast=SimpleNamespace(equivalent=[]))
    return {"route": "refined", "action": "confirmed", "law": law, "law_as_confirmed": law, "reported": True,
            "tentative_law": None, "tentative_reason": None, "refined": SimpleNamespace(path=path)}


def test_the_second_version_is_in_force_unless_the_environment_says_otherwise(monkeypatch) -> None:
    monkeypatch.delenv(vouched_route_v2.ENVIRONMENT_VARIABLE, raising=False)
    assert vouched_route_v2.version() == "second" and vouched_route_v2.in_force()
    for value, expected in (("off", "off"), (" OFF ", "off"), ("first", "first"), ("on", "first"), ("ON", "first"),
                            ("", "second"), ("yes", "second"), ("second", "second")):
        monkeypatch.setenv(vouched_route_v2.ENVIRONMENT_VARIABLE, value)
        assert vouched_route_v2.version() == expected, value


def test_by_default_a_wrong_law_is_withheld_and_the_law_is_kept(monkeypatch) -> None:
    monkeypatch.delenv(vouched_route_v2.ENVIRONMENT_VARIABLE, raising=False)
    X, y, pool = _rows()
    wrong = vouched_route_v2.checked(_record(FAR), X, y, pool)
    assert wrong["law"] is None and wrong["reported"] is False and wrong["tentative_law"] == FAR
    assert wrong["tentative_reason"].startswith(vouched_route_v2.NOT_VOUCHED)
    assert wrong["vouching"]["version"] == "second" and wrong["vouching"]["applied"] is True
    assert wrong["vouching"]["evidence"] == vouched_route_v2.EVIDENCE
    right = vouched_route_v2.checked(_record(LAW), X, y, pool)
    assert right["reported"] is True and right["law"] == LAW and right["vouching"]["applied"] is True


def test_off_gives_the_route_as_it_is_and_first_gives_the_first_version(monkeypatch) -> None:
    X, y, pool = _rows()
    monkeypatch.setenv(vouched_route_v2.ENVIRONMENT_VARIABLE, "off")
    plain = vouched_route_v2.checked(_record(FAR), X, y, pool)
    assert plain["law"] == FAR and plain["reported"] is True
    assert plain["vouching"]["applied"] is False and plain["vouching"]["in_force"] is False and "off" in plain["vouching"]["why_not"]
    monkeypatch.setenv(vouched_route_v2.ENVIRONMENT_VARIABLE, "first")
    mine = vouched_route_v2.checked(_record(FAR), X, y, pool)
    theirs = vouched_route.checked(_record(FAR), X, y, pool)
    assert mine["vouching"]["version"] == "first"
    strip = {"version", "note"}
    assert {k: v for k, v in mine["vouching"].items() if k not in strip} == theirs["vouching"]
    assert {k: v for k, v in mine.items() if k not in ("refined", "vouching")} == {
        k: v for k, v in theirs.items() if k not in ("refined", "vouching")}


def test_what_the_checks_do_not_touch(monkeypatch) -> None:
    monkeypatch.delenv(vouched_route_v2.ENVIRONMENT_VARIABLE, raising=False)
    X, y, pool = _rows()
    for record, why in (({**_record(FAR), "route": "main"}, "refined path only"),
                        ({**_record(FAR), "law": None, "reported": False}, "reported no law")):
        out = vouched_route_v2.checked(record, X, y, pool)
        assert out["vouching"]["applied"] is False and why in out["vouching"]["why_not"]
        assert out["law"] == record["law"]


def test_the_whole_route_and_what_it_stands_on(monkeypatch) -> None:
    monkeypatch.delenv(vouched_route_v2.ENVIRONMENT_VARIABLE, raising=False)
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (300, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))
    y = -4.0 * X[:, 0] - 0.2 * X[:, 1]
    columns = {"a": X[:, 0], "b": X[:, 1]}
    out = vouched_route_v2.discover_law(columns, y, pool, lambda rows: -4.0 * pool[rows, 0] - 0.2 * pool[rows, 1])
    plain = default_route.discover_law(columns, y, pool, lambda rows: -4.0 * pool[rows, 0] - 0.2 * pool[rows, 1])
    assert set(out) == set(plain) | {"vouching"} and out["law_as_confirmed"] == plain["law_as_confirmed"]
    # it stands on a recorded run in which all five expectations held
    summary = json.loads((ROOT / "preregistration" / "RESULT-VOUCHING-V2-2026-11" / "results.json").read_text())["summary"]
    assert all(summary["expectations"][f"{k}_holds"] for k in range(1, 6))
    # and the modules it wraps are the ones that run measured
    digest = json.loads((ROOT / "preregistration" / "DIGEST-VOUCHING-V2.json").read_text())["sha256"]
    for rel in ("apedp_dyn/discovery_l2/vouching_v2.py", "apedp_dyn/discovery_l2/vouching.py"):
        assert digest[rel] == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
