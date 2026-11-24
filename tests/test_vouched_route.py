"""The table route with the checks as its last step: what stays, what gives way, what is withheld, and what is left alone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import default_route, vouched_route, vouching
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase

ROOT = Path(__file__).resolve().parents[1]
LAW = "0.8*x0 + 1.9/(x0 + 0.6)"
FAR = "0.52*x0 + 2.2"                  # a line: far from the law


def _rows(noise: float = 0.01, n: int = 700, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    clean = 0.8 * x + 1.9 / (x + 0.6)
    y = clean + noise * float(np.sqrt(np.mean(clean ** 2))) * rng.standard_normal(n)
    return {"x0": x}, y, rng.uniform(1.0, 5.0, (300, 1))


def _record(law: str | None, equivalent=(), route: str = "refined", reported: bool = True) -> dict:
    """What ``default_route.discover_law`` hands on, as far as the checks read it."""
    path = SimpleNamespace(contrast=SimpleNamespace(equivalent=list(equivalent)))
    return {"route": route, "action": "equivalent_class" if equivalent else "confirmed", "law": law, "law_as_confirmed": law,
            "reported": reported, "tentative_law": None, "tentative_reason": None, "refined": SimpleNamespace(path=path)}


@pytest.fixture(autouse=True)
def _in_force(monkeypatch):
    monkeypatch.delenv(vouched_route.ENVIRONMENT_VARIABLE, raising=False)


def test_a_law_that_passes_is_reported_as_the_route_reported_it() -> None:
    X, y, pool = _rows()
    out = vouched_route.checked(_record(LAW), X, y, pool)
    assert out["law"] == LAW and out["reported"] and out["tentative_law"] is None
    state = out["vouching"]
    assert state["in_force"] and state["applied"] and state["reason"] == "" and not state["switched"]
    assert state["route_law"] == LAW and state["rows"] == 700 and state["evidence"] == vouched_route.EVIDENCE


def test_a_wrong_law_gives_way_to_a_member_of_its_class_that_passes() -> None:
    X, y, pool = _rows()
    out = vouched_route.checked(_record(FAR, [LAW]), X, y, pool)
    assert out["law"] == LAW and out["law_as_confirmed"] == LAW and out["reported"]
    assert out["vouching"]["switched"] and out["vouching"]["route_law"] == FAR and out["vouching"]["refused"] == 1


def test_a_wrong_law_with_nothing_beside_it_is_withheld_and_shown_with_the_reason() -> None:
    X, y, pool = _rows()
    out = vouched_route.checked(_record(FAR), X, y, pool)
    assert out["law"] is None and out["reported"] is False
    assert out["tentative_law"] == FAR and out["law_as_confirmed"] == FAR             # still shown, as not vouched for
    assert out["tentative_reason"].startswith(vouched_route.NOT_VOUCHED + ": ") and "structure" in out["tentative_reason"]


def test_the_switch_and_the_cases_the_checks_were_not_measured_on_leave_the_record_alone(monkeypatch) -> None:
    X, y, pool = _rows()
    for record, why in ((_record(FAR, route="main_search"), "refined path only"), (_record(None, reported=False), "no law"),
                        (_record(FAR, reported=False), "no law")):
        out = vouched_route.checked(record, X, y, pool)
        assert {k: out[k] for k in record} == record and not out["vouching"]["applied"] and why in out["vouching"]["why_not"]
    monkeypatch.setenv(vouched_route.ENVIRONMENT_VARIABLE, "OFF")
    assert not vouched_route.vouching_in_force()
    out = vouched_route.checked(_record(FAR), X, y, pool)
    assert out["law"] == FAR and out["reported"] and out["vouching"] == {
        "in_force": False, "applied": False, "evidence": vouched_route.EVIDENCE,
        "why_not": f"switched off ({vouched_route.ENVIRONMENT_VARIABLE}=off)"}


def test_checks_that_crash_vouch_for_nothing(monkeypatch) -> None:
    X, y, pool = _rows()

    def broken(*args, **kwargs):
        raise MemoryError("too many pairs")

    monkeypatch.setattr(vouching, "vouch", broken)
    out = vouched_route.checked(_record(LAW), X, y, pool)
    assert out["law"] is None and not out["reported"] and out["tentative_law"] == LAW
    assert "the checks failed: MemoryError" in out["tentative_reason"]


def test_wide_tables_are_given_fewer_rows_and_narrow_ones_what_the_checks_were_measured_with() -> None:
    assert vouched_route.rows_given(700, 1) == 700 and vouched_route.rows_given(50, 3) == 50
    assert vouched_route.rows_given(14_000, 3) == vouching.MAX_ROWS == vouched_route.rows_given(14_000, 7)
    assert vouched_route.rows_given(14_000, 12) == 1154 and vouched_route.rows_given(14_000, 5_000) == vouched_route.FEWEST_ROWS
    # more rows than the checks' limit: the same draw the checks would have made themselves
    X, y, pool = _rows(n=2_000)
    seen = {}

    def spy(law, others, names, dev_X, dev_y, more, **kwargs):
        seen["rows"] = dev_X[:, 0].copy()
        return vouching.Vouched(law, "")

    import unittest.mock as mock

    with mock.patch.object(vouching, "vouch", spy):
        vouched_route.checked(_record(LAW), X, y, pool)
    theirs = np.sort(np.random.default_rng(0).choice(2_000, vouching.MAX_ROWS, replace=False))
    assert np.array_equal(seen["rows"], X["x0"][theirs])


class _Scripted(ProposerBase):
    """A search stage that proposes fixed formulas: stands in for PySR."""

    def __init__(self, formulas):
        self.formulas = formulas

    @property
    def proposer_id(self) -> str:
        return "scripted"

    def fit_and_propose(self, train_data, target_col=None, guesses=None, **kwargs):
        return tuple(CandidateProposal(proposer_id="scripted", candidate_id=f"s{i}", formula=f, strategy_archetype=None,
                                       declared_scale_invariant=False) for i, f in enumerate(self.formulas))


def test_the_whole_route_takes_the_same_arguments_and_adds_what_the_checks_did() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 5.0, (300, 2))
    pool = rng.uniform(1.0, 5.0, (600, 2))

    def law(a, b):
        return -4.0 * a - 0.2 * b

    y = law(*X.T) * (1 + 0.002 * rng.standard_normal(300))
    noise = np.random.default_rng(1)

    def measure(rows):
        return law(*pool[rows].T) * (1 + 0.002 * noise.standard_normal(len(rows)))

    columns = {"a": X[:, 0], "b": X[:, 1]}
    stage = _Scripted(["a + b", "a*b", "a - b**2"])
    plain = default_route.discover_law(columns, y, pool, measure, proposer=stage)
    noise = np.random.default_rng(1)
    out = vouched_route.discover_law(columns, y, pool, measure, proposer=stage)
    assert out["route"] == "refined" and out["action"] == plain["action"] and set(plain) | {"vouching"} == set(out)
    state = out["vouching"]
    if plain["reported"]:
        assert state["applied"] and state["route_law"] == plain["law"]
        assert out["law"] in (plain["law"], None) or state["switched"]
        if out["law"] is None:
            assert out["tentative_law"] == plain["law"] and out["tentative_reason"].startswith(vouched_route.NOT_VOUCHED)
    else:
        assert not state["applied"] and out["law"] is None
    # the earlier routes are handed on as they were measured
    earlier = vouched_route.discover_law(columns, y, pool, measure, route_name="main_search", proposer=stage)
    assert earlier["route"] == "main_search" and not earlier["vouching"]["applied"]


def test_the_module_stands_on_a_recorded_result_and_the_frozen_modules_are_not_edited() -> None:
    """PREREG-COUNTERFEIT made its third, fourth and fifth expectations the condition for putting the checks in force."""
    record = json.loads((ROOT / "preregistration" / "RESULT-COUNTERFEIT-2026-11" / "results.json").read_text(encoding="utf-8"))
    expectations = record["summary"]["expectations"]
    assert expectations["3_holds"] and expectations["4_holds"] and expectations["5_holds"]
    assert (ROOT / vouched_route.EVIDENCE).is_file()
    digest = json.loads((ROOT / "preregistration" / "DIGEST-COUNTERFEIT.json").read_text(encoding="utf-8"))["sha256"]
    assert record["frozen"] == digest
    for path in ("apedp_dyn/discovery_l2/vouching.py", "apedp_dyn/discovery_l2/refined_search.py"):
        assert digest[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    sums = (ROOT / "preregistration" / "RESULT-COUNTERFEIT-2026-11" / "SHA256SUMS").read_text(encoding="utf-8").split()
    for want, name in zip(sums[0::2], sums[1::2], strict=True):
        assert hashlib.sha256((ROOT / "preregistration" / "RESULT-COUNTERFEIT-2026-11" / name).read_bytes()).hexdigest() == want
