"""Pins for PREREG-MMLM-ORDER-2026-11 (scripts/run_prereg_mmlm_order.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("run_prereg_mmlm_order", ROOT / "scripts" / "run_prereg_mmlm_order.py")
assert _spec is not None and _spec.loader is not None
mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mo)


def test_the_sets_and_their_sizes() -> None:
    final = mo.entries(mo.LAW_SEED)
    assert sum(e["set"] == "invented" for e in final.values()) == 50
    assert sum(e["set"] == "other_fields" for e in final.values()) == 24
    assert len(mo.units()) == 2 * 74


def test_the_rehearsal_reads_no_law_of_the_final_sets() -> None:
    final = {e["formula"] for e in mo.entries(mo.LAW_SEED).values()}
    rehearsal = [u[0] for u in mo.units(mo.REHEARSAL_SEED, rehearsal=True)]
    assert len(rehearsal) == 2 * mo.REHEARSAL_LAWS
    assert all(e["set"] == "invented" for e in rehearsal)
    assert not {e["formula"] for e in rehearsal} & final


def test_invented_laws_leave_out_the_memory_test_laws() -> None:
    import build_invented_laws as grammar
    import sympy as sp

    memory = {grammar.structure(sp.sympify(law["text"]))
              for law in json.loads((ROOT / "benchmarks" / "memory_blind_laws.json").read_text())["laws"]}
    drawn = {grammar.structure(sp.sympify(law["text"])) for law in mo.invented_laws()}
    assert not drawn & memory


def test_the_prior_is_mmlms_and_orders_candidates() -> None:
    prior = mo.mmlm_prior()
    assert callable(prior)


def test_the_summary_reads_each_expectation() -> None:
    def row(arm: str, problem: str, s: str, symbolic: bool, seconds: float) -> dict:
        return {"arm": arm, "problem": problem, "set": s, "noise": 0.0, "status": "ok", "symbolic": symbolic,
                "accuracy_solution": symbolic, "model": "x0" if symbolic else None, "seconds": seconds}
    rows = []
    for i in range(40):                              # 48 reported per arm, none wrong: the bound is under 10 %
        for s in ("invented", "other_fields"):
            rows.append(row("default", f"p{i}{s}", s, i < 24, 10.0))
            rows.append(row("mmlm", f"p{i}{s}", s, i < 24, 7.0))
    out = mo.summarise(rows)["expectations"]
    assert out["1_holds"] and out["2_holds"] and out["3_holds"]
    rows[1]["symbolic"] = False                      # MMLM loses one law
    rows[1]["accuracy_solution"] = False
    assert not mo.summarise(rows)["expectations"]["1_holds"]
    few = [r for r in rows if int(r["problem"][1:].rstrip("abcdefghijklmnopqrstuvwxyz_")) < 10]
    assert not mo.summarise(few)["expectations"]["3_holds"]   # 12 reported: too few for the bound, so it fails


def test_the_upper_bound() -> None:
    assert mo.upper_bound(0, 0) == 1.0
    assert 0.0 < mo.upper_bound(0, 60) < 0.06
    assert mo.upper_bound(2, 44) > 0.1


def test_the_final_run_refuses_a_changed_plan() -> None:
    if not mo.DIGEST.exists():
        pytest.skip("not frozen yet")
    mo.check_frozen()
