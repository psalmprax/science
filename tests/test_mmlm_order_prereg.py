"""Pins for PREREG-MMLM-ORDER-2026-11 (scripts/run_prereg_mmlm_order.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
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


@pytest.fixture(scope="module")
def final() -> dict:
    return mo.entries()


def test_the_sets_and_their_sizes(final: dict) -> None:
    sizes = Counter(e["set"] for e in final.values())
    assert sizes == {"other_fields": 23, "odebench": 12, "power_laws": 30, "guard": 30}
    assert len(mo.units()) == 2 * 95
    assert "field_reynolds_number" not in final                         # the same law as the ideal gas's
    assert sum(bool(e.get("training_shape")) for e in final.values()) == len(mo.TRAINING_SHAPES)


def test_every_familiar_law_is_reachable_and_one_per_shape(final: dict) -> None:
    keys = [mo._key(e["formula"]) for e in final.values() if e["set"] in ("odebench", "power_laws")]
    assert all(k[1] is not None for k in keys) and len(keys) == len(set(keys))


def test_the_rehearsal_reads_no_shape_of_the_final_sets(final: dict) -> None:
    import build_invented_laws as grammar
    import sympy as sp

    rehearsal = mo.entries(rehearsal=True)
    assert Counter(e["set"] for e in rehearsal.values()) == {"guard": 8, "power_laws": 8}
    reach = {mo._key(e["formula"]) for e in final.values() if e["set"] != "guard"}
    assert not reach & {mo._key(e["formula"]) for e in rehearsal.values() if e["set"] == "power_laws"}
    shape = {grammar.structure(sp.sympify(e["formula"])) for e in final.values() if e["set"] == "guard"}
    assert not shape & {grammar.structure(sp.sympify(e["formula"])) for e in rehearsal.values() if e["set"] == "guard"}


def test_the_guard_leaves_out_the_memory_test_laws(final: dict) -> None:
    import build_invented_laws as grammar
    import sympy as sp

    memory = {grammar.structure(sp.sympify(law["text"]))
              for law in json.loads((ROOT / "benchmarks" / "memory_blind_laws.json").read_text())["laws"]}
    assert not memory & {grammar.structure(sp.sympify(e["formula"])) for e in final.values() if e["set"] == "guard"}


def test_both_noise_levels_of_a_law_share_a_shard() -> None:
    dealt = mo.shards(False, 20)
    sizes = Counter(dealt.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1


def _row(arm: str, problem: str, s: str, symbolic: bool, seconds: float = 10.0, noise: float = 0.0,
         status: str = "ok") -> dict:
    return {"arm": arm, "problem": problem, "set": s, "noise": noise, "status": status, "symbolic": symbolic,
            "accuracy_solution": symbolic, "model": "x0" if symbolic else None, "seconds": seconds,
            "end_to_end_seconds": 240.0 + seconds}


def _rows(gained: int, lost: int, ties: int = 20) -> list[dict]:
    rows = []
    for i in range(gained):
        rows += [_row("mmlm", f"g{i}", "power_laws", True), _row("default", f"g{i}", "power_laws", False)]
    for i in range(lost):
        rows += [_row("mmlm", f"l{i}", "odebench", False), _row("default", f"l{i}", "odebench", True)]
    for i in range(ties):
        rows += [_row(a, f"t{i}", "guard", True) for a in mo.ARMS]
    return rows


def _units(rows: list[dict]) -> int:
    return len({(r["problem"], r["noise"]) for r in rows})


def test_the_summary_reads_each_expectation() -> None:
    rows = _rows(8, 1)
    out = mo.summarise(rows, _units(rows))
    assert out["expectations"]["read"] and all(out["expectations"]["held"].values())
    assert out["outcome"].startswith("all four held")
    rows = _rows(3, 1)                                           # more gained, but p = 0.31
    out = mo.summarise(rows, _units(rows))
    assert not out["expectations"]["held"]["1"] and out["outcome"].startswith("list MMLM")
    rows = [*_rows(8, 1), _row("mmlm", "w", "guard", False), _row("default", "w", "guard", False)]
    rows[-2]["model"] = "x1"                                     # MMLM reports a wrong law, the default nothing
    assert not mo.summarise(rows, _units(rows))["expectations"]["held"]["3"]
    slow = [{**r, "end_to_end_seconds": 400.0} if r["arm"] == "mmlm" else r for r in _rows(8, 1)]
    assert not mo.summarise(slow, _units(slow))["expectations"]["held"]["4"]
    guard_loss = [*_rows(8, 1), _row("mmlm", "gl", "guard", False), _row("default", "gl", "guard", True)]
    assert not mo.summarise(guard_loss, _units(guard_loss))["expectations"]["held"]["2"]


def test_missing_units_or_failed_searches_leave_it_not_read() -> None:
    rows = _rows(8, 1)
    assert not mo.summarise(rows, _units(rows) + 1)["expectations"]["read"]
    failed = rows + [_row(a, f"f{i}", "guard", False, status="search_failed") for i in range(10) for a in mo.ARMS]
    out = mo.summarise(failed, _units(failed))
    assert not out["expectations"]["read"] and "failed search" in out["expectations"]["why_not_read"]


def test_one_arm_failing_in_its_own_process_leaves_the_other() -> None:
    def boom() -> None:
        raise RuntimeError("only this arm")
    assert mo._run(boom, (), 30)[0] == "error"
    assert mo._run(lambda: {"x": 1}, (), 30)[:2] == ("ok", {"x": 1})


def test_the_sign_test() -> None:
    assert mo.sign_test(0, 0) == 1.0
    assert abs(mo.sign_test(8, 1) - 10 / 512) < 1e-9
    assert mo.sign_test(5, 0) == pytest.approx(1 / 32)


def test_the_prior_is_mmlms() -> None:
    prior = mo.mmlm_prior()
    from apedp_dyn.discovery_l2 import exhaustive_search as es

    nodes = es.enumerate_expressions(2, 3)
    bits = [prior(node) for node, _c in nodes[:50]]
    assert len(set(bits)) > 1 and all(b > 0 for b in bits)


def test_the_final_run_refuses_a_changed_plan() -> None:
    if not mo.DIGEST.exists():
        pytest.skip("not frozen yet")
    mo.check_frozen()
