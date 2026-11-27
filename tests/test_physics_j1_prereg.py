"""Pins for PREREG-PHYSICS-J1-2026-11 (scripts/run_prereg_physics_j1.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("run_prereg_physics_j1", ROOT / "scripts" / "run_prereg_physics_j1.py")
assert _spec is not None and _spec.loader is not None
j1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j1)

DEV = ("teleport", "freeze", "vertical_reversal", "time_reversal")


def test_development_seeds_and_streams() -> None:
    assert j1.seeds("dev") == list(range(30))
    # Streams are named by a hash of the breakage, so two names of one length do not share one.
    a = j1.stream(3, "freeze").random(4)
    b = j1.stream(3, "smear6").random(4)
    assert (a != b).any()
    assert (j1.stream(3, "freeze").random(4) == a).all()


def _rows(per_kind: dict[str, tuple[int, dict[str, int]]], n: int = 60, unseen: tuple[str, ...] = ("smear", "drift"),
          ) -> list[dict]:
    rows = []
    for kind in j1.GATE_KINDS:
        true_pass, missed = per_kind.get(kind, per_kind["*"])
        for s in range(n):
            rows.append({"source": "rendered", "kind": kind, "seed": s, "variant": "true",
                         "outcome": "pass" if s < true_pass else "fail"})
            for how in (*DEV, *unseen):
                rows.append({"source": "rendered", "kind": kind, "seed": s, "variant": how,
                             "group": "unseen" if how in unseen else "development",
                             "outcome": "pass" if s < missed.get(how, 0) else "fail"})
    return rows


def _expected(n: int = 60) -> dict[str, list[int]]:
    return {k: list(range(n)) for k in j1.GATE_KINDS}


def test_the_gate_reads_every_scene_kind() -> None:
    ok = j1.gate(_rows({"*": (55, {"teleport": 2, "smear": 5})}), _expected(), ("smear", "drift"))
    assert ok["state"] == "holds", ok["per_kind_failures"]
    # One scene kind's true clips at 75 %: the pooled reading still holds, so the owner decides.
    one = j1.gate(_rows({"*": (55, {}), "pendulum": (45, {})}), _expected(), ("smear", "drift"))
    assert one["state"] == "the owner decides" and one["per_kind_failures"] and not one["pooled_failures"]
    # Everywhere 7 of 60 of a kind not caught: the pooled reading fails too.
    bad = j1.gate(_rows({"*": (55, {"smear": 7})}), _expected(), ("smear", "drift"))
    assert bad["state"] == "fails" and bad["pooled_failures"]


def test_a_broken_clip_is_caught_only_by_a_fail() -> None:
    rows = _rows({"*": (55, {})})
    for r in rows:
        if r["variant"] == "smear" and r["seed"] < 7:
            r["outcome"] = "untestable"                   # untestable is not caught
    out = j1.gate(rows, _expected(), ("smear", "drift"))
    assert out["state"] == "fails"


def test_without_two_reviewer_kinds_it_is_not_read() -> None:
    rows = _rows({"*": (55, {})}, unseen=("smear",))
    assert j1.gate(rows, _expected(), ("smear",))["state"] == "not read"


def test_a_missing_clip_leaves_it_not_read() -> None:
    rows = _rows({"*": (55, {})})
    rows = [r for r in rows if not (r["kind"] == "roll" and r["seed"] == 7)]
    assert j1.gate(rows, _expected(), ("smear", "drift"))["state"] == "not read"


def test_a_small_cell_fails_on_three_misses() -> None:
    rows = _rows({"*": (55, {"smear": 3})})
    for r in rows:
        if r["variant"] == "smear" and r["seed"] >= 20:
            r["outcome"] = "not applicable"               # applies to 20 clips per kind: a small cell
    out = j1.gate(rows, _expected(), ("smear", "drift"))
    assert any("small cell" in f for f in out["per_kind_failures"])


def test_untestable_true_clips_count_as_not_passing() -> None:
    rows = _rows({"*": (60, {})})
    for r in rows:
        if r["variant"] == "true" and r["seed"] < 13:
            r["outcome"] = "untestable"
    assert j1.gate(rows, _expected(), ("smear", "drift"))["state"] == "fails"


def test_bounds() -> None:
    assert j1.bound(0, 0, upper=True) is None
    assert 0.04 < j1.bound(0, 60, upper=True) < 0.05
    assert 0.75 < j1.bound(240, 300, upper=False) < 0.77


def test_the_final_run_refuses_a_changed_judge() -> None:
    if not j1.JUDGE_DIGEST.exists() or not j1.DIGEST.exists():
        pytest.skip("not frozen yet")
    j1.check_frozen()
