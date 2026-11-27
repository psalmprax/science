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


def test_blind_and_development_seeds_are_apart() -> None:
    from apedp_dyn.video.physics_clips_pybullet import BLIND_FROM

    dev, final = set(j1.seeds("dev")), set(j1.seeds("final"))
    assert not dev & final and min(final) >= BLIND_FROM and max(dev) < BLIND_FROM
    assert len(final) == 60 and len(dev) == 30


def _rows(true_pass: int, broken_pass: dict[str, int], n: int = 60, unseen: str = "smear") -> list[dict]:
    rows = []
    for kind in j1.GATE_KINDS:
        for s in range(n):
            rows.append({"source": "rendered", "kind": kind, "seed": s, "variant": "true",
                         "outcome": "pass" if s < true_pass else "fail"})
            for how, k in broken_pass.items():
                rows.append({"source": "rendered", "kind": kind, "seed": s, "variant": how,
                             "group": "unseen" if how == unseen else "development",
                             "outcome": "pass" if s < k else "fail"})
    return rows


def test_the_gate_reads_both_expectations() -> None:
    ok = j1.gate(_rows(55, {"teleport": 2, "smear": 5}))
    assert ok["true_holds"] and ok["part_1_holds"]
    assert not j1.gate(_rows(45, {"teleport": 2, "smear": 5}))["part_1_holds"]        # 75 % of true clips
    bad = j1.gate(_rows(55, {"teleport": 2, "smear": 7}))                             # 11.7 % of one kind
    assert not bad["broken_holds"]["smear"] and not bad["part_1_holds"]


def test_without_a_reviewer_kind_part_1_is_not_read() -> None:
    out = j1.gate(_rows(55, {"teleport": 0}, unseen="none"))
    assert not out["unseen_kinds_read"] and not out["part_1_holds"]


def test_a_kind_applied_to_fewer_than_twenty_clips_is_reported_not_read() -> None:
    rows = _rows(55, {"teleport": 0, "smear": 0})
    for r in rows:
        if r["variant"] == "smear" and r["seed"] >= 3:
            r["outcome"] = "not applicable"                   # applies to 3 clips per kind: 15 in all
    out = j1.gate(rows)
    assert "smear" not in out["broken_holds"] and not out["unseen_kinds_read"]


def test_untestable_true_clips_count_as_not_passing() -> None:
    rows = _rows(60, {"teleport": 0, "smear": 0})
    for r in rows:
        if r["variant"] == "true" and r["seed"] < 13:
            r["outcome"] = "untestable"
    assert not j1.gate(rows)["true_holds"]                   # 47 of 60 pass


def test_the_final_run_refuses_a_changed_judge() -> None:
    if not j1.JUDGE_DIGEST.exists():
        pytest.skip("not frozen yet")
    if not j1.DIGEST.exists():
        pytest.skip("stage 2 not frozen yet")
    j1.check_frozen()
