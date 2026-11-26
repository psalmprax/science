"""The physics judge (apedp_dyn/epistemic/physics_judge.py), its perturbations and its synthetic worlds.

These tests pin the judge's mechanics on synthetic clips; its validation is on PhysInOne
(docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md, gate J), not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from apedp_dyn.epistemic import physics_judge as pj
from apedp_dyn.epistemic.physics_judge import FAIL, PASS, UNTESTABLE, Track, judge
from apedp_dyn.video.physics_perturbations import (
    PERTURBATIONS,
    loses_energy,
    perturb,
    time_reversal,
)
from apedp_dyn.video.physics_tracks_synthetic import Camera, make


def test_projected_model_fits_a_projected_parabola_exactly() -> None:
    cam = Camera()
    t = np.arange(30) / 30.0
    pts = np.stack([-1.0 + 1.5 * t, 0.5 - 2.0 * t, 1.0 + 3.0 * t - 0.5 * 9.81 * t**2], axis=1)
    xy, _ = cam.project(pts, 0.1)
    seg = pj._segment(t, xy, 0, t.size - 1, 0.25, math.log(60))
    assert seg is not None and seg.model == "projected"
    assert np.max(np.abs(seg.pos(t) - xy)) < 1e-3
    # Its velocity is the derivative of its position.
    h = 1e-5
    v_num = (seg.pos(t[10] + h) - seg.pos(t[10] - h)) / (2 * h)
    assert np.allclose(seg.vel(t[10]), v_num, rtol=1e-4, atol=1e-3)


def test_a_parabola_in_the_image_is_not_split() -> None:
    t = np.arange(40) / 30.0
    xy = np.stack([100 + 200 * t, 50 + 50 * t + 300 * t**2], axis=1)
    segs = pj._partition(t, xy, 0.25, 2)
    assert len(segs) == 1


@pytest.mark.parametrize("kind", pj.SCENE_KINDS)
def test_true_synthetic_clips_pass(kind: str) -> None:
    v = judge(kind, make(kind, 2))
    assert v.outcome == PASS, v.reasons


@pytest.mark.parametrize(("kind", "how"), [("free_fall", "teleport"), ("throw", "freeze"),
                                          ("bounce", "vertical_reversal"), ("free_fall", "time_reversal"),
                                          ("pendulum", "freeze"), ("collision", "teleport")])
def test_development_perturbations_fail(kind: str, how: str) -> None:
    broken = perturb(kind, make(kind, 2), how, np.random.default_rng(7))
    assert broken is not None
    v = judge(kind, broken)
    assert v.outcome == FAIL, v.stats


def test_a_still_object_is_untestable() -> None:
    t = np.arange(60) / 30.0
    xy = np.tile([[320.0, 240.0]], (60, 1)) + np.random.default_rng(0).normal(0, 0.2, (60, 2))
    v = judge("free_fall", [Track(t, xy, np.full(60, 20.0))])
    assert v.outcome == UNTESTABLE


def test_an_object_mostly_out_of_view_is_untestable() -> None:
    tr = make("free_fall", 1)[0]
    xy = tr.xy.copy()
    xy[20:] = np.nan
    v = judge("free_fall", [Track(tr.t, xy, tr.size)])
    assert v.outcome == UNTESTABLE


def test_frames_whose_mask_area_jumps_are_dropped() -> None:
    tr = make("throw", 3)[0]
    size = tr.size.copy()
    xy = tr.xy.copy()
    size[30] *= 3.0  # an occluder merged into the mask for one frame
    xy[30] += 40.0
    assert judge("throw", [Track(tr.t, xy, size)]).outcome == PASS


def test_time_reversal_applies_only_where_energy_is_lost() -> None:
    falling = make("free_fall", 0)
    assert loses_energy(falling, "free_fall")
    assert perturb("free_fall", falling, "time_reversal", np.random.default_rng(0)) is not None
    t = np.arange(60) / 30.0
    flying = [Track(t, np.stack([100 + 300 * t, 400 - 400 * t + 250 * t**2], axis=1), np.full(60, 20.0))]
    assert not loses_energy(flying, "throw")
    assert perturb("throw", flying, "time_reversal", np.random.default_rng(0)) is None


def test_vertical_reversal_needs_vertical_motion() -> None:
    # Two balls rolling across a table: nothing vertical to mirror.
    assert perturb("collision", make("collision", 0), "vertical_reversal", np.random.default_rng(0)) is None


def test_perturbations_keep_the_clip_length() -> None:
    tracks = make("bounce", 4)
    for how in PERTURBATIONS:
        broken = perturb("bounce", tracks, how, np.random.default_rng(1))
        if broken is not None:
            assert broken[0].xy.shape == tracks[0].xy.shape
            assert broken[0].size is not None and broken[0].size.shape == tracks[0].size.shape
    rev = time_reversal(tracks[0])
    assert np.array_equal(rev.xy[0], tracks[0].xy[-1])


def test_a_collision_without_reaction_fails() -> None:
    a, b = make("collision", 3)
    t = b.t
    # B keeps its approach as if A were not there: A changes velocity, B does not.
    xb = np.stack([b.xy[0, 0] + (b.xy[5, 0] - b.xy[0, 0]) / 5 * np.arange(t.size), b.xy[:, 1]], axis=1)
    v = judge("collision", [a, Track(t, xb, b.size)])
    assert v.outcome == FAIL


def test_judge_checks_its_arguments() -> None:
    tr = make("free_fall", 0)
    with pytest.raises(ValueError):
        judge("levitation", tr)
    with pytest.raises(ValueError):
        judge("collision", tr)
    with pytest.raises(ValueError):
        Track(np.arange(3.0), np.zeros((4, 2)))


def test_verdicts_are_deterministic() -> None:
    tr = make("roll", 5)
    assert judge("roll", tr).to_dict() == judge("roll", tr).to_dict()


def test_the_development_script_refuses_the_blind_pool() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_physics_judge_dev.py"
    spec = importlib.util.spec_from_file_location("run_physics_judge_dev", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit, match="blind"):
        mod.main(["fetch", "--kind", "throw", "--pool", "blind", "--out", "/dev/null"])
