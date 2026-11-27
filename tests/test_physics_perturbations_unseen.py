"""Pins for the reviewer's unseen breakages (apedp_dyn/video/physics_perturbations_unseen.py, PREREG-PHYSICS-J1-2026-11).

Development clips only (seeds below 30); the judge is never called here.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.epistemic.physics_judge import Track
from apedp_dyn.video import physics_perturbations_unseen as unseen
from apedp_dyn.video.physics_perturbations import PERTURBATIONS

ROOT = Path(__file__).resolve().parents[1]
APPLIES = {"sideways_push": ("free_fall", "throw", "bounce"), "period_drift": ("pendulum",),
           "floor_speedup": ("roll", "throw", "collision")}


def _rng(seed: int, how: str) -> np.random.Generator:
    return np.random.default_rng([seed, int(hashlib.sha256(how.encode()).hexdigest()[:8], 16)])


def _swing(amplitude: float, n: int = 90) -> Track:
    t = np.arange(n) / 30.0
    x = 160.0 + amplitude * np.cos(2 * np.pi * t / 1.4)
    y = 120.0 + 0.02 * (x - 160.0) ** 2
    return Track(t, np.stack([x, y], axis=1), np.full(n, 12.0), (160.0, 120.0))


def _shift(broken: list[Track], tracks: list[Track]) -> float:
    return max(float(np.nanmax(np.linalg.norm(b.xy - t.xy, axis=1))) for b, t in zip(broken, tracks, strict=True))


def test_the_interface() -> None:
    assert len(unseen.UNSEEN_KINDS) >= 2 and len(set(unseen.UNSEEN_KINDS)) == len(unseen.UNSEEN_KINDS)
    assert not set(unseen.UNSEEN_KINDS) & {"true", *PERTURBATIONS}
    assert set(unseen.LAWS) == set(unseen.UNSEEN_KINDS) and len(set(unseen.LAWS.values())) == len(unseen.LAWS)
    with pytest.raises(ValueError):
        unseen.perturb_unseen("free_fall", [_swing(30.0)], "teleport", _rng(0, "teleport"))


def test_it_names_the_stage_1_digest() -> None:
    digest = ROOT / "preregistration" / "DIGEST-PHYSICS-J1-JUDGE.json"
    if not digest.exists():
        pytest.skip("stage 1 not frozen in this checkout")
    assert unseen.STAGE1_SHA256 == hashlib.sha256(digest.read_bytes()).hexdigest()


def test_none_where_it_does_not_apply() -> None:
    t = np.arange(90) / 30.0
    still = Track(t, np.tile([160.0, 120.0], (90, 1)), np.full(90, 12.0), (160.0, 120.0))
    for how in unseen.UNSEEN_KINDS:
        for kind in ("free_fall", "throw", "roll", "bounce", "pendulum"):
            assert unseen.perturb_unseen(kind, [still], how, _rng(0, how)) is None, (how, kind)
        assert unseen.perturb_unseen("collision", [still, still], how, _rng(0, how)) is None
    swing = _swing(30.0)
    for how, kinds in APPLIES.items():
        for kind in {"free_fall", "throw", "roll", "bounce", "pendulum"} - set(kinds):
            assert unseen.perturb_unseen(kind, [swing], how, _rng(0, how)) is None, (how, kind)


def test_the_floor_of_five_pixels() -> None:
    small, large = _swing(1.0), _swing(30.0)
    assert unseen.perturb_unseen("pendulum", [small], "period_drift", _rng(0, "period_drift")) is None
    broken = unseen.perturb_unseen("pendulum", [large], "period_drift", _rng(0, "period_drift"))
    assert broken is not None and _shift(broken, [large]) >= unseen.MIN_SHIFT_PX


def test_a_still_bob_is_not_moved_by_noise() -> None:
    rng = np.random.default_rng(5)
    noisy = _swing(2.0)
    noisy = Track(noisy.t, noisy.xy + rng.normal(0, 0.3, noisy.xy.shape), noisy.size, noisy.centre)
    assert unseen.perturb_unseen("pendulum", [noisy], "period_drift", _rng(1, "period_drift")) is None


@pytest.fixture(scope="module")
def clips() -> dict[str, list[Track]]:
    pytest.importorskip("pybullet")
    from apedp_dyn.video.physics_clips_pybullet import make_clip

    return {kind: make_clip(kind, 3)[0] for kind in ("free_fall", "throw", "roll", "bounce", "pendulum", "collision")}


@pytest.mark.parametrize("how", unseen.UNSEEN_KINDS)
def test_deterministic_pure_and_above_the_floor_on_development_clips(clips: dict[str, list[Track]], how: str) -> None:
    applied = 0
    for kind, tracks in clips.items():
        before = copy.deepcopy(tracks)
        first = unseen.perturb_unseen(kind, tracks, how, _rng(3, how))
        again = unseen.perturb_unseen(kind, tracks, how, _rng(3, how))
        for a, b in zip(tracks, before, strict=True):              # the tracks it was given are unchanged
            assert np.array_equal(a.xy, b.xy, equal_nan=True) and np.array_equal(a.t, b.t)
            assert (a.size is None) == (b.size is None) and (a.size is None or np.array_equal(a.size, b.size, equal_nan=True))
        assert (first is None) == (again is None)
        if first is None:
            continue
        assert again is not None
        for a, b in zip(first, again, strict=True):
            assert np.array_equal(a.xy, b.xy, equal_nan=True)
        assert len(first) == len(tracks) and _shift(first, tracks) >= unseen.MIN_SHIFT_PX
        applied += 1
    assert applied >= 1
