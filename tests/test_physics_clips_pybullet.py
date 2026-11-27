"""The clips simulated and rendered here for the physics judge (apedp_dyn/video/physics_clips_pybullet.py).

Skipped where PyBullet is not installed (it is not a dependency of the package; the judge's workflows install it).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from apedp_dyn.epistemic.physics_judge import PASS, judge
from apedp_dyn.video import physics_clips_pybullet as pc


def test_the_blind_pool_is_refused_without_a_frozen_run() -> None:
    with pytest.raises(ValueError, match="blind"):
        pc.make_clip("free_fall", pc.BLIND_FROM)


@pytest.mark.parametrize("kind", pc.KINDS)
def test_a_clip_has_its_tracks_and_facts_and_is_the_same_each_time(kind: str) -> None:
    tracks, facts = pc.make_clip(kind, 0)
    again, facts_again = pc.make_clip(kind, 0)
    assert len(tracks) == (2 if kind == "collision" else 1)
    assert facts == facts_again and facts["kind"] == kind and facts["seed"] == 0
    for a, b in zip(tracks, again, strict=True):
        assert a.xy.shape == (pc.FRAMES, 2)
        assert np.array_equal(a.xy, b.xy, equal_nan=True)
        assert np.isfinite(a.xy).mean() > 0.6          # the object is in view for most of the clip


def test_the_camera_is_nearly_level_and_at_the_side() -> None:
    for kind in pc.KINDS:
        cam = pc.make_clip(kind, 1)[1]["camera"]
        assert -10.0 <= cam["pitch_deg"] <= 3.0 and abs(cam["yaw_deg"]) <= 20.0


def test_true_rendered_clips_pass_the_judge() -> None:
    for kind in ("free_fall", "throw", "pendulum"):
        tracks, _ = pc.make_clip(kind, 2)
        v = judge(kind, tracks)
        assert v.outcome == PASS, (kind, v.reasons)
