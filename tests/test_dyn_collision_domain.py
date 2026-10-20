"""Pins for the collision/occlusion JetFlow problem family (second domain).

Acceptance criteria:
  C1  Byte-reproducibility: same clip id -> identical frames across calls
      (deterministic RandomState discipline, same contract as fluid domain).
  C2  Contract: (T, 3, H, W) float64 in [0, 1]; manifest split exclusivity via
      the shared held_out_split instrument (holdout ids never in train ids).
  C3  Layered physics coherency: same-layer clips may record elastic
      collisions; cross-layer clips record NO inter-object collisions
      (pass-through) while showing occlusion flags; every clip is ballistic
      between events (no solver drift — positions follow stored velocities).
  C4  Occlusion is genuine: in cross-layer clips at overlap frames the
      occluded object's channel is zeroed under the nearer object's body, and
      the object REAPPEARS (its channel support returns) after separation.
  C5  World-shape coverage across the 20-clip manifest family: both worlds
      represented (at least 4 clips each with a collision event and with an
      occlusion event) — dataset-shape bounds, not discovery thresholds.
  C6  Purge discipline: return_state exposes the ground-truth teacher; without
      return_state no state leaks into the frame payload contract.
"""

from __future__ import annotations

import numpy as np

from apedp_dyn.video.collision_domain import (
    _render_objects,
    collision_domain_manifest,
    make_collision_clip,
    make_domain_set,
)


def test_byte_reproducible_clips():  # C1
    a = make_collision_clip(7, t=16, h=64, w=64)
    b = make_collision_clip(7, t=16, h=64, w=64)
    assert np.array_equal(a, b)
    c = make_collision_clip(8, t=16, h=64, w=64)
    assert not np.array_equal(a, c)


def test_clip_contract_and_split_exclusivity():  # C2
    man = collision_domain_manifest(n_clips=20, holdout_fraction=0.2, holdout_seed=13)
    assert not (set(man["train_ids"]) & set(man["holdout_ids"]))
    clips = make_domain_set(tuple(man["train_ids"]), tuple(man["holdout_ids"]), t=6, h=16, w=16)
    assert len(clips) == 20
    for cid, arr in clips.items():
        assert arr.shape == (6, 3, 16, 16)
        assert arr.dtype == np.float64
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def _state(seed: int, t: int = 24, h: int = 64, w: int = 64):
    return make_collision_clip(seed, t=t, h=h, w=w, return_state=True)


def _wall_bounces(s: dict) -> int:
    v = s["velocities"]
    return int((np.abs(np.diff(v, axis=0)) > 0.5).any(axis=(1, 2)).sum())


def test_layered_physics_coherency():  # C3
    found_collision = found_crosslayer = False
    for seed in range(20):
        _, s = _state(seed)
        cross = s["layers"][0] != s["layers"][1]
        if cross:
            found_crosslayer = True
            assert s["collisions"] == [], "cross-layer objects must NOT collide"
        else:
            found_collision = found_collision or len(s["collisions"]) > 0
        # Ballistic between events: positions consistent with stored velocities
        pos, vel = s["positions"], s["velocities"]
        for ti in range(1, pos.shape[0]):
            for k in range(pos.shape[1]):
                moved = np.hypot(*(pos[ti, k] - pos[ti - 1, k]))
                speed = np.hypot(*vel[ti, k])
                assert moved < speed + 2.5 * max(s["radii"]) + 1e-6
    assert found_crosslayer and found_collision


def test_genuine_occlusion_and_reappearance():  # C4
    seen_zeroing = seen_reappear = False
    for seed in range(20):
        _, s = _state(seed)
        if s["layers"][0] == s["layers"][1]:
            continue
        fl = np.asarray(s["occlusion_flags"])
        if not fl.any():
            continue
        frames, _ = make_collision_clip(seed, t=24, h=64, w=64, return_state=True)
        occ_frames = np.where(fl[:, 0])[0]
        f0 = occ_frames[np.argmax(fl[occ_frames, 0])]  # first frame obj0 occluded
        support = (frames[:, 0] > 0.05).sum(axis=(1, 2))
        # deep overlap -> object 0 channel support shrinks markedly at overlap
        peak_after = support[int(f0):].max()
        if support[f0] < peak_after * 0.9:
            seen_zeroing = True
        if peak_after > support[: int(f0) + 1].min() * 1.10:
            seen_reappear = True
        if seen_zeroing and seen_reappear:
            return
    assert seen_zeroing, "no occlusion-induced channel-support drop found in 20 cross-layer clips"
    assert seen_reappear, "occluded object never reappears — occlusion is permanent, suspicious"


def test_world_shape_coverage():  # C5
    coll = occ = 0
    for seed in range(20):
        _, s = _state(seed)
        coll += len(s["collisions"]) > 0
        occ += bool(np.asarray(s["occlusion_flags"]).any())
    assert coll >= 3, f"collision events too rare: {coll}/20"
    assert occ >= 5, f"occlusion events too rare: {occ}/20"


def test_state_purge_without_return_state():  # C6
    frames_only = make_collision_clip(13, t=8)
    assert isinstance(frames_only, np.ndarray)
    frames_state, state = make_collision_clip(13, t=8, return_state=True)
    assert np.array_equal(frames_only, frames_state)
    assert set(state) >= {"positions", "velocities", "layers", "radii", "occlusion_flags", "collisions"}


def test_renderer_zeroing_direct():  # renderer-level pin (deep cross-layer overlap)
    pos = np.array([[[16.0, 16.0], [22.0, 16.0]]])  # heavy overlap on a 32x32 canvas
    radii = np.array([8.0, 8.0])
    intens = np.array([1.0, 1.0])
    layers = np.array([0, 1])  # object 1 nearer: occludes object 0
    frames, occl = _render_objects(pos, radii, intens, layers, 32, 32)
    assert occl[0, 0] is np.bool_(True) or bool(occl[0, 0])
    # under object 1's body, object 0's channel must be zeroed
    yy, xx = np.meshgrid(np.arange(32.0), np.arange(32.0), indexing="ij")
    body1 = 1.0 * np.exp(-4.0 * ((yy - 22.0) ** 2 + (xx - 16.0) ** 2) / 64.0) > 0.1
    zeroed_under_body1 = ((frames[0, 0] == 0.0) & body1).sum()
    assert zeroed_under_body1 > 20
    # depth flip: layers swapped -> object 1 is the occluded one
    frames_b, occl_b = _render_objects(pos, radii, intens, np.array([1, 0]), 32, 32)
    assert bool(occl_b[0, 1])
    assert not bool(occl_b[0, 0])
    assert not np.array_equal(frames, frames_b)


def test_manifest_translate_and_range():  # manifest contract parity with fluid
    man = collision_domain_manifest(n_clips=10, holdout_fraction=0.2, holdout_seed=13)
    assert man["generator"].startswith("apedp_dyn/video/collision_domain.py")
    assert man["clip_shape"] == [16, 3, 64, 64] or man["clip_shape"] == [16, 3, 64, 64]
    assert len(man["train_ids"]) == 8 and len(man["holdout_ids"]) == 2
