r"""Domain Scoping: 2D Rigid-Body Collision + Occlusion Scenes (SECOND problem family).

Status: experimental (domain-scoped dataset generator + scope manifest)

Second member of the JetFlow matched-benchmark protocol family. Whereas the
first domain (fluid tracer advection, `fluid_domain.py`) stress-tests
continuum transport, this domain stress-tests the DISCONTINUOUS regime:
ballistic motion, elastic wall/pair collisions, and genuine occlusion, i.e.
the scene class the external evaluation recommended as the second JetFlow
benchmark family.

Physics discipline (same honesty contract as the fluid domain):
* Layered world: each object has a fixed depth layer. Objects in the SAME
  layer collide elastically (equal-mass: swap normal velocity components) and
  bounce off walls; objects in DIFFERENT layers pass through each other's
  (y, x) tracks untouched — the nearer layer's rendered body occludes the
  farther one's channel wherever its body dominates. Occlusion is therefore a
  physical consequence of parallax (real disappearance/reappearance), not a
  stylized effect, and same-layer clips keep genuine collision dynamics.
* Motion is exactly ballistic between events — no solver drift.
* Deterministic: clip id <-> seed, byte-reproducible across processes
  (RandomState only). Same burnable-holdout discipline via eval_gate's
  held_out_split — holdout ids are exclusive resources.
* ``return_state=True`` exposes the ground-truth scene state (positions,
  velocities, layers, collisions) for TRAIN-CLIP-ONLY supervised use;
  holdout clips must NOT consume it (train/holdout discipline).
"""
from __future__ import annotations

import numpy as np


def _render_objects(
    positions: np.ndarray,  # (T, n_obj, 2): [y, x]
    radii: np.ndarray,  # (n_obj,)
    intensities: np.ndarray,  # (n_obj,)
    layers: np.ndarray,  # (n_obj,) integer depth; higher = nearer (occludes)
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render (T, 3, H, W) frames with depth-ordered occlusion composition.

    Object k's template is a soft disc intensity_k * exp(-4 (d/r)^2) in
    channel k (objects 0, 1, 2 -> channels 0, 1, 2; n_obj <= 3). The occluder
    relation is by (layer, id) rank: object j occludes object k iff
    (layers[j], j) > (layers[k], k) — nearer layer wins, ties break by id.
    The occluded channel is zeroed wherever an occluder's rendered BODY
    (value > 0.1 * its intensity) covers the pixel: full-body occlusion for
    cross-layer pass-throughs and a contact crescent for same-layer
    collisions. Returns frames and per-frame occlusion flags (T, n_obj).
    """
    t_tot, n_obj = positions.shape[0], positions.shape[1]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    frames = np.zeros((t_tot, 3, h, w), dtype=np.float64)
    occl = np.zeros((t_tot, n_obj), dtype=bool)

    def occludes(j: int, k: int) -> bool:
        # Object j occludes object k iff j is strictly NEARER, or on equal
        # layers the higher id is on top. Direct comparison: no positional
        # rank assumptions (layers may arrive in any order).
        return (layers[j] > layers[k]) or (layers[j] == layers[k] and j > k)

    for ti in range(t_tot):
        discs, bodies = [], []
        for k in range(n_obj):
            cy, cx = positions[ti, k]
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            discs.append(intensities[k] * np.exp(-4.0 * d2 / max(1e-9, radii[k] ** 2)))
            bodies.append(discs[k] > 0.1 * intensities[k])
        for k in range(n_obj):
            hi_mask = np.zeros((h, w), dtype=bool)
            for j in range(n_obj):
                if j != k and occludes(j, k):
                    hi_mask |= bodies[j]
            own = ((yy - positions[ti, k, 0]) ** 2 + (xx - positions[ti, k, 1]) ** 2) <= radii[k] ** 2
            occl[ti, k] = bool((hi_mask & own).any())
            frames[ti, k % 3] += np.where(hi_mask, 0.0, discs[k])
    return frames, occl


def make_collision_clip(
    seed: int,
    t: int = 16,
    h: int = 64,
    w: int = 64,
    n_obj: int = 2,
    canvas_margin: float = 0.15,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generate one deterministic 2D rigid-body collision/occlusion clip.

    Output in [0, 1], float64, (T, 3, H, W). Byte-reproducible (RandomState).
    With ``return_state=True`` ALSO returns ground-truth scene state per
    frame: {"positions": (T, n_obj, 2), "velocities": (T, n_obj, 2),
    "layers": (n_obj,), "radii": (n_obj,), "collisions": [(frame, a, b), ...]}.
    """
    if t < 4:
        raise ValueError(f"Need at least 4 frames, got t={t}")
    if n_obj < 2:
        raise ValueError("collision domain requires >= 2 objects")
    if n_obj > 3:
        raise ValueError("3-channel clip format supports up to 3 objects")
    if not 0.0 < canvas_margin < 0.5:
        raise ValueError("canvas_margin must be in (0, 0.5)")
    rng = np.random.RandomState(abs(seed) * 7907 + 29)

    radii = rng.uniform(0.08, 0.16, size=n_obj) * min(h, w)
    intensities = rng.uniform(0.6, 1.0, size=n_obj)
    # Depth layers: same-layer clips (~50%) keep elastic inter-object
    # collisions; cross-layer clips (~50%) produce genuine pass-through
    # occlusion. Both worlds are exercised across the manifest family.
    cross_layer = bool(rng.uniform() < 0.5)
    layers = np.arange(n_obj, dtype=np.int64) if cross_layer else np.zeros(n_obj, dtype=np.int64)

    # Spawn non-overlapping, inboard.
    margin = float(canvas_margin * min(h, w))
    positions0 = np.zeros((n_obj, 2), dtype=np.float64)
    for k in range(n_obj):
        pad = radii[k] + margin
        for _ in range(200):
            p = np.array([rng.uniform(pad, h - pad), rng.uniform(pad, w - pad)])
            if all(np.hypot(*(positions0[j] - p)) > radii[j] + radii[k] + 0.05 * min(h, w) for j in range(k)):
                positions0[k] = p
                break
        else:
            positions0[k] = np.array([(k + 1) * h / (n_obj + 1), w * 0.5])

    speeds = rng.uniform(1.5, 4.0, size=(n_obj, 2))
    velocities = speeds * rng.choice([-1.0, 1.0], size=(n_obj, 2))
    # Interception spawning (~60%, seeded): object 1 is given a course toward
    # object 0 (slightly rotated), so a collision (same layer) or an
    # occlusion pass-through (cross layer) occurs within the clip window.
    if rng.uniform() < 0.6:
        dirv = positions0[0] - positions0[1]
        dist = float(np.hypot(*dirv))
        if dist > 1e-9:
            ang = rng.uniform(-0.35, 0.35)
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            s1 = float(np.hypot(*velocities[1]))
            velocities[1] = s1 * (rot @ (dirv / dist))

    positions = np.zeros((t, n_obj, 2), dtype=np.float64)
    velocities_t = np.zeros((t, n_obj, 2), dtype=np.float64)
    collisions: list[tuple[int, int, int]] = []
    dt = 1.0
    positions[0], velocities_t[0] = positions0.copy(), velocities.copy()
    for ti in range(1, t):
        p = positions[ti - 1].copy()
        v = velocities_t[ti - 1].copy()
        p = p + dt * v
        # Elastic wall bounce (per object, all layers).
        for k in range(n_obj):
            if (p[k, 0] < radii[k] and v[k, 0] < 0) or (p[k, 0] > h - radii[k] and v[k, 0] > 0):
                v[k, 0] *= -1.0
            if (p[k, 1] < radii[k] and v[k, 1] < 0) or (p[k, 1] > w - radii[k] and v[k, 1] > 0):
                v[k, 1] *= -1.0
        p[:, 0] = np.clip(p[:, 0], radii, h - radii)
        p[:, 1] = np.clip(p[:, 1], radii, w - radii)
        # Pairwise equal-mass elastic collisions — SAME LAYER ONLY: cross-layer
        # objects pass through each others' (y, x) track (parallax worlds).
        for a in range(n_obj):
            for b in range(a + 1, n_obj):
                if layers[a] != layers[b]:
                    continue  # cross-layer: no collision, pass-through
                d_vec = p[b] - p[a]
                dist = float(np.hypot(*d_vec))
                if dist < radii[a] + radii[b]:
                    n_hat = d_vec / max(1e-9, dist)
                    va_n = float(np.dot(v[a], n_hat))
                    vb_n = float(np.dot(v[b], n_hat))
                    if va_n - vb_n > 0:  # approaching
                        v[a] += (vb_n - va_n) * n_hat
                        v[b] += (va_n - vb_n) * n_hat
                        collisions.append((ti, a, b))
                    overlap = (radii[a] + radii[b] - dist) * 0.5
                    p[a] -= n_hat * overlap
                    p[b] += n_hat * overlap
        positions[ti], velocities_t[ti] = p, v

    frames, occl_flags = _render_objects(positions, radii, intensities, layers, h, w)
    for ci in range(3):
        lo, hi = frames[:, ci].min(), frames[:, ci].max()
        frames[:, ci] = (frames[:, ci] - lo) / max(1e-9, hi - lo)
    if return_state:
        return frames, {"positions": positions, "velocities": velocities_t,
                        "layers": layers.tolist(), "radii": radii.tolist(),
                        "occlusion_flags": occl_flags, "collisions": collisions}
    return frames


def make_domain_set(
    train_ids: tuple[str, ...], holdout_ids: tuple[str, ...], t: int = 16, h: int = 64,
    w: int = 64,
) -> dict[str, np.ndarray]:
    """Deterministic domain clips keyed by the manifest ids (holdout exclusive)."""
    clips: dict[str, np.ndarray] = {}
    for cid in tuple(train_ids) + tuple(holdout_ids):
        seed = int(cid.replace("collision_clip_", ""), 10)
        clips[cid] = make_collision_clip(seed, t=t, h=h, w=w)
    return clips


def collision_domain_manifest(
    n_clips: int = 20, holdout_fraction: float = 0.2, holdout_seed: int = 13,
    t: int = 16, h: int = 64, w: int = 64,
) -> dict[str, object]:
    """Scope manifest for the collision family (same contract as the fluid one)."""
    from apedp_dyn.video.eval_gate import held_out_split

    ids = [f"collision_clip_{i:04d}" for i in range(n_clips)]
    train, hold = held_out_split(ids, holdout_fraction=holdout_fraction, seed=holdout_seed)
    return {
        "domain": "2D rigid-body collision + occlusion scenes (layered world: same-layer elastic, cross-layer pass-through)",
        "generator": "apedp_dyn/video/collision_domain.py::make_collision_clip (deterministic, RandomState)",
        "physics_knobs_correspondence": {
            "convection_velocity": "ballistic translation between events",
            "viscosity_nu": "no continuum analog (discontinuous regime)",
        },
        "n_clips": n_clips,
        "clip_shape": [t, 3, h, w],
        "train_ids": list(train),
        "holdout_ids": list(hold),
    }
