r"""Domain Scoping: 2D Incompressible Navier-Stokes Fluid (Tracer Advection).

Status: experimental (domain-scoped dataset generator + scope manifest)

Photorealism lever: DOMAIN SELECTION. One physically coherent domain is
learnable at small model scale where "everything on the internet" is not.
Chosen domain: 2D incompressible fluid tracer evolution — the natural-scene
statistics class covered by wind/water/smoke/cloud dynamics.

Why this domain is honest for the physics claims in this repo:
* The velocity field is built from a streamfunction (u = curl psi), so
  incompressibility (nabla . u = 0) holds to grid precision — the
  Noether divergence-free projector finally has a physical object it
  genuinely applies to (latent channel pairing does not).
* Convection / viscosity controls in ``apedp_dyn.video.physics_knobs``
  map 1:1 onto the advection and diffusion terms of this dataset.

Everything is deterministic: clip id <-> seed, byte-reproducible across
processes (RandomState only). Couple with ``eval_gate.held_out_split``
for the train/holdout discipline; the holdout ids are exclusive.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _streamfunction_velocity(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocity from streamfunction: u = dpsi/dy, v = -dpsi/dx (exactly
    grid-divergence-free up to floating-point roundoff)."""
    u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
    v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    return u, v


def _vortex_streamfunction(
    h: int, w: int, centers: Sequence[tuple[float, float]], strengths: Sequence[float]
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    psi = np.zeros((h, w), dtype=np.float64)
    for (cy, cx), s in zip(centers, strengths, strict=False):
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        psi += s * np.exp(-r2 / (0.25 * min(h, w) ** 2))
    return psi


def make_fluid_clip(
    seed: int,
    t: int = 16,
    h: int = 64,
    w: int = 64,
    n_vortices: int = 3,
    diffusion: float = 0.01,
    diffusion_steps: int = 1,
    return_flow: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one deterministic 2D incompressible fluid tracer clip.

    The dye (3 channels: vorticity magnitude + two advected scalars) is
    advected by semi-Lagrangian backtracing under the evolving
    streamfunction field. Output in [0, 1], float64 (T, 3, H, W).

    With ``return_flow=True`` ALSO returns the per-frame analytic velocity
    the generator itself used to advect the dye: ``(u, v)`` grids of shape
    (T, H, W) each, where ``u`` is the y-velocity and ``v`` the x-velocity
    (the streamfunction convention ``u = dpsi/dy, v = -dpsi/dx``), in the
    same px/frame semi-Lagrangian units as the dye advection. This is the
    train-clip-only flow teacher for supervised flow-head training
    (TESTS pin the (u, v) convention so callers cannot silently swap them).
    Holdout clips must NOT consume it (train/holdout discipline).
    """
    if t < 4:
        raise ValueError(f"Need at least 4 frames, got t={t}")
    rng = np.random.RandomState(abs(seed) * 1000003 + 17)

    # Vortex field: centers slowly advected by the field itself (rigid drift
    # of centers keeps this a bounded simulation); strengths oscillate slowly
    # so the velocity field evolves in time.
    centers = []
    for _ in range(n_vortices):
        centers.append((rng.uniform(0.2 * h, 0.8 * h), rng.uniform(0.2 * w, 0.8 * w)))
    base_strength = rng.uniform(3.0, 9.0, size=n_vortices) * rng.choice([-1.0, 1.0], size=n_vortices)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")

    # Initial dye: 3 independent smooth scalar blobs
    dye = np.zeros((3, h, w), dtype=np.float64)
    for ci in range(3):
        n_blobs = 2
        for _ in range(n_blobs):
            cy, cx = rng.uniform(0.15 * h, 0.85 * h), rng.uniform(0.15 * w, 0.85 * w)
            rad = rng.uniform(0.08, 0.25) * min(h, w)
            dye[ci] += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2))
    dye /= max(1e-9, dye.max())

    frames = np.zeros((t, 3, h, w), dtype=np.float64)
    flow_u = np.zeros((t, h, w), dtype=np.float64)
    flow_v = np.zeros((t, h, w), dtype=np.float64)
    for ti in range(t):
        phase = 2 * np.pi * ti / max(1, t - 1)
        strengths = base_strength * (0.8 + 0.2 * np.sin(phase + 1.7 * np.arange(n_vortices)))
        # Rigid slow drift of vortex centers (bounded, periodic)
        centers_now = []
        for k, (cy, cx) in enumerate(centers or []):
            drift_y = 0.15 * h * np.sin(phase + 0.9 * k)
            drift_x = 0.15 * w * np.sin(phase + 1.3 * k + 0.5)
            centers_now.append((cy + drift_y, cx + drift_x))

        psi = _vortex_streamfunction(h, w, centers_now, strengths)
        u, v = _streamfunction_velocity(psi)
        flow_u[ti], flow_v[ti] = u, v

        # Perceptual frame 0: vorticity (+two advected scalar channels)
        frames[ti, 0] = np.abs(_curl_from_velocity(u, v))
        advected = _advect(dye, u, v, 1.0)
        if diffusion > 0.0:
            for _ in range(max(1, diffusion_steps)):
                advected = advected + diffusion * (
                    np.roll(advected, -1, axis=-1) + np.roll(advected, 1, axis=-1)
                    + np.roll(advected, -1, axis=-2) + np.roll(advected, 1, axis=-2)
                    - 4.0 * advected
                )
        dye = advected
        # Renormalize to keep the dye channel statistics stable
        m = dye.max()
        if m > 1e-9:
            dye = dye / m
        frames[ti, 1:3] = dye[1:3]

    # Normalize each channel into [0, 1] with the joint per-clip range
    for ci in range(3):
        lo, hi = frames[:, ci].min(), frames[:, ci].max()
        frames[:, ci] = (frames[:, ci] - lo) / max(1e-9, hi - lo)
    if return_flow:
        return frames, flow_u, flow_v
    return frames


def _curl_from_velocity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.asarray(
        (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) * 0.5
        - (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) * 0.5
    )


def _advect(field: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Semi-Lagrangian backtrace advection of a (3, H, W) dye field."""
    h, w = field.shape[1], field.shape[2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    back_y = yy - dt * u
    back_x = xx - dt * v
    y0 = np.floor(back_y).astype(np.int64) % h
    x0 = np.floor(back_x).astype(np.int64) % w
    y1 = (y0 + 1) % h
    x1 = (x0 + 1) % w
    wy = back_y - np.floor(back_y)
    wx = back_x - np.floor(back_x)
    out = np.zeros_like(field)
    for ci in range(field.shape[0]):
        f = field[ci]
        out[ci] = (
            (1 - wy) * (1 - wx) * f[y0, x0] + (1 - wy) * wx * f[y0, x1]
            + wy * (1 - wx) * f[y1, x0] + wy * wx * f[y1, x1]
        )
    return out


def make_domain_set(
    train_ids: tuple[str, ...], holdout_ids: tuple[str, ...], t: int = 16, h: int = 64,
    w: int = 64,
) -> dict[str, np.ndarray]:
    """Deterministic domain clips keyed by the manifest ids: holdout ids stay
    exclusive resources (same burnable-holdout discipline as the RIME stack)."""
    clips: dict[str, np.ndarray] = {}
    for cid in tuple(train_ids) + tuple(holdout_ids):
        seed = int(cid.replace("fluid_clip_", ""), 10)
        clips[cid] = make_fluid_clip(seed, t=t, h=h, w=w)
    return clips


def fluid_domain_manifest(
    n_clips: int = 20, holdout_fraction: float = 0.2, holdout_seed: int = 13,
    t: int = 16, h: int = 64, w: int = 64, diffusion: float = 0.01,
) -> dict[str, object]:
    """Scope manifest: declares the domain, the deterministic generator, and a
    held-out split (exclusive). Thresholds are NOT part of the manifest."""
    from apedp_dyn.video.eval_gate import held_out_split

    ids = [f"fluid_clip_{i:04d}" for i in range(n_clips)]
    train, hold = held_out_split(ids, holdout_fraction=holdout_fraction, seed=holdout_seed)
    return {
        "domain": "2D incompressible Navier-Stokes tracer (vorticity + dye)",
        "generator": "apedp_dyn/video/fluid_domain.py::make_fluid_clip (deterministic, RandomState)",
        "incompressibility": "velocity from streamfunction; grid-divergence-free",
        "physics_knobs_correspondence": {
            "convection_velocity": "semi-Lagrangian advection term",
            "viscosity_nu": "explicit laplacian diffusion term",
        },
        "n_clips": n_clips,
        "clip_shape": [t, 3, h, w],
        "train_ids": list(train),
        "holdout_ids": list(hold),
    }
