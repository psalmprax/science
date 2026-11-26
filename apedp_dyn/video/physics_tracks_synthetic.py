"""Synthetic object tracks for the physics judge's unit tests (experimental, audit use only).

Small rigid-body worlds simulated in metres and seconds, seen by a static pinhole camera at 30 frames
per second for 90 frames, with Gaussian noise on the image position and size of each ball. These are
the tests' worlds, not the judge's validation data: that is PhysInOne (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md).
"""

from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.physics_judge import Track

G = 9.81
FPS = 30.0
FRAMES = 90
SUBSTEPS = 40


class Camera:
    def __init__(self, position: tuple[float, float, float] = (0.0, -9.0, 2.5),
                 target: tuple[float, float, float] = (0.0, 0.0, 0.8), focal: float = 900.0,
                 width: int = 1280, height: int = 720) -> None:
        self.c: np.ndarray = np.asarray(position, float)
        forward = np.asarray(target, float) - self.c
        self.forward = forward / np.linalg.norm(forward)
        right = np.cross(self.forward, np.array([0.0, 0.0, 1.0]))
        self.right = right / np.linalg.norm(right)
        self.down = np.cross(self.forward, self.right)
        self.f, self.cx, self.cy = focal, width / 2.0, height / 2.0

    def project(self, points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        rel = points - self.c
        depth = rel @ self.forward
        x = self.f * (rel @ self.right) / depth + self.cx
        y = self.f * (rel @ self.down) / depth + self.cy
        return np.stack([x, y], axis=1), 2.0 * radius * self.f / depth * np.sqrt(np.pi) / 2.0


def _sample(states: list[np.ndarray]) -> np.ndarray:
    out: np.ndarray = np.asarray(states[::SUBSTEPS][:FRAMES])
    return out


def _ballistic(p: np.ndarray, v: np.ndarray, radius: float, restitution: float, friction: float) -> np.ndarray:
    """A ball under gravity bouncing on the floor z = radius, sliding to rest with friction."""
    dt = 1.0 / (FPS * SUBSTEPS)
    out: list[np.ndarray] = []
    p, v = p.astype(float).copy(), v.astype(float).copy()
    resting = False
    for _ in range(FRAMES * SUBSTEPS + 1):
        out.append(p.copy())
        if resting:
            hv: np.ndarray = v[:2]
            sp = float(np.linalg.norm(hv))
            if sp > 0:
                dec = min(sp, friction * G * dt)
                v[:2] = hv * (1 - dec / sp)
            p += v * dt
            continue
        v[2] -= G * dt
        p += v * dt
        if p[2] < radius:
            p[2] = radius
            if abs(v[2]) * restitution < 0.4:
                v[2] = 0.0
                resting = True
            else:
                v[2] = -restitution * v[2]
                v[:2] *= 0.85
    return _sample(out)


def _track(camera: Camera, points: np.ndarray, radius: float, rng: np.random.Generator,
           noise_px: float) -> Track:
    xy, size = camera.project(points, radius)
    xy = xy + rng.normal(0.0, noise_px, xy.shape)
    size = size * (1.0 + rng.normal(0.0, 0.01, size.shape))
    t = np.arange(xy.shape[0]) / FPS
    return Track(t, xy, size, (camera.cx, camera.cy))


def make(kind: str, seed: int, noise_px: float = 0.3) -> list[Track]:
    """One true clip of a scene kind: its tracks as the camera sees them."""
    rng = np.random.default_rng(seed)
    cam = Camera()
    r = 0.15
    if kind == "free_fall":
        p = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(2.0, 3.0)])
        pts = _ballistic(p, np.zeros(3), r, rng.uniform(0.3, 0.5), 0.3)
        return [_track(cam, pts, r, rng, noise_px)]
    if kind == "bounce":
        p = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(2.0, 3.0)])
        pts = _ballistic(p, np.zeros(3), r, rng.uniform(0.6, 0.8), 0.3)
        return [_track(cam, pts, r, rng, noise_px)]
    if kind == "throw":
        p = np.array([rng.uniform(-1.5, -1.0), rng.uniform(-0.3, 0.3), 1.0])
        v = np.array([rng.uniform(1.0, 2.0), rng.uniform(-0.4, 0.4), rng.uniform(2.5, 4.0)])
        pts = _ballistic(p, v, r, rng.uniform(0.3, 0.6), 0.3)
        return [_track(cam, pts, r, rng, noise_px)]
    if kind == "roll":
        theta = np.radians(rng.uniform(15, 30))
        length = 1.6
        top = np.array([-1.8, rng.uniform(-0.3, 0.3), r + length * np.sin(theta)])
        down = np.array([np.cos(theta), 0.0, -np.sin(theta)])
        a = 5.0 / 7.0 * G * np.sin(theta)
        dt = 1.0 / (FPS * SUBSTEPS)
        out, s, v, on_floor, p, vx = [], 0.0, 0.0, False, top.copy(), 0.0
        decel = rng.uniform(0.3, 0.8)
        for _ in range(FRAMES * SUBSTEPS + 1):
            out.append(p.copy())
            if not on_floor:
                v += a * dt
                s += v * dt
                p = top + s * down
                if s >= length:
                    on_floor = True
                    vx = v * np.cos(theta)
            else:
                vx = max(0.0, vx - decel * dt)
                p = p + np.array([vx * dt, 0.0, 0.0])
                p[2] = r
        return [_track(cam, _sample(out), r, rng, noise_px)]
    if kind == "pendulum":
        pivot = np.array([0.0, rng.uniform(-0.2, 0.2), 2.6])
        length = rng.uniform(0.8, 1.4)
        th, om = np.radians(rng.uniform(20, 40)), 0.0
        damping = rng.uniform(0.0, 0.3)
        dt = 1.0 / (FPS * SUBSTEPS)
        out = []
        for _ in range(FRAMES * SUBSTEPS + 1):
            out.append(pivot + length * np.array([np.sin(th), 0.0, -np.cos(th)]))
            om += (-(G / length) * np.sin(th) - damping * om) * dt
            th += om * dt
        return [_track(cam, _sample(out), 0.08, rng, noise_px)]
    if kind == "collision":
        ma, mb = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)
        e = rng.uniform(0.5, 0.9)
        y = rng.uniform(-0.3, 0.3)
        xa, xb = -1.6, 1.6
        va, vb = rng.uniform(1.2, 2.0), -rng.uniform(0.6, 1.4)
        decel = 0.3
        dt = 1.0 / (FPS * SUBSTEPS)
        pa, pb, hit = [], [], False
        for _ in range(FRAMES * SUBSTEPS + 1):
            pa.append(np.array([xa, y, r]))
            pb.append(np.array([xb, y, r]))
            if not hit and xb - xa <= 2 * r and va > vb:
                u = (ma * va + mb * vb) / (ma + mb)
                va, vb = u - e * mb * (va - vb) / (ma + mb), u + e * ma * (va - vb) / (ma + mb)
                hit = True
            va = np.sign(va) * max(0.0, abs(va) - decel * dt)
            vb = np.sign(vb) * max(0.0, abs(vb) - decel * dt)
            xa += va * dt
            xb += vb * dt
        return [_track(cam, _sample(pa), r, rng, noise_px), _track(cam, _sample(pb), r, rng, noise_px)]
    raise ValueError(f"unknown scene kind {kind!r}")
