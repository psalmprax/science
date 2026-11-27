"""Clips simulated and rendered here, to test the physics judge (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md, 1a).

Status: audit-only

On the owner's word of 27 November ("go ahead to test all"): the judge is tested mainly on clips we simulate and
render ourselves, PhysInOne kept as a second check. PyBullet (zlib licence) simulates rigid bodies at 240 Hz; its
software renderer (no GPU) draws 320 x 240 frames at 30 per second from a nearly level side camera, and its
segmentation buffer gives each object's exact mask. A track is the mask's centre and the square root of its
area per frame; frames where the object is out of view are missing. The engine, its contact model, friction,
spin and tumbling cubes are not those of the judge's own synthetic test worlds
(``apedp_dyn/video/physics_tracks_synthetic.py``), so the judge was not built on this generator; from now on it
is developed on it, so its blind test also needs traps from elsewhere (AGENTS.md 1.6).

Pools: seeds below ``BLIND_FROM`` are development clips; seeds from ``BLIND_FROM`` up are the blind set, read
only by a frozen run (``allow_blind=True``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from apedp_dyn.epistemic.physics_judge import Track

KINDS = ("free_fall", "throw", "roll", "bounce", "pendulum", "collision")
BLIND_FROM = 1_000_000
WIDTH, HEIGHT, FPS, SUBSTEPS = 320, 240, 30, 8      #: 240 Hz physics, a frame every 8 steps
FRAMES = 90


def _pb() -> Any:
    import pybullet as p

    return p


def _body(p: Any, shape: str, size: float, mass: float, pos: list[float], colour: list[float],
          orientation: list[float] | None = None) -> int:
    if shape == "sphere":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=colour)
    else:
        half = [size, size, size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=colour)
    return int(p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=pos, baseOrientation=orientation or [0, 0, 0, 1]))


def _camera(p: Any, rng: np.random.Generator, target: list[float], distance: float) -> tuple[Any, Any, dict[str, float]]:
    yaw = math.radians(rng.uniform(-20, 20))
    pitch = math.radians(rng.uniform(-10, 3))
    eye = [target[0] + distance * math.sin(yaw) * math.cos(pitch),
           target[1] - distance * math.cos(yaw) * math.cos(pitch),
           target[2] - distance * math.sin(pitch)]
    fov = float(rng.uniform(45, 60))
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=WIDTH / HEIGHT, nearVal=0.05, farVal=50)
    return view, proj, {"yaw_deg": round(math.degrees(yaw), 1), "pitch_deg": round(math.degrees(pitch), 1),
                        "fov_deg": round(fov, 1), "distance": round(distance, 2)}


def _scene(p: Any, kind: str, rng: np.random.Generator) -> tuple[list[int], list[float], float, dict[str, Any]]:
    """Builds one scene; returns the tracked bodies, the camera's target, its distance, and the scene's facts."""
    plane = p.createCollisionShape(p.GEOM_PLANE)
    floor = p.createMultiBody(0, plane)
    friction = float(rng.uniform(0.3, 0.8))
    p.changeDynamics(floor, -1, lateralFriction=friction)
    facts: dict[str, Any] = {"kind": kind, "friction": round(friction, 3)}
    red, blue = [0.85, 0.15, 0.1, 1], [0.1, 0.3, 0.85, 1]
    if kind in ("free_fall", "bounce", "throw"):
        shape = "sphere" if kind == "bounce" or rng.random() < 0.6 else "box"
        size = float(rng.uniform(0.06, 0.14))
        height = float(rng.uniform(1.2, 2.2))
        x0 = float(rng.uniform(-0.8, -0.4)) if kind == "throw" else float(rng.uniform(-0.3, 0.3))
        quat = p.getQuaternionFromEuler(list(rng.uniform(-0.5, 0.5, 3))) if shape == "box" else None
        body = _body(p, shape, size, float(rng.uniform(0.2, 2.0)), [x0, 0.0, height], red, quat)
        e = float(rng.uniform(0.6, 0.9) if kind == "bounce" else rng.uniform(0.1, 0.5))
        p.changeDynamics(body, -1, restitution=e, lateralFriction=friction, rollingFriction=0.001)
        p.changeDynamics(floor, -1, restitution=e)
        if kind == "throw":
            p.resetBaseVelocity(body, linearVelocity=[float(rng.uniform(0.8, 2.0)), 0.0, float(rng.uniform(1.5, 3.5))])
        facts.update(shape=shape, size=round(size, 3), height=round(height, 3), restitution=round(e, 3))
        return [body], [0.0, 0.0, 0.8], float(rng.uniform(4.0, 5.5)), facts
    if kind == "roll":
        angle = float(rng.uniform(0.2, 0.45))
        length = 1.6
        slope_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, 0.4, 0.02])
        slope_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, 0.4, 0.02], rgbaColor=[0.6, 0.6, 0.6, 1])
        centre = [-0.9, 0.0, (length / 2) * math.sin(angle)]
        slope = p.createMultiBody(0, slope_col, slope_vis, centre, p.getQuaternionFromEuler([0, angle, 0]))
        p.changeDynamics(slope, -1, lateralFriction=friction)
        r = float(rng.uniform(0.05, 0.1))
        top = [centre[0] - (length / 2 - 0.1) * math.cos(angle), 0.0,
               centre[2] + (length / 2 - 0.1) * math.sin(angle) + r + 0.02]
        ball = _body(p, "sphere", r, float(rng.uniform(0.2, 1.0)), top, red)
        p.changeDynamics(ball, -1, lateralFriction=friction, rollingFriction=float(rng.uniform(0.0005, 0.003)))
        facts.update(slope_deg=round(math.degrees(angle), 1), radius=round(r, 3))
        return [ball], [-0.2, 0.0, 0.4], float(rng.uniform(3.8, 5.0)), facts
    if kind == "pendulum":
        length = float(rng.uniform(0.35, 0.8))           # periods 1.2 to 1.8 s: at least three turns in 3 s
        pivot = [0.0, 0.0, 2.0]
        theta = math.radians(rng.uniform(25, 50))
        pos = [pivot[0] + length * math.sin(theta), 0.0, pivot[2] - length * math.cos(theta)]
        bob = _body(p, "sphere", float(rng.uniform(0.05, 0.09)), float(rng.uniform(0.3, 1.5)), pos, red)
        p.createConstraint(bob, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0],
                           [-length * math.sin(theta), 0.0, length * math.cos(theta)], pivot)
        damping = float(rng.uniform(0.0, 0.4))           # from none to a swing that visibly dies down
        p.changeDynamics(bob, -1, linearDamping=damping, angularDamping=0.0)
        facts.update(length=round(length, 3), amplitude_deg=round(math.degrees(theta), 1), damping=round(damping, 3))
        return [bob], [0.0, 0.0, pivot[2] - 0.6 * length], float(rng.uniform(1.6, 2.2)) + 2.0 * length, facts
    if kind == "collision":
        ra, rb = float(rng.uniform(0.06, 0.1)), float(rng.uniform(0.06, 0.1))
        a = _body(p, "sphere", ra, float(rng.uniform(0.3, 2.0)), [-1.0, 0.0, ra], red)
        b = _body(p, "sphere", rb, float(rng.uniform(0.3, 2.0)), [1.0, 0.0, rb], blue)
        e = float(rng.uniform(0.3, 0.9))
        for body in (a, b):
            p.changeDynamics(body, -1, restitution=e, lateralFriction=friction, rollingFriction=0.002)
        p.resetBaseVelocity(a, linearVelocity=[float(rng.uniform(1.0, 2.0)), 0, 0])
        p.resetBaseVelocity(b, linearVelocity=[-float(rng.uniform(0.0, 1.5)), 0, 0])
        facts.update(restitution=round(e, 3), radii=[round(ra, 3), round(rb, 3)])
        return [a, b], [0.0, 0.0, 0.15], float(rng.uniform(3.0, 4.0)), facts
    raise ValueError(f"unknown kind {kind!r}")


def make_clip(kind: str, seed: int, allow_blind: bool = False) -> tuple[list[Track], dict[str, Any]]:
    """One true clip: the tracks of its tracked objects and the facts of its scene and camera."""
    if seed >= BLIND_FROM and not allow_blind:
        raise ValueError("blind clips are read only by a frozen run")
    p = _pb()
    rng = np.random.default_rng([seed, KINDS.index(kind)])
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(1.0 / (FPS * SUBSTEPS), physicsClientId=cid)
        bodies, target, distance, facts = _scene(p, kind, rng)
        view, proj, cam = _camera(p, rng, target, distance)
        n = len(bodies)
        xy = np.full((n, FRAMES, 2), np.nan)
        size = np.full((n, FRAMES), np.nan)
        for k in range(FRAMES):
            _, _, _, _, seg = p.getCameraImage(WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER)
            seg = np.asarray(seg).reshape(HEIGHT, WIDTH)
            for i, body in enumerate(bodies):
                ys, xs = np.nonzero(seg == body)
                if xs.size:
                    xy[i, k] = (xs.mean(), ys.mean())
                    size[i, k] = math.sqrt(xs.size)
            for _ in range(SUBSTEPS):
                p.stepSimulation()
    finally:
        p.disconnect(cid)
    t = np.arange(FRAMES) / FPS
    tracks = [Track(t, xy[i], size[i], (WIDTH / 2.0, HEIGHT / 2.0)) for i in range(n)]
    return tracks, {**facts, "camera": cam, "seed": seed}
