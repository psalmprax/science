"""A physics judge for object tracks in video (step 1a of docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md).

Status: experimental. Nothing reads it in force; its settings are development settings, proposed and not
pinned. Before any gate uses them they need the owner's word, a pin and a CHANGELOG entry (AGENTS.md 1.3).

The judge gets the track of one object (two for a collision) from a static camera: per frame, the time in
seconds, the image position of the object's centre in pixels (x to the right, y down) and, if known, its
size (the square root of its mask's area). It says ``pass``, ``fail`` or ``untestable``, with reasons.

What it tests is the form of the law, never its constants: pixel scale and frame rate are not known, so a
fall must show constant acceleration in the camera's projection, not 9.81 m/s^2.

How:

1. Frames whose mask area jumps against its neighbours (occlusion, a tracker slip) are dropped; too few
   visible frames, or too little motion, make the clip untestable.
2. The noise is estimated from third differences (a projected parabola has none to speak of).
3. The track is cut into segments by optimal partitioning with a Bayesian information penalty. Each segment
   is fitted with the projection of constant-acceleration motion through a pinhole camera: both image
   coordinates are ratios of quadratics in time sharing one denominator (the depth). A contact (a bounce, a
   landing, a hit) is where one segment ends and the next begins.
4. At each contact: the position must be continuous (no jump the motion cannot explain) and the speed must
   not grow (no energy from nowhere: an object at rest does not start moving by itself).
5. Under gravity (falls, throws, bounces, rolls): no segment may accelerate upward against the clip's own
   gravity, and the clip must show the law it is about (a fall must accelerate downward somewhere; a roll
   must speed up somewhere).
6. A pendulum's bob must keep to one conic (a circle seen in projection), keep its velocity continuous,
   swing with a steady half-period and an amplitude that does not grow.
7. A collision: both objects change velocity at the same moment, their changes point in opposite
   directions (momentum, with the mass ratio unknown but positive), the kinetic energy implied does not
   grow, and they do not pass through each other.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCENE_KINDS = ("free_fall", "throw", "roll", "bounce", "pendulum", "collision")
GRAVITY_KINDS = ("free_fall", "throw", "roll", "bounce")
PASS, FAIL, UNTESTABLE = "pass", "fail", "untestable"


@dataclass(frozen=True)
class Track:
    """One object's track: times (s), image positions (px, NaN where not seen), optional size (px)."""

    t: np.ndarray
    xy: np.ndarray
    size: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        xy = np.asarray(self.xy, dtype=float)
        if t.ndim != 1 or xy.shape != (t.size, 2):
            raise ValueError("a track needs t of shape (n,) and xy of shape (n, 2)")
        if self.size is not None and np.asarray(self.size).shape != t.shape:
            raise ValueError("size must have the shape of t")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "xy", xy)
        if self.size is not None:
            object.__setattr__(self, "size", np.asarray(self.size, dtype=float))


@dataclass(frozen=True)
class JudgeSettings:
    """Development settings. Every number is proposed; none is pinned or in force."""

    min_visible_fraction: float = 0.6
    min_motion_sizes: float = 1.0
    min_motion_px: float = 8.0
    noise_floor_px: float = 0.25
    min_segment: int = 2
    contact_min_frames: int = 4
    jump_sigma: float = 6.0
    jump_speed_frames: float = 0.5
    speed_gain_tol: float = 0.5
    rebound_tol: float = 0.1
    stop_rising_share: float = 0.5
    rest_share: float = 0.05
    speed_noise_sigma: float = 4.0
    accel_z: float = 4.0
    min_accel_frames: int = 6
    upward_accel_share: float = 0.25
    upward_accel_sizes: float = 2.0
    area_jump_ratio: float = 1.5
    pendulum_conic_sigma: float = 6.0
    pendulum_velocity_jump: float = 0.35
    pendulum_window: int = 6
    period_tol: float = 0.2
    amplitude_growth_tol: float = 0.15
    turning_prominence: float = 0.2
    collision_sync_frames: int = 2
    collision_cos: float = 0.5
    collision_event_share: float = 0.15
    energy_gain_tol: float = 0.1
    overlap_share: float = 0.5


DEVELOPMENT = JudgeSettings()


@dataclass
class Verdict:
    kind: str
    outcome: str
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------
# The segment model: the pinhole projection of constant-acceleration motion
# ---------------------------------------------------------------------------------------------------------


MODEL_PARAMETERS = {"rest": 2, "line": 4, "parabola": 6, "projected": 8}
MODEL_MIN_FRAMES = {"rest": 2, "line": 3, "parabola": 4, "projected": 6}


@dataclass(frozen=True)
class _Segment:
    """A stretch of track between contacts and the motion model that fits it best.

    ``projected`` is constant-acceleration motion seen through a pinhole camera: both image coordinates
    are ratios of quadratics in time with one shared denominator (the depth). ``parabola`` (constant
    image acceleration), ``line`` (constant image velocity) and ``rest`` are its simpler cases, used where
    they fit as well, since the penalty counts parameters.
    """

    start: int
    stop: int  # inclusive, in the index of the cleaned track
    model: str
    t0: float
    span: float
    mu: np.ndarray
    scale: float
    coef: np.ndarray  # projected: a0 a1 a2 b0 b1 b2 c1 c2; polynomials: (degree + 1, 2), normalised units
    cost: float

    def _tau(self, t: np.ndarray | float) -> np.ndarray:
        out: np.ndarray = (np.atleast_1d(np.asarray(t, dtype=float)) - self.t0) / self.span
        return out

    def pos(self, t: np.ndarray | float) -> np.ndarray:
        tau = self._tau(t)
        if self.model != "projected":
            powers = np.stack([tau**k for k in range(self.coef.shape[0])], axis=1)
            poly: np.ndarray = self.mu + self.scale * (powers @ self.coef)
            return poly
        a0, a1, a2, b0, b1, b2, c1, c2 = self.coef
        q = 1.0 + c1 * tau + c2 * tau**2
        p = np.stack([a0 + a1 * tau + a2 * tau**2, b0 + b1 * tau + b2 * tau**2], axis=-1)
        out: np.ndarray = self.mu + self.scale * p / q[:, None]
        return out

    def vel(self, t: np.ndarray | float) -> np.ndarray:
        tau = self._tau(t)
        if self.model != "projected":
            deg = self.coef.shape[0] - 1
            if deg == 0:
                return np.zeros((tau.size, 2))
            powers = np.stack([k * tau ** (k - 1) for k in range(1, deg + 1)], axis=1)
            dpoly: np.ndarray = self.scale * (powers @ self.coef[1:]) / self.span
            return dpoly
        a0, a1, a2, b0, b1, b2, c1, c2 = self.coef
        q = 1.0 + c1 * tau + c2 * tau**2
        p = np.stack([a0 + a1 * tau + a2 * tau**2, b0 + b1 * tau + b2 * tau**2], axis=-1)
        dp = np.stack([a1 + 2 * a2 * tau, b1 + 2 * b2 * tau], axis=-1)
        dq = c1 + 2 * c2 * tau
        out: np.ndarray = self.scale * (dp * q[:, None] - p * dq[:, None]) / (q[:, None] ** 2) / self.span
        return out


def _fit_projected(tau: np.ndarray, z: np.ndarray, iterations: int = 2) -> np.ndarray | None:
    """P(tau)/Q(tau) for both coordinates with Q shared: linear least squares re-weighted by 1/Q."""
    n = tau.size
    ridge = 1e-6
    weights = np.ones(n)
    coef: np.ndarray | None = None
    for _ in range(iterations + 1):
        a = np.zeros((2 * n + 2, 8))
        rhs = np.zeros(2 * n + 2)
        w = weights
        for j, off in ((0, 0), (1, 3)):
            rows = slice(j * n, (j + 1) * n)
            a[rows, off] = w
            a[rows, off + 1] = w * tau
            a[rows, off + 2] = w * tau**2
            a[rows, 6] = -w * z[:, j] * tau
            a[rows, 7] = -w * z[:, j] * tau**2
            rhs[rows] = w * z[:, j]
        a[2 * n, 6] = math.sqrt(ridge)
        a[2 * n + 1, 7] = math.sqrt(ridge)
        coef, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        q = 1.0 + coef[6] * tau + coef[7] * tau**2
        if np.any(q <= 0.05):
            return None
        weights = 1.0 / q
    if coef is None:
        return None
    coef = _refine_projected(tau, z, coef)
    grid = np.linspace(-0.1, 1.1, 25)
    if np.any(1.0 + coef[6] * grid + coef[7] * grid**2 <= 0.05):
        return None
    return coef


def _projected_residual(tau: np.ndarray, z: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a0, a1, a2, b0, b1, b2, c1, c2 = coef
    q = 1.0 + c1 * tau + c2 * tau**2
    pu = a0 + a1 * tau + a2 * tau**2
    pv = b0 + b1 * tau + b2 * tau**2
    r = np.concatenate([z[:, 0] - pu / q, z[:, 1] - pv / q])
    n = tau.size
    jac = np.zeros((2 * n, 8))
    for k in range(3):
        jac[:n, k] = tau**k / q
        jac[n:, 3 + k] = tau**k / q
    for j, col in ((1, 6), (2, 7)):
        jac[:n, col] = -pu * tau**j / q**2
        jac[n:, col] = -pv * tau**j / q**2
    return r, jac


def _refine_projected(tau: np.ndarray, z: np.ndarray, coef: np.ndarray, iterations: int = 6) -> np.ndarray:
    """Levenberg-Marquardt on the image residual itself, from the linearised fit."""
    r, jac = _projected_residual(tau, z, coef)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iterations):
        jtj = jac.T @ jac
        step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), jac.T @ r)
        trial = coef + step
        q = 1.0 + trial[6] * tau + trial[7] * tau**2
        if np.all(q > 0.05):
            r_new, jac_new = _projected_residual(tau, z, trial)
            new_cost = float(r_new @ r_new)
            if new_cost < cost:
                coef, r, jac, cost = trial, r_new, jac_new, new_cost
                lam *= 0.3
                continue
        lam *= 10.0
    return coef


def _segment(t: np.ndarray, xy: np.ndarray, start: int, stop: int, sigma: float,
             penalty_per_parameter: float) -> _Segment | None:
    """The best-penalised model for frames start..stop, or None if none applies."""
    tt, pts = t[start:stop + 1], xy[start:stop + 1]
    n = tt.size
    t0 = float(tt[0])
    span = float(tt[-1] - tt[0]) if n > 1 else 1.0
    if span <= 0:
        return None
    tau = (tt - t0) / span
    mu = pts.mean(axis=0)
    scale = float(max(np.abs(pts - mu).max(), 1e-9))
    z = (pts - mu) / scale
    best: _Segment | None = None
    best_score = np.inf
    for model, k in MODEL_PARAMETERS.items():
        if n < MODEL_MIN_FRAMES[model]:
            continue
        if best_score <= k * penalty_per_parameter:
            break  # a richer model's penalty alone exceeds the best score so far
        if model == "projected":
            coef = _fit_projected(tau, z)
            if coef is None:
                continue
        else:
            deg = {"rest": 0, "line": 1, "parabola": 2}[model]
            design = np.stack([tau**j for j in range(deg + 1)], axis=1)
            coef, *_ = np.linalg.lstsq(design, z, rcond=None)
        seg = _Segment(start, stop, model, t0, span, mu, scale, coef, 0.0)
        resid = pts - seg.pos(tt)
        cost = float(np.sum(resid**2) / sigma**2)
        score = cost + k * penalty_per_parameter
        if score < best_score:
            best_score = score
            best = _Segment(start, stop, model, t0, span, mu, scale, coef, cost)
    return best


def _partition(t: np.ndarray, xy: np.ndarray, sigma: float, min_len: int) -> list[_Segment]:
    """Optimal partitioning into motion segments, with a Bayesian information penalty per parameter."""
    n = t.size
    per_parameter = math.log(max(2 * n, 3))
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    choice: list[tuple[int, _Segment | None]] = [(-1, None)] * (n + 1)
    for stop in range(min_len - 1, n):
        for start in range(0, stop - min_len + 2):
            if not np.isfinite(best[start]):
                continue
            seg = _segment(t, xy, start, stop, sigma, per_parameter)
            if seg is None:
                continue
            total = best[start] + seg.cost + MODEL_PARAMETERS[seg.model] * per_parameter
            if total < best[stop + 1]:
                best[stop + 1] = total
                choice[stop + 1] = (start, seg)
    if not np.isfinite(best[n]):
        return []
    segments: list[_Segment] = []
    end = n
    while end > 0:
        start, seg = choice[end]
        if seg is None:
            raise RuntimeError("partition back-pointer missing")
        segments.append(seg)
        end = start
    return segments[::-1]


# ---------------------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------------------


def _clean(track: Track, settings: JudgeSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    ok = np.all(np.isfinite(track.xy), axis=1) & np.isfinite(track.t)
    size = track.size
    if size is not None:
        ok &= np.isfinite(size) & (size > 0)
        s = np.where(ok, size, np.nan)
        for k in np.flatnonzero(ok):
            lo, hi = max(0, k - 3), min(s.size, k + 4)
            neighbours = np.concatenate([s[lo:k], s[k + 1:hi]])
            neighbours = neighbours[np.isfinite(neighbours)]
            if neighbours.size >= 2:
                ratio = s[k] / float(np.median(neighbours))
                if ratio > settings.area_jump_ratio or ratio < 1.0 / settings.area_jump_ratio:
                    ok[k] = False
    t = track.t[ok]
    xy = track.xy[ok]
    sz = None if size is None else size[ok]
    return t, xy, sz, int(track.t.size)


def _noise(xy: np.ndarray, floor: float) -> float:
    if xy.shape[0] < 5:
        return floor
    d3 = np.diff(xy, n=3, axis=0)
    mad = float(np.median(np.abs(d3 - np.median(d3, axis=0))))
    return max(1.4826 * mad / math.sqrt(20.0), floor)


def _preflight(track: Track, settings: JudgeSettings, kind: str) -> tuple[Verdict | None, dict[str, Any]]:
    t, xy, size, total = _clean(track, settings)
    info: dict[str, Any] = {"t": t, "xy": xy, "size": size}
    visible = t.size / max(total, 1)
    if t.size < 2 * settings.min_segment or visible < settings.min_visible_fraction:
        return Verdict(kind, UNTESTABLE, [f"object seen in {visible:.0%} of frames"],
                       {"visible": visible}), info
    size_med = float(np.median(size)) if size is not None else float("nan")
    extent = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    need = settings.min_motion_px
    if size is not None:
        need = max(need, settings.min_motion_sizes * size_med)
    if extent < need:
        return Verdict(kind, UNTESTABLE, [f"object moves {extent:.1f} px, less than {need:.1f}"],
                       {"visible": visible, "extent": extent}), info
    sigma = _noise(xy, settings.noise_floor_px)
    info.update(sigma=sigma, size_med=size_med, extent=extent, visible=visible,
                dt=float(np.median(np.diff(t))))
    return None, info


@dataclass(frozen=True)
class _Event:
    frame: int  # index in the cleaned track of the first frame after the contact
    time: float
    gap: float
    gap_allowed: float
    v_before: np.ndarray
    v_after: np.ndarray
    v_noise: float
    rest_speed: float

    @property
    def dv(self) -> np.ndarray:
        out: np.ndarray = self.v_after - self.v_before
        return out


def _max_speed(t: np.ndarray, segments: list[_Segment]) -> float:
    top = 0.0
    for seg in segments:
        v = seg.vel(t[seg.start:seg.stop + 1])
        top = max(top, float(np.max(np.linalg.norm(v, axis=1))))
    return top


def _events(t: np.ndarray, segments: list[_Segment], sigma: float, dt: float,
            settings: JudgeSettings) -> list[_Event]:
    """Contacts between consecutive segments, each at the moment the two segments' motions meet.

    At a real contact both models pass through the contact point at the contact time, so their distance
    there is near zero; after a jump in position it stays near the jump's size.
    """
    events = []
    top = _max_speed(t, segments)
    # Segments shorter than a contact's frames are part of the contact between their long neighbours.
    long = [seg for seg in segments if seg.stop - seg.start + 1 >= settings.contact_min_frames]
    for a, b in itertools.pairwise(long):
        grid = np.linspace(t[a.stop] - 0.5 * dt, t[b.start] + 0.5 * dt, 41)
        d = np.linalg.norm(a.pos(grid) - b.pos(grid), axis=1)
        k = int(np.argmin(d))
        tc = float(grid[k])
        va, vb = a.vel(tc)[0], b.vel(tc)[0]
        n = min(a.stop - a.start + 1, b.stop - b.start + 1)
        v_se = 3.0 * sigma / dt * math.sqrt(12.0 / max(n * (n * n - 1.0), 6.0))
        speed = max(float(np.linalg.norm(va)), float(np.linalg.norm(vb)))
        allowed = settings.jump_sigma * sigma + settings.jump_speed_frames * speed * dt
        v_noise = settings.speed_noise_sigma * v_se
        events.append(_Event(b.start, tc, float(d[k]), float(allowed), va, vb, v_noise,
                             v_noise + settings.rest_share * top))
    return events


def _vertical_accel(t: np.ndarray, y: np.ndarray, sigma: float) -> tuple[float, float]:
    """Image-vertical acceleration of a quadratic fit and its standard error (px/s^2)."""
    tc = t - t.mean()
    x = np.stack([np.ones_like(tc), tc, tc**2], axis=1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    cov = sigma**2 * np.linalg.inv(x.T @ x)
    return 2.0 * float(coef[2]), 2.0 * math.sqrt(float(cov[2, 2]))


def _along_accel(t: np.ndarray, xy: np.ndarray, sigma: float) -> tuple[float, float]:
    """Acceleration along the segment's net direction of travel, and its standard error."""
    d = xy[-1] - xy[0]
    norm = float(np.linalg.norm(d))
    if norm == 0:
        return 0.0, float("inf")
    return _vertical_accel(t, xy @ (d / norm), sigma)


def _event_summary(e: _Event) -> dict[str, Any]:
    return {"frame": e.frame, "gap": round(e.gap, 2), "gap_allowed": round(e.gap_allowed, 2),
            "rest_speed": round(e.rest_speed, 2),
            "speed_before": round(float(np.linalg.norm(e.v_before)), 2),
            "speed_after": round(float(np.linalg.norm(e.v_after)), 2), "v_noise": round(e.v_noise, 2)}


def _contact_reasons(events: list[_Event], settings: JudgeSettings,
                     skip: frozenset[int] | set[int] = frozenset(), gravity: bool = False) -> list[str]:
    """Continuity of position at every contact; no speed from nowhere at a contact outside ``skip``.

    Under gravity two more: a rebound (falling before, rising after) is no faster vertically than the fall
    (restitution at most one), and a rising object does not stop rising at a contact (nothing is above it).
    """
    reasons = []
    for e in events:
        if e.gap > e.gap_allowed:
            reasons.append(f"position jumps {e.gap:.1f} px at frame {e.frame} (allowed {e.gap_allowed:.1f})")
        if e.frame in skip:
            continue
        before = float(np.linalg.norm(e.v_before))
        after = float(np.linalg.norm(e.v_after))
        if before <= e.rest_speed and after > e.rest_speed + e.v_noise:
            reasons.append(f"starts moving from rest by itself at frame {e.frame} ({after:.1f} px/s)")
        elif before > e.rest_speed and after > (1.0 + settings.speed_gain_tol) * before + e.v_noise:
            reasons.append(f"speed grows at a contact at frame {e.frame}: {before:.1f} -> {after:.1f} px/s")
        if not gravity:
            continue
        vy0, vy1 = float(e.v_before[1]), float(e.v_after[1])
        if vy0 > e.rest_speed and vy1 < -e.v_noise and -vy1 > (1.0 + settings.rebound_tol) * vy0 + e.v_noise:
            reasons.append(f"rebounds faster than it fell at frame {e.frame}: {vy0:.1f} -> {-vy1:.1f} px/s")
        if vy0 < -e.rest_speed and vy1 - vy0 > max(settings.stop_rising_share * -vy0, e.v_noise):
            reasons.append(f"stops rising abruptly at frame {e.frame} ({-vy0:.1f} px/s up before)")
    return reasons


# ---------------------------------------------------------------------------------------------------------
# One object under gravity: falls, throws, bounces, rolls
# ---------------------------------------------------------------------------------------------------------


def _judge_gravity(track: Track, kind: str, settings: JudgeSettings) -> Verdict:
    early, info = _preflight(track, settings, kind)
    if early is not None:
        return early
    t, xy, sigma, dt = info["t"], info["xy"], info["sigma"], info["dt"]
    size_med = info["size_med"]
    segments = _partition(t, xy, sigma, settings.min_segment)
    if not segments:
        return Verdict(kind, FAIL, ["no segment of constant-acceleration motion fits"], {"sigma": sigma})
    events = _events(t, segments, sigma, dt, settings)
    reasons = _contact_reasons(events, settings, gravity=True)

    accels = []
    for seg in segments:
        n = seg.stop - seg.start + 1
        if n < settings.min_accel_frames:
            continue
        tt, yy = t[seg.start:seg.stop + 1], xy[seg.start:seg.stop + 1]
        a, se = _vertical_accel(tt, yy[:, 1], sigma)
        along, along_se = _along_accel(tt, yy, sigma)
        accels.append((seg, a, se, along, along_se))
    down = [a for _, a, se, _, _ in accels if a > settings.accel_z * se]
    a_ref = max(down) if down else None
    limit = (settings.upward_accel_share * a_ref if a_ref is not None
             else settings.upward_accel_sizes * (size_med if math.isfinite(size_med) else 10.0))
    for seg, a, se, _, _ in accels:
        if a < -settings.accel_z * se and -a > limit:
            reasons.append(f"accelerates upward ({a:.0f} px/s^2) over frames {seg.start}-{seg.stop}")
    if kind in ("free_fall", "throw", "bounce") and not down:
        reasons.append("no segment shows a downward acceleration")
    if kind == "roll" and not any(al > settings.accel_z * ase for _, _, _, al, ase in accels):
        reasons.append("no segment speeds up along its path")
    stats = {"sigma": round(sigma, 3), "segments": [[s.start, s.stop] for s in segments],
             "events": [_event_summary(e) for e in events],
             "accels": [[s.start, s.stop, round(a, 1), round(se, 1)] for s, a, se, _, _ in accels],
             "visible": round(info["visible"], 3)}
    return Verdict(kind, FAIL if reasons else PASS, reasons, stats)


# ---------------------------------------------------------------------------------------------------------
# A pendulum
# ---------------------------------------------------------------------------------------------------------


def _conic_residual(xy: np.ndarray) -> float:
    """Robust spread (px) of the points about the best conic, Taubin's fit (a circle seen in projection)."""
    mu = xy.mean(axis=0)
    sc = float(max(np.abs(xy - mu).max(), 1e-9))
    z = (xy - mu) / sc
    x, y = z[:, 0], z[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    d = np.stack([x * x, x * y, y * y, x, y, one], axis=1)
    dx = np.stack([2 * x, y, zero, one, zero, zero], axis=1)
    dy = np.stack([zero, x, 2 * y, zero, one, zero], axis=1)
    m = d.T @ d
    nmat = dx.T @ dx + dy.T @ dy
    # Taubin: minimise v'Mv / v'Nv. N is singular in the constant term; eliminate it.
    m11, m12, m22 = m[:5, :5], m[:5, 5:], m[5:, 5:]
    reduced = m11 - m12 @ np.linalg.pinv(m22) @ m12.T
    n11 = nmat[:5, :5]
    w, v = np.linalg.eig(np.linalg.pinv(n11) @ reduced)
    w = np.real(w)
    v = np.real(v)
    k = int(np.argmin(np.where(w > -1e-12, w, np.inf)))
    head = v[:, k]
    tail = -np.linalg.pinv(m22) @ m12.T @ head
    coef = np.concatenate([head, tail])
    f = d @ coef
    grad = np.sqrt((dx @ coef) ** 2 + (dy @ coef) ** 2)
    geo = np.abs(f) / np.maximum(grad, 1e-12)
    return float(1.4826 * np.median(geo) * sc)


def _local_breaks(t: np.ndarray, xy: np.ndarray, sigma: float, window: int) -> list[dict[str, float]]:
    """At each frame boundary, the gap in position and velocity between parabolas fitted on either side."""
    out = []
    n = t.size
    for k in range(window, n - window + 1):
        left, right = slice(k - window, k), slice(k, k + window)
        tm = 0.5 * (t[k - 1] + t[k])
        preds = []
        for sl in (left, right):
            tt = t[sl] - tm
            design = np.stack([np.ones_like(tt), tt, tt**2], axis=1)
            coef, *_ = np.linalg.lstsq(design, xy[sl], rcond=None)
            cov = np.linalg.inv(design.T @ design)
            preds.append((coef[0], coef[1], sigma * math.sqrt(cov[0, 0]), sigma * math.sqrt(cov[1, 1])))
        (p0, v0, sp0, sv0), (p1, v1, sp1, sv1) = preds
        out.append({"frame": k, "dp": float(np.linalg.norm(p1 - p0)), "dv": float(np.linalg.norm(v1 - v0)),
                    "sp": math.hypot(sp0, sp1), "sv": math.hypot(sv0, sv1),
                    "speed": max(float(np.linalg.norm(v0)), float(np.linalg.norm(v1)))})
    return out


def _judge_pendulum(track: Track, settings: JudgeSettings) -> Verdict:
    kind = "pendulum"
    early, info = _preflight(track, settings, kind)
    if early is not None:
        return early
    t, xy, sigma, dt = info["t"], info["xy"], info["sigma"], info["dt"]
    reasons = []
    conic = _conic_residual(xy)
    if conic > settings.pendulum_conic_sigma * sigma:
        reasons.append(f"the bob leaves its arc: {conic:.2f} px off the best conic (noise {sigma:.2f})")
    breaks = _local_breaks(t, xy, sigma, settings.pendulum_window)
    for b in breaks:
        if b["dp"] > settings.jump_sigma * b["sp"] + settings.jump_speed_frames * b["speed"] * dt:
            reasons.append(f"position jumps {b['dp']:.1f} px at frame {b['frame']}")
            break
    for b in breaks:
        if b["dv"] > settings.pendulum_velocity_jump * b["speed"] + settings.speed_noise_sigma * b["sv"]:
            reasons.append(f"velocity jumps at frame {b['frame']}: {b['dv']:.1f} px/s")
            break
    segments = _partition(t, xy, sigma, settings.min_segment)
    events = _events(t, segments, sigma, dt, settings)
    # Swing along the principal axis: steady half-period, amplitude that does not grow.
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    s = centred @ vt[0]
    if segments:
        vel = np.concatenate([seg.vel(t[seg.start:seg.stop + 1]) @ vt[0] for seg in segments])
    else:
        vel = np.gradient(s, t)
    sign = np.sign(vel)
    rng = float(s.max() - s.min())
    turns: list[int] = []
    for k in range(1, s.size):
        if sign[k] == 0 or sign[k - 1] == 0 or sign[k] == sign[k - 1]:
            continue
        ref = s[turns[-1]] if turns else s[0]
        if abs(s[k] - ref) >= settings.turning_prominence * rng and (not turns or k - turns[-1] >= 3):
            turns.append(k)
    stats: dict[str, Any] = {"sigma": round(sigma, 3), "conic_px": round(conic, 3), "turns": turns,
                             "events": [_event_summary(e) for e in events]}
    # Extremes of the swing: the turning points, and the ends of the clip where the bob is at rest there.
    speed = np.abs(vel)
    rest = settings.rest_share * float(speed.max()) if speed.size else 0.0
    extremes = list(turns)
    if speed.size and speed[0] <= 2 * rest and (not extremes or extremes[0] > 2):
        extremes.insert(0, 0)
    if speed.size and speed[-1] <= 2 * rest and (not extremes or extremes[-1] < s.size - 3):
        extremes.append(s.size - 1)
    if len(extremes) < 2:
        if reasons:
            return Verdict(kind, FAIL, reasons, stats)
        return Verdict(kind, UNTESTABLE, ["fewer than two turning points seen"], stats)
    centre = float(s[int(np.argmax(speed))])
    amps = np.abs(s[extremes] - centre)
    stats.update(extremes=extremes, amplitudes=[round(float(a), 1) for a in amps])
    if len(turns) >= 3:
        halves = np.diff(t[turns])
        stats["half_periods"] = [round(float(h), 3) for h in halves]
        if halves.max() > (1.0 + settings.period_tol) * halves.min():
            reasons.append(f"half-periods vary from {halves.min():.3f} to {halves.max():.3f} s")
    for k in range(1, amps.size):
        if amps[k] > (1.0 + settings.amplitude_growth_tol) * amps[k - 1]:
            reasons.append(f"the swing grows: {amps[k - 1]:.1f} -> {amps[k]:.1f} px")
    return Verdict(kind, FAIL if reasons else PASS, reasons, stats)


# ---------------------------------------------------------------------------------------------------------
# A collision of two objects
# ---------------------------------------------------------------------------------------------------------


def _judge_collision(a: Track, b: Track, settings: JudgeSettings) -> Verdict:
    kind = "collision"
    parts = []
    for name, tr in (("A", a), ("B", b)):
        early, info = _preflight(tr, settings, kind)
        if early is not None and "moves" not in " ".join(early.reasons):
            return Verdict(kind, UNTESTABLE, [f"object {name}: {r}" for r in early.reasons], early.stats)
        if early is not None:  # an object at rest until hit is fine; it must still be seen
            t, xy, size, _ = _clean(tr, settings)
            info = {"t": t, "xy": xy, "size": size, "sigma": _noise(xy, settings.noise_floor_px),
                    "dt": float(np.median(np.diff(t))),
                    "size_med": float(np.median(size)) if size is not None else float("nan")}
        parts.append(info)
    reasons: list[str] = []
    per: list[tuple[list[_Segment], list[_Event]]] = []
    for name, info in zip(("A", "B"), parts, strict=True):
        segs = _partition(info["t"], info["xy"], info["sigma"], settings.min_segment)
        if not segs:
            return Verdict(kind, FAIL, [f"object {name}: no segment of constant-acceleration motion fits"], {})
        per.append((segs, _events(info["t"], segs, info["sigma"], info["dt"], settings)))
    dt = min(parts[0]["dt"], parts[1]["dt"])
    max_speed = []
    for segs, _ in per:
        speeds = [float(np.max(np.linalg.norm(s.vel(np.linspace(s.t0, s.t0 + s.span, 5)), axis=1))) for s in segs]
        max_speed.append(max(speeds))
    significant = []
    for (_segs, evs), top in zip(per, max_speed, strict=True):
        significant.append([e for e in evs if float(np.linalg.norm(e.dv)) > max(e.v_noise, settings.collision_event_share * top)])
    pairs = []
    for ea in significant[0]:
        for eb in significant[1]:
            if abs(ea.time - eb.time) <= settings.collision_sync_frames * dt + 1e-9:
                pairs.append((ea, eb))
    paired_a = {ea.frame for ea, _ in pairs}
    paired_b = {eb.frame for _, eb in pairs}
    for name, (_, evs), skip in (("A", per[0], paired_a), ("B", per[1], paired_b)):
        reasons += [f"object {name}: {r}" for r in _contact_reasons(evs, settings, skip)]
    # Pass-through: centres closer than half the summed sizes at any common frame.
    ta, tb = parts[0]["t"], parts[1]["t"]
    common, ia, ib = np.intersect1d(np.round(ta, 6), np.round(tb, 6), return_indices=True)
    stats: dict[str, Any] = {"pairs": len(pairs), "events_a": [_event_summary(e) for e in per[0][1]],
                             "events_b": [_event_summary(e) for e in per[1][1]]}
    if common.size and parts[0]["size"] is not None and parts[1]["size"] is not None:
        dist = np.linalg.norm(parts[0]["xy"][ia] - parts[1]["xy"][ib], axis=1)
        reach = settings.overlap_share * 0.5 * (parts[0]["size"][ia] + parts[1]["size"][ib])
        overlap = bool(np.any(dist < reach))
        stats["closest"] = round(float(np.min(dist)), 1)
    else:
        overlap = False
    if not pairs:
        if overlap and not significant[0] and not significant[1]:
            reasons.append("the objects pass through each other")
        elif significant[0] or significant[1]:
            reasons.append("one object changes velocity and the other does not react")
        if not reasons:
            return Verdict(kind, UNTESTABLE, ["no collision seen"], stats)
        return Verdict(kind, FAIL, reasons, stats)
    for ea, eb in pairs:
        da, db = ea.dv, eb.dv
        na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
        cos = float(np.dot(da, -db) / max(na * nb, 1e-12))
        if cos < settings.collision_cos:
            reasons.append(f"velocity changes are not opposite at the collision (cos {cos:.2f})")
            continue
        ratio = nb / max(na, 1e-12)  # m_A / m_B from momentum
        before = ratio * float(np.dot(ea.v_before, ea.v_before)) + float(np.dot(eb.v_before, eb.v_before))
        after = ratio * float(np.dot(ea.v_after, ea.v_after)) + float(np.dot(eb.v_after, eb.v_after))
        if after > (1.0 + settings.energy_gain_tol) * before:
            reasons.append(f"kinetic energy grows at the collision ({before:.0f} -> {after:.0f}, mass ratio {ratio:.2f})")
        stats.setdefault("collisions", []).append({"cos": round(cos, 3), "mass_ratio": round(ratio, 3),
                                                   "energy_before": round(before, 1), "energy_after": round(after, 1)})
    return Verdict(kind, FAIL if reasons else PASS, reasons, stats)


# ---------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------


def judge(kind: str, tracks: list[Track], settings: JudgeSettings = DEVELOPMENT) -> Verdict:
    """The verdict for one clip of a scene kind; ``tracks`` holds one track (two for a collision)."""
    if kind not in SCENE_KINDS:
        raise ValueError(f"unknown scene kind {kind!r}")
    if kind == "collision":
        if len(tracks) != 2:
            raise ValueError("a collision needs two tracks")
        return _judge_collision(tracks[0], tracks[1], settings)
    if len(tracks) != 1:
        raise ValueError(f"{kind} needs one track")
    if kind == "pendulum":
        return _judge_pendulum(tracks[0], settings)
    return _judge_gravity(tracks[0], kind, settings)
