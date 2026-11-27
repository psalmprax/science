"""Breakages the physics judge was not developed against: the independent reviewer's arm of PREREG-PHYSICS-J1-2026-11.

Status: experimental

Written by the independent reviewer, who took no part in building the judge, after stage 1 was frozen
(``preregistration/DIGEST-PHYSICS-J1-JUDGE.json``, named below by its SHA-256), under the plan's seven rules for the
reviewer's breakages (AGENTS.md 1.6). The builder did not see this file before it was committed. Each kind breaks one
named law (``LAWS``). Where it applies is decided from the true track alone; this module never calls the judge.

Three kinds:

- ``sideways_push`` (falls, throws, bounces). From release to the first landing, a constant horizontal acceleration
  of 25 % to 60 % of the clip's own downward acceleration is added, with its sign drawn at random. The landing point
  keeps the offset it reached. A dropped object then drifts sideways along a curve; a thrown one gains or loses
  horizontal speed in flight. Under the generator's camera (no roll) gravity is image-vertical. On the development
  clips the true first flights' horizontal acceleration is at most a tenth of their downward one (median 0.4 %); the
  broken ones' is at least a quarter. Not a changed g, a time scaling or a mirror: the horizontal and vertical
  motions no longer share one law.
- ``period_drift`` (pendulums). The bob keeps exactly its true path, but the clip's tempo drifts steadily. The rate
  of its time goes from 1 - b to 1 + b over the clip, or from 1 + b to 1 - b, with b drawn in [0.10, 0.20]; the clip
  still runs from the true first frame to the true last. A uniform time scaling would be lawful; a drift is not. It
  applies only where two conditions hold, both found from the true turning points and the drift:
  - the drifted clip's last half-period, against the first on the same side, lies outside 0.94 to 1.06;
  - the bob moves at least the floor below.
  Where the drawn b falls short of either, b is raised until it does not (at most 0.35). On the development clips the
  30 true pendulums' same-side ratio lies within 0.975 to 1.011 (damping and amplitude dependence), and every broken
  one, measured on its own track, lies outside 0.94 to 1.06.
- ``floor_speedup`` (rolls, throws, collisions). On a level stretch of floor, the ball surges: its speed grows by 50 %
  to 100 % within 0.25 to 0.4 s, then falls back to its true speed, so it ends ahead of its true position. The
  stretch is after a roll leaves the slope, after a thrown object lands, or after a collision (for the ball moving
  away from the other, which it never comes closer to). Once on the level floor nothing pushes a ball forward:
  friction only slows it, and a rolling ball's spin cannot add speed. On the development clips the true speed along
  the floor, over a window as long as the surge's rise, is at most 0.98 of itself for rolls and 1.21 for thrown
  objects after landing (small hops); the broken ones' is at least 1.11 and 1.35.

Where each applies (rules 5 and 6):
- The broken track moves at least 5 px and at least 10 noise widths away from the true one somewhere. The noise
  width is the larger of two robust estimates from the true track: the median absolute deviation of its second and
  of its third differences.
- A broken position that would leave the image becomes unseen, as a camera would lose it, and a surge must stay in
  view throughout.
- A breakage that would cost more than 9 frames of view, or leave less than 70 % of the clip in view, does not
  apply, so that a broken clip is not made untestable by construction.

Applicability on the 180 development clips (seeds 0 to 29, ``make_clip(kind, seed)``, with the runner's streams
``stream(seed, how)``), clips it applies to out of 30:

| Scene kind | sideways_push | period_drift | floor_speedup |
| :--- | :--- | :--- | :--- |
| free_fall | 30 | 0 | 0 |
| throw | 29 | 0 | 14 |
| roll | 0 | 0 | 22 |
| bounce | 30 | 0 | 0 |
| pendulum | 0 | 30 | 0 |
| collision | 0 | 0 | 5 |
| all six | 89 of 180 | 30 of 180 | 41 of 180 |

No call raised on those clips; none changed the tracks it was given, and each gave the same result twice. The reviewer
did not run the judge on these breakages.
"""

from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.physics_judge import Track

#: SHA-256 of the bytes of preregistration/DIGEST-PHYSICS-J1-JUDGE.json, the stage-1 freeze these were written against.
STAGE1_SHA256 = "9f4cc10b382577406090c4a5d15ea15a0884bf3d271ee1dece5b5e26a28c225a"

UNSEEN_KINDS = ("sideways_push", "period_drift", "floor_speedup")

LAWS = {
    "sideways_push": "In free flight the only force is gravity, which is vertical: the horizontal velocity stays "
                     "constant (a dropped object falls straight down, a thrown one keeps its horizontal speed until "
                     "it lands).",
    "period_drift": "Isochronism: a pendulum's period is set by its length and stays the same through the clip "
                    "(amplitude dependence at these swings is a few per cent at most).",
    "floor_speedup": "Newton's first law with friction: once on the level floor (after the slope, after landing or "
                     "after the hit) nothing pushes a ball forward, so its speed along the floor cannot grow; it "
                     "keeps it or loses it.",
}

MIN_SHIFT_PX = 5.0          #: rule 5: the breakage moves the track at least this far ...
NOISE_WIDTHS = 10.0         #: ... and at least this many noise widths of the true track
MIN_VISIBLE = 0.7           #: rule 6: a broken clip keeps at least this share of its frames in view ...
MAX_FRAMES_LOST = 9         #: ... and loses at most this many frames of view to the breakage
PUSH_SHARE = (0.25, 0.60)   #: sideways_push: the horizontal acceleration, as a share of the clip's own downward one
DRIFT = (0.10, 0.20)        #: period_drift: the rate of time goes between 1 - b and 1 + b over the clip
MAX_DRIFT = 0.35            #: period_drift: the largest b, where the drawn one leaves the change too small to see
LAWFUL_RATIOS = (0.94, 1.06)  #: period_drift: the broken clip's same-side half-period ratio must lie outside this
                              #: (the 30 true development pendulums lie within 0.975 to 1.011)
SURGE_GAIN = (0.5, 1.0)     #: floor_speedup: the share by which the speed grows
SURGE_SECONDS = (0.25, 0.40)  #: floor_speedup: how long the speed grows (it then falls back over as long)


# ---------------------------------------------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------------------------------------------


def _noise_px(xy: np.ndarray) -> float:
    """The track's position noise (px): the larger of the robust spreads of its second and third differences."""
    widths = []
    for order, var_factor in ((2, 6.0), (3, 20.0)):
        d = np.diff(xy, n=order, axis=0)
        d = d[np.all(np.isfinite(d), axis=1)]
        if d.shape[0] < 5:
            continue
        mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
        widths.append(float(np.max(1.4826 * mad / np.sqrt(var_factor))))
    return max(widths) if widths else 0.0


def _floor_px(track: Track) -> float:
    return max(MIN_SHIFT_PX, NOISE_WIDTHS * _noise_px(track.xy))


def _seen(track: Track) -> np.ndarray:
    out: np.ndarray = np.all(np.isfinite(track.xy), axis=1)
    return out


def _in_image(track: Track, xy: np.ndarray) -> np.ndarray:
    """Which positions are inside the image (the image centre gives its size); all, where it is not known."""
    ok: np.ndarray = np.all(np.isfinite(xy), axis=1)
    if track.centre is None:
        return ok
    width, height = 2.0 * track.centre[0], 2.0 * track.centre[1]
    inside: np.ndarray = np.zeros(xy.shape[0], dtype=bool)
    inside[ok] = (xy[ok, 0] >= 0) & (xy[ok, 0] <= width) & (xy[ok, 1] >= 0) & (xy[ok, 1] <= height)
    return inside


def _finish(track: Track, xy: np.ndarray, size: np.ndarray | None) -> Track | None:
    """The broken track, with positions outside the image unseen; None if it moves less than the floor or loses
    too much of the view."""
    seen_true = _seen(track)
    inside = _in_image(track, xy)
    xy = xy.copy()
    xy[~inside] = np.nan
    size = None if size is None else np.where(inside, size, np.nan)
    both = seen_true & inside
    if not both.any():
        return None
    shift = float(np.max(np.linalg.norm(xy[both] - track.xy[both], axis=1)))
    if not shift >= _floor_px(track):
        return None
    lost = int(seen_true.sum() - inside.sum())
    if inside.mean() < MIN_VISIBLE or lost > MAX_FRAMES_LOST:
        return None
    return Track(track.t.copy(), xy, size, track.centre)


def _copy(track: Track) -> Track:
    return Track(track.t.copy(), track.xy.copy(), None if track.size is None else track.size.copy(), track.centre)


def _run_from(seen: np.ndarray, start: int) -> int:
    """The last index of the run of seen frames that starts at ``start``."""
    end = start
    while end + 1 < seen.size and seen[end + 1]:
        end += 1
    return end


# ---------------------------------------------------------------------------------------------------------------
# sideways_push: falls, throws, bounces
# ---------------------------------------------------------------------------------------------------------------


def _first_flight(track: Track) -> tuple[int, int] | None:
    """From the first seen frame (the release or the throw) to the first landing: the first frame after a descent of
    at least four frames where the object stops descending in the image."""
    seen = _seen(track)
    if not seen.any():
        return None
    start = int(np.argmax(seen))
    end = _run_from(seen, start)
    y = track.xy[start:end + 1, 1]
    descending = 0
    for j, step in enumerate(np.diff(y)):
        if step > 0:
            descending += 1
        elif descending >= 4:
            return start, start + j
        else:
            descending = 0
    return None


def sideways_push(track: Track, rng: np.random.Generator) -> Track | None:
    flight = _first_flight(track)
    if flight is None:
        return None
    start, land = flight
    if land - start < 8:
        return None
    t = track.t
    tau = t[start:land] - t[start]
    coef = np.polyfit(tau, track.xy[start:land, 1], 2)
    down = 2.0 * float(coef[0])                       # px/s^2, image y grows downward
    if not (np.isfinite(down) and down > 0):
        return None
    share = float(rng.uniform(*PUSH_SHARE))
    sign = float(rng.choice([-1.0, 1.0]))
    dx = np.zeros(t.size)
    dx[start:land + 1] = 0.5 * share * down * (t[start:land + 1] - t[start]) ** 2
    dx[land + 1:] = dx[land]
    for s in (sign, -sign):                           # the drawn direction, or the other if it leaves the image
        xy = track.xy.copy()
        xy[:, 0] = xy[:, 0] + s * dx
        out = _finish(track, xy, None if track.size is None else track.size.copy())
        if out is not None:
            return out
    return None


# ---------------------------------------------------------------------------------------------------------------
# period_drift: pendulums
# ---------------------------------------------------------------------------------------------------------------


def _catmull_rom(values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Values on a uniform grid of indices 0..n-1, read at fractional indices ``at`` (a smooth cubic through them)."""
    n = values.shape[0]
    i = np.clip(np.floor(at).astype(int), 0, n - 2)
    f = (at - i)[:, None] if values.ndim == 2 else at - i
    p0 = values[np.clip(i - 1, 0, n - 1)]
    p1 = values[i]
    p2 = values[np.clip(i + 1, 0, n - 1)]
    p3 = values[np.clip(i + 2, 0, n - 1)]
    out: np.ndarray = 0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f**2
                             + (-p0 + 3 * p1 - 3 * p2 + p3) * f**3)
    return out


def _along_swing(track: Track) -> np.ndarray:
    """The bob's position along its main axis of motion, centred between its extremes."""
    xy = track.xy
    c = xy - xy.mean(axis=0)
    s = c @ np.linalg.svd(c, full_matrices=False)[2][0]
    out: np.ndarray = s - 0.5 * (float(s.max()) + float(s.min()))
    return out


def _turns(track: Track) -> np.ndarray:
    """The times of the swing's turning points: the first frame if the bob is let go there at an extreme, then each
    reversal beyond a fifth of the largest excursion, refined by a parabola through five frames; reversals closer
    than five frames are one (a flat turn)."""
    s, t = _along_swing(track), track.t
    top = float(np.max(np.abs(s)))
    if not top > 0:
        none: np.ndarray = np.array([])
        return none
    found: list[tuple[int, float]] = []
    if abs(s[0]) > 0.8 * top:
        found.append((0, float(t[0])))
    for k in range(2, s.size - 2):
        if (s[k] - s[k - 1]) * (s[k + 1] - s[k]) > 0 or abs(s[k]) <= 0.2 * top:
            continue
        a2, a1, _ = np.polyfit(t[k - 2:k + 3] - t[k], s[k - 2:k + 3], 2)
        at = float(t[k] + (-a1 / (2 * a2) if a2 != 0 else 0.0))
        if found and k - found[-1][0] < 5:
            if abs(s[k]) > abs(s[found[-1][0]]):
                found[-1] = (k, at)
            continue
        found.append((k, at))
    times: np.ndarray = np.array([at for _, at in found])
    return times


def _same_parity(halves: np.ndarray) -> float | None:
    """The last half-period against the first of the same parity (one side of the swing against the same side)."""
    last = halves.size - 1 if (halves.size - 1) % 2 == 0 else halves.size - 2
    return float(halves[last] / halves[0]) if last >= 2 else None


def _drifted_ratio(track: Track, clock: np.ndarray) -> float | None:
    """The same-parity half-period ratio the broken clip will show, from the true turning points and the clock (the
    true clip's time, as a share of its length, at each frame of the broken clip)."""
    turns = _turns(track)
    if turns.size < 4:
        return None
    span = track.t[-1] - track.t[0]
    seen_at = np.interp(turns, track.t[0] + clock * span, track.t)   # when the broken clip shows each true turn
    return _same_parity(np.diff(seen_at))


def period_drift(track: Track, rng: np.random.Generator) -> Track | None:
    seen = _seen(track)
    n = track.t.size
    if n < 30 or not seen.all():
        return None
    dt = np.diff(track.t)
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-9):
        return None
    turns = _turns(track)
    true_ratio = _same_parity(np.diff(turns)) if turns.size >= 4 else None
    if true_ratio is None:
        return None
    b = float(rng.uniform(*DRIFT))
    quicker = bool(rng.random() < 0.5)
    u = (track.t - track.t[0]) / (track.t[-1] - track.t[0])
    # The rate of the clip's time goes from 1 - b to 1 + b (the swing quickens) or from 1 + b to 1 - b; the clip
    # still runs from the true first frame to the true last. Where the drawn b would leave the change inside what
    # true swings show, or move the bob less than the floor, b is raised (up to MAX_DRIFT); if that is not enough,
    # the kind does not apply.
    floor = _floor_px(track)
    while b <= MAX_DRIFT:
        clock = u - b * (u - u**2) if quicker else u + b * (u - u**2)
        ratio = _drifted_ratio(track, clock)
        at = clock * (n - 1)
        xy = _catmull_rom(track.xy, at)
        outside = ratio is not None and (ratio <= LAWFUL_RATIOS[0] or ratio >= LAWFUL_RATIOS[1])
        if outside and float(np.max(np.linalg.norm(xy - track.xy, axis=1))) >= floor:
            size = None if track.size is None else _catmull_rom(track.size, at)
            return _finish(track, xy, size)
        b = round(b + 0.01, 6)
    return None


# ---------------------------------------------------------------------------------------------------------------
# floor_speedup: rolls, throws, collisions
# ---------------------------------------------------------------------------------------------------------------


def _level_motion(track: Track, start: int, end: int) -> tuple[np.ndarray, float] | None:
    """The direction and speed (px/s) of a nearly level, steady stretch of floor motion, or None."""
    if end - start + 1 < 12:
        return None
    tt = track.t[start:end + 1]
    vx = float(np.polyfit(tt, track.xy[start:end + 1, 0], 1)[0])
    vy = float(np.polyfit(tt, track.xy[start:end + 1, 1], 1)[0])
    speed = float(np.hypot(vx, vy))
    if not (np.isfinite(speed) and speed >= 20.0) or abs(vy) > 0.2 * abs(vx):
        return None
    return np.array([vx, vy]) / speed, speed


def _surge(track: Track, start: int, end: int, rng: np.random.Generator) -> Track | None:
    """The ball surges forward somewhere in frames start..end of a level floor stretch."""
    motion = _level_motion(track, start, end)
    if motion is None:
        return None
    direction, speed = motion
    gain = float(rng.uniform(*SURGE_GAIN))
    rise = float(rng.uniform(*SURGE_SECONDS))
    dt = float(np.median(np.diff(track.t)))
    width = int(np.ceil(2 * rise / dt))
    if end - start < width:
        return None
    begin = int(rng.integers(start, end - width + 1))
    t0 = float(track.t[begin])
    accel = gain * speed / rise
    tau = np.clip(track.t - t0, 0.0, 2 * rise)
    # Extra speed grows by accel over `rise`, then falls back to zero over as long: the offset it leaves is the area.
    extra = np.where(tau <= rise, 0.5 * accel * tau**2,
                     accel * rise**2 - 0.5 * accel * (2 * rise - tau) ** 2)
    xy = track.xy + extra[:, None] * direction[None, :]
    if not _in_image(track, xy)[begin:begin + width + 1].all():
        return None                                    # the whole surge must be seen, not run out of the image
    return _finish(track, xy, None if track.size is None else track.size.copy())


def _floor_stretch(track: Track) -> tuple[int, int] | None:
    """The stretch after the object has come down onto the floor (down the slope, or at its landing): from three
    frames after it first reaches the floor's height in the image to the end of that run of seen frames."""
    seen = _seen(track)
    idx = np.flatnonzero(seen)
    if idx.size < 20:
        return None
    y = track.xy[:, 1]
    floor_y = float(np.median(y[idx[-max(5, idx.size // 4):]]))
    drop = floor_y - float(y[idx[0]])
    if not drop > 0:
        return None
    reach = [k for k in idx if y[k] >= floor_y - max(2.0, 0.1 * drop)]
    if not reach:
        return None
    start = int(reach[0]) + 3
    if start >= seen.size or not seen[start]:
        return None
    return start, _run_from(seen, start)


def _collision_after(a: Track, b: Track) -> int | None:
    """The first frame after the two objects' closest approach (their contact)."""
    both = _seen(a) & _seen(b)
    if both.sum() < 20:
        return None
    dist = np.where(both, np.linalg.norm(np.nan_to_num(a.xy) - np.nan_to_num(b.xy), axis=1), np.inf)
    return int(np.argmin(dist)) + 4


def floor_speedup(kind: str, tracks: list[Track], rng: np.random.Generator) -> list[Track] | None:
    if kind in ("roll", "throw"):
        stretch = _floor_stretch(tracks[0])
        if stretch is None:
            return None
        broken = _surge(tracks[0], *stretch, rng)
        return None if broken is None else [broken]
    # A collision: the ball moving away from the other after the hit, the faster one if both do.
    a, b = tracks
    after = _collision_after(a, b)
    if after is None:
        return None
    options = []
    for i, (me, other) in enumerate(((a, b), (b, a))):
        seen = _seen(me) & _seen(other)
        if after >= seen.size or not seen[after]:
            continue
        end = _run_from(seen, after)
        motion = _level_motion(me, after, end)
        if motion is None:
            continue
        direction, speed = motion
        if float(np.dot(direction, me.xy[after] - other.xy[after])) <= 0:
            continue                                   # moving toward the other: a surge would push it into it
        options.append((speed, i, end))
    if not options:
        return None
    _, i, end = max(options)
    broken = _surge(tracks[i], after, end, rng)
    if broken is None:
        return None
    other = tracks[1 - i]
    both = _seen(broken) & _seen(other)
    true_gap = np.linalg.norm(tracks[i].xy - other.xy, axis=1)
    gap = np.linalg.norm(broken.xy - other.xy, axis=1)
    if np.any(gap[both] < true_gap[both] - 1e-9):
        return None                                    # the surge must not bring the balls closer than they were
    out = [_copy(tr) for tr in tracks]
    out[i] = broken
    return out


# ---------------------------------------------------------------------------------------------------------------
# Entry point (the runner's interface)
# ---------------------------------------------------------------------------------------------------------------


def perturb_unseen(kind: str, tracks: list[Track], how: str, rng: np.random.Generator) -> list[Track] | None:
    """The clip's tracks broken by the reviewer's kind ``how``, or None where it does not apply. Pure: the tracks it
    is given are not changed, and the same tracks and stream give the same result."""
    if how not in UNSEEN_KINDS:
        raise ValueError(f"unknown kind of breakage {how!r}")
    if how == "sideways_push":
        if kind not in ("free_fall", "throw", "bounce") or len(tracks) != 1:
            return None
        broken = sideways_push(tracks[0], rng)
        return None if broken is None else [broken]
    if how == "period_drift":
        if kind != "pendulum" or len(tracks) != 1:
            return None
        broken = period_drift(tracks[0], rng)
        return None if broken is None else [broken]
    if (kind in ("roll", "throw") and len(tracks) == 1) or (kind == "collision" and len(tracks) == 2):
        return floor_speedup(kind, tracks, rng)
    return None
