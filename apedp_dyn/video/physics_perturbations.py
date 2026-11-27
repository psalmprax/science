"""Development perturbations for the physics judge (docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md, part 1a).

Status: experimental. Four ways to break a clip's physics, applied to its object tracks:

- ``teleport``: from one frame on, the object is shifted by three times its size (a jump in position);
- ``freeze``: the object stops for a while and then carries on as if nothing happened (a hold);
- ``vertical_reversal``: for part of the clip the vertical motion runs mirrored (it accelerates upward),
  with the position kept continuous at both ends; applied where it moves the object at least
  ``MIN_SHIFT_PX``, contacts included;
- ``time_reversal``: the clip runs backwards; applied only to clips that lose energy (they end at rest,
  or their swing decays by a tenth and four noise widths), because a clip without losses looks lawful
  backwards too.

Where each applies is set by the physics and the track's noise, not by the judge's own tolerances (the
independent review of 27 November found two rules of that day that had been).

These are the development kinds. The blind test adds kinds the judge was not developed against, written
by the independent reviewer (AGENTS.md 1.6); they are not in this file.
"""

from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.physics_judge import Track

PERTURBATIONS = ("teleport", "freeze", "vertical_reversal", "time_reversal")
MIN_SHIFT_PX = 5.0      #: a mirrored stretch must move the object at least this far: a floor set apart from the judge
LOSS_SHARE = 0.10       #: a swing loses energy when its last swing is at least this share below its first ...
NOISE_SIGMAS = 4.0      #: ... and by at least this many widths of the track's own noise


def _noise_px(track: Track) -> float:
    """The track's position noise, from its second differences (median absolute deviation), in pixels."""
    ok = np.all(np.isfinite(track.xy), axis=1)
    xy = track.xy[ok]
    if xy.shape[0] < 5:
        return 0.0
    d2 = np.diff(xy, n=2, axis=0)
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
    return float(np.mean(1.4826 * mad / np.sqrt(6.0)))


def _speed(track: Track) -> np.ndarray:
    xy = track.xy
    v = np.full(xy.shape[0], np.nan)
    ok = np.all(np.isfinite(xy), axis=1)
    if ok.sum() < 3:
        empty: np.ndarray = v
        return empty
    t = track.t
    grad = np.stack([np.gradient(np.where(ok, xy[:, j], np.nan), t) for j in range(2)], axis=1)
    sp = np.linalg.norm(grad, axis=1)
    out = sp.copy()
    for k in range(sp.size):  # running median of five, ignoring NaN
        w = sp[max(0, k - 2):k + 3]
        w = w[np.isfinite(w)]
        out[k] = float(np.median(w)) if w.size else np.nan
    result: np.ndarray = out
    return result


def _size(track: Track) -> float:
    if track.size is None:
        return 10.0
    s = track.size[np.isfinite(track.size)]
    return float(np.median(s)) if s.size else 10.0


def _moving_frame(track: Track, rng: np.random.Generator, lo: float = 0.1, hi: float = 0.8,
                  vertical: bool = False, span: int = 0) -> int | None:
    """A frame where the object moves (vertically, if asked); with ``span``, one from which it keeps moving
    for at least half of the next ``span`` frames, so a stretch begun there is motion, not a landing."""
    sp = _speed(track)
    if vertical:
        ok = np.all(np.isfinite(track.xy), axis=1)
        vy = np.gradient(np.where(ok, track.xy[:, 1], np.nan), track.t)
        sp = np.abs(vy)
    top = np.nanmax(sp) if np.any(np.isfinite(sp)) else 0.0
    if not top > 0:
        return None
    n = sp.size
    moving = np.isfinite(sp) & (sp > 0.3 * top)
    idx = [k for k in range(int(lo * n), int(hi * n)) if moving[k]
           and (span == 0 or float(np.mean(moving[k:k + span])) >= 0.5)]
    if not idx:
        return None
    return int(rng.choice(idx))


def loses_energy(tracks: list[Track], kind: str) -> bool:
    """Whether a true clip loses energy, so that running it backwards breaks its physics."""
    if kind == "pendulum":
        tr = tracks[0]
        ok = np.all(np.isfinite(tr.xy), axis=1)
        xy = tr.xy[ok]
        if xy.shape[0] < 10:
            return False
        centred = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        s = centred @ vt[0]
        # The swing's amplitude at its extremes (local extremes of the position along the swing), the first against
        # the last on the same side (a camera at an angle makes the near side look larger). A loss counts when it is
        # a tenth of the swing and four noise widths: set by the physics and the noise, not by the judge's limits.
        mid = 0.5 * (float(s.max()) + float(s.min()))
        a = np.abs(s - mid)
        peaks = [k for k in range(1, a.size - 1) if a[k] >= a[k - 1] and a[k] >= a[k + 1] and a[k] > 0.3 * a.max()]
        side = [k for k in peaks if np.sign(s[k] - mid) == np.sign(s[peaks[0]] - mid)] if peaks else []
        if len(side) < 2:
            return False
        first, last = float(a[side[0]]), float(a[side[-1]])
        return last <= (1.0 - LOSS_SHARE) * first and first - last >= NOISE_SIGMAS * _noise_px(tr)
    for tr in tracks:
        sp = _speed(tr)
        if not np.any(np.isfinite(sp)):
            continue
        top = float(np.nanmax(sp))
        tail = sp[-10:]
        tail = tail[np.isfinite(tail)]
        if top > 0 and tail.size and float(np.median(tail)) < 0.2 * top:
            return True
    return False


def _with(track: Track, xy: np.ndarray, size: np.ndarray | None) -> Track:
    return Track(track.t.copy(), xy, size, track.centre)


def teleport(track: Track, rng: np.random.Generator) -> Track | None:
    k = _moving_frame(track, rng)
    if k is None:
        return None
    angle = rng.uniform(0, 2 * np.pi)
    shift = 3.0 * _size(track) * np.array([np.cos(angle), np.sin(angle)])
    xy = track.xy.copy()
    xy[k:] += shift
    return _with(track, xy, None if track.size is None else track.size.copy())


def freeze(track: Track, rng: np.random.Generator) -> Track | None:
    k = _moving_frame(track, rng)
    if k is None or k < 1:
        return None
    n = track.t.size
    hold = max(8, n // 8)
    if k + hold >= n:
        return None
    xy = np.concatenate([track.xy[:k], np.repeat(track.xy[k - 1:k], hold, axis=0), track.xy[k:n - hold]])
    size = None
    if track.size is not None:
        size = np.concatenate([track.size[:k], np.repeat(track.size[k - 1:k], hold), track.size[k:n - hold]])
    return _with(track, xy, size)


def vertical_reversal(track: Track, rng: np.random.Generator) -> Track | None:
    n = track.t.size
    length = max(10, n // 5)
    k = _moving_frame(track, rng, 0.05, 0.7, vertical=True)
    if k is None:
        return None
    if k + length >= n:
        return None
    xy = track.xy.copy()
    y = track.xy[:, 1]
    xy[k:k + length, 1] = 2.0 * y[k] - y[k:k + length]
    xy[k + length:, 1] = y[k + length:] - 2.0 * (y[k + length - 1] - y[k])
    if not np.nanmax(np.abs(xy[:, 1] - y)) >= MIN_SHIFT_PX:
        return None  # the mirror moves the object by less than the floor: not a test of anything
    return _with(track, xy, None if track.size is None else track.size.copy())


def time_reversal(track: Track) -> Track:
    return _with(track, track.xy[::-1].copy(), None if track.size is None else track.size[::-1].copy())


def perturb(kind: str, tracks: list[Track], how: str, rng: np.random.Generator) -> list[Track] | None:
    """The clip's tracks broken one way, or None where that way does not apply to this clip."""
    if how not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {how!r}")
    if how == "time_reversal":
        if not loses_energy(tracks, kind):
            return None
        return [time_reversal(tr) for tr in tracks]
    # Break the object that moves most; in a collision the other one stays as it was.
    extents = []
    for tr in tracks:
        ok = np.all(np.isfinite(tr.xy), axis=1)
        xy = tr.xy[ok]
        extents.append(float(np.max(np.linalg.norm(xy - xy[0], axis=1))) if xy.shape[0] else 0.0)
    target = int(np.argmax(extents))
    fn = {"teleport": teleport, "freeze": freeze, "vertical_reversal": vertical_reversal}[how]
    broken = fn(tracks[target], rng)
    if broken is None:
        return None
    out = list(tracks)
    out[target] = broken
    return out
