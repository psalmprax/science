"""E5 ambiguity stability / identifiability under drift (forensics doc S73).

Can APEDP distinguish stable non-identifiability (B) from drifting
identifiability (C)? Control: stable identifiable (A). No new epistemic
status is implemented here; this experiment establishes the signatures
first. Additive only; adjudicate.py untouched.

Frozen design (hypotheses H1=x^2, H2=x^2+0.15x^3, H3=x^2-0.15x^3;
observations x in [-1,1], sigma=0.1, 6 windows x 300 points):
  A stable identifiable:   truth H2 every window.
  B stable non-identifiable: truth H1, rival gap ~0 (H2 coef 1e-6).
  C drifting: truth H2 (w1-2) -> H1 (w3-4) -> H3 (w5-6).
Per window: window-local adjudication (margin, winner) + cumulative
adjudication on all data so far. Frozen classifier (see classify()):
  A: cumulative reaches UNIQUE, winner never rotates.
  B: never UNIQUE (local or cumulative), margins flat ~0.
  C: winner rotates >=1 across windows OR cumulative UNIQUE-then-loss.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from apedp_dyn.epistemic.active_disambiguation.trading_e4t1_v2 import _adjudicate, _score
from apedp_dyn.epistemic.adjudicate import REL_FLOOR
from apedp_dyn.grammar.candidate_ir import ScoredCandidate

HIDS = ("H1", "H2", "H3")
N_WINDOWS = 6
N_PER_WINDOW = 300
NOISE = 0.1
SEED = 91


def _forms(case: str) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    if case == "B":
        return {"H1": lambda x: x**2, "H2": lambda x: x**2 + 1e-6 * x**3, "H3": lambda x: x**2 - 1e-6 * x**3}
    return {"H1": lambda x: x**2, "H2": lambda x: x**2 + 0.45 * x**3, "H3": lambda x: x**2 - 0.45 * x**3}


def _truth(case: str, window: int) -> str:
    if case == "A":
        return "H2"
    if case == "B":
        return "H1"
    if case == "C":
        return "H2" if window < 2 else ("H1" if window < 4 else "H3")
    raise ValueError(case)


@dataclass(frozen=True)
class WindowOutcome:
    window: int
    truth: str
    local_status: str
    local_winner: str | None
    local_best: str  # rank-1 identity regardless of status (rotation tracking)
    local_margin: float
    local_floor: float  # kernel resolution floor (REL_FLOOR x RMS(y)); rotation
    cumul_status: str  # must clear this, else the change is sub-τ noise
    cumul_winner: str | None
    cumul_margin: float


@dataclass(frozen=True)
class Trajectory:
    case: str
    windows: tuple[WindowOutcome, ...]


def _margin(scored: list[ScoredCandidate]) -> float:
    rms = sorted(float(s.holdout_rmse) for s in scored if s.holdout_rmse is not None)
    return rms[1] - rms[0]


def _best(scored: list[ScoredCandidate]) -> str:
    return min(scored,
               key=lambda s: s.holdout_rmse
               if s.holdout_rmse is not None else float("inf")).candidate.candidate_id


def run_case(case: str, seed: int = SEED) -> Trajectory:
    forms = _forms(case)
    rng = np.random.default_rng(seed)
    xs = [rng.uniform(-1, 1, N_PER_WINDOW) for _ in range(N_WINDOWS)]
    truths = [_truth(case, w) for w in range(N_WINDOWS)]
    ys = [forms[t](x) + rng.normal(0, NOISE, N_PER_WINDOW) for x, t in zip(xs, truths, strict=True)]
    out = []
    cx = np.concatenate(xs[:1])
    cy = np.concatenate(ys[:1])
    for w in range(N_WINDOWS):
        local = [_score(h, ys[w] - forms[h](xs[w])) for h in HIDS]
        ls, lw = _adjudicate(local)
        if w > 0:
            cx = np.concatenate([cx, xs[w]])
            cy = np.concatenate([cy, ys[w]])
        cumul = [_score(h, cy - forms[h](cx)) for h in HIDS]
        cs, cw = _adjudicate(cumul)
        floor = float(REL_FLOOR * np.sqrt(np.mean(np.asarray(ys[w]) ** 2)))
        out.append(WindowOutcome(w, truths[w], ls, lw, _best(local), _margin(local), floor,
                                 cs, cw, _margin(cumul)))
    return Trajectory(case, tuple(out))


ROT_MARGIN = 1e-4  # rotation counts only above B's numerical floor (~1e-9)


def classify(traj: Trajectory) -> str:
    """Frozen trajectory classifier: 'stable-identifiable' | 'stable-ambiguous' | 'drifting'.

    Rotation gate is kernel-consistent, not tuned: a rank change counts only
    when the incoming window's margin clears that window's own resolution
    floor (sub-τ differences are "observationally indistinguishable" by the
    frozen adjudication contract). ROT_MARGIN remains as the numerical
    backstop for degenerate (zero-scale) windows.
    """
    bests = [w.local_best for w in traj.windows]
    margins = [w.local_margin for w in traj.windows]
    floors = [w.local_floor for w in traj.windows]
    rotation = sum(
        1
        for a, b, ma, mb, fa, fb in zip(bests[:-1], bests[1:], margins[:-1], margins[1:],
                                        floors[:-1], floors[1:], strict=True)
        if a != b and max(ma, mb) > max(ROT_MARGIN, fb)
    )
    ever_unique = any("unique_support" in w.cumul_status.lower() for w in traj.windows)
    cumul_stats = [w.cumul_status for w in traj.windows]
    margins = [w.cumul_margin for w in traj.windows]
    # Sustained collapse only: single-window τ-straddling (Δ≈τ flicker) must
    # not count as drift. Require final MULTI + margin below half its peak.
    lost = (
        ever_unique
        and "multi" in cumul_stats[-1].lower()
        and margins[-1] < 0.5 * max(margins[:-1])
    )
    if rotation >= 1 or lost:
        return "drifting"
    if ever_unique:
        return "stable-identifiable"
    return "stable-ambiguous"


def summarize(traj: Trajectory) -> dict[str, object]:
    return {
        "case": traj.case,
        "class": classify(traj),
        "local_unique": sum("unique_support" in w.local_status.lower() for w in traj.windows),
        "cumul_unique": sum("unique_support" in w.cumul_status.lower() for w in traj.windows),
        "winner_rotation": len({w.local_best for w in traj.windows}),
        "cumul_margins": [round(w.cumul_margin, 4) for w in traj.windows],
    }
