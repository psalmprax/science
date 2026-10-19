"""Floor-rule fitting and proposal construction (neutral shared layer).

Moved verbatim from `apedp_dyn.epistemic.prior_guided_search` so that proposer
modules never import underscore-private helpers from the referee package
(`apedp_dyn.epistemic`, AGENTS.md §1.1). Both the referee-side prior-guided
search adapter and `apedp_dyn.proposers.rime_ad_proposer` import from this
neutral layer.

All fits operate on dev rows only; this module never sees holdout slices and
never promotes.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype

PATH_EXTRAPOLATABLE_RULE = "extrapolatable_rule_form"
FLOOR_RULE_PROPOSER_ID = "prior_guided_search"


def fit_floor_rule(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    thresholds: Sequence[float],
    *,
    global_widths: int = 30,
    phase_steps: int = 15,
    refine_radius: float = 0.05,
    refine_steps: int = 41,
) -> tuple[float, float, float, float, float] | None:
    """Fit q0*floor((x-off)/w) + b by deterministic two-stage gridded least squares.

    Stage 1 (global): log-spaced widths from twice the finest resolvable scale
    (2 * median adjacent-x spacing * min plausible run) up to half the dev span,
    each phased to the first estimated change-point. Stage 2 (refine): dense
    local grid around the stage-1 winner. All fits are OLS on dev rows only.
    Returns (q0, w, off, b, dev_r2), or None when fewer than two thresholds
    exist to phase the rule. All grid sizes are explicit caller arguments.
    """
    thr = np.asarray(sorted(float(t) for t in thresholds), dtype=float)
    if thr.size < 2:
        return None
    dx = np.diff(np.sort(np.asarray(x_dev, dtype=float).ravel()))
    dx_med = float(np.median(dx[dx > 0])) if np.any(dx > 0) else 1.0
    span = float(np.max(x_dev) - np.min(x_dev))
    w_lo = max(2.0 * dx_med, span / max(len(x_dev), 2))
    w_hi = max(span / 2.0, w_lo * 1.5)
    t0 = float(thr[0])

    def _ols(w: float, off: float) -> tuple[float, float, float, float]:
        pred = np.floor((x_dev - off) / w)
        A = np.column_stack([pred, np.ones_like(pred)])
        coef, *_ = np.linalg.lstsq(A, y_dev, rcond=None)
        fit = A @ coef
        ss = float(np.sum((y_dev - fit) ** 2))
        return ss, float(coef[0]), float(coef[1]), float(off)

    def _margin(w: float, off: float) -> float:
        """Max-margin tie-break: distance from the nearest jump to any dev point.

        When several (w, off) classify dev rows identically, the one placing
        jumps mid-gap generalizes best. Phase distance of each x to the jump
        lattice, minimized over rows.
        """
        ph = np.mod(x_dev - off, w)
        return float(np.min(np.minimum(ph, w - ph)))

    evals: list[tuple[float, float, float, float, float]] = []

    def _scan(w: float, off: float) -> None:
        ss, q0, b, oo = _ols(w, off)
        evals.append((ss, q0, w, oo, b))

    for w in np.exp(np.linspace(np.log(w_lo), np.log(w_hi), global_widths)):
        for off in np.linspace(t0 - w, t0, phase_steps):
            _scan(float(w), float(off))
    if not evals:
        return None
    seed_ss = min(e[0] for e in evals)
    seed = min((e for e in evals if e[0] <= seed_ss * (1.0 + 1e-6) + 1e-20),
               key=lambda e: -_margin(e[2], e[3]))
    _, _q0, w, off, _b = seed
    for w_try in np.linspace(w * (1.0 - refine_radius), w * (1.0 + refine_radius), refine_steps):
        if w_try <= 0:
            continue
        for oo in np.linspace(off - w_try / 2.0, off + w_try / 2.0, refine_steps):
            _scan(float(w_try), float(oo))
    best_ss = min(e[0] for e in evals)
    tol = best_ss * (1.0 + 1e-6) + 1e-20
    tied = [e for e in evals if e[0] <= tol]
    ss, _q0, w, off, _b = max(tied, key=lambda e: _margin(e[2], e[3]))
    pred = np.floor((x_dev - off) / w)
    A = np.column_stack([pred, np.ones_like(pred)])
    coef, *_ = np.linalg.lstsq(A, y_dev, rcond=None)
    ss_tot = float(np.sum((y_dev - y_dev.mean()) ** 2))
    r2 = 1.0 - ss / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(w), float(off), float(coef[1]), r2


def floor_rule_candidate(
    x_dev: np.ndarray,
    q0: float,
    w: float,
    off: float,
    b: float,
    dev_r2: float,
    feature_name: str,
) -> CandidateProposal:
    """Extrapolatable rule-form proposal. Generalizes past the last threshold."""

    def _scorer(df: pd.DataFrame) -> np.ndarray:
        xv = np.asarray(df[feature_name], dtype=float).ravel()
        return q0 * np.floor((xv - off) / w) + b

    formula = f"{q0:.6g} * floor(({feature_name} - {off:.6g}) / {w:.6g}) + {b:.6g}"
    return CandidateProposal(
        proposer_id=FLOOR_RULE_PROPOSER_ID,
        candidate_id="guided_floor_rule_data_phased",
        formula=formula,
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=False,
        declared_nominal_dependencies=(feature_name,),
        compute_score=_scorer,
        parameters={"quantum_step": float(q0), "step_width": float(w),
                    "offset": float(off), "level": float(b)},
        metadata={"r2": dev_r2, "method": PATH_EXTRAPOLATABLE_RULE,
                  "note": "width/offset phased from data-estimated change-points; "
                          "extrapolates past the dev window"},
    )
