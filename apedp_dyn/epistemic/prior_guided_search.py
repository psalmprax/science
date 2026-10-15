"""Prior-guided search adapter: closes the profiler -> search loop.

What was open. `profile_dataset` emits grammar/loss/transmutation
recommendations and `staircase_prior` emits step families, but no search
consumed either: the recommendations were advisory strings. This adapter is
the consumer. It reads a shape prior and a dataset profile computed on DEV
rows only, selects a search path through a FIXED, preregistered table, fits
only on dev, and emits `CandidateProposal` objects. It never promotes, never
scores holdout, never sees holdout rows (AGENTS.md 1.1/1.2): the function
signature accepts dev arrays only, and holdout evaluation is the CALLER's job
(own the split, score with `candidate.compute_score`, never refit).

What is deliberately unreachable. Human-chosen operator primitives. The
selectable paths are table rows keyed by (family, profile flags); this module
contains no operator-name literals used for selection. That property is
machine-enforced by `test_no_human_primitive_literals`, which scans this
file's source for quoted primitive tokens. Routing to the transmutation fitter
on `piecewise_constant` data is table-driven selection with data-estimated
parameters (step width, offset, quantum), not per-equation human declaration:
the table cannot name an equation, only a family.

Two forms for stepped data, stated plainly. The sign-step basis
(`c0 + sum ci*sign(x - ti)`, thresholds from the data) recovers dev rows
exactly but does NOT extrapolate past the last threshold -- fixed thresholds
cannot reach unseen x. The extrapolatable rule form (fitted quantum, width,
offset) is what generalizes to holdout. The adapter returns both, primary
first: exactness is demonstrated on dev, generalization is measured on
holdout, and neither is claimed for the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.autonomous_profiler import DatasetProfile, profile_dataset
from apedp_dyn.epistemic.coordinate_transmuter import TransmutationResult, transmute_and_propose
from apedp_dyn.epistemic.shape_inference import (
    FAMILY_PIECEWISE_CONSTANT,
    ShapePrior,
    StepFamilySeeds,
    fit_step_family,
    staircase_prior,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype

PATH_STEP_BASIS = "step_basis_exact_dev"
PATH_EXTRAPOLATABLE_RULE = "extrapolatable_rule_form"
PATH_TRANSMUTED_SINGULAR = "transmuted_singular_or_power"
PATH_ABSTAIN = "abstain_no_supported_family"

_GUIDED_PROPOSER_ID = "prior_guided_search"


@dataclass(frozen=True)
class GuidedProposal:
    """Everything the closed loop produced from dev rows. Proposals, not verdicts."""

    path: str
    family: str
    primary: CandidateProposal | None
    supporting: tuple[CandidateProposal, ...]
    formula: str
    parameters: dict[str, float]
    thresholds: tuple[float, ...]
    dev_r2: float
    plan_rationale: str
    abstained: bool
    abstain_reason: str

    def candidates(self) -> tuple[CandidateProposal, ...]:
        out = (self.primary,) if self.primary is not None else ()
        return out + self.supporting

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "family": self.family,
            "formula": self.formula,
            "parameters": dict(self.parameters),
            "thresholds": list(self.thresholds),
            "dev_r2": self.dev_r2,
            "plan_rationale": self.plan_rationale,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "n_candidates": len(self.candidates()),
        }


def _fit_floor_rule(
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
    _, q0, w, off, b = seed
    for w in np.linspace(w * (1.0 - refine_radius), w * (1.0 + refine_radius), refine_steps):
        if w <= 0:
            continue
        for oo in np.linspace(off - w / 2.0, off + w / 2.0, refine_steps):
            _scan(float(w), float(oo))
    best_ss = min(e[0] for e in evals)
    tol = best_ss * (1.0 + 1e-6) + 1e-20
    tied = [e for e in evals if e[0] <= tol]
    ss, q0, w, off, b = max(tied, key=lambda e: _margin(e[2], e[3]))
    pred = np.floor((x_dev - off) / w)
    A = np.column_stack([pred, np.ones_like(pred)])
    coef, *_ = np.linalg.lstsq(A, y_dev, rcond=None)
    ss_tot = float(np.sum((y_dev - y_dev.mean()) ** 2))
    r2 = 1.0 - ss / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(w), float(off), float(coef[1]), r2


def _floor_rule_candidate(
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
        proposer_id=_GUIDED_PROPOSER_ID,
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


def _sign_basis_candidate(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    seeds: StepFamilySeeds,
    feature_name: str,
) -> tuple[CandidateProposal, str, float]:
    """Dev-exact sign-step proposal. Exact on dev by construction; NOT holdout-capable."""
    beta = np.asarray(seeds.coefficients, dtype=float)
    thr = np.asarray(seeds.thresholds, dtype=float)

    def _scorer(df: pd.DataFrame) -> np.ndarray:
        xv = np.asarray(df[feature_name], dtype=float).ravel()
        out = np.full_like(xv, beta[0])
        for c, t in zip(beta[1:], thr):
            out = out + c * np.where(xv >= t, 1.0, -1.0)
        return out

    terms = " + ".join(f"{c:.6g}*(sign({feature_name}-{t:.6g}))" for c, t in zip(beta[1:], thr))
    formula = f"{beta[0]:.6g} + {terms}" if terms else f"{beta[0]:.6g}"
    cand = CandidateProposal(
        proposer_id=_GUIDED_PROPOSER_ID,
        candidate_id=f"guided_step_basis_{len(thr)}steps",
        formula=formula,
        strategy_archetype=StrategyArchetype.TACTICAL,
        declared_scale_invariant=False,
        declared_nominal_dependencies=(feature_name,),
        compute_score=_scorer,
        parameters={"intercept": float(beta[0]),
                    **{f"step_{i}": float(c) for i, c in enumerate(beta[1:])}},
        metadata={"r2": seeds.r2, "method": PATH_STEP_BASIS,
                  "note": "dev-exact by construction; fixed thresholds do not extrapolate"},
    )
    return cand, formula, seeds.r2


def select_search_path(
    prior: ShapePrior,
    profile: DatasetProfile,
) -> str:
    """The fixed table. Family and flags in, path out. No equation names, no primitives.

    Row 1: stepped data (prior verdict) -> fit the step basis for dev-exactness
    AND the extrapolatable rule form for holdout. Both parameters from data.
    Row 2: singular/power-law flags (profile verdict) -> transmutation fitter.
    Row 3: anything else -> abstain. The adapter declines rather than invents.
    """
    if prior.family == FAMILY_PIECEWISE_CONSTANT and prior.n_change_points > 0:
        return PATH_EXTRAPOLATABLE_RULE
    if profile.is_arrhenius_tunneling or profile.is_pure_power_law:
        return PATH_TRANSMUTED_SINGULAR
    return PATH_ABSTAIN


def propose_unsolicited(
    x_dev: Sequence[float],
    y_dev: Sequence[float],
    feature_name: str,
    *,
    shape_kwargs: dict[str, Any] | None = None,
    min_rows: int = 40,
) -> GuidedProposal:
    """Run the closed loop on dev rows. The ONLY human input is calling this function.

    `shape_kwargs` carries the caller's explicit cutoffs for the prior (see
    `shape_inference.infer_shape_prior`); this adapter defines none of its own.
    """
    xs = np.asarray(x_dev, dtype=float).ravel()
    ys = np.asarray(y_dev, dtype=float).ravel()
    if xs.size < min_rows:
        return GuidedProposal(PATH_ABSTAIN, "absent", None, (), "", {}, (), float("nan"),
                              "too few dev rows for a prior", True,
                              f"n_dev={xs.size} < min_rows={min_rows}")

    sk = dict(shape_kwargs) if shape_kwargs else {}
    prior, seeds = staircase_prior(xs, ys, **sk)
    profile = profile_dataset(xs, ys)
    path = select_search_path(prior, profile)

    if path == PATH_ABSTAIN:
        return GuidedProposal(PATH_ABSTAIN, prior.family, None, (), "", {}, (), float("nan"),
                              f"table selects abstain for family={prior.family}", True,
                              f"family={prior.family}; no supported path")

    if path == PATH_TRANSMUTED_SINGULAR:
        res: TransmutationResult = transmute_and_propose(xs, ys, profile,
                                                         feature_name=feature_name)
        if not res.success or res.candidate is None:
            return GuidedProposal(PATH_ABSTAIN, prior.family, None, (), "", {}, (), float("nan"),
                                  "transmutation fitter declined", True,
                                  "fitter returned success=False")
        return GuidedProposal(path, prior.family, res.candidate, (), res.formula,
                              dict(res.parameters), (), res.r2,
                              f"table row 2: profile flags -> transmutation fit on dev",
                              False, "")

    # Row 1 needs step seeds; the table guarantees them, belt-and-braces anyway.
    if seeds is None:
        return GuidedProposal(PATH_ABSTAIN, prior.family, None, (), "", {}, (), float("nan"),
                              "table row 1 without change-points: cannot phase a rule", True,
                              "no estimated thresholds")

    # Row 1: stepped data. Supporting evidence = dev-exact step basis;
    # primary = extrapolatable rule form fitted on the same dev rows by the
    # adapter's own phased grid (uniform path: no dependence on another
    # module's internal gating, which was measured to decline stepped data
    # under noise that the shape prior identifies).
    step_cand, step_formula, step_r2 = _sign_basis_candidate(xs, ys, seeds, feature_name)
    step_params = {"intercept": seeds.intercept,
                   **{f"step_{i}": c for i, c in enumerate(seeds.coefficients[1:])}}
    rule = _fit_floor_rule(xs, ys, seeds.thresholds)
    if rule is None:
        return GuidedProposal(PATH_STEP_BASIS, prior.family, step_cand, (), step_formula,
                              step_params, seeds.thresholds, step_r2,
                              "table row 1, rule form unphasable (<2 thresholds): "
                              "dev-exact basis only, no holdout claim",
                              False, "")
    q0, w, off, b, rule_r2 = rule
    rule_cand = _floor_rule_candidate(xs, q0, w, off, b, rule_r2, feature_name)
    return GuidedProposal(PATH_EXTRAPOLATABLE_RULE, prior.family, rule_cand, (step_cand,),
                          rule_cand.formula,
                          {"quantum_step": q0, "step_width": w, "offset": off, "level": b},
                          seeds.thresholds, rule_r2,
                          f"table row 1: {prior.n_change_points} data-estimated thresholds; "
                          f"dev-exact basis R2={step_r2:.6f}, rule form dev R2={rule_r2:.6f}",
                          False, "")


def score_on_holdout(
    candidate: CandidateProposal,
    x_hold: Sequence[float],
    y_hold: Sequence[float],
    feature_name: str,
) -> dict[str, float]:
    """Score a proposal on holdout rows. No refit. Caller-owned split discipline."""
    xh = np.asarray(x_hold, dtype=float).ravel()
    yh = np.asarray(y_hold, dtype=float).ravel()
    pred = np.asarray(candidate.compute_score(pd.DataFrame({feature_name: xh})),
                      dtype=float).ravel()
    ss_res = float(np.sum((yh - pred) ** 2))
    ss_tot = float(np.sum((yh - yh.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(ss_res / xh.size)) if xh.size else float("nan"),
        "n_holdout": int(xh.size),
    }
