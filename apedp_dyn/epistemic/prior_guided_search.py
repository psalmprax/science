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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.epistemic.autonomous_profiler import DatasetProfile, profile_dataset
from apedp_dyn.epistemic.coordinate_transmuter import TransmutationResult, transmute_and_propose
from apedp_dyn.epistemic.shape_inference import (
    FAMILY_PIECEWISE_CONSTANT,
    ShapePrior,
    StepFamilySeeds,
    staircase_prior,
)
from apedp_dyn.grammar.floor_rules import (
    PATH_EXTRAPOLATABLE_RULE,
    fit_floor_rule,
    floor_rule_candidate,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype

PATH_STEP_BASIS = "step_basis_exact_dev"
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


_fit_floor_rule = fit_floor_rule
_floor_rule_candidate = floor_rule_candidate


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
        for c, t in zip(beta[1:], thr, strict=False):
            out = out + c * np.where(xv >= t, 1.0, -1.0)
        return out

    terms = " + ".join(f"{c:.6g}*(sign({feature_name}-{t:.6g}))" for c, t in zip(beta[1:], thr, strict=False))
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
                              "table row 2: profile flags -> transmutation fit on dev",
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
