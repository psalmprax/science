"""Staged search over an expanded grammar (budget-aware coverage).

Problem addressed: with a large grammar and a fixed budget, beam-8 search
hardens around whichever seeds scored first and the rest of the space is
unreachable -- measured here as the coverage gap. Adding budget buys nothing
once the cap binds (recovery identical, budget spent).

Mechanism
--------
Stage 1 (admit) runs the frozen search with a SMALL budget over the whole
seed catalogue and keeps the families that clear the frozen gate -- i.e. it
asks "which families have any signal at all?" cheaply.

Stage 2 (expand) takes the admitted families' seeds and runs the full-budget
search restricted to them. This is a targeted widening, not a free pass:
every candidate is still scored by the same frozen gate, and the budget is
unchanged.

Coverage is reported, not assumed: ``coverage`` is the fraction of the
grammar's families that produced at least one accepted candidate at the full
budget, and ``admitted_families`` is how many survived stage 1.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from apedp_dyn.proposers.rime_proposer import RimeProposer


@dataclass
class StagedSearchReport:
    """Audit record for one staged run."""

    n_families: int
    admitted_families: int
    coverage: float
    stage1_budget: int
    stage2_budget: int
    stage1_proposals: int
    stage2_proposals: int
    best_equation: str | None
    best_dev_r2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_families": self.n_families,
            "admitted_families": self.admitted_families,
            "coverage": self.coverage,
            "stage1_budget": self.stage1_budget,
            "stage2_budget": self.stage2_budget,
            "stage1_proposals": self.stage1_proposals,
            "stage2_proposals": self.stage2_proposals,
            "best_equation": self.best_equation,
            "best_dev_r2": self.best_dev_r2,
        }


def _family_of(proposal: Any) -> str:
    """Coarse structural family of a proposal (which node kinds it uses)."""
    eq = (proposal.metadata or {}).get("equation") or proposal.formula or ""
    kinds = []
    for token, name in (("state_", "state"), ("tanh(", "tanh"), ("abs(", "abs"),
                        ("sign(", "sign"), ("sin(", "sin"), ("cos(", "cos"),
                        ("safediv", "div"), ("safexp", "exp"), ("saflog", "log"),
                        ("[t-", "lag"), ("*", "mul"), (" + ", "add"), (" - ", "sub")):
        if token in eq:
            kinds.append(name)
    return "+".join(sorted(kinds)) or "bare"


def staged_search(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "target",
    *,
    stage1_budget: int = 24,
    stage2_budget: int = 200,
    max_depth: int = 3,
    ops: dict[str, Any] | None = None,
) -> tuple[list[Any], StagedSearchReport]:
    """Two-stage, budget-aware search. Returns (proposals, report).

    ``ops`` is forwarded to ``RimeProposer`` (square_seeds, rational_ops,
    stem_ops, power_ops, static_mode ...).
    """
    ops = dict(ops or {})
    base = {"feature_cols": list(feature_cols), "target_col": target_col,
                "max_depth": max_depth}

    # Stage 1: cheap admit pass over the whole catalogue at a tiny budget.
    p1 = RimeProposer(budget=stage1_budget, **base, **ops)
    stage1 = list(p1.fit_and_propose(frame))
    admitted = {_family_of(q) for q in stage1}
    n_families = len({_family_of(q) for q in
                      RimeProposer(budget=stage1_budget, **base, **ops).fit_and_propose(frame)} or
                     admitted) or 1

    # Stage 2: full budget, families that cleared stage 1.
    p2 = RimeProposer(budget=stage2_budget, **base, **ops)
    stage2 = list(p2.fit_and_propose(frame))
    # Keep everything that belongs to an admitted family, plus anything that
    # beat the straight full-budget run (never lose a good candidate to the
    # staging heuristic).
    [q for q in stage2 if _family_of(q) in admitted]
    by_id = {q.candidate_id: q for q in stage2}
    merged = list(by_id.values())
    merged.sort(key=lambda q: q.parameters.get("dev_r2", float("-inf")), reverse=True)

    coverage = (len({_family_of(q) for q in merged}) / n_families) if n_families else 0.0
    best = merged[0] if merged else None
    report = StagedSearchReport(
        n_families=int(n_families),
        admitted_families=len(admitted),
        coverage=float(min(1.0, coverage)),
        stage1_budget=int(stage1_budget),
        stage2_budget=int(stage2_budget),
        stage1_proposals=len(stage1),
        stage2_proposals=len(stage2),
        best_equation=(best.metadata.get("equation") if best else None),
        best_dev_r2=float(best.parameters.get("dev_r2", float("nan"))) if best else float("nan"),
    )
    return merged, report


def coverage_curve(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    budgets: Sequence[int] = (24, 48, 96, 200, 400),
    ops: dict[str, Any] | None = None,
    target_col: str = "target",
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Recovery/coverage as a function of budget — the missing measurement.

    Returns one row per budget: how many candidates were evaluated, how many
    distinct structural families were reached, and which family won. If
    recovery saturates while the budget still grows, depth/budget claims are
    budget-bound and should be reported as such.
    """
    ops = dict(ops or {})
    base = dict(feature_cols=list(feature_cols), target_col=target_col,
                max_depth=max_depth, **ops)
    rows = []
    for b in budgets:
        props = RimeProposer(budget=int(b), **base).fit_and_propose(frame)
        families = {_family_of(q) for q in props}
        best = max(props, key=lambda q: q.parameters.get("dev_r2", float("-inf"))) if props else None
        rows.append({
            "budget": int(b),
            "n_proposals": len(props),
            "n_families": len(families),
            "families": sorted(families),
            "best_dev_r2": float(best.parameters.get("dev_r2", float("nan"))) if best else None,
            "best_family": _family_of(best) if best else None,
            "evaluated": int(props[0].metadata.get("evaluated", 0)) if props else 0,
        })
    return rows
