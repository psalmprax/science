"""Proposer contract for continuous ODE and coupled dynamical system discovery.

Adheres strictly to AGENTS.md Section 1.1:
Generates candidate dynamical hypotheses from training data without inspecting holdouts.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.ode.dispatch import discover_coupled_ode

PROPOSER_ID = "coupled_ode_proposer"


class CoupledOdeProposer(ProposerBase):
    """Proposer engine for coupled multi-variable non-linear ODE systems."""

    def __init__(
        self,
        dt: float = 0.05,
        max_polynomial_degree: int = 2,
        include_cross_interactions: bool = True,
        sparsity_threshold: float = 0.02,
        strategy_archetype: StrategyArchetype = StrategyArchetype.CONTINUOUS_FACTOR,
    ):
        self._dt = dt
        self._max_polynomial_degree = max_polynomial_degree
        self._include_cross_interactions = include_cross_interactions
        self._sparsity_threshold = sparsity_threshold
        self._strategy_archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return PROPOSER_ID

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "x0",
    ) -> Sequence[CandidateProposal]:
        """Fit candidate coupled dynamical system on development data."""
        if isinstance(train_data, dict):
            df = next(iter(train_data.values()))
        else:
            df = train_data

        state_cols = [c for c in df.columns if not c.startswith("_") and c != "t"]
        if len(state_cols) < 2:
            return ()

        series_dict = {col: df[col].to_numpy(dtype=float) for col in state_cols}
        try:
            res = discover_coupled_ode(
                series_dict,
                dt=self._dt,
                max_polynomial_degree=self._max_polynomial_degree,
                include_cross_interactions=self._include_cross_interactions,
                sparsity_threshold=self._sparsity_threshold,
            )
        except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            return ()

        candidates: list[CandidateProposal] = []
        for eq in res.equations:
            cid_hash = hashlib.sha256(f"{eq.state_variable}_{eq.formula}".encode()).hexdigest()[:12]
            param_dict = {term: float(coeff) for term, coeff in zip(eq.active_terms, eq.coefficients, strict=False)}

            # FLAW-02 remediation: the score function must EVALUATE the
            # discovered right-hand side, not echo the input state column.
            # Term encoding mirrors the kernel library built in
            # rust/apedp-kernel/src/rime/coupled_system.rs: "1" (constant),
            # "x_i" (linear), "x_i^2" / "x_i*x_j" (degree-2 squares and
            # cross-interactions), "x_i^3" (degree-3 self-powers).
            active = [(term, float(coeff)) for term, coeff in
                      zip(eq.active_terms, eq.coefficients, strict=False)]

            def _term_value(term: str, row: dict[str, float]) -> float:
                if term == "1":
                    return 1.0
                if "*" in term:
                    left, right = term.split("*", 1)
                    return row[left] * row[right]
                if term.endswith("^2"):
                    return row[term[:-2]] ** 2
                if term.endswith("^3"):
                    return row[term[:-2]] ** 3
                return row[term]

            def make_score_fn(eq_info=eq, active_terms=active):
                def _scorer(eval_df: pd.DataFrame) -> np.ndarray:
                    if eq_info.state_variable not in eval_df.columns:
                        return np.zeros(len(eval_df))
                    missing = [v for term, _ in active_terms
                               for v in [(term=="1" and "1") or
                                         (term[:-2] if (term.endswith("^2") or term.endswith("^3"))
                                          else (term.split("*", 1)[0] if "*" in term else term))]
                               if v != "1" and v not in eval_df.columns]
                    if missing:
                        raise ValueError(
                            f"cannot evaluate discovered term(s): state column(s) "
                            f"{sorted(set(missing))} absent from eval frame"
                        )
                    rows = eval_df.to_dict(orient="records")
                    out = np.zeros(len(eval_df), dtype=float)
                    for k, row in enumerate(rows):
                        acc = 0.0
                        for term, coeff in active_terms:
                            try:
                                acc += coeff * _term_value(term, row)
                            except KeyError as exc:  # never present at this point; belt-and-braces for exotic frames
                                raise ValueError(
                                    f"discovered term '{term}' references absent column {exc}"
                                ) from exc
                        out[k] = acc
                    # Predict state derivative / forward step (the discovered
                    # d{x_i}/dt = sum(coeff * basis(x)), not x_i itself).
                    return np.asarray(out)
                return _scorer

            proposal = CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"coupled_ode_{eq.state_variable}_{cid_hash}",
                formula=eq.formula,
                strategy_archetype=self._strategy_archetype,
                declared_scale_invariant=True,
                declared_nominal_dependencies=tuple(state_cols),
                compute_score=make_score_fn(eq),
                parameters=param_dict,
                metadata={
                    "state_variable": eq.state_variable,
                    "r2": float(eq.r2),
                    "residual_rmse": float(eq.residual_rmse),
                    "mean_system_r2": float(res.mean_system_r2),
                },
            )
            candidates.append(proposal)

        return tuple(candidates)
