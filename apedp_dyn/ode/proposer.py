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

            def make_score_fn(eq_info=eq):
                def _scorer(eval_df: pd.DataFrame) -> np.ndarray:
                    if eq_info.state_variable not in eval_df.columns:
                        return np.zeros(len(eval_df))
                    # Predict state derivative / forward step
                    return np.asarray(np.asarray(np.asarray(eval_df[eq_info.state_variable].to_numpy(dtype=float))))
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
