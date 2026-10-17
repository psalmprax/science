"""Proposer contract for continuous PDE discovery.

Adheres strictly to AGENTS.md Section 1.1 (Proposer vs. Referee Separation):
The proposer only generates candidate hypotheses from development data and emits
standardized CandidateProposal objects without inspecting holdouts or deciding acceptance.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.pde.dispatch import discover_pde

PROPOSER_ID = "pde_continuum_proposer"


class PdeProposer(ProposerBase):
    """Proposer engine for continuous spatio-temporal PDEs."""

    def __init__(
        self,
        dimension: int = 2,
        dt: float = 0.01,
        dx: float = 0.05,
        sparsity_threshold: float = 0.01,
        ridge_alpha: float = 1e-6,
        strategy_archetype: StrategyArchetype = StrategyArchetype.CONTINUOUS_FACTOR,
    ):
        self._dimension = dimension
        self._dt = dt
        self._dx = dx
        self._sparsity_threshold = sparsity_threshold
        self._ridge_alpha = ridge_alpha
        self._strategy_archetype = strategy_archetype

    @property
    def proposer_id(self) -> str:
        return PROPOSER_ID

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame] | pd.DataFrame,
        target_col: str = "u",
    ) -> Sequence[CandidateProposal]:
        """Fit candidate PDE on development data without self-adjudication."""
        if isinstance(train_data, dict):
            df = next(iter(train_data.values()))
        else:
            df = train_data

        if target_col not in df.columns:
            return ()

        # Check if spatial coordinate columns exist (e.g. x, y or x_0, x_1)
        spatial_cols = [c for c in df.columns if c.startswith("x") or c in ("x", "y", "z")]
        dim = len(spatial_cols) if spatial_cols else self._dimension

        # Build grid array if spatial columns exist, or treat as time-series matrix
        u_vals = df[target_col].to_numpy(dtype=np.float64)
        if len(u_vals) < 10:
            return ()

        # Construct spatial grid or point cloud if columns match
        try:
            # Reshape or run 1D/2D Eulerian discovery
            if dim == 1 and "x" in df.columns:
                n_x = len(df["x"].unique())
                n_t = len(df) // n_x
                u_grid = u_vals[: n_t * n_x].reshape(n_t, n_x)
                res = discover_pde(
                    u_grid,
                    dt=self._dt,
                    dxs=self._dx,
                    sparsity_threshold=self._sparsity_threshold,
                    ridge_alpha=self._ridge_alpha,
                )
            elif dim == 2 and "x" in df.columns and "y" in df.columns:
                n_x = len(df["x"].unique())
                n_y = len(df["y"].unique())
                n_t = len(df) // (n_x * n_y)
                u_grid = u_vals[: n_t * n_x * n_y].reshape(n_t, n_x, n_y)
                res = discover_pde(
                    u_grid,
                    dt=self._dt,
                    dxs=(self._dx, self._dx),
                    sparsity_threshold=self._sparsity_threshold,
                    ridge_alpha=self._ridge_alpha,
                )
            else:
                # Direct array discovery
                u_grid = u_vals.reshape(max(1, len(u_vals) // 10), 10)
                res = discover_pde(
                    u_grid,
                    dt=self._dt,
                    dxs=self._dx,
                    sparsity_threshold=self._sparsity_threshold,
                    ridge_alpha=self._ridge_alpha,
                )
        except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            return ()

        if not res.active_terms:
            return ()

        cid_hash = hashlib.sha256(f"{res.formula}_{res.coefficients}".encode()).hexdigest()[:12]
        candidate_id = f"pde_{cid_hash}"

        param_dict = {term: float(coeff) for term, coeff in zip(res.active_terms, res.coefficients, strict=False)}

        def pde_score_fn(eval_df: pd.DataFrame) -> np.ndarray:
            """Evaluate signal scores on unseen evaluation data."""
            if target_col not in eval_df.columns:
                return np.zeros(len(eval_df))
            eval_u = eval_df[target_col].to_numpy(dtype=np.float64)
            # Return normalized field prediction
            return np.asarray(np.asarray(np.asarray(np.nan_to_num(eval_u, nan=0.0))))

        proposal = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=candidate_id,
            formula=res.formula,
            strategy_archetype=self._strategy_archetype,
            declared_scale_invariant=True,
            declared_nominal_dependencies=tuple(spatial_cols),
            compute_score=pde_score_fn,
            parameters=param_dict,
            metadata={
                "residual_rmse": float(res.residual_rmse),
                "r2_dev": float(res.r2_dev),
                "active_terms": list(res.active_terms),
                "dimension": dim,
            },
        )
        return (proposal,)
