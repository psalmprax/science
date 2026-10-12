"""Distributed Lag & Feature Interaction Proposer.

Generates candidate hypotheses expanding the discovery search space beyond static
pointwise features into:
  1. Distributed lag kernels (exponential memory decay across windows k in {1, 2, 5, 10}).
  2. Pairwise bilinear feature interactions (e.g., Order Flow Imbalance x Relative Spread).
  3. L1/L2 penalized parameter selection (ElasticNet) with cross-validation.

Emits standardized CandidateProposal instances conforming to ProposerBase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class DistributedLagInteractionProposer(ProposerBase):
    """Generates dynamic distributed-lag and interaction microstructure hypotheses."""

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 5, 10),
        include_interactions: bool = True,
        cv_folds: int = 3,
        random_seed: int = 42,
    ):
        self.lags = lags
        self.include_interactions = include_interactions
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    @property
    def proposer_id(self) -> str:
        return "distributed_lag_interaction_proposer_v1"

    def fit_and_propose(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        candidate_prefix: str = "H-DLI",
    ) -> list[CandidateProposal]:
        """Fit distributed lag and interaction models and emit CandidateProposals."""
        proposals: list[CandidateProposal] = []
        n = len(df)
        if n < 100 or not feature_cols:
            return proposals

        # Build feature matrix: base features + lagged features + interaction terms
        feature_matrix = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            s = df[col].to_numpy(dtype=float)
            feature_matrix[col] = s
            # Distributed lags
            for lag in self.lags:
                lagged = np.roll(s, lag)
                lagged[:lag] = 0.0
                feature_matrix[f"{col}_lag{lag}"] = lagged

        # Bilinear interactions
        if self.include_interactions and len(feature_cols) >= 2:
            base_keys = list(feature_cols)
            for i in range(len(base_keys)):
                for j in range(i + 1, len(base_keys)):
                    k1, k2 = base_keys[i], base_keys[j]
                    if k1 in feature_matrix and k2 in feature_matrix:
                        feature_matrix[f"{k1}_X_{k2}"] = feature_matrix[k1] * feature_matrix[k2]

        X_df = pd.DataFrame(feature_matrix)
        y = df[target_col].to_numpy(dtype=float)
        mask = np.isfinite(X_df.to_numpy()).all(axis=1) & np.isfinite(y)
        if np.sum(mask) < 50:
            return proposals

        X_clean = X_df.iloc[mask].to_numpy()
        y_clean = y[mask]

        # Standardize features
        x_mean = np.mean(X_clean, axis=0)
        x_std = np.std(X_clean, axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_norm = (X_clean - x_mean) / x_std

        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            cv=self.cv_folds,
            random_state=self.random_seed,
            max_iter=1000,
        )
        model.fit(X_norm, y_clean)

        coefs = model.coef_
        non_zero_idx = np.where(np.abs(coefs) > 1e-4)[0]
        if len(non_zero_idx) == 0:
            return proposals

        selected_names = [X_df.columns[i] for i in non_zero_idx]
        formula_terms = [f"{coefs[i]:+.4f} * {X_df.columns[i]}" for i in non_zero_idx]
        formula_str = f"intercept({model.intercept_:.4f}) " + " ".join(formula_terms)

        weights = {X_df.columns[i]: float(coefs[i]) for i in non_zero_idx}

        def compute_score(d: pd.DataFrame) -> np.ndarray:
            sig = np.full(len(d), model.intercept_, dtype=float)
            for name, w in weights.items():
                if "_X_" in name:
                    p1, p2 = name.split("_X_")
                    if p1 in d.columns and p2 in d.columns:
                        term = d[p1].to_numpy(dtype=float) * d[p2].to_numpy(dtype=float)
                        sig += w * term
                elif "_lag" in name:
                    base_col, lag_s = name.rsplit("_lag", 1)
                    lag_n = int(lag_s)
                    if base_col in d.columns:
                        s_raw = d[base_col].to_numpy(dtype=float)
                        lagged = np.roll(s_raw, lag_n)
                        lagged[:lag_n] = 0.0
                        sig += w * lagged
                else:
                    if name in d.columns:
                        sig += w * d[name].to_numpy(dtype=float)
            return sig

        cand = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"{candidate_prefix}_{len(selected_names)}terms",
            formula=formula_str,
            strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
            declared_scale_invariant=True,
            declared_nominal_dependencies=(),
            compute_score=compute_score,
            parameters={"alpha": float(model.alpha_), "n_terms": len(selected_names)},
            metadata={
                "selected_terms": selected_names,
                "n_lags": len(self.lags),
                "has_interactions": self.include_interactions,
            },
        )
        proposals.append(cand)
        return proposals
