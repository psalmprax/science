"""Distributed Lag & Feature Interaction Proposer.

Generates candidate hypotheses expanding the discovery search space beyond static
pointwise features into:
  1. Distributed lag kernels (exponential memory decay across windows k in {1, 2, 5, 10}).
  2. Pairwise bilinear feature interactions (e.g., Order Flow Imbalance x Relative Spread).
  3. L1/L2 penalized parameter selection (ElasticNet) with cross-validation.

Emits standardized CandidateProposal instances conforming to ProposerBase.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

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

        # Build feature matrix: base features + lagged features + interaction terms.
        # `column_spec` maps each generated column to its recipe so the scorer
        # never has to re-parse names by substring (feature names may themselves
        # contain "_X_" / "_lag").
        feature_matrix: dict[str, np.ndarray] = {}
        column_spec: dict[str, tuple] = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            s = df[col].to_numpy(dtype=float)
            feature_matrix[col] = s
            column_spec[col] = ("base", col)
            # Distributed lags. The first `lag` values are unknowable (there is
            # no earlier observation): they are marked NaN so the finite-row
            # mask below excludes them from the fit, instead of being fabricated
            # as zeros (which taught the model a signal that does not exist and
            # made the scorer disagree with the fit on exactly those rows).
            for lag in self.lags:
                lagged = np.roll(s, lag)
                lagged[:lag] = np.nan
                lag_name = f"{col}_lag{lag}"
                feature_matrix[lag_name] = lagged
                column_spec[lag_name] = ("lag", col, lag)

        # Bilinear interactions
        if self.include_interactions and len(feature_cols) >= 2:
            base_keys = list(feature_cols)
            for i in range(len(base_keys)):
                for j in range(i + 1, len(base_keys)):
                    k1, k2 = base_keys[i], base_keys[j]
                    if k1 in feature_matrix and k2 in feature_matrix:
                        inter_name = f"{k1}_X_{k2}"
                        feature_matrix[inter_name] = feature_matrix[k1] * feature_matrix[k2]
                        column_spec[inter_name] = ("interaction", k1, k2)

        X_df = pd.DataFrame(feature_matrix)
        y = df[target_col].to_numpy(dtype=float)
        mask = np.isfinite(X_df.to_numpy()).all(axis=1) & np.isfinite(y)
        if np.sum(mask) < 50:
            return proposals

        X_clean = X_df.iloc[mask].to_numpy()
        y_clean = y[mask]

        # Standardize features for the ElasticNet fit (preconditioning only).
        x_mean = np.mean(X_clean, axis=0)
        x_std = np.std(X_clean, axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_norm = (X_clean - x_mean) / x_std

        # Cross-validation must respect time ordering: plain KFold trains each
        # fold on rows that come AFTER it, leaking future information into
        # hyperparameter selection. TimeSeriesSplit only ever validates forward.
        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            cv=TimeSeriesSplit(n_splits=self.cv_folds),
            random_state=self.random_seed,
            max_iter=1000,
        )
        model.fit(X_norm, y_clean)

        coefs = model.coef_
        non_zero_idx = np.where(np.abs(coefs) > 1e-4)[0]
        if len(non_zero_idx) == 0:
            return proposals

        selected_names = [X_df.columns[i] for i in non_zero_idx]
        # Map coefficients back to the RAW feature scale so the scorer is
        # exactly the fitted model: y = bias + sum(w_raw * raw_feature).
        raw_weights = {X_df.columns[i]: float(coefs[i] / x_std[i]) for i in non_zero_idx}
        bias = float(model.intercept_ - np.sum(coefs[non_zero_idx] * x_mean[non_zero_idx] / x_std[non_zero_idx]))
        formula_terms = [f"{raw_weights[name]:+.4f} * {name}" for name in selected_names]
        formula_str = f"intercept({bias:.4f}) " + " ".join(formula_terms)

        max_lag = max((spec[2] for spec in column_spec.values() if spec[0] == "lag"), default=0)
        # Causal state carried across calls: a lag term evaluated on a streaming
        # chunk shorter than the lag must see the observations that preceded the
        # chunk, not zeros. `_lag_state` holds the tail of the most recent
        # series; a fresh (non-streaming) frame resets it.
        lag_state: dict[str, Any] = {"buffer": None}

        def compute_score(d: pd.DataFrame) -> np.ndarray:
            sig = np.full(len(d), bias, dtype=float)
            for name, w in raw_weights.items():
                spec = column_spec[name]
                if spec[0] == "interaction":
                    _, p1, p2 = spec
                    if p1 in d.columns and p2 in d.columns:
                        term = d[p1].to_numpy(dtype=float) * d[p2].to_numpy(dtype=float)
                        sig += w * term
                elif spec[0] == "lag":
                    _, base_col, lag_n = spec
                    if base_col in d.columns:
                        s_raw = d[base_col].to_numpy(dtype=float)
                        buffer = lag_state["buffer"]
                        if buffer is not None and buffer[0] == base_col:
                            history = np.concatenate([buffer[1], s_raw])
                        else:
                            history = s_raw
                        # lagged[t] = history[t] shifted by lag_n within the
                        # concatenated series, aligned to the current frame.
                        # lagged[t] = series value lag_n steps before frame
                        # position t. Rows whose predecessor lies before the
                        # available history are unknowable and stay NaN.
                        lagged = np.full(len(s_raw), np.nan)
                        if len(history) >= len(s_raw) + lag_n:
                            lagged = history[len(history) - len(s_raw) - lag_n: len(history) - lag_n]
                        else:
                            available = len(history) - lag_n
                            if available > 0:
                                lagged[len(s_raw) - available:] = history[:available]
                        lag_state["buffer"] = (base_col, s_raw[-max_lag:].copy() if max_lag else s_raw[-1:].copy())
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
            # NOT scale invariant: the fitted model carries a non-zero intercept
            # and raw-scale weights, so f(cX) != c*f(X). Declaring invariance
            # here made referee invariance tests falsify a true claim.
            declared_scale_invariant=False,
            declared_nominal_dependencies=(),
            compute_score=compute_score,
            parameters={"alpha": float(model.alpha_), "n_terms": len(selected_names), "intercept": bias},
            metadata={
                "selected_terms": selected_names,
                "n_lags": len(self.lags),
                "has_interactions": self.include_interactions,
                "fit_standardized": True,
                "coefficients_mapped_to_raw_scale": True,
            },
        )
        proposals.append(cand)
        return proposals
