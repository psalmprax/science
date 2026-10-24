"""Regime-Switching / Threshold Microstructure Proposer.

Generates candidate hypotheses that adapt to volatility or liquidity regimes
using piecewise linear threshold models:
    y_t = beta_{low} * x_t  if vol_t <= tau
    y_t = beta_{high} * x_t if vol_t > tau

Emits standardized CandidateProposal instances conforming to ProposerBase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class RegimeSwitchingProposer(ProposerBase):
    """Generates regime-dependent piecewise linear candidate hypotheses."""

    def __init__(
        self,
        vol_window: int = 20,
        regime_threshold_quantile: float = 0.50,
        random_seed: int = 42,
    ):
        self.vol_window = vol_window
        self.threshold_quantile = regime_threshold_quantile
        self.random_seed = random_seed

    @property
    def proposer_id(self) -> str:
        return "regime_switching_proposer_v1"

    def fit_and_propose(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        vol_col: str | None = None,
        candidate_prefix: str = "H-RSW",
    ) -> list[CandidateProposal]:
        """Fit piecewise regime models and emit CandidateProposals."""
        proposals: list[CandidateProposal] = []
        n = len(df)
        if n < 100 or not feature_cols:
            return proposals

        # Compute volatility indicator for regime segmentation
        if vol_col and vol_col in df.columns:
            regime_metric = df[vol_col].to_numpy(dtype=float)
        else:
            # Trailing standard deviation of first feature. No backfill: early
            # rows have no observable volatility yet and are excluded from both
            # regime fits rather than assigned a fabricated one (bfill would
            # leak future information into the warmup rows).
            s0 = df[feature_cols[0]].to_numpy(dtype=float)
            regime_metric = pd.Series(s0).rolling(self.vol_window, min_periods=5).std().to_numpy()

        metric_ok = np.isfinite(regime_metric)
        tau = float(np.nanquantile(regime_metric, self.threshold_quantile))
        regime_high = regime_metric > tau

        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)

        mask_low = metric_ok & (~regime_high) & np.isfinite(y) & np.isfinite(X).all(axis=1)
        mask_high = metric_ok & regime_high & np.isfinite(y) & np.isfinite(X).all(axis=1)

        if np.sum(mask_low) < 30 or np.sum(mask_high) < 30:
            return proposals

        # Fit OLS in each regime with ridge regularization
        def fit_ridge(X_sub, y_sub, lam=1.0):
            XtX = X_sub.T @ X_sub + lam * np.eye(X_sub.shape[1])
            return np.linalg.solve(XtX, X_sub.T @ y_sub)

        beta_low = fit_ridge(X[mask_low], y[mask_low])
        beta_high = fit_ridge(X[mask_high], y[mask_high])

        formula_str = (
            f"IF vol > {tau:.4f} THEN "
            + " + ".join(f"{b:+.4f}*{c}" for b, c in zip(beta_high, feature_cols, strict=True))
            + " ELSE "
            + " + ".join(f"{b:+.4f}*{c}" for b, c in zip(beta_low, feature_cols, strict=True))
        )

        def compute_score(d: pd.DataFrame) -> np.ndarray:
            if not all(c in d.columns for c in feature_cols):
                return np.zeros(len(d))
            X_eval = d[feature_cols].to_numpy(dtype=float)
            if vol_col and vol_col in d.columns:
                v_eval = d[vol_col].to_numpy(dtype=float)
            else:
                s_eval = d[feature_cols[0]].to_numpy(dtype=float)
                v_eval = pd.Series(s_eval).rolling(self.vol_window, min_periods=5).std().to_numpy()

            is_high = v_eval > tau
            sig = np.zeros(len(d))
            sig[is_high] = X_eval[is_high] @ beta_high
            sig[~is_high] = X_eval[~is_high] @ beta_low
            # Rows without an observable volatility indicator (insufficient
            # warmup history or non-finite vol input) cannot be assigned a
            # regime honestly: score them NaN so callers mask them out.
            sig[~np.isfinite(v_eval)] = np.nan
            return sig

        cand = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"{candidate_prefix}_vol_q{int(self.threshold_quantile*100)}",
            formula=formula_str,
            strategy_archetype=StrategyArchetype.TACTICAL,
            # NOT scale invariant: the regime split branches on a volatility
            # threshold tau estimated on the training scale. Rescaling the input
            # rescales volatility and moves observations across tau, swapping
            # which branch's parameters apply -- f(cX) != c*f(X).
            declared_scale_invariant=False,
            declared_nominal_dependencies=(),
            compute_score=compute_score,
            parameters={
                "threshold_tau": tau,
                "vol_window": self.vol_window,
            },
            metadata={
                "n_features": len(feature_cols),
                "samples_low": int(np.sum(mask_low)),
                "samples_high": int(np.sum(mask_high)),
                "samples_metric_missing": int(np.sum(~metric_ok)),
            },
        )
        proposals.append(cand)
        return proposals
