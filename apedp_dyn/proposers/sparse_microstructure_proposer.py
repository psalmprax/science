"""Sparse Microstructure Hypothesis Proposer.

Generates sparse, regularized linear microstructure alpha candidates using ElasticNet/Lasso
over strictly causal order flow and liquidity features.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


def extract_microstructure_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract causal microstructure features and forward 1-period returns from bar dataframe."""
    p = df["price"].to_numpy(dtype=float) if "price" in df.columns else df["close"].to_numpy(dtype=float)
    v = df["V"].to_numpy(dtype=float) if "V" in df.columns else df["vol"].to_numpy(dtype=float)
    tb = df["taker_buy"].to_numpy(dtype=float) if "taker_buy" in df.columns else v * 0.5

    r = np.zeros(len(p))
    r[1:] = np.log(np.maximum(p[1:], 1e-12) / np.maximum(p[:-1], 1e-12))

    # Causal features
    # 1. 5-period momentum
    mom5 = pd.Series(p).pct_change(5).fillna(0.0).to_numpy()
    # 2. Cumulative volume delta proxy
    vc = np.maximum(v, 1e-12)
    cvd = (2.0 * tb - vc) / vc
    # 3. Normalized order flow imbalance (OFI proxy)
    ofi = np.sign(r) * (v / np.maximum(pd.Series(v).rolling(20, min_periods=1).mean().to_numpy(), 1e-12))
    # 4. Volatility ratio (5-bar vs 20-bar realized std)
    std5 = pd.Series(r).rolling(5, min_periods=2).std().fillna(0.0).to_numpy()
    std20 = pd.Series(r).rolling(20, min_periods=5).std().fillna(1e-4).to_numpy()
    vol_ratio = std5 / np.maximum(std20, 1e-6)

    feats = pd.DataFrame({
        "mom5": mom5,
        "cvd": cvd,
        "ofi": ofi,
        "vol_ratio": vol_ratio,
    })

    # Forward 1-period return target
    fwd_r = np.zeros(len(p))
    fwd_r[:-1] = r[1:]
    return feats, fwd_r


class SparseMicrostructureProposer(ProposerBase):
    """Proposes sparse linear microstructure models via ElasticNet regularization."""

    def __init__(self, l1_ratios: Sequence[float] = (0.5, 0.9), alphas: Sequence[float] = (1e-4, 1e-3, 1e-2)):
        self._l1_ratios = tuple(l1_ratios)
        self._alphas = tuple(alphas)

    @property
    def proposer_id(self) -> str:
        return "sparse_microstructure_elasticnet"

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame],
        target_col: str = "price",
    ) -> Sequence[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        all_X: list[pd.DataFrame] = []
        all_y: list[np.ndarray] = []

        for _, df in train_data.items():
            if len(df) < 30:
                continue
            X, y = extract_microstructure_features(df)
            valid = np.isfinite(y) & np.all(np.isfinite(X.to_numpy()), axis=1)
            all_X.append(X[valid].iloc[25:])
            all_y.append(y[valid][25:])

        if not all_X:
            return proposals

        X_mat = pd.concat(all_X, ignore_index=True)
        y_vec = np.concatenate(all_y)
        feat_names = list(X_mat.columns)

        # Standardize features for balanced L1 shrinkage across disparate scales
        means = X_mat.mean()
        stds = X_mat.std().replace(0.0, 1.0)
        X_norm = (X_mat - means) / stds

        cand_idx = 0
        alphas_to_try = list(self._alphas) + [1e-5]
        for alpha in alphas_to_try:
            for l1_ratio in self._l1_ratios:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=2000, random_state=42)
                model.fit(X_norm.to_numpy(), y_vec)
                # Unstandardize coefficients back to natural feature units
                raw_coefs = model.coef_ / stds.to_numpy()
                coefs = dict(zip(feat_names, raw_coefs, strict=True))
                active_terms = {k: float(v) for k, v in coefs.items() if abs(v) > 1e-6}
                if not active_terms:
                    continue

                formula_parts = [f"{v:+.4f}*{k}" for k, v in active_terms.items()]
                formula_str = " ".join(formula_parts)
                cand_idx += 1

                # Closure for score computation
                def _make_scorer(weights: dict[str, float]):
                    def _score_fn(df: pd.DataFrame) -> np.ndarray:
                        X_sub, _ = extract_microstructure_features(df)
                        score = np.zeros(len(df))
                        for k, w in weights.items():
                            if k in X_sub.columns:
                                score += w * X_sub[k].to_numpy()
                        return score
                    return _score_fn

                proposals.append(
                    CandidateProposal(
                        proposer_id=self.proposer_id,
                        candidate_id=f"{self.proposer_id}-a{alpha}-l1{l1_ratio}-{cand_idx}",
                        formula=formula_str,
                        strategy_archetype=StrategyArchetype.CONTINUOUS_FACTOR,
                        declared_scale_invariant=True,
                        compute_score=_make_scorer(active_terms),
                        parameters=active_terms,
                        metadata={
                            "alpha": alpha,
                            "l1_ratio": l1_ratio,
                            "intercept": float(model.intercept_),
                            "n_active_features": len(active_terms),
                        },
                    )
                )

        return proposals
