"""Order-Book Resiliency Baseline Hypothesis Proposer.

Implements the classic event-driven exponential recovery baseline:
  ΔP_{t+k} = α + β * Shock_t * exp(-γ * k)
based on established market microstructure literature:
  - Biais, Hillion, Spatt (1995), Journal of Finance
  - Large (2007), Journal of Financial Markets
  - Bouchaud et al. (2009), Quantitative Finance

Serves as the rigorous literature benchmark against which new liquidity alphas
must demonstrate incremental predictive power and structural novelty.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


def _exp_decay(k: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    return beta * np.exp(-gamma * k)


class OrderBookResiliencyProposer(ProposerBase):
    """Proposes event-driven order-book resiliency bounce candidates."""

    def __init__(self, shock_std_threshold: float = 2.0, max_horizon: int = 5):
        self._shock_std_threshold = shock_std_threshold
        self._max_horizon = max_horizon

    @property
    def proposer_id(self) -> str:
        return "order_book_resiliency_biais_large"

    def fit_and_propose(
        self,
        train_data: dict[str, pd.DataFrame],
        target_col: str = "price",
    ) -> Sequence[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        shocks: list[float] = []
        recoveries: list[np.ndarray] = []

        for _, df in train_data.items():
            p = df["price"].to_numpy(dtype=float) if "price" in df.columns else df["close"].to_numpy(dtype=float)
            if len(p) < 50:
                continue
            r = np.zeros(len(p))
            r[1:] = np.log(np.maximum(p[1:], 1e-12) / np.maximum(p[:-1], 1e-12))
            sigma = pd.Series(r).rolling(20, min_periods=5).std().fillna(1e-4).to_numpy()

            # Identify liquidity shock bars: |r_t| > threshold * sigma_t
            shock_mask = np.abs(r) > (self._shock_std_threshold * sigma)
            shock_indices = np.where(shock_mask)[0]

            for idx in shock_indices:
                if idx + self._max_horizon < len(p):
                    # Measure subsequent price displacement trajectory relative to shock bar
                    shock_sign = float(np.sign(r[idx]))
                    disp = [r[idx + k] * (-shock_sign) for k in range(1, self._max_horizon + 1)]
                    shocks.append(float(r[idx]))
                    recoveries.append(np.array(disp))

        if len(recoveries) < 10:
            # Fallback to calibrated textbook decay parameters if sample shocks are sparse
            beta_fit, gamma_fit = 0.35, 0.45
        else:
            rec_mat = np.array(recoveries)
            mean_traj = np.mean(rec_mat, axis=0)
            k_steps = np.arange(1, self._max_horizon + 1, dtype=float)
            try:
                popt, _ = curve_fit(_exp_decay, k_steps, mean_traj, p0=[0.5, 0.5], maxfev=1000)
                beta_fit, gamma_fit = float(popt[0]), max(float(popt[1]), 0.05)
            except Exception:
                beta_fit, gamma_fit = 0.35, 0.45

        formula_str = f"Shock_t * {beta_fit:.4f} * exp(-{gamma_fit:.4f} * k)"

        def _make_resiliency_scorer(th: float, beta: float, gamma: float):
            def _scorer(df: pd.DataFrame) -> np.ndarray:
                p_arr = df["price"].to_numpy(dtype=float) if "price" in df.columns else df["close"].to_numpy(dtype=float)
                ret = np.zeros(len(p_arr))
                ret[1:] = np.log(np.maximum(p_arr[1:], 1e-12) / np.maximum(p_arr[:-1], 1e-12))
                sig = pd.Series(ret).rolling(20, min_periods=5).std().fillna(1e-4).to_numpy()
                is_shock = np.abs(ret) > (th * sig)
                # Signal predicts negative of shock sign (mean reversion bounce)
                signal = np.where(is_shock, -np.sign(ret) * beta * np.exp(-gamma), 0.0)
                return signal
            return _scorer

        proposals.append(
            CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"{self.proposer_id}-decay-b{beta_fit:.2f}-g{gamma_fit:.2f}",
                formula=formula_str,
                strategy_archetype=StrategyArchetype.TAIL_RISK,
                declared_scale_invariant=True,
                compute_score=_make_resiliency_scorer(self._shock_std_threshold, beta_fit, gamma_fit),
                parameters={"beta": beta_fit, "gamma": gamma_fit, "threshold": self._shock_std_threshold},
                metadata={
                    "literature_family": "ORDER_BOOK_RESILIENCY",
                    "citation": "Large (2007) Journal of Financial Markets",
                    "n_shocks_calibrated": len(recoveries),
                },
            )
        )
        return proposals
