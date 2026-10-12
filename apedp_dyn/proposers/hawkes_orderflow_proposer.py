"""Hawkes Self-Exciting Process Order-Flow Proposer.

Models event-level order arrival clustering and quote replenishment via
univariate self-exciting point processes:
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

Branching ratio eta = alpha / beta measures system reflexivity / fragility.
Signals trigger on cluster intensity surges representing liquidity depletion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class HawkesOrderFlowProposer(ProposerBase):
    """Generates event-arrival intensity hypotheses using self-exciting Hawkes processes."""

    def __init__(
        self,
        decay_beta: float = 1.0,
        intensity_threshold_quantile: float = 0.90,
    ):
        self.decay_beta = decay_beta
        self.intensity_threshold_quantile = intensity_threshold_quantile

    @property
    def proposer_id(self) -> str:
        return "hawkes_orderflow_proposer_v1"

    def fit_and_propose(
        self,
        event_timestamps: np.ndarray | pd.Series,
        candidate_prefix: str = "H-HWK",
    ) -> list[CandidateProposal]:
        """Fit Hawkes process parameters via maximum likelihood and emit CandidateProposals."""
        proposals: list[CandidateProposal] = []
        t_arr = np.sort(np.asarray(event_timestamps, dtype=float))
        n = len(t_arr)
        if n < 50:
            return proposals

        t_start, t_end = t_arr[0], t_arr[-1]
        T = max(t_end - t_start, 1.0)
        # Shift to start at 0
        t_norm = t_arr - t_start

        # Negative log-likelihood function with fixed beta
        beta = self.decay_beta

        def neg_log_lik(params):
            mu, alpha = params[0], params[1]
            if mu <= 1e-5 or alpha <= 1e-5 or alpha >= beta:
                return 1e9
            # Compute recursive recursive intensity terms R[i] = exp(-beta * dt) * (1 + R[i-1])
            dt = np.diff(t_norm)
            R = np.zeros(n)
            for i in range(1, n):
                R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])

            lam_i = mu + alpha * R
            log_terms = np.sum(np.log(np.maximum(lam_i, 1e-8)))
            # Integrated intensity: mu * T + (alpha / beta) * sum(1 - exp(-beta * (T - t_i)))
            int_term = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t_norm)))
            return float(int_term - log_terms)

        init_mu = float(n / T)
        init_alpha = 0.2 * beta
        res = minimize(
            neg_log_lik,
            x0=[init_mu, init_alpha],
            bounds=[(1e-4, 50.0), (1e-4, beta * 0.95)],
            method="L-BFGS-B",
        )

        mu_hat, alpha_hat = float(res.x[0]), float(res.x[1])
        branching_ratio = float(alpha_hat / beta)

        # Compute empirical intensity series
        R = np.zeros(n)
        dt = np.diff(t_norm)
        for i in range(1, n):
            R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])
        intensities = mu_hat + alpha_hat * R

        tau_intensity = float(np.quantile(intensities, self.intensity_threshold_quantile))
        formula_str = (
            f"hawkes_intensity(mu={mu_hat:.4f}, alpha={alpha_hat:.4f}, beta={beta:.4f}, "
            f"branching={branching_ratio:.3f}) > {tau_intensity:.4f}"
        )

        def compute_score(df: pd.DataFrame) -> np.ndarray:
            # If timestamp column available, compute intensity; otherwise zero
            if "t" in df.columns or "timestamp" in df.columns:
                col = "t" if "t" in df.columns else "timestamp"
                ts = df[col].to_numpy(dtype=float)
                # Simple trailing exponential sum approximation
                sig = np.zeros(len(ts))
                for idx in range(1, len(ts)):
                    delta_t = max(ts[idx] - ts[idx - 1], 1e-6)
                    sig[idx] = np.exp(-beta * delta_t) * (1.0 + sig[idx - 1])
                return (mu_hat + alpha_hat * sig > tau_intensity).astype(float)
            return np.zeros(len(df))

        cand = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"{candidate_prefix}_branching{int(branching_ratio*100)}",
            formula=formula_str,
            strategy_archetype=StrategyArchetype.TAIL_RISK,
            declared_scale_invariant=True,
            declared_nominal_dependencies=(),
            compute_score=compute_score,
            parameters={
                "mu": mu_hat,
                "alpha": alpha_hat,
                "beta": beta,
                "branching_ratio": branching_ratio,
                "tau_intensity": tau_intensity,
            },
            metadata={
                "n_events": n,
                "duration_seconds": float(T),
            },
        )
        proposals.append(cand)
        return proposals
