"""Hawkes Self-Exciting Process Order-Flow Proposer.

Models event-level order arrival clustering and quote replenishment via
univariate self-exciting point processes:
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

Branching ratio eta = alpha / beta measures system reflexivity / fragility.
Signals trigger on cluster intensity surges representing liquidity depletion.

Archetype contract: the candidate declares TAIL_RISK (binary surge) only when
the fitted threshold binds on training data (both classes present). A
degenerate threshold means the emitted signal is a continuous standardized
intensity and the candidate is declared CONTINUOUS_FACTOR. If an evaluation
frame lacks the timestamp/volume features used in fitting, the binary surge
can still collapse; the scorer then falls back to the continuous standardized
intensity on that frame and metadata records the score mode.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


def _extract_timestamps(data: Any) -> np.ndarray:
    if isinstance(data, dict):
        all_ts = []
        for v in data.values():
            if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)):
                ts = _extract_timestamps(v)
                if len(ts) > 0:
                    all_ts.append(ts)
        return np.sort(np.concatenate(all_ts)) if all_ts else np.array([], dtype=float)
    if isinstance(data, pd.DataFrame):
        for col in ("timestamp", "t", "time"):
            if col in data.columns:
                return np.sort(np.asarray(data[col].to_numpy(dtype=float)))
        if isinstance(data.index, pd.DatetimeIndex):
            return np.sort(data.index.astype(np.int64).to_numpy() / 1e9)
        return np.sort(np.asarray(data.index.to_numpy(dtype=float)))
    if isinstance(data, pd.Series):
        return np.sort(np.asarray(data.to_numpy(dtype=float)))
    if isinstance(data, (np.ndarray, list, tuple)):
        return np.sort(np.asarray(data, dtype=float))
    return np.array([], dtype=float)


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
        event_data: dict[str, pd.DataFrame] | pd.DataFrame | np.ndarray | pd.Series | None = None,
        candidate_prefix: str = "H-HWK",
        event_timestamps: np.ndarray | pd.Series | None = None,
        **kwargs: Any,
    ) -> list[CandidateProposal]:
        """Fit Hawkes process parameters via maximum likelihood and emit CandidateProposals."""
        proposals: list[CandidateProposal] = []
        input_data = event_data if event_data is not None else event_timestamps
        if input_data is None:
            return proposals

        t_arr = _extract_timestamps(input_data)
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
            # Compute recursive intensity terms R[i] = exp(-beta * dt) * (1 + R[i-1])
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
        # Finding 10 fix: the standardized-intensity scorer must not use the
        # evaluated frame's own mean/std — over a holdout that folds the whole
        # future distribution into every t. The scoring standardization couples
        # only to the FIT-time intensity distribution (the same constants the
        # surge threshold quantile already uses), so scoring stays causal.
        fit_intens_mean = float(np.mean(intensities))
        fit_intens_std = float(np.std(intensities))

        fit_mean_v: float | None = None
        fit_mean_dp: float | None = None
        if isinstance(input_data, pd.DataFrame):
            for vcol in ("volume", "vol", "size", "trades", "count"):
                if vcol in input_data.columns:
                    rv = np.nan_to_num(input_data[vcol].to_numpy(dtype=float), nan=0.0)
                    mv = float(np.mean(rv))
                    if mv > 1e-12:
                        fit_mean_v = mv
                    break
            for pcol in ("close", "price"):
                if pcol in input_data.columns:
                    p = input_data[pcol].to_numpy(dtype=float)
                    dp = np.diff(p, prepend=p[0])
                    mdp = float(np.mean(np.abs(dp)))
                    if mdp > 1e-12:
                        fit_mean_dp = mdp
                    break

        tau_intensity = float(np.quantile(intensities, self.intensity_threshold_quantile))
        # Archetype contract: a binary rare-event detector (TAIL_RISK) is only
        # declared when the fitted surge actually binds on the training data
        # (both classes present). A degenerate threshold means the emitted
        # signal is a continuous standardized intensity, which must be declared
        # as CONTINUOUS_FACTOR so the ledger does not mis-adjudicate it under
        # tail-risk exposure rules.
        binary_viable = bool(np.any(intensities > tau_intensity) and not np.all(intensities > tau_intensity))
        archetype = StrategyArchetype.TAIL_RISK if binary_viable else StrategyArchetype.CONTINUOUS_FACTOR
        score_mode = "binary_tail_with_continuous_fallback" if binary_viable else "continuous_standardized_intensity"
        formula_str = (
            f"hawkes_intensity(mu={mu_hat:.4f}, alpha={alpha_hat:.4f}, beta={beta:.4f}, "
            f"branching={branching_ratio:.3f}) > {tau_intensity:.4f}"
        )

        def compute_score(df: pd.DataFrame) -> np.ndarray:
            n_samples = len(df)
            if n_samples == 0:
                return np.zeros(0, dtype=float)

            # 1. Identify arrival weights / order-flow volume if available
            # Normalized against fit-time distribution or causal expanding mean (no future leak)
            weights = np.ones(n_samples, dtype=float)
            has_vol = False
            for vcol in ("volume", "vol", "size", "trades", "count"):
                if vcol in df.columns:
                    raw_v = np.nan_to_num(df[vcol].to_numpy(dtype=float), nan=0.0)
                    if fit_mean_v is not None and fit_mean_v > 1e-12:
                        weights = raw_v / fit_mean_v
                        has_vol = True
                    else:
                        cum_v = np.cumsum(raw_v)
                        exp_v = cum_v / np.arange(1, n_samples + 1)
                        weights = np.where(exp_v > 1e-12, raw_v / np.maximum(exp_v, 1e-12), 1.0)
                        has_vol = True
                    break

            # If no volume, check for price activity
            if not has_vol:
                for pcol in ("close", "price"):
                    if pcol in df.columns:
                        p = df[pcol].to_numpy(dtype=float)
                        dp = np.diff(p, prepend=p[0])
                        abs_dp = np.abs(dp)
                        if fit_mean_dp is not None and fit_mean_dp > 1e-12:
                            weights = abs_dp / fit_mean_dp
                            has_vol = True
                        else:
                            cum_dp = np.cumsum(abs_dp)
                            exp_dp = cum_dp / np.arange(1, n_samples + 1)
                            weights = np.where(exp_dp > 1e-12, abs_dp / np.maximum(exp_dp, 1e-12), 1.0)
                            has_vol = True
                        break

            # 2. Extract timestamps or default to unit time steps
            ts = None
            for col in ("timestamp", "t", "time"):
                if col in df.columns:
                    ts = df[col].to_numpy(dtype=float)
                    break
            if ts is None and isinstance(df.index, pd.DatetimeIndex):
                ts = df.index.astype(np.int64).to_numpy() / 1e9

            sig = np.zeros(n_samples, dtype=float)
            if ts is not None and len(ts) == n_samples:
                for idx in range(1, n_samples):
                    delta_t = max(ts[idx] - ts[idx - 1], 1e-6)
                    sig[idx] = np.exp(-beta * delta_t) * (weights[idx - 1] + sig[idx - 1])
            else:
                delta_t = 1.0
                for idx in range(1, n_samples):
                    sig[idx] = np.exp(-beta * delta_t) * (weights[idx - 1] + sig[idx - 1])

            intens = mu_hat + alpha_hat * sig
            # Degenerate fit (no binding threshold on training data): emit the
            # continuous standardized intensity, never a fake binary vector.
            # Standardized against the fit-coupling constants, not eval-frame
            # statistics (Finding 10: eval-frame mean/std is full-future leak).
            if not binary_viable:
                if fit_intens_std > 1e-8:
                    return (intens - fit_intens_mean) / fit_intens_std
                return intens - fit_intens_mean
            surge = (intens > tau_intensity).astype(float)
            # FLAW-09: an eval-frame collapse (e.g. a low-volatility interval
            # where the surge never binds) used to switch the scorer's return
            # type to an unbounded continuous z-score — mutating the declared
            # binary-regime contract {0, 1} and corrupting downstream sizing.
            # The honest binary output in a collapsed regime is the collapsed
            # binary itself (all zeros: no surges fired), so the return TYPE
            # stays constant. The fit-time continuous declaration is the
            # separate, deliberate path (binary_viable=False → the candidate
            # is declared CONTINUOUS_FACTOR above) and is untouched.
            return surge

        cand = CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"{candidate_prefix}_branching{int(branching_ratio*100)}",
            formula=formula_str,
            strategy_archetype=archetype,
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
                "score_mode": score_mode,
                "binary_surge_viable_on_fit": binary_viable,
            },
        )
        proposals.append(cand)
        return proposals
