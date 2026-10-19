"""Candidate Trial Ledger & Deflated Sharpe Ratio (DSR) Accounting.

Implements rigorous multiple-testing control across all competing hypothesis proposers
per Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting, and Non-Normality", Journal of Portfolio Management.

Maintains an immutable trial ledger recording every attempted candidate to prevent
selection-bias-induced false discoveries.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from apedp_dyn.conf.canonical_spec import DSR_THRESHOLD

EULER_MASCHERONI = 0.57721566490153286


@dataclass(frozen=True)
class CandidateTrialRecord:
    """Immutable audit record for one candidate evaluation in the trial family."""
    candidate_id: str
    proposer_id: str
    formula: str
    strategy_archetype: str
    in_sample_sharpe: float
    in_sample_skew: float
    in_sample_kurt: float
    n_bars: int
    status: str
    timestamp: str
    metadata: dict[str, Any]


class CandidateTrialLedger:
    """Immutable audit ledger tracking candidate evaluations across all proposers."""

    def __init__(self, persistence_path: Path | str | None = None):
        self._records: list[CandidateTrialRecord] = []
        self._persistence_path = Path(persistence_path) if persistence_path else None
        if self._persistence_path and self._persistence_path.is_file():
            self._load_from_disk()

    @classmethod
    def from_env(cls, filename: str = "trial_ledger.jsonl") -> CandidateTrialLedger:
        """Initialize trial ledger using persistent storage path configured by notebook/cloud runtime adapter."""
        from apedp_dyn.env import get_ledger_path
        return cls(persistence_path=get_ledger_path(filename))

    def record_trial(
        self,
        candidate_id: str,
        proposer_id: str,
        formula: str,
        strategy_archetype: str,
        in_sample_sharpe: float,
        returns: np.ndarray | None = None,
        in_sample_skew: float = 0.0,
        in_sample_kurt: float = 3.0,
        n_bars: int = 0,
        status: str = "EVALUATED",
        metadata: dict[str, Any] | None = None,
    ) -> CandidateTrialRecord:
        """Record an attempted hypothesis evaluation in the ledger."""
        if returns is not None and len(returns) > 3:
            r = np.asarray(returns, dtype=float)
            r = r[np.isfinite(r)]
            n_bars = len(r)
            if n_bars > 3 and np.std(r) > 1e-9:
                in_sample_skew = float(stats.skew(r))
                in_sample_kurt = float(stats.kurtosis(r, fisher=False))  # Pearson kurtosis (normal=3.0)

        record = CandidateTrialRecord(
            candidate_id=candidate_id,
            proposer_id=proposer_id,
            formula=formula,
            strategy_archetype=strategy_archetype,
            in_sample_sharpe=float(in_sample_sharpe),
            in_sample_skew=float(in_sample_skew),
            in_sample_kurt=max(float(in_sample_kurt), 1.0),
            n_bars=int(n_bars),
            status=status,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._records.append(record)

        if self._persistence_path:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")

        return record

    @property
    def trial_count(self) -> int:
        """Total number of attempted hypotheses evaluated in this trial family."""
        return len(self._records)

    def get_records(self) -> list[CandidateTrialRecord]:
        return list(self._records)

    def get_sharpe_distribution(self) -> tuple[float, float]:
        """Compute (mean, variance) of Sharpe ratios across all tested candidates."""
        if not self._records:
            return 0.0, 0.0
        srs = np.array([r.in_sample_sharpe for r in self._records], dtype=float)
        mean_sr = float(np.mean(srs))
        var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
        return mean_sr, var_sr

    def _load_from_disk(self) -> None:
        assert self._persistence_path is not None
        with open(self._persistence_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self._records.append(CandidateTrialRecord(**d))


def compute_expected_max_sharpe(n_trials: int, var_trials: float, min_var: float = 1e-4) -> float:
    """Expected maximum Sharpe ratio under the null hypothesis of uninformative search.

    E[max_N SR] ≈ σ_trials * ((1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e)))
    using Bailey & López de Prado (2014) closed-form asymptotic approximation.
    """
    if n_trials <= 1:
        return 0.0

    # Ensure trial multiplicity is not bypassed when identical/constant trials are submitted
    effective_var = max(var_trials, min_var) if min_var > 0 else var_trials
    if effective_var <= 1e-12:
        return 0.0

    sigma_sr = np.sqrt(effective_var)
    expected_z = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials) + \
                 EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma_sr * expected_z)


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    var_trials: float,
    n_bars: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 1.0,
) -> dict[str, Any]:
    """Compute Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).

    Args:
        observed_sharpe: Estimated Sharpe ratio of the candidate.
        n_trials: Total number of hypothesis trials attempted across all proposers.
        var_trials: Variance of Sharpe ratios across the trial family.
        n_bars: Sample length (number of periodic returns).
        skewness: Estimated skewness of returns (γ_3).
        kurtosis: Estimated Pearson kurtosis of returns (γ_4, normal=3.0).
        annualization_factor: Factor to de-annualize SR for estimation error math.

    Returns:
        dict containing:
          - dsr: Deflated Sharpe Ratio probability ∈ [0, 1]
          - p_value: Probability that observed SR is explained by search multiplicity
          - expected_max_sr: Threshold hurdle accounting for trial multiplicity
          - se_sharpe: Standard error of the Sharpe ratio under non-normality
          - is_significant: True if DSR >= 0.95 (p < 0.05)
    """
    if n_bars <= 2:
        return {
            "dsr": 0.0,
            "p_value": 1.0,
            "expected_max_sr": 0.0,
            "se_sharpe": 1.0,
            "is_significant": False,
        }

    # Standardize to per-period units for standard error derivation
    sr_period = observed_sharpe / np.sqrt(annualization_factor) if annualization_factor > 0 else observed_sharpe

    # Asymptotic standard error of Sharpe estimator under Mertens (2002) non-normality
    kurt = max(kurtosis, 1.0)
    denom_var = (1.0 - skewness * sr_period + (kurt - 1.0) / 4.0 * (sr_period ** 2))
    denom_var = max(denom_var, 1e-8)
    se_sr_period = np.sqrt(denom_var / (n_bars - 1.0))
    se_sr_annual = float(se_sr_period * np.sqrt(annualization_factor))

    # Expected max SR under null
    exp_max_sr = compute_expected_max_sharpe(n_trials, var_trials)

    # Deflated Sharpe test statistic
    z_stat = (observed_sharpe - exp_max_sr) / max(se_sr_annual, 1e-8)
    dsr = float(stats.norm.cdf(z_stat))
    p_val = float(1.0 - dsr)

    return {
        "dsr": dsr,
        "p_value": p_val,
        "expected_max_sr": exp_max_sr,
        "se_sharpe": se_sr_annual,
        "is_significant": bool(dsr >= DSR_THRESHOLD),
    }
