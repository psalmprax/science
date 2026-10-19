"""Decoupled Predictive Incrementality and Structural Novelty Adjudication.

Decouples empirical predictive power from structural/literature novelty:
  1. PREDICTIVE_INCREMENTALITY: Tests out-of-sample loss reduction against incumbent
     baselines using Diebold-Mariano tests with HAC standard errors (p < 0.01).
  2. STRUCTURAL_NOVELTY: Tests functional orthogonality against a canonical library
     of established academic market microstructure and asset pricing literature.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.conf.canonical_spec import (
    DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
    MAX_LITERATURE_R2,
)


@dataclass(frozen=True)
class PredictiveIncrementalityResult:
    verdict: str  # "PREDICTIVELY_INCREMENTAL" | "INCREMENTALITY_UNPROVEN"
    loss_reduction: float  # Mean loss(base) - loss(cand)
    dm_statistic: float  # Diebold-Mariano test statistic
    p_value: float  # Asymptotic p-value under HAC variance
    is_significant: bool  # True if p_value < alpha and loss_reduction > 0
    alpha: float
    details: dict[str, Any]


@dataclass(frozen=True)
class StructuralNoveltyResult:
    verdict: str  # "STRUCTURALLY_NOVEL" | "REPRESENTATION_REIDENTIFICATION"
    max_projection_r2: float  # R^2 against closest literature baseline
    closest_literature_baseline: str  # Identifier of collinear baseline
    literature_citation: str
    orthogonal_variance_fraction: float  # 1 - max_r2
    novelty_threshold_r2: float  # Ceiling R^2 (default 0.85)
    details: dict[str, Any]


def compute_hac_variance(d: np.ndarray, max_lags: int | None = None) -> float:
    """Compute Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) variance."""
    n = len(d)
    if n <= 1:
        return 1.0
    if max_lags is None:
        max_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lags = max(min(max_lags, n - 1), 0)

    d_dem = d - np.mean(d)
    gamma_0 = float(np.mean(d_dem ** 2))
    var_hac = gamma_0

    for lag in range(1, max_lags + 1):
        weight = 1.0 - (lag / (max_lags + 1.0))
        gamma_k = float(np.mean(d_dem[lag:] * d_dem[:-lag]))
        var_hac += 2.0 * weight * gamma_k

    return max(float(var_hac), 1e-12)


def evaluate_predictive_incrementality(
    y_true: np.ndarray,
    y_candidate_pred: np.ndarray,
    y_incumbent_pred: np.ndarray,
    loss_fn: str = "mse",
    alpha: float = DIEBOLD_MARIANO_P_VALUE_THRESHOLD,
) -> PredictiveIncrementalityResult:
    """Perform out-of-sample Diebold-Mariano test of forecast accuracy improvement."""
    y = np.asarray(y_true, dtype=float)
    y_cand = np.asarray(y_candidate_pred, dtype=float)
    y_base = np.asarray(y_incumbent_pred, dtype=float)

    mask = np.isfinite(y) & np.isfinite(y_cand) & np.isfinite(y_base)
    y, y_cand, y_base = y[mask], y_cand[mask], y_base[mask]
    n = len(y)

    if n < 20:
        return PredictiveIncrementalityResult(
            verdict="INCREMENTALITY_UNPROVEN",
            loss_reduction=0.0,
            dm_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
            details={"error": f"Sample size n={n} < 20 insufficient for DM test."},
        )

    if loss_fn == "mae":
        loss_base = np.abs(y - y_base)
        loss_cand = np.abs(y - y_cand)
    else:  # default mse
        loss_base = (y - y_base) ** 2
        loss_cand = (y - y_cand) ** 2

    d = loss_base - loss_cand
    mean_d = float(np.mean(d))
    hac_var = compute_hac_variance(d)
    se_d = np.sqrt(hac_var / n)

    dm_stat = float(mean_d / max(se_d, 1e-12))
    # One-sided p-value for H_0: mean_d <= 0 vs H_1: mean_d > 0
    from scipy.stats import norm
    p_val = float(1.0 - norm.cdf(dm_stat))
    is_sig = bool(mean_d > 0 and p_val < alpha)
    verdict = "PREDICTIVELY_INCREMENTAL" if is_sig else "INCREMENTALITY_UNPROVEN"

    return PredictiveIncrementalityResult(
        verdict=verdict,
        loss_reduction=mean_d,
        dm_statistic=dm_stat,
        p_value=p_val,
        is_significant=is_sig,
        alpha=alpha,
        details={
            "n_obs": n,
            "mean_loss_base": float(np.mean(loss_base)),
            "mean_loss_candidate": float(np.mean(loss_cand)),
            "hac_se": se_d,
        },
    )


# Academic Literature Baseline Reference Registry
CANONICAL_LITERATURE_REGISTRY: dict[str, dict[str, str]] = {
    "KYLE_LAMBDA": {
        "name": "Kyle's Lambda (Linear Order Flow Impact)",
        "citation": "Kyle, A. S. (1985). Continuous Auctions and Insider Trading. Econometrica, 53(6), 1315-1335.",
    },
    "AMIHUD_ILLIQUIDITY": {
        "name": "Amihud Illiquidity Measure",
        "citation": "Amihud, Y. (2002). Illiquidity and stock returns. Journal of Financial Markets, 5(1), 31-56.",
    },
    "ORDER_BOOK_RESILIENCY": {
        "name": "Biais-Large-Bouchaud Order Book Resiliency (Exponential Recovery)",
        "citation": "Biais et al. (1995); Large (2007). Measuring the resiliency of an electronic limit order book. JFM.",
    },
    "CARHART_MOMENTUM": {
        "name": "Carhart 12-Month Momentum Factor (UMD)",
        "citation": "Carhart, M. M. (1997). On Persistence in Mutual Fund Performance. The Journal of Finance, 52(1), 57-82.",
    },
    "SHORT_TERM_REVERSAL": {
        "name": "Jegadeesh Short-Term Weekly Reversal",
        "citation": "Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns. The Journal of Finance, 45(3), 881-898.",
    },
}


def evaluate_structural_novelty(
    candidate_signal: np.ndarray,
    literature_features: dict[str, np.ndarray],
    novelty_threshold_r2: float = MAX_LITERATURE_R2,
) -> StructuralNoveltyResult:
    """Evaluate structural novelty against literature baselines via linear projection."""
    s = np.asarray(candidate_signal, dtype=float)
    max_r2 = 0.0
    closest_baseline = "NONE"
    citation = "NONE"
    fit_details = {}

    valid_cols = []
    col_names = []
    for name, feat in literature_features.items():
        f = np.asarray(feat, dtype=float)
        mask = np.isfinite(s) & np.isfinite(f)
        if np.sum(mask) < 20 or np.std(s[mask]) < 1e-9 or np.std(f[mask]) < 1e-9:
            continue
        corr = float(np.corrcoef(s[mask], f[mask])[0, 1])
        r2 = corr ** 2
        fit_details[name] = {"r2": r2, "corr": corr}
        if r2 > max_r2:
            max_r2 = r2
            closest_baseline = name
            citation = CANONICAL_LITERATURE_REGISTRY.get(name, {}).get("citation", "Standard literature baseline.")
        valid_cols.append(f)
        col_names.append(name)

    # Joint linear span projection: candidate onto joint literature matrix [F_1, ..., F_m, 1]
    if len(valid_cols) > 1:
        joint_mask = np.isfinite(s)
        for col in valid_cols:
            joint_mask &= np.isfinite(col)
        if int(np.sum(joint_mask)) > len(valid_cols) + 5:
            s_j = s[joint_mask]
            var_s = float(np.var(s_j))
            if var_s > 1e-12:
                A = np.column_stack([col[joint_mask] for col in valid_cols] + [np.ones(len(s_j))])
                coef, *_ = np.linalg.lstsq(A, s_j, rcond=None)
                pred_j = A @ coef
                joint_r2 = float(max(0.0, 1.0 - np.mean((s_j - pred_j) ** 2) / var_s))
                fit_details["JOINT_LITERATURE_SPAN"] = {"r2": joint_r2}
                if max_r2 < novelty_threshold_r2 and joint_r2 >= novelty_threshold_r2:
                    max_r2 = joint_r2
                    closest_baseline = f"JOINT_SPAN({', '.join(col_names)})"
                    citation = "Linear combination of literature baselines."
                elif max_r2 < novelty_threshold_r2 and joint_r2 > max_r2:
                    max_r2 = joint_r2
                elif max_r2 >= novelty_threshold_r2 and joint_r2 > max_r2:
                    max_r2 = joint_r2

    ortho_var = float(max(0.0, 1.0 - max_r2))
    is_novel = bool(max_r2 < novelty_threshold_r2)
    verdict = "STRUCTURALLY_NOVEL" if is_novel else "REPRESENTATION_REIDENTIFICATION"

    return StructuralNoveltyResult(
        verdict=verdict,
        max_projection_r2=max_r2,
        closest_literature_baseline=closest_baseline,
        literature_citation=citation,
        orthogonal_variance_fraction=ortho_var,
        novelty_threshold_r2=novelty_threshold_r2,
        details=fit_details,
    )


# ── Priority 3: Regime Robustness Diagnostic ────────────────────────────────

@dataclass(frozen=True)
class RegimeDiagnosticDetails:
    regime_name: str
    n_obs: int
    mean_loss_reduction: float
    dm_statistic: float
    dm_p_value: float
    ic: float


@dataclass(frozen=True)
class RegimeRobustnessDiagnosticResult:
    status: str  # "REGIME_STABLE" | "REGIME_VULNERABLE" | "REGIME_FRAGILE"
    survival_fraction: float  # Fraction of regimes with non-negative incremental gain
    mean_loss_reduction: float
    is_robust: bool
    per_regime_results: dict[str, RegimeDiagnosticDetails]
    details: dict[str, Any]


def evaluate_regime_robustness_diagnostic(
    y_true: np.ndarray,
    y_candidate_pred: np.ndarray,
    y_incumbent_pred: np.ndarray,
    regime_labels: np.ndarray | Sequence[str],
    min_survival_rate: float = 0.60,
) -> RegimeRobustnessDiagnosticResult:
    """Evaluate whether candidate predictive incrementality survives across distinct regimes.

    Implements Reviewer Correction B: Uses regime stability as a diagnostic rather than a
    rigid residual-homoskedasticity veto. Tests whether incremental forecast improvements
    persist across changing volatility environments and temporal slices.
    """
    y = np.asarray(y_true, dtype=float)
    y_c = np.asarray(y_candidate_pred, dtype=float)
    y_b = np.asarray(y_incumbent_pred, dtype=float)
    regs = np.asarray(regime_labels)

    unique_regimes = np.unique(regs)
    per_regime = {}
    positive_regimes = 0
    total_valid_regimes = 0
    all_loss_reductions = []

    for r in unique_regimes:
        idx = (regs == r) & np.isfinite(y) & np.isfinite(y_c) & np.isfinite(y_b)
        n_r = int(np.sum(idx))
        if n_r < 15:
            continue

        total_valid_regimes += 1
        d_r = (y[idx] - y_b[idx]) ** 2 - (y[idx] - y_c[idx]) ** 2
        mean_d_r = float(np.mean(d_r))
        all_loss_reductions.append(mean_d_r)

        hac_var = compute_hac_variance(d_r)
        se_r = np.sqrt(hac_var / n_r)
        dm_stat = float(mean_d_r / max(se_r, 1e-12))
        from scipy.stats import norm
        p_val = float(1.0 - norm.cdf(dm_stat))
        corr = float(np.corrcoef(y_c[idx], y[idx])[0, 1]) if np.std(y_c[idx]) > 1e-9 else 0.0

        if mean_d_r >= -1e-6:
            positive_regimes += 1

        per_regime[str(r)] = RegimeDiagnosticDetails(
            regime_name=str(r),
            n_obs=n_r,
            mean_loss_reduction=mean_d_r,
            dm_statistic=dm_stat,
            dm_p_value=p_val,
            ic=corr,
        )

    if total_valid_regimes == 0:
        return RegimeRobustnessDiagnosticResult(
            status="REGIME_FRAGILE",
            survival_fraction=0.0,
            mean_loss_reduction=0.0,
            is_robust=False,
            per_regime_results={},
            details={"error": "Zero regimes with sufficient sample size."},
        )

    survival_frac = float(positive_regimes / total_valid_regimes)
    overall_mean = float(np.mean(all_loss_reductions)) if all_loss_reductions else 0.0

    is_robust = bool(survival_frac >= min_survival_rate and overall_mean > 0)
    if survival_frac >= 0.80:
        status = "REGIME_STABLE"
    elif survival_frac >= min_survival_rate:
        status = "REGIME_VULNERABLE"
    else:
        status = "REGIME_FRAGILE"

    return RegimeRobustnessDiagnosticResult(
        status=status,
        survival_fraction=survival_frac,
        mean_loss_reduction=overall_mean,
        is_robust=is_robust,
        per_regime_results=per_regime,
        details={
            "n_regimes_evaluated": total_valid_regimes,
            "min_survival_rate_threshold": min_survival_rate,
        },
    )

