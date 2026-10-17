"""Calibrated Three-Tier Benchmark Suite.

Implements the reviewer's architecture separating discovery power from referee discipline:
  1. RefereeBenchmark: Measures false promotion (FPR), correct abstention (CAR),
     and valid retention (VRR) with Wilson score confidence intervals.
  2. DiscoveryBenchmark: Measures recovery power (1 - β) across a calibrated grid of
     realistic effect sizes (IC ∈ [0.02, 0.05, 0.15]), sample sizes, fat tails, and regimes.
  3. TransferBenchmark: Measures generalization retention (IC_OOS / IC_IS) on untouched OOD data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from apedp_dyn.conf.canonical_spec import (
    REFEREE_MAX_FPR,
    REFEREE_MIN_CAR,
    REFEREE_MIN_VRR,
)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p_hat = k / n
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z ** 2
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - margin) if k > 0 else 0.0
    upper = min(1.0, center + margin) if k < n else 1.0
    return float(lower), float(upper)


# ── Tier 1: Referee Benchmark ────────────────────────────────────────────────

@dataclass
class RefereeBenchmarkReport:
    n_null: int
    n_ambiguous: int
    n_warranted: int
    false_promotions: int
    correct_abstentions: int
    valid_retained: int
    fpr: float
    fpr_ci: tuple[float, float]
    car: float
    car_ci: tuple[float, float]
    vrr: float
    vrr_ci: tuple[float, float]
    passed_referee_contract: bool


def run_calibrated_referee_benchmark(
    referee_adjudicator_fn,
    n_trials: int = 40,
    seed: int = 42,
) -> RefereeBenchmarkReport:
    """Evaluate referee on calibrated null, ambiguous, and warranted synthetic cases."""
    rng = np.random.RandomState(seed)
    false_promos = 0
    correct_abstains = 0
    valid_retained = 0

    # 1. Null noise cases: purely uninformative (expect REJECT or ABSTAIN)
    for _ in range(n_trials):
        x = rng.randn(250)
        y = rng.randn(250)  # zero true relation
        verdict = referee_adjudicator_fn(x, y, is_null=True)
        if verdict == "PROMOTED":
            false_promos += 1
        elif verdict in ("REJECTED", "ABSTAINED"):
            pass

    # 2. Ambiguous cases: very low SNR (IC ~ 0.01, expect ABSTAIN)
    for _ in range(n_trials):
        x = rng.randn(250)
        y = 0.01 * x + rng.randn(250)
        verdict = referee_adjudicator_fn(x, y, is_ambiguous=True)
        if verdict == "ABSTAINED":
            correct_abstains += 1

    # 3. Warranted cases: high SNR (IC ~ 0.15, expect PROMOTE)
    for _ in range(n_trials):
        x = rng.randn(250)
        y = 0.20 * x + rng.randn(250)
        verdict = referee_adjudicator_fn(x, y, is_warranted=True)
        if verdict == "PROMOTED":
            valid_retained += 1

    fpr = float(false_promos / max(n_trials, 1))
    car = float(correct_abstains / max(n_trials, 1))
    vrr = float(valid_retained / max(n_trials, 1))

    report = RefereeBenchmarkReport(
        n_null=n_trials,
        n_ambiguous=n_trials,
        n_warranted=n_trials,
        false_promotions=false_promos,
        correct_abstentions=correct_abstains,
        valid_retained=valid_retained,
        fpr=fpr,
        fpr_ci=wilson_ci(false_promos, n_trials),
        car=car,
        car_ci=wilson_ci(correct_abstains, n_trials),
        vrr=vrr,
        vrr_ci=wilson_ci(valid_retained, n_trials),
        passed_referee_contract=(fpr <= REFEREE_MAX_FPR and car >= REFEREE_MIN_CAR and vrr >= REFEREE_MIN_VRR),
    )
    return report


# ── Tier 2: Calibrated Discovery Benchmark ───────────────────────────────────

@dataclass
class DiscoveryConditionResult:
    condition_name: str
    target_ic: float
    sample_size: int
    noise_type: str  # "gaussian", "fat_tailed_t4", "regime_switching"
    trials: int
    recovered_count: int
    recovery_rate_power: float
    power_ci: tuple[float, float]
    mean_estimated_ic: float


@dataclass
class DiscoveryBenchmarkSuiteReport:
    conditions: list[DiscoveryConditionResult]
    overall_mean_power: float
    weak_signal_power: float  # IC ~ 0.02
    moderate_signal_power: float  # IC ~ 0.05
    strong_signal_power: float  # IC ~ 0.15


def run_calibrated_discovery_benchmark(
    proposer_discovery_fn,
    trials_per_condition: int = 15,
    seed: int = 42,
) -> DiscoveryBenchmarkSuiteReport:
    """Measure recovery statistical power (1 - β) across a calibrated grid of effect sizes."""
    rng = np.random.RandomState(seed)
    conditions: list[DiscoveryConditionResult] = []

    # Grid: target_ic in {0.02, 0.05, 0.15}, sample_size in {250, 500}, noise types
    grid = [
        ("weak_gaussian_250", 0.02, 250, "gaussian"),
        ("moderate_gaussian_250", 0.05, 250, "gaussian"),
        ("strong_gaussian_250", 0.15, 250, "gaussian"),
        ("moderate_fat_tailed_t4", 0.05, 250, "fat_tailed_t4"),
        ("moderate_regime_switch", 0.05, 500, "regime_switching"),
    ]

    for cond_name, ic_target, n_bars, noise_type in grid:
        recovered = 0
        est_ics = []
        for _ in range(trials_per_condition):
            # Generate feature
            f = rng.randn(n_bars)
            # Generate noise
            if noise_type == "fat_tailed_t4":
                noise = rng.standard_t(df=4, size=n_bars) * 0.7
            else:
                noise = rng.randn(n_bars)

            # Generate target
            if noise_type == "regime_switching":
                # Beta is active only in first half
                beta = np.zeros(n_bars)
                beta[: n_bars // 2] = ic_target * 2.0
                y = beta * f + noise
            else:
                y = ic_target * f + noise

            # Proposer attempts recovery
            success, estimated_ic = proposer_discovery_fn(f, y)
            if success:
                recovered += 1
            est_ics.append(float(estimated_ic))

        power = float(recovered / trials_per_condition)
        conditions.append(
            DiscoveryConditionResult(
                condition_name=cond_name,
                target_ic=ic_target,
                sample_size=n_bars,
                noise_type=noise_type,
                trials=trials_per_condition,
                recovered_count=recovered,
                recovery_rate_power=power,
                power_ci=wilson_ci(recovered, trials_per_condition),
                mean_estimated_ic=float(np.mean(est_ics)),
            )
        )

    powers = [c.recovery_rate_power for c in conditions]
    weak_power = np.mean([c.recovery_rate_power for c in conditions if c.target_ic == 0.02])
    mod_power = np.mean([c.recovery_rate_power for c in conditions if c.target_ic == 0.05])
    strong_power = np.mean([c.recovery_rate_power for c in conditions if c.target_ic == 0.15])

    return DiscoveryBenchmarkSuiteReport(
        conditions=conditions,
        overall_mean_power=float(np.mean(powers)),
        weak_signal_power=float(weak_power),
        moderate_signal_power=float(mod_power),
        strong_signal_power=float(strong_power),
    )


# ── Tier 3: Transfer Benchmark ───────────────────────────────────────────────

@dataclass
class TransferBenchmarkReport:
    in_sample_mean_ic: float
    out_of_sample_mean_ic: float
    retention_ratio: float  # OOS IC / IS IC
    passed_transfer: bool
    details: dict[str, Any]


def run_transfer_benchmark(
    discovered_model_eval_fn,
    test_environments: dict[str, tuple[np.ndarray, np.ndarray]],
) -> TransferBenchmarkReport:
    """Evaluate surviving models across unseen OOD environments."""
    is_ics, oos_ics = [], []
    for _env_name, (X_test, y_test) in test_environments.items():
        is_ic, oos_ic = discovered_model_eval_fn(X_test, y_test)
        is_ics.append(is_ic)
        oos_ics.append(oos_ic)

    mean_is = float(np.mean(is_ics))
    mean_oos = float(np.mean(oos_ics))
    retention = float(mean_oos / max(mean_is, 1e-4)) if mean_is > 0 else 0.0

    return TransferBenchmarkReport(
        in_sample_mean_ic=mean_is,
        out_of_sample_mean_ic=mean_oos,
        retention_ratio=retention,
        passed_transfer=bool(mean_oos > 0.01 and retention >= 0.30),
        details={"per_env_oos_ic": dict(zip(test_environments.keys(), oos_ics, strict=True))},
    )


# ── Priority 0: Detailed Power Calibration Grid ─────────────────────────────

@dataclass
class PowerCalibrationCell:
    sample_size: int
    target_ic: float
    theoretical_max_power: float
    empirical_power: float
    power_ci: tuple[float, float]
    discovery_efficiency_ratio: float  # empirical / theoretical_max


def compute_theoretical_gaussian_power(n: int, ic: float, alpha: float = 0.05) -> float:
    """Compute theoretical upper-bound statistical power for correlation detection."""
    z_alpha = stats.norm.ppf(1.0 - alpha)
    # Non-centrality parameter under asymptotic normality: delta = sqrt(N) * rho
    delta = np.sqrt(max(n - 3, 1)) * ic
    theoretical_power = float(1.0 - stats.norm.cdf(z_alpha - delta))
    return float(np.clip(theoretical_power, 0.0, 1.0))


def generate_power_curve_matrix(
    proposer_discovery_fn,
    sample_sizes: list[int] | None = None,
    target_ics: list[float] | None = None,
    trials_per_cell: int = 10,
    noise_type: str = "gaussian",
    seed: int = 42,
) -> list[PowerCalibrationCell]:
    """Generate empirical vs theoretical power matrix across sample sizes and effect sizes."""
    if sample_sizes is None:
        sample_sizes = [250, 500, 1000]
    if target_ics is None:
        target_ics = [0.02, 0.05, 0.15]

    rng = np.random.RandomState(seed)
    cells: list[PowerCalibrationCell] = []

    for n in sample_sizes:
        for ic in target_ics:
            theo_power = compute_theoretical_gaussian_power(n, ic, alpha=0.05)
            recovered = 0
            for _ in range(trials_per_cell):
                f = rng.randn(n)
                if noise_type == "fat_tailed_t4":
                    noise = rng.standard_t(df=4, size=n) * 0.7
                elif noise_type == "ar1_correlated":
                    # AR(1) noise with phi = 0.3
                    eps = rng.randn(n)
                    noise = np.zeros(n)
                    for t in range(1, n):
                        noise[t] = 0.3 * noise[t - 1] + eps[t]
                else:
                    noise = rng.randn(n)

                y = ic * f + noise
                success, _ = proposer_discovery_fn(f, y)
                if success:
                    recovered += 1

            emp_power = float(recovered / trials_per_cell)
            ci = wilson_ci(recovered, trials_per_cell)
            efficiency = float(emp_power / max(theo_power, 1e-4))

            cells.append(
                PowerCalibrationCell(
                    sample_size=n,
                    target_ic=ic,
                    theoretical_max_power=theo_power,
                    empirical_power=emp_power,
                    power_ci=ci,
                    discovery_efficiency_ratio=efficiency,
                )
            )

    return cells

