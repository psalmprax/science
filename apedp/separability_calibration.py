"""Empirical calibration for separability analyzer and coupling residual thresholds.

Tests the separability detection and coupling residual gate across:
1. Exact separable models: f(x1, x2) = g(x1) + h(x2)
2. Weak interactions: f(x1, x2) = g(x1) + h(x2) + eps * x1 * x2  (eps in [0.02, 0.10])
3. Strong interactions: f(x1, x2) = g(x1) + h(x2) + x1 * x2
4. Noise levels: sigma in [0.0, 0.01, 0.05, 0.10]
5. Unequal variable scales: x1 in [10, 100], x2 in [0.01, 0.1]
6. Heteroscedastic noise: sigma(x) = sigma_0 * (1 + |x1|)

Measures false-decomposition rate, missed-interaction rate, and false-alarm interaction rate
as the threshold varies, identifying the calibrated operational threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp.separability import SeparabilityAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationTrialResult:
    regime: str
    noise_level: float
    is_interactive: bool
    unexplained_std_ratio: float
    separability_detected_additive: bool
    trial_seed: int
    details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    false_decomposition_rate: float  # interactive model where coupling residual <= threshold
    missed_interaction_rate: float   # interactive model where neither analyzer nor residual triggered
    false_alarm_rate: float         # purely separable model where residual > threshold
    accuracy: float                 # overall correct classification (interactive vs separable)


def generate_calibration_dataset(
    regime: str,
    n: int = 60,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
    scale_ratio: float = 1.0,
    heteroscedastic: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    """Generates synthetic datasets across scientific regimes.

    Returns:
        (x_dict, y, is_interactive)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if scale_ratio != 1.0:
        x1 = rng.uniform(10.0, 100.0, n)
        x2 = rng.uniform(0.01, 0.1, n)
    else:
        x1 = rng.uniform(0.5, 3.0, n)
        x2 = rng.uniform(0.5, 3.0, n)

    if regime == "exact_separable":
        # Purely additive
        if scale_ratio != 1.0:
            y_clean = 0.05 * (x1**2) + 500.0 * (x2**2)
        else:
            y_clean = 2.5 * (x1**2) + 1.8 * np.sin(x2)
        is_interactive = False

    elif regime == "weak_interaction":
        # Weak multiplicative coupling
        eps = 0.05
        if scale_ratio != 1.0:
            y_clean = 0.05 * (x1**2) + 500.0 * (x2**2) + eps * (x1 * x2)
        else:
            y_clean = 2.5 * (x1**2) + 1.8 * np.sin(x2) + eps * (x1 * x2)
        is_interactive = True

    elif regime == "strong_interaction":
        # Strong multiplicative coupling
        if scale_ratio != 1.0:
            y_clean = 0.05 * (x1**2) + 500.0 * (x2**2) + 0.8 * (x1 * x2)
        else:
            y_clean = 2.5 * (x1**2) + 1.8 * np.sin(x2) + 1.5 * (x1 * x2)
        is_interactive = True

    elif regime == "rational_interaction":
        # Non-separable rational coupling
        y_clean = (x1**2) / (1.0 + x2**2)
        is_interactive = True

    elif regime == "3var_separable":
        x3 = rng.uniform(0.5, 3.0, n)
        y_clean = 1.5 * (x1**2) + 0.8 * np.exp(-x2) + 0.4 / x3
        is_interactive = False
        x_dict = {"x1": x1, "x2": x2, "x3": x3}
        return _add_noise(x_dict, y_clean, noise_level, rng, heteroscedastic, is_interactive)

    else:
        raise ValueError(f"Unknown regime: {regime}")

    x_dict = {"x1": x1, "x2": x2}
    return _add_noise(x_dict, y_clean, noise_level, rng, heteroscedastic, is_interactive)


def _add_noise(
    x_dict: dict[str, np.ndarray],
    y_clean: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
    heteroscedastic: bool,
    is_interactive: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    s_y = float(np.std(y_clean)) + 1e-12
    if noise_level > 0.0:
        if heteroscedastic:
            x1 = x_dict["x1"]
            x1_norm = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1) + 1e-12)
            sigma = noise_level * s_y * (0.5 + 1.5 * x1_norm)
            noise = rng.normal(0.0, sigma, size=len(y_clean))
        else:
            noise = rng.normal(0.0, noise_level * s_y, size=len(y_clean))
        y = y_clean + noise
    else:
        y = y_clean.copy()
    return x_dict, y, is_interactive


def compute_coupling_residual_ratio(
    x_dict: dict[str, np.ndarray],
    y: np.ndarray,
) -> float:
    """Computes the unexplained residual standard deviation ratio after fitting separate coordinates.

    Uses independent coordinate polynomial projections (degree 2) plus constant.
    """
    n = len(y)
    cols = [np.ones(n)]
    for _k, v in sorted(x_dict.items()):
        v_arr = np.asarray(v, dtype=float)
        cols.append(v_arr)
        cols.append(v_arr**2)

    Phi = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    y_pred = Phi @ coef
    residual = y - y_pred
    s_y = float(np.std(y)) + 1e-12
    return float(np.std(residual) / s_y)


def run_single_calibration_trial(
    regime: str,
    noise_level: float,
    scale_ratio: float = 1.0,
    heteroscedastic: bool = False,
    n_points: int = 60,
    seed: int = 42,
) -> CalibrationTrialResult:
    """Executes one calibration trial across analyzer and coupling residual check."""
    rng = np.random.default_rng(seed)
    x_dict, y, is_interactive = generate_calibration_dataset(
        regime=regime,
        n=n_points,
        noise_level=noise_level,
        rng=rng,
        scale_ratio=scale_ratio,
        heteroscedastic=heteroscedastic,
    )

    analyzer = SeparabilityAnalyzer(tolerance=0.35, min_points=15)
    decomp = analyzer.decompose(x_dict, y)
    is_additive = decomp.is_fully_additive

    ratio = compute_coupling_residual_ratio(x_dict, y)

    return CalibrationTrialResult(
        regime=regime,
        noise_level=noise_level,
        is_interactive=is_interactive,
        unexplained_std_ratio=ratio,
        separability_detected_additive=is_additive,
        trial_seed=seed,
        details={"scale_ratio": scale_ratio, "heteroscedastic": float(heteroscedastic)},
    )


def run_full_calibration_experiment(
    thresholds: list[float] | None = None,
    n_trials_per_cell: int = 10,
    base_seed: int = 500,
) -> dict[str, Any]:
    """Runs a full factorial calibration across models, noise, scale, and heteroscedasticity."""
    if thresholds is None:
        thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    regimes = [
        ("exact_separable", 1.0, False),
        ("weak_interaction", 1.0, False),
        ("strong_interaction", 1.0, False),
        ("rational_interaction", 1.0, False),
        ("3var_separable", 1.0, False),
        ("exact_separable", 100.0, False),     # Unequal scale
        ("weak_interaction", 100.0, False),    # Unequal scale with interaction
        ("exact_separable", 1.0, True),        # Heteroscedastic noise
        ("weak_interaction", 1.0, True),       # Heteroscedastic noise with interaction
    ]

    noise_levels = [0.0, 0.01, 0.05, 0.10]

    all_trials: list[CalibrationTrialResult] = []
    trial_idx = 0

    for regime_name, scale_ratio, heteroscedastic in regimes:
        for noise in noise_levels:
            for _rep in range(n_trials_per_cell):
                seed = base_seed + trial_idx
                trial_idx += 1
                res = run_single_calibration_trial(
                    regime=regime_name,
                    noise_level=noise,
                    scale_ratio=scale_ratio,
                    heteroscedastic=heteroscedastic,
                    seed=seed,
                )
                all_trials.append(res)

    # Compute metrics for each candidate threshold
    threshold_metrics: list[ThresholdMetrics] = []
    interactive_trials = [t for t in all_trials if t.is_interactive]
    separable_trials = [t for t in all_trials if not t.is_interactive]

    for thresh in thresholds:
        # False decomposition: interactive model where coupling residual was NOT triggered (ratio <= thresh)
        false_decomps = sum(1 for t in interactive_trials if t.unexplained_std_ratio <= thresh)
        false_decomp_rate = false_decomps / max(1, len(interactive_trials))

        # Missed interaction: interactive model where analyzer said additive AND residual was <= thresh
        missed = sum(
            1 for t in interactive_trials
            if t.separability_detected_additive and t.unexplained_std_ratio <= thresh
        )
        missed_rate = missed / max(1, len(interactive_trials))

        # False alarm: purely separable model where residual > thresh (unwarranted search for interaction)
        false_alarms = sum(1 for t in separable_trials if t.unexplained_std_ratio > thresh)
        false_alarm_rate = false_alarms / max(1, len(separable_trials))

        # Correct classification:
        # Interactive correctly identified if ratio > thresh
        # Separable correctly identified if ratio <= thresh
        correct = sum(1 for t in interactive_trials if t.unexplained_std_ratio > thresh) + sum(
            1 for t in separable_trials if t.unexplained_std_ratio <= thresh
        )
        accuracy = correct / len(all_trials)

        threshold_metrics.append(
            ThresholdMetrics(
                threshold=thresh,
                false_decomposition_rate=false_decomp_rate,
                missed_interaction_rate=missed_rate,
                false_alarm_rate=false_alarm_rate,
                accuracy=accuracy,
            )
        )

    # Find optimal threshold: max accuracy while keeping missed interaction rate <= 0.10
    admissible_thresholds = [m for m in threshold_metrics if m.missed_interaction_rate <= 0.10]
    if admissible_thresholds:
        best_metric = max(admissible_thresholds, key=lambda m: (m.accuracy, -m.false_alarm_rate))
        recommended_thresh = best_metric.threshold
    else:
        best_metric = max(threshold_metrics, key=lambda m: m.accuracy)
        recommended_thresh = best_metric.threshold

    return {
        "total_trials": len(all_trials),
        "n_interactive": len(interactive_trials),
        "n_separable": len(separable_trials),
        "threshold_metrics": [
            {
                "threshold": m.threshold,
                "false_decomposition_rate": m.false_decomposition_rate,
                "missed_interaction_rate": m.missed_interaction_rate,
                "false_alarm_rate": m.false_alarm_rate,
                "accuracy": m.accuracy,
            }
            for m in threshold_metrics
        ],
        "recommended_threshold": recommended_thresh,
        "provisional_threshold_0_15_metrics": next(
            (
                {
                    "threshold": m.threshold,
                    "false_decomposition_rate": m.false_decomposition_rate,
                    "missed_interaction_rate": m.missed_interaction_rate,
                    "false_alarm_rate": m.false_alarm_rate,
                    "accuracy": m.accuracy,
                }
                for m in threshold_metrics
                if abs(m.threshold - 0.15) < 1e-4
            ),
            None,
        ),
    }


__all__ = [
    "CalibrationTrialResult",
    "ThresholdMetrics",
    "compute_coupling_residual_ratio",
    "generate_calibration_dataset",
    "run_full_calibration_experiment",
    "run_single_calibration_trial",
]
