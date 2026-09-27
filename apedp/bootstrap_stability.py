"""Bootstrap Structural Stability Runner for APEDP.

Quantifies discovery stability across data perturbations and random seeds:
A single successful run could be a lucky symbolic artifact.
This module executes N independent bootstrap evaluations:
    Bootstrap resample D* ~ D (with optional noise jitter and train-window perturbation)
    -> Equation discovery
    -> Canonicalize to structural family
    -> Compute empirical recovery probability S(f) = P(family f | D*)

Distinguishes robust physical laws (S(f) >= 80%) from sporadic artifacts (S(f) <= 10%).
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.model_synthesis import AutonomousModelSynthesizer
from apedp.symbolic_equivalence import CanonicalForm, SymbolicEquivalenceEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapTrial:
    """Outcome of a single bootstrap trial."""

    trial_index: int
    seed: int
    raw_expression: str
    canonical_family: str
    fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapPolicy:
    """Configurable gate for structural-ambiguity detection (no magic constants).

    The engine measures evidence; the harness interprets test status from it.
    - min_family_stability: AMBIGUOUS when family_stability <= this value.
    - selection_margin: AMBIGUOUS when mse_gap <= this value. None disables
      the margin gate (stability alone decides).
    - min_trials: fewer trials than this yields INSUFFICIENT (cannot measure).
    """

    min_family_stability: float = 0.80
    selection_margin: float | None = None
    min_trials: int = 10


class IdentifiabilityStatus(str, Enum):
    """Engine-side diagnosis (NOT a pytest status)."""

    IDENTIFIABLE = "IDENTIFIABLE"
    AMBIGUOUS = "AMBIGUOUS"  # finite-sample structural ambiguity (Category B)
    INSUFFICIENT = "INSUFFICIENT"  # too few usable trials to measure
    UNKNOWN = "UNKNOWN"  # legacy path without ambiguity diagnostics


@dataclass(frozen=True)
class BootstrapStabilityReport:
    """Summary of structural recovery stability across bootstrap trials."""

    dominant_family: str
    stability_score: float  # S(f) in [0.0, 1.0]
    is_stable: bool
    stability_threshold: float
    family_frequencies: dict[str, float]
    family_counts: dict[str, int]
    n_trials: int
    trials: list[BootstrapTrial] = field(default_factory=list)
    family_stability: float = 0.0  # proportion of trials selecting dominant family
    family_mean_mse: dict[str, float] = field(default_factory=dict)  # mean val_MSE per family
    mse_gap: float | None = None  # second_best_mean_mse - best_mean_mse; None if <2 families
    identifiability: str = IdentifiabilityStatus.UNKNOWN.value
    selection_margin: float | None = None  # margin rule actually applied (None = stability-only)
    reason: str = ""  # audit trail: measurements + rule applied

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_family": self.dominant_family,
            "stability_score": float(self.stability_score),
            "is_stable": self.is_stable,
            "stability_threshold": float(self.stability_threshold),
            "family_frequencies": {k: float(v) for k, v in self.family_frequencies.items()},
            "family_counts": dict(self.family_counts),
            "n_trials": self.n_trials,
            "trials": [t.to_dict() for t in self.trials],
            "family_stability": float(self.family_stability),
            "family_mean_mse": {k: float(v) for k, v in self.family_mean_mse.items()},
            "mse_gap": float(self.mse_gap) if self.mse_gap is not None else None,
            "identifiability": self.identifiability,
            "selection_margin": float(self.selection_margin) if self.selection_margin is not None else None,
            "reason": self.reason,
        }

    def diagnostic_summary(self) -> str:
        """Human-readable audit trail separating measurements from interpretation."""
        gap_str = f"{self.mse_gap:.3e}" if self.mse_gap is not None else "n/a (<2 families)"
        margin_str = f"{self.selection_margin:.3e}" if self.selection_margin is not None else "n/a (stability-only)"
        lines = [
            "Bootstrap stability",
            "--------------------",
            f"Trials:              {self.n_trials}",
            f"Dominant family:     {self.dominant_family}",
            f"Family stability:    {self.family_stability:.2f}",
            f"MSE separation:      {gap_str}",
            f"Selection margin:    {margin_str}",
            f"Identifiability:     {self.identifiability}",
            f"Reason:              {self.reason}",
        ]
        return "\n".join(lines)


def suggest_verdict(report: BootstrapStabilityReport) -> str:
    """Harness-side mapping from engine diagnosis to test verdict.

    Returns "PASS" | "XFAIL" | "FAIL". The engine never imports pytest;
    the benchmark harness maps this string to pytest actions.
    - AMBIGUOUS -> XFAIL (expected finite-sample structural ambiguity)
    - INSUFFICIENT -> FAIL (cannot measure: trials failed or too few)
    - IDENTIFIABLE + stable -> PASS; otherwise FAIL.
    """
    if report.identifiability == IdentifiabilityStatus.AMBIGUOUS.value:
        return "XFAIL"
    if report.identifiability == IdentifiabilityStatus.INSUFFICIENT.value:
        return "FAIL"
    if report.identifiability == IdentifiabilityStatus.IDENTIFIABLE.value and report.is_stable:
        return "PASS"
    if report.identifiability == IdentifiabilityStatus.UNKNOWN.value and report.is_stable:
        return "PASS"
    return "FAIL"


class BootstrapStabilityRunner:
    """Executes bootstrap perturbations and measures empirical recovery stability."""

    def __init__(
        self,
        default_trials: int = 50,
        subsample_ratio: float = 0.85,
        stability_threshold: float = 0.60,
    ):
        self.default_trials = default_trials
        self.subsample_ratio = subsample_ratio
        self.stability_threshold = stability_threshold

    def evaluate_stability(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        discovery_fn: Callable[[dict[str, np.ndarray], np.ndarray, int], str | sp.Expr],
        n_trials: int | None = None,
        noise_jitter: float = 0.0,
        stability_threshold: float | None = None,
        seed: int = 42,
        known_variables: list[str] | None = None,
    ) -> BootstrapStabilityReport:
        """Runs N bootstrap evaluations and computes structural stability metric S(f)."""
        trials_count = n_trials or self.default_trials
        threshold = stability_threshold if stability_threshold is not None else self.stability_threshold

        var_names = sorted(x_data.keys())
        n_samples = len(y_data)
        if n_samples == 0:
            raise ValueError("Dataset is empty; cannot compute bootstrap stability")

        sub_n = max(3, round(n_samples * self.subsample_ratio))
        trials: list[BootstrapTrial] = []
        family_counts: dict[str, int] = {}

        for t in range(trials_count):
            trial_seed = seed + t
            rng = np.random.default_rng(trial_seed)

            # Bootstrap resampling with replacement
            indices = rng.choice(n_samples, size=sub_n, replace=True)
            x_sample = {v: np.asarray(x_data[v])[indices] for v in var_names}
            y_sample = np.asarray(y_data)[indices]

            # Optional noise jitter
            if noise_jitter > 0.0:
                y_sample = y_sample + rng.normal(0.0, noise_jitter, size=sub_n)

            # Run discovery on the perturbed sample
            try:
                raw_result = discovery_fn(x_sample, y_sample, trial_seed)
                canon: CanonicalForm = SymbolicEquivalenceEngine.canonicalize(
                    raw_result, known_variables=known_variables or var_names
                )
                family_key = canon.structural_family
                raw_str = str(raw_result)
                fp = canon.fingerprint
            except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
                logger.warning("Discovery trial %d failed with error: %s", t, exc)
                family_key = "FAILED_EVALUATION"
                raw_str = "None"
                fp = ""

            trials.append(
                BootstrapTrial(
                    trial_index=t,
                    seed=trial_seed,
                    raw_expression=raw_str,
                    canonical_family=family_key,
                    fingerprint=fp,
                )
            )
            family_counts[family_key] = family_counts.get(family_key, 0) + 1

        # Calculate frequency distribution
        family_frequencies = {k: v / trials_count for k, v in family_counts.items()}
        # Identify dominant structural family
        dominant = max(family_counts.keys(), key=lambda k: family_counts[k])
        stability_score = family_frequencies[dominant]
        is_stable = bool(stability_score >= threshold and dominant != "FAILED_EVALUATION")

        return BootstrapStabilityReport(
            dominant_family=dominant,
            stability_score=stability_score,
            is_stable=is_stable,
            stability_threshold=threshold,
            family_frequencies=family_frequencies,
            family_counts=family_counts,
            n_trials=trials_count,
            trials=trials,
        )

    def evaluate_synthesis_stability(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        n_trials: int | None = None,
        noise_jitter: float = 0.0,
        stability_threshold: float | None = None,
        seed: int = 42,
        known_variables: list[str] | None = None,
        policy: BootstrapPolicy | None = None,
    ) -> BootstrapStabilityReport:
        """Run the full AutonomousModelSynthesizer synthesize() pipeline N times
        with bootstrap resampling and optional noise jitter, then aggregate
        structural-family recovery frequencies.

        This directly addresses the highest-priority gap: "Does APEDP recover the same
        equation when the experiment is repeated under comparable noise?" (Case 808.)

        Returns a BootstrapStabilityReport where dominant_family is the most-recovered
        structural family, family_stability = P(family | repeated noisy experiments),
        and mse_gap = second_best_mean_MSE - best_mean_MSE (or None if <2 families).

        The engine produces evidence (family_stability, mse_gap, identifiability,
        reason); the harness interprets test status via suggest_verdict().
        """
        trials_count = n_trials or self.default_trials
        threshold = stability_threshold if stability_threshold is not None else self.stability_threshold
        gate = policy or BootstrapPolicy()

        var_names = sorted(x_data.keys())
        n_samples = len(y_data)
        if n_samples == 0:
            raise ValueError("Dataset is empty; cannot compute synthesis stability")

        # Flat arrays for synthesis (synthesize() expects 1-D arrays)
        x_flat = np.asarray(x_data[var_names[0]]) if var_names else np.asarray(y_data)
        y_flat = np.asarray(y_data)
        N = n_samples  # full dataset size for bootstrap

        sub_n = max(3, round(N * 0.8))

        # Tracking structures for ambiguity diagnostics
        family_trial_counts: Counter[str] = Counter()
        family_val_mses_dict: dict[str, list[float]] = {}

        for t in range(trials_count):
            trial_seed = seed + t
            trial_rng = np.random.default_rng(trial_seed)

            # Bootstrap resampling with replacement: indices into full dataset
            abs_indices = trial_rng.choice(N, size=sub_n, replace=True)  # indices in [0, N-1]
            x_sample_flat = x_flat[abs_indices]     # size-sub_n subsample
            y_sample_flat = y_flat[abs_indices]

            # Optional noise jitter on top of bootstrap
            if noise_jitter > 0.0:
                y_sample_flat = y_sample_flat + trial_rng.normal(0.0, noise_jitter, size=sub_n)

            # Local 0..sub_n-1 indices for train/val split of the subsample
            train_n = max(3, sub_n - round(sub_n * 0.25))
            val_n = sub_n - train_n

            # Random partition of the subsample into train/val
            perm = trial_rng.permutation(sub_n)
            train_local = perm[:train_n]
            val_local = perm[train_n:train_n + val_n]

            x_train_flat = x_sample_flat[train_local]
            y_train_flat = y_sample_flat[train_local]
            x_val_flat = x_sample_flat[val_local]
            y_val_flat = y_sample_flat[val_local]

            # Run full synthesis pipeline on this perturbed sample
            try:
                synthesizer = AutonomousModelSynthesizer()
                models = synthesizer.synthesize(
                    x_train_flat, y_train_flat, x_val_flat, y_val_flat,
                )
                if not models:
                    family_trial_counts["NO_MODEL"] += 1
                    continue
                # Canonicalize the best model's family
                best = models[0]  # already sorted by val_mse
                canon: CanonicalForm = SymbolicEquivalenceEngine.canonicalize(
                    best.expression_str, known_variables=known_variables or [var_names[0]] if var_names else []
                )
                family_key = canon.structural_family

                # Track val_MSE per family (using the best model's val_mse)
                # The model's val_mse is the validation MSE from the synthesize call
                best_val_mse = best.val_mse
                if family_key not in family_val_mses_dict:
                    family_val_mses_dict[family_key] = []
                family_val_mses_dict[family_key].append(best_val_mse)

                family_trial_counts[family_key] += 1
            except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
                logger.warning("Synthesis trial %d failed with error: %s", t, exc)
                family_trial_counts["SYNTHESIS_ERROR"] += 1

        # Calculate frequency distribution
        family_frequencies = {k: v / trials_count for k, v in family_trial_counts.items()}
        # Identify dominant structural family
        if not family_trial_counts:
            dominant = "NO_TRIALS"
            stability_score = 0.0
        else:
            dominant = max(family_trial_counts.keys(), key=lambda k: family_trial_counts[k])
            stability_score = family_frequencies[dominant]
        is_stable = bool(stability_score >= threshold and dominant not in {"INSUFFICIENT_DATA", "NO_MODEL", "SYNTHESIS_ERROR"})

        # Compute family_stability: proportion of trials selecting dominant family
        family_stability = family_trial_counts[dominant] / trials_count if family_trial_counts else 0.0

        # Compute family_mean_mse: mean val_MSE per family
        family_mean_mse: dict[str, float] = {}
        for family, mses in family_val_mses_dict.items():
            family_mean_mse[family] = sum(mses) / len(mses) if mses else 0.0

        # Compute mse_gap: second_best_mean_mse - best_mean_mse.
        # Means are per family across noise realizations (NOT two arbitrary trials).
        if len(family_mean_mse) >= 2:
            ranked = sorted(family_mean_mse.items(), key=lambda kv: kv[1])
            best_mse = ranked[0][1]
            second_best_mse = ranked[1][1]
            mse_gap: float | None = second_best_mse - best_mse
        else:
            # Fewer than two families observed; no separation to compute.
            mse_gap = None

        # Engine-side diagnosis (harness maps to PASS/XFAIL/FAIL via suggest_verdict).
        failed = dominant in {"INSUFFICIENT_DATA", "NO_MODEL", "SYNTHESIS_ERROR", "NO_TRIALS"}
        if failed or trials_count < gate.min_trials:
            identifiability = IdentifiabilityStatus.INSUFFICIENT.value
            reason = (
                f"cannot measure: dominant={dominant}, usable_trials={trials_count}, "
                f"min_trials={gate.min_trials}"
            )
        else:
            low_stability = family_stability <= gate.min_family_stability
            if gate.selection_margin is None:
                narrow_gap = True
            elif mse_gap is None:
                # No competing family observed: no evidence of indistinguishability.
                narrow_gap = False
            else:
                narrow_gap = mse_gap <= gate.selection_margin
            if low_stability and narrow_gap:
                identifiability = IdentifiabilityStatus.AMBIGUOUS.value
            else:
                identifiability = IdentifiabilityStatus.IDENTIFIABLE.value
            gap_str = f"{mse_gap:.3e}" if mse_gap is not None else "n/a"
            margin_str = f"{gate.selection_margin:.3e}" if gate.selection_margin is not None else "n/a"
            reason = (
                f"family_stability={family_stability:.2f} (gate<={gate.min_family_stability:.2f}); "
                f"mse_gap={gap_str} (margin<={margin_str}); rule=AMBIGUOUS iff low stability and narrow gap"
            )

        return BootstrapStabilityReport(
            dominant_family=dominant,
            stability_score=stability_score,
            is_stable=is_stable,
            stability_threshold=threshold,
            family_frequencies=family_frequencies,
            family_counts=dict(family_trial_counts),
            n_trials=trials_count,
            trials=[],
            family_stability=family_stability,
            family_mean_mse=family_mean_mse,
            mse_gap=mse_gap,
            identifiability=identifiability,
            selection_margin=gate.selection_margin,
            reason=reason,
        )


__all__ = [
    "BootstrapPolicy",
    "BootstrapStabilityReport",
    "BootstrapStabilityRunner",
    "BootstrapTrial",
    "IdentifiabilityStatus",
    "suggest_verdict",
]