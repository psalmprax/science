"""APEDP-X Agnostic Domain Adapter (Level 2 Epistemic Control Layer).

Provides a generic declared-invariant interface and validation pipeline for
arbitrary scientific domains without requiring hardcoded domain-specific rules.

Core Epistemic Principles:
  1. The core adjudication kernel (Claim Ladder + EEDR + AEH) is domain-independent.
  2. Rather than hard-coding PDE Lax entropy, Kepler density bounds, or financial
     slippage into the referee, the problem or candidate model declares its
     expected invariants, symmetries, bounds, and conservation laws.
  3. The Agnostic Adapter evaluates:
     - Empirical fit & holdout generalization (noise floor, holdout degradation)
     - Declared invariants & physical/mathematical conservation properties
     - Structural rival equivalence at resolution floor tau_epi
     - Input perturbation stability (Lipschitz/bounded response)
     - Extrapolative regime transfer
  4. Synthesizes an EvidenceBundle and routes to the frozen referee kernel.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
    EvidenceRequirement,
    adjudicate_scientific_claim,
)
from apedp_dyn.epistemic.eedr import (
    EEDRAdjudicationResult,
    execute_eedr_loop,
)


class InvariantKind(str, Enum):
    """Generic taxonomy of scientific invariants that any domain can declare."""
    CONSERVATION_LAW = "conservation_law"         # Delta Q <= epsilon (mass, energy, momentum, flux)
    SYMMETRY_TRANSFORMATION = "symmetry_transform" # f(T(x)) == rho(T) f(x) (parity, rotation, translation)
    BOUND_POSITIVITY = "bound_positivity"         # y >= y_min, L <= y <= U (non-negativity, probability)
    MONOTONICITY = "monotonicity"                 # dy/dx >= 0 or order-preserving (entropy, priority)
    LIPSCHITZ_SMOOTHNESS = "lipschitz_smoothness" # ||f(x1) - f(x2)|| <= L ||x1 - x2||
    SCALE_HOMOGENEITY = "scale_homogeneity"       # f(lambda * x) == lambda^k * f(x)
    ASYMPTOTIC_LIMIT = "asymptotic_limit"         # lim_{x -> inf} f(x) == C (saturation, bounded tail)
    INTERVENTIONAL_INVARIANCE = "interventional_inv" # Invariance under targeted do(X) perturbation
    CUSTOM_PREDICATE = "custom_predicate"         # Arbitrary user/problem-supplied verification callable


@dataclass
class DeclaredInvariant:
    """Specification of an invariant claimed or required for a candidate model."""
    name: str
    kind: InvariantKind
    description: str = ""
    tolerance: float = 1e-3
    target_value: float | None = None
    required_for_tier: ClaimLevel = ClaimLevel.C3_STRUCTURAL
    evaluator: Callable[[Any, Any], tuple[bool, float, str]] | None = None


@dataclass
class InvariantVerificationResult:
    """Outcome of verifying a declared invariant against test data or trajectories."""
    name: str
    kind: InvariantKind
    passed: bool
    observed_metric: float
    tolerance: float
    diagnostic_message: str


@dataclass
class AgnosticCandidateSpecification:
    """Candidate model specification presented to the Agnostic Adapter."""
    candidate_id: str
    proposed_claim: ClaimLevel
    formula_repr: str
    model_callable: Callable[[np.ndarray], np.ndarray] | None = None
    declared_invariants: list[DeclaredInvariant] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgnosticProblemDataset:
    """Generic dataset representing an empirical or simulated scientific problem."""
    domain_label: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    X_extrap: np.ndarray | None = None
    y_extrap: np.ndarray | None = None
    rival_models: list[AgnosticCandidateSpecification] = field(default_factory=list)
    tau_epi: float | None = None
    noise_floor: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgnosticAdjudicationOutcome:
    """Complete referee adjudication outcome via the Agnostic Adapter."""
    candidate_id: str
    proposed_claim: ClaimLevel
    justified_claim: ClaimLevel
    verdict: EpistemicAdjudicationVerdict
    promoted: bool
    final_status: str
    verified_invariants: list[InvariantVerificationResult]
    failed_invariants: list[InvariantVerificationResult]
    empirical_metrics: dict[str, float]
    evidence_bundle: EvidenceBundle
    eedr_result: EEDRAdjudicationResult | None = None


class AgnosticAdapter:
    """Domain-Agnostic Adapter implementing the Level 2 Declared Invariant Interface."""

    def __init__(self, default_tau_epi: float = 0.02, max_repair_budget: int = 1):
        self.default_tau_epi = default_tau_epi
        self.max_repair_budget = max_repair_budget

    def evaluate_empirical_fit(
        self,
        candidate: AgnosticCandidateSpecification,
        dataset: AgnosticProblemDataset,
    ) -> dict[str, float]:
        """Compute standard empirical fit and holdout generalization metrics."""
        metrics: dict[str, float] = {}

        if candidate.model_callable is None:
            if "precomputed_metrics" in candidate.metadata:
                return dict(candidate.metadata["precomputed_metrics"])
            return {
                "no_model_callable": 1.0,
                "execution_error": 1.0,
                "in_sample_r2": -999.0,
                "holdout_rmse": 999.0,
            }

        try:
            # 1. In-sample fit
            y_pred_train = candidate.model_callable(dataset.X_train)
            if np.any(np.isnan(y_pred_train)) or np.any(np.isinf(y_pred_train)):
                metrics["train_nan_or_inf"] = 1.0
                metrics["in_sample_r2"] = -999.0
                metrics["train_rmse"] = 999.0
            else:
                train_ss_res = np.sum((dataset.y_train - y_pred_train) ** 2)
                train_ss_tot = np.sum((dataset.y_train - np.mean(dataset.y_train)) ** 2) + 1e-12
                metrics["in_sample_r2"] = float(1.0 - (train_ss_res / train_ss_tot))
                metrics["train_rmse"] = float(np.sqrt(np.mean((dataset.y_train - y_pred_train) ** 2)))

            # 2. Out-of-sample holdout
            y_pred_holdout = candidate.model_callable(dataset.X_holdout)
            if np.any(np.isnan(y_pred_holdout)) or np.any(np.isinf(y_pred_holdout)):
                metrics["holdout_nan_or_inf"] = 1.0
                metrics["holdout_rmse"] = 999.0
            else:
                metrics["holdout_rmse"] = float(np.sqrt(np.mean((dataset.y_holdout - y_pred_holdout) ** 2)))

            # 3. Extrapolation holdout (if available)
            if dataset.X_extrap is not None and dataset.y_extrap is not None:
                y_pred_extrap = candidate.model_callable(dataset.X_extrap)
                if np.any(np.isnan(y_pred_extrap)) or np.any(np.isinf(y_pred_extrap)):
                    metrics["extrap_nan_or_inf"] = 1.0
                    metrics["extrap_rmse"] = 999.0
                else:
                    metrics["extrap_rmse"] = float(np.sqrt(np.mean((dataset.y_extrap - y_pred_extrap) ** 2)))
        except Exception:  # noqa: BLE001 - deliberate catch-all: best-effort probe path reports or skips the failure, never crashes
            metrics["execution_error"] = 1.0
            metrics["holdout_rmse"] = 999.0
            metrics["in_sample_r2"] = -999.0

        return metrics

    def verify_invariant(
        self,
        invariant: DeclaredInvariant,
        candidate: AgnosticCandidateSpecification,
        dataset: AgnosticProblemDataset,
    ) -> InvariantVerificationResult:
        """Verify an individual declared invariant against the candidate and dataset."""
        if invariant.evaluator is not None:
            passed, obs_val, msg = invariant.evaluator(candidate, dataset)
            return InvariantVerificationResult(
                name=invariant.name,
                kind=invariant.kind,
                passed=passed,
                observed_metric=obs_val,
                tolerance=invariant.tolerance,
                diagnostic_message=msg,
            )

        # Standard built-in generic invariant evaluators
        if candidate.model_callable is None:
            key = f"invariant_{invariant.name}"
            if key in candidate.metadata:
                meta_passed = bool(candidate.metadata[key])
                return InvariantVerificationResult(
                    name=invariant.name,
                    kind=invariant.kind,
                    passed=meta_passed,
                    observed_metric=0.0 if meta_passed else 1.0,
                    tolerance=invariant.tolerance,
                    diagnostic_message="Verified from candidate specification metadata" if meta_passed else "Violation recorded in specification",
                )
            return InvariantVerificationResult(
                name=invariant.name,
                kind=invariant.kind,
                passed=False,
                observed_metric=0.0,
                tolerance=invariant.tolerance,
                diagnostic_message="Unverified: no evaluator, no model callable, and no recorded result",
            )

        f = candidate.model_callable

        if invariant.kind == InvariantKind.BOUND_POSITIVITY:
            y_pred = f(dataset.X_train)
            min_val = float(np.min(y_pred))
            lower_bound = invariant.target_value if invariant.target_value is not None else 0.0
            passed = min_val >= (lower_bound - invariant.tolerance)
            msg = f"Minimum observed value {min_val:.5f} (bound: >= {lower_bound})"
            return InvariantVerificationResult(invariant.name, invariant.kind, passed, min_val, invariant.tolerance, msg)

        if invariant.kind == InvariantKind.MONOTONICITY:
            y_pred = f(dataset.X_train)
            diffs = np.diff(y_pred)
            min_slope = float(np.min(diffs)) if len(diffs) > 0 else 0.0
            passed = min_slope >= -invariant.tolerance
            msg = f"Minimum step difference {min_slope:.5f} (required: >= {-invariant.tolerance})"
            return InvariantVerificationResult(invariant.name, invariant.kind, passed, min_slope, invariant.tolerance, msg)

        if invariant.kind == InvariantKind.LIPSCHITZ_SMOOTHNESS:
            # Evaluate perturbation amplification
            rng = np.random.RandomState(42)
            noise = rng.normal(0, 1e-4, size=dataset.X_train.shape)
            y_clean = f(dataset.X_train)
            y_perturbed = f(dataset.X_train + noise)
            dx = np.linalg.norm(noise) + 1e-12
            dy = np.linalg.norm(y_perturbed - y_clean)
            lip_ratio = float(dy / dx)
            max_lip = invariant.target_value if invariant.target_value is not None else 100.0
            passed = lip_ratio <= (max_lip + invariant.tolerance)
            msg = f"Observed empirical Lipschitz ratio {lip_ratio:.3f} (max: {max_lip})"
            return InvariantVerificationResult(invariant.name, invariant.kind, passed, lip_ratio, invariant.tolerance, msg)

        if invariant.kind == InvariantKind.ASYMPTOTIC_LIMIT:
            # Test large input horizon
            X_large = dataset.X_train * 100.0
            y_large = f(X_large)
            has_nan_inf = np.any(np.isnan(y_large)) or np.any(np.isinf(y_large))
            max_abs = float(np.max(np.abs(y_large))) if not has_nan_inf else 1e9
            ceiling = invariant.target_value if invariant.target_value is not None else 1e6
            passed = (not has_nan_inf) and (max_abs <= ceiling)
            msg = f"Asymptotic max absolute value {max_abs:.3f} (ceiling: {ceiling})"
            return InvariantVerificationResult(invariant.name, invariant.kind, passed, max_abs, invariant.tolerance, msg)

        return InvariantVerificationResult(
            invariant.name,
            invariant.kind,
            False,
            0.0,
            invariant.tolerance,
            "Unverified: no evaluator and no built-in check for this invariant kind",
        )

    def evaluate_structural_rivals(
        self,
        candidate: AgnosticCandidateSpecification,
        dataset: AgnosticProblemDataset,
        cand_rmse: float,
    ) -> tuple[bool, float, list[str]]:
        """Check if candidate can reject structural rivals within resolution floor tau_epi."""
        tau = dataset.tau_epi if dataset.tau_epi is not None else self.default_tau_epi
        if not dataset.rival_models:
            return False, 999.0, []

        indistinguishable: list[str] = []
        min_delta = 999.0

        for rival in dataset.rival_models:
            if rival.candidate_id == candidate.candidate_id:
                continue
            r_metrics = self.evaluate_empirical_fit(rival, dataset)
            r_rmse = r_metrics.get("holdout_rmse", 999.0)
            delta = abs(r_rmse - cand_rmse)
            if delta < min_delta:
                min_delta = delta
            if delta <= tau:
                indistinguishable.append(rival.candidate_id)

        all_rejected = len(indistinguishable) == 0
        return all_rejected, min_delta, indistinguishable

    def synthesize_evidence_bundle(
        self,
        candidate: AgnosticCandidateSpecification,
        dataset: AgnosticProblemDataset,
    ) -> tuple[EvidenceBundle, list[InvariantVerificationResult], list[InvariantVerificationResult]]:
        """Build an EvidenceBundle from empirical fit, structural rivals, and declared invariants."""
        metrics = self.evaluate_empirical_fit(candidate, dataset)
        bundle = EvidenceBundle(metrics=dict(metrics))

        # In-sample convergence check
        r2 = metrics.get("in_sample_r2", -999.0)
        train_rmse = metrics.get("train_rmse", 999.0)
        if r2 >= 0.85 and train_rmse < 0.5:
            bundle.add(EvidenceRequirement.IN_SAMPLE_CONVERGENCE)
        else:
            bundle.failures.append(f"insufficient_in_sample_convergence_r2_{r2:.3f}")

        # Out-of-sample holdout check
        holdout_rmse = metrics.get("holdout_rmse", 999.0)
        if holdout_rmse < 0.25 and not metrics.get("holdout_nan_or_inf", 0.0):
            bundle.add(EvidenceRequirement.OUT_OF_SAMPLE_HOLDOUT)
        else:
            bundle.failures.append(f"excessive_holdout_rmse_{holdout_rmse:.4f}")

        # Verify declared invariants
        verified_invs: list[InvariantVerificationResult] = []
        failed_invs: list[InvariantVerificationResult] = []

        for inv in candidate.declared_invariants:
            res = self.verify_invariant(inv, candidate, dataset)
            if res.passed:
                verified_invs.append(res)
                # Map specific invariant kinds to standard EvidenceRequirements
                if inv.kind == InvariantKind.CONSERVATION_LAW:
                    bundle.add(EvidenceRequirement.PHYSICAL_CONSERVATION)
                elif inv.kind == InvariantKind.BOUND_POSITIVITY:
                    bundle.add(EvidenceRequirement.GEOMETRIC_FEASIBILITY)
                elif inv.kind == InvariantKind.LIPSCHITZ_SMOOTHNESS:
                    bundle.add(EvidenceRequirement.NUMERICAL_CFL_STABILITY)
                elif inv.kind == InvariantKind.MONOTONICITY:
                    bundle.add(EvidenceRequirement.ENTROPY_CONDITION)
                elif inv.kind == InvariantKind.INTERVENTIONAL_INVARIANCE:
                    bundle.add(EvidenceRequirement.INTERVENTIONAL_INVARIANCE)
            else:
                failed_invs.append(res)
                bundle.failures.append(f"invariant_violated_{inv.name}_{res.diagnostic_message}")

        # Check structural rivals for C3+
        rivals_rejected, min_delta, indist_ids = self.evaluate_structural_rivals(candidate, dataset, holdout_rmse)
        bundle.metrics["delta_rival"] = float(min_delta)
        if rivals_rejected:
            bundle.add(EvidenceRequirement.EQUIVALENCE_RIVALS_REJECTED)
        else:
            bundle.failures.append(f"indistinguishable_rivals_{','.join(indist_ids)}")

        # Check extrapolation holdout for C5+
        if dataset.X_extrap is not None:
            extrap_rmse = metrics.get("extrap_rmse", 999.0)
            if extrap_rmse < 0.35 and not metrics.get("extrap_nan_or_inf", 0.0):
                bundle.add(EvidenceRequirement.REGIME_TRANSFER_EXTRAPOLATION)
            else:
                bundle.failures.append(f"extrapolation_divergence_rmse_{extrap_rmse:.4f}")

        # Dual bound / formal certificate if provided in metadata
        if candidate.metadata.get("dual_bound_certified", False):
            bundle.add(EvidenceRequirement.DUAL_BOUND_CERTIFICATE)

        return bundle, verified_invs, failed_invs

    def adjudicate(
        self,
        candidate: AgnosticCandidateSpecification,
        dataset: AgnosticProblemDataset,
        run_eedr: bool = True,
    ) -> AgnosticAdjudicationOutcome:
        """Adjudicate candidate under the Agnostic Adapter against the Claim Ladder and EEDR."""
        bundle, verified_invs, failed_invs = self.synthesize_evidence_bundle(candidate, dataset)

        # 1. Direct adjudication against frozen Claim Ladder
        adj = adjudicate_scientific_claim(
            candidate_id=candidate.candidate_id,
            proposed_claim=candidate.proposed_claim,
            evidence=bundle,
            domain="agnostic",
        )

        final_verdict = adj.verdict
        justified_claim = adj.justified_claim
        final_status = "PROMOTED_ORIGINAL" if adj.verdict == EpistemicAdjudicationVerdict.PROMOTED else "REJECTED"
        eedr_res: EEDRAdjudicationResult | None = None

        # 2. Bounded EEDR loop if failing/unresolved
        if run_eedr and adj.verdict != EpistemicAdjudicationVerdict.PROMOTED:
            eedr_res = execute_eedr_loop(
                candidate_id=candidate.candidate_id,
                proposed_claim=candidate.proposed_claim,
                evidence=bundle,
                domain="agnostic",
                max_repair_budget=self.max_repair_budget,
            )
            final_status = eedr_res.final_status
            justified_claim = eedr_res.final_justified_claim
            if eedr_res.final_status == "PROMOTED_REPAIRED":
                final_verdict = EpistemicAdjudicationVerdict.PROMOTED

        promoted = (final_verdict == EpistemicAdjudicationVerdict.PROMOTED)

        return AgnosticAdjudicationOutcome(
            candidate_id=candidate.candidate_id,
            proposed_claim=candidate.proposed_claim,
            justified_claim=justified_claim,
            verdict=final_verdict,
            promoted=promoted,
            final_status=final_status,
            verified_invariants=verified_invs,
            failed_invariants=failed_invs,
            empirical_metrics=bundle.metrics,
            evidence_bundle=bundle,
            eedr_result=eedr_res,
        )
