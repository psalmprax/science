"""Parameter-Recovery Diagnostics for Scientific Equation Discovery.

Disentangles prediction error (MSE, R^2) from parameter recovery accuracy and
structural fidelity. Categorizes synthetic equation discovery results into:
- Category A: Correct structural family with parameter recovery within tolerance.
- Category B: Numerically indistinguishable impostor/surrogate family (high R^2, wrong structure).
- Category C: Representation or optimization failure (poor fit, diverging parameters, or unidentifiable).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from apedp.discovery_types import DictCompatibleResult


class RecoveryCategory(str, Enum):
    """Scientific categorization of discovery outcome against known ground truth."""
    CATEGORY_A = "CATEGORY_A"  # Correct structural family + close parameterization
    CATEGORY_B = "CATEGORY_B"  # Numerically competitive surrogate / impostor family
    CATEGORY_C = "CATEGORY_C"  # Representation or optimization failure


@dataclass(frozen=True)
class ParameterConfidenceInterval(DictCompatibleResult):
    """Estimated standard error and 95% confidence interval for a model parameter."""
    param_name: str
    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95

    @property
    def interval_width(self) -> float:
        return self.ci_upper - self.ci_lower


@dataclass(frozen=True)
class ParameterUncertaintyReport(DictCompatibleResult):
    """Local Gauss-Newton parameter covariance and identifiability estimates for continuous parameters."""
    intervals: dict[str, ParameterConfidenceInterval] = field(default_factory=dict)
    covariance_matrix: list[list[float]] = field(default_factory=list)
    correlation_matrix: list[list[float]] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)
    parameter_identifiability: dict[str, str] = field(default_factory=dict)
    overall_identifiability: str = "STRONG"
    collinear_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    residual_variance: float = 0.0
    degrees_of_freedom: int = 1
    jacobian_condition_number: float = 1.0
    is_well_identified: bool = True
    method_label: str = "Local Gauss-Newton 95% parameter confidence intervals"


@dataclass(frozen=True)
class ParameterError(DictCompatibleResult):
    """Detailed error metric for an individual model parameter."""
    param_name: str
    ground_truth: float
    estimated: float
    absolute_error: float
    relative_error: float


@dataclass(frozen=True)
class ParameterRecoveryReport(DictCompatibleResult):
    """Comprehensive diagnostic report on parameter recovery and structural fidelity."""
    category: RecoveryCategory
    is_family_correct: bool
    recovered_family: str
    target_family: str
    val_mse: float
    ood_mse: float | None = None
    proof_tier: str = "P0"
    param_errors: dict[str, ParameterError] = field(default_factory=dict)
    max_relative_param_error: float = float("inf")
    mean_relative_param_error: float = float("inf")
    verdict_summary: str = ""

    @property
    def diagnostic_vector(self) -> tuple[bool, float, float, float | None, str]:
        """Returns the scientific diagnostic tuple:
        (family_accuracy, max_relative_param_error, val_mse, ood_mse, proof_tier).
        """
        return (
            self.is_family_correct,
            self.max_relative_param_error,
            self.val_mse,
            self.ood_mse,
            self.proof_tier,
        )

    @property
    def is_category_a(self) -> bool:
        return self.category == RecoveryCategory.CATEGORY_A

    @property
    def is_category_b(self) -> bool:
        return self.category == RecoveryCategory.CATEGORY_B

    @property
    def is_category_c(self) -> bool:
        return self.category == RecoveryCategory.CATEGORY_C


def compute_parameter_recovery(
    recovered_family: str,
    target_family: str,
    recovered_params: dict[str, float],
    target_params: dict[str, float],
    val_mse: float,
    ood_mse: float | None = None,
    proof_tier: str = "P0",
    *,
    val_mse_threshold: float = 1e-4,
    param_rel_tolerance: float = 0.05,
    param_abs_tolerance: float = 0.05,
) -> ParameterRecoveryReport:
    """Computes parameter errors and classifies the recovery outcome.

    Parameters:
        recovered_family: Structural family identified by the synthesizer.
        target_family: Canonical ground-truth structural family.
        recovered_params: Dictionary of estimated continuous parameter values.
        target_params: Dictionary of true parameters.
        val_mse: Measured validation mean-squared error.
        val_mse_threshold: Upper limit for considering the model predictive.
        param_rel_tolerance: Maximum relative error to accept as Category A.
        param_abs_tolerance: Maximum absolute error fallback for parameters near zero.

    Returns:
        ParameterRecoveryReport dataclass with category and fine-grained errors.
    """
    # Normalize family strings (e.g. rational_transition vs rational_saturation)
    def normalize_fam(fam: str) -> str:
        f = fam.strip().lower()
        if f in {"rational_transition", "rational_saturation"}:
            return "rational"
        return f

    fam_match = normalize_fam(recovered_family) == normalize_fam(target_family)

    param_errors: dict[str, ParameterError] = {}
    rel_errors: list[float] = []

    for name, p_true in target_params.items():
        if name in recovered_params:
            p_est = float(recovered_params[name])
            abs_err = abs(p_est - p_true)
            denom = abs(p_true) if abs(p_true) > 1e-6 else 1.0
            rel_err = abs_err / denom
            param_errors[name] = ParameterError(
                param_name=name,
                ground_truth=p_true,
                estimated=p_est,
                absolute_error=abs_err,
                relative_error=rel_err,
            )
            rel_errors.append(rel_err)

    max_rel_err = max(rel_errors) if rel_errors else float("inf")
    mean_rel_err = (sum(rel_errors) / len(rel_errors)) if rel_errors else float("inf")

    params_within_tolerance = False
    if rel_errors and len(rel_errors) == len(target_params):
        params_within_tolerance = all(
            err.relative_error <= param_rel_tolerance or err.absolute_error <= param_abs_tolerance
            for err in param_errors.values()
        )

    is_predictive = val_mse <= val_mse_threshold

    if fam_match and is_predictive and params_within_tolerance:
        category = RecoveryCategory.CATEGORY_A
        summary = (
            f"Category A: Exact structural family '{recovered_family}' recovered with high "
            f"parameter fidelity (max relative error = {max_rel_err:.2%}, MSE = {val_mse:.2e})."
        )
    elif is_predictive and not fam_match:
        category = RecoveryCategory.CATEGORY_B
        summary = (
            f"Category B: Empirically predictive surrogate (MSE = {val_mse:.2e}), but resolved to "
            f"impostor family '{recovered_family}' instead of target '{target_family}'."
        )
    elif fam_match and is_predictive and not params_within_tolerance:
        category = RecoveryCategory.CATEGORY_B
        summary = (
            f"Category B: Structural family '{recovered_family}' correctly identified, but continuous "
            f"parameters shifted outside tolerance (max relative error = {max_rel_err:.2%}) due to "
            f"local optimization basin or finite-sample noise."
        )
    else:
        category = RecoveryCategory.CATEGORY_C
        summary = (
            f"Category C: Representation or optimization failure. Family match={fam_match}, "
            f"Val MSE = {val_mse:.2e} exceeding threshold {val_mse_threshold:.2e}."
        )

    return ParameterRecoveryReport(
        category=category,
        is_family_correct=fam_match,
        recovered_family=recovered_family,
        target_family=target_family,
        val_mse=val_mse,
        ood_mse=ood_mse,
        proof_tier=proof_tier,
        param_errors=param_errors,
        max_relative_param_error=max_rel_err,
        mean_relative_param_error=mean_rel_err,
        verdict_summary=summary,
    )


def compute_parameter_uncertainty(
    eval_fn: Callable[[np.ndarray, dict[str, float]], np.ndarray],
    param_values: dict[str, float],
    x_data: np.ndarray,
    y_data: np.ndarray,
    confidence_level: float = 0.95,
    eps: float = 1e-6,
) -> ParameterUncertaintyReport:
    """Computes parameter standard errors and confidence intervals via empirical Hessian/Jacobian."""
    param_names = list(param_values.keys())
    p = len(param_names)
    n = len(y_data)
    dof = max(n - p, 1)

    if p == 0:
        return ParameterUncertaintyReport(degrees_of_freedom=dof)

    y_pred = eval_fn(x_data, param_values)
    residuals = y_data - y_pred
    res_var = float(np.sum(residuals**2) / dof)

    # Finite-difference Jacobian J of model predictions w.r.t parameters: J_ij = df(x_i)/dtheta_j
    J = np.zeros((n, p), dtype=float)
    for j, name in enumerate(param_names):
        theta_val = param_values[name]
        h = max(abs(theta_val) * eps, eps)

        perturbed_plus = dict(param_values)
        perturbed_plus[name] = theta_val + h
        y_plus = eval_fn(x_data, perturbed_plus)

        perturbed_minus = dict(param_values)
        perturbed_minus[name] = theta_val - h
        y_minus = eval_fn(x_data, perturbed_minus)

        J[:, j] = (y_plus - y_minus) / (2.0 * h)

    # Gauss-Newton approximation: J^T J
    JtJ = J.T @ J
    cond_num = float(np.linalg.cond(JtJ)) if JtJ.size > 0 else float("inf")

    # Covariance matrix: sigma^2 * (J^T J)^(-1)
    try:
        if cond_num > 1e12:
            inv_JtJ = np.linalg.pinv(JtJ, rcond=1e-10)
            well_identified = False
        else:
            inv_JtJ = np.linalg.inv(JtJ)
            well_identified = True
        cov = res_var * inv_JtJ
    except np.linalg.LinAlgError:
        cov = np.eye(p) * float("inf")
        well_identified = False

    # Two-sided normal critical value (1.96 for 95%)
    z = 1.95996 if abs(confidence_level - 0.95) < 1e-3 else 2.576

    intervals: dict[str, ParameterConfidenceInterval] = {}
    for j, name in enumerate(param_names):
        var_j = float(cov[j, j])
        se_j = float(np.sqrt(max(var_j, 0.0))) if np.isfinite(var_j) else float("inf")
        est = float(param_values[name])
        ci_l = est - z * se_j if np.isfinite(se_j) else float("-inf")
        ci_u = est + z * se_j if np.isfinite(se_j) else float("inf")
        intervals[name] = ParameterConfidenceInterval(
            param_name=name,
            point_estimate=est,
            standard_error=se_j,
            ci_lower=ci_l,
            ci_upper=ci_u,
            confidence_level=confidence_level,
        )

    # Correlation matrix: rho_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
    corr = np.zeros((p, p), dtype=float)
    std_devs = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    for i in range(p):
        for j in range(p):
            if np.isfinite(cov[i, j]) and np.isfinite(std_devs[i]) and np.isfinite(std_devs[j]):
                corr[i, j] = float(np.clip(cov[i, j] / (std_devs[i] * std_devs[j]), -1.0, 1.0))
            else:
                corr[i, j] = 1.0 if i == j else 0.0

    collinear_pairs: list[tuple[str, str, float]] = []
    for i in range(p):
        for j in range(i + 1, p):
            if abs(corr[i, j]) > 0.85:
                collinear_pairs.append((param_names[i], param_names[j], float(corr[i, j])))

    # Per-parameter identifiability classification
    ident_status: dict[str, str] = {}
    weak_count = 0
    for name in param_names:
        ci = intervals[name]
        est = abs(ci.point_estimate)
        rel_err = ci.standard_error / max(est, 1e-12)
        if not np.isfinite(ci.standard_error) or rel_err > 0.50 or (ci.ci_lower < 0 < ci.ci_upper and est > 1e-4):
            ident_status[name] = "WEAKLY_IDENTIFIED"
            weak_count += 1
        else:
            ident_status[name] = "IDENTIFIED"

    if weak_count == 0:
        overall_ident = "STRONG"
    elif weak_count < p:
        overall_ident = "PARTIAL"
    else:
        overall_ident = "POOR"

    return ParameterUncertaintyReport(
        intervals=intervals,
        covariance_matrix=cov.tolist(),
        correlation_matrix=corr.tolist(),
        parameter_names=param_names,
        parameter_identifiability=ident_status,
        overall_identifiability=overall_ident,
        collinear_pairs=collinear_pairs,
        residual_variance=res_var,
        degrees_of_freedom=dof,
        jacobian_condition_number=cond_num,
        is_well_identified=well_identified,
        method_label="Local Gauss-Newton 95% parameter confidence intervals",
    )


@dataclass(frozen=True)
class StageD2EvidenceReport(DictCompatibleResult):
    """Decomposed, auditable evidence components for Stage D2 model selection.

    Avoids collapsing multiple distinct statistical signals into an opaque single scalar.
    Preserves forensic auditability across likelihood, complexity, extrapolation, and stability.
    """
    model_expression: str
    structural_family: str
    likelihood_evidence: float        # Δ ln L or relative log-likelihood gain
    complexity_penalty: float         # k * ln(n) or structural description length
    ood_penalty: float                # Extrapolation degradation penalty
    parameter_stability: float        # Variance / bootstrap stability of continuous parameters
    family_stability: float           # Empirical recovery frequency of structural family
    bootstrap_selection_frequency: float  # P(model selected across bootstrap trials)
    final_decision: str               # "ACCEPTED", "COMPETITIVE_SURROGATE", "REJECTED_COMPLEXITY", "REJECTED_OOD"
    parameter_uncertainty: dict[str, Any] = field(default_factory=dict)
    physics_evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def make_stage_d2_evidence_report(
    model_expression: str,
    structural_family: str,
    val_mse: float,
    baseline_mse: float,
    n_samples: int,
    complexity: float,
    near_ood_mse: float | None = None,
    condition_number: float = 1.0,
    decision_verdict: str = "ACCEPTED",
    bootstrap_frequency: float = 1.0,
    family_frequency: float = 1.0,
    parameter_uncertainty: dict[str, Any] | None = None,
    physics_evidence: dict[str, Any] | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
) -> StageD2EvidenceReport:
    """Builds an auditable, decomposed Stage D2 evidence report without scalar collapsing."""
    safe_baseline = max(baseline_mse, 1e-12)
    safe_val = max(val_mse, 1e-12)
    likelihood_evidence = float(-0.5 * n_samples * np.log(safe_val / safe_baseline))
    complexity_penalty = float(complexity * np.log(max(n_samples, 2)))

    if near_ood_mse is not None and np.isfinite(near_ood_mse):
        ood_penalty = float(max(near_ood_mse / safe_val, 1.0))
    else:
        ood_penalty = 1.0

    param_stability = float(1.0 / (1.0 + np.log10(max(condition_number, 1.0))))

    return StageD2EvidenceReport(
        model_expression=model_expression,
        structural_family=structural_family,
        likelihood_evidence=likelihood_evidence,
        complexity_penalty=complexity_penalty,
        ood_penalty=ood_penalty,
        parameter_stability=param_stability,
        family_stability=family_frequency,
        bootstrap_selection_frequency=bootstrap_frequency,
        final_decision=decision_verdict,
        parameter_uncertainty=dict(parameter_uncertainty or {}),
        physics_evidence=dict(physics_evidence or {}),
        diagnostics=dict(extra_diagnostics or {}),
    )

