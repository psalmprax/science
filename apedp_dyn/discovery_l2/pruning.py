"""Three-Stage Pruning Pipeline for L2 Open Discovery.

Implements:
  - Gate A (Algebraic Validity): expressions are well-formed, non-singular, non-constant.
  - Gate B (Physical Validity): expression output dimension matches target LHS dimension.
  - Gate C (Evidential Validity): parameter fitting on train split, holdout RMSE evaluation,
    residual whiteness check, and Pareto complexity selection.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from apedp_dyn.discovery_l2.ast_generator import ASTNode, BinaryOpNode
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.epistemic.adjudicate import BenchmarkLevel, DataOrigin
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.gates.intervention import observation_scale
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _additive_terms(node: ASTNode, sign: float = 1.0) -> list[tuple[float, ASTNode]]:
    """Split an expression into top-level additive terms with signs.

    (+ a b) -> [(+1, a), (+1, b)]; (- a b) -> [(+1, a), (-1, b)], with
    the sign threading through nested sums, so -(a+b) -> [(-1, a),
    (-1, b)]. Anything else (products, powers, transcendentals, leaves)
    is a single term. Each term gets its OWN fitted coefficient
    downstream: fitting one global scale for a multi-term shape
    collapses expressiveness (measured 5000x RMSE penalty on logistic
    -x^2+x), so the design matrix carries one column per additive term.
    """
    if isinstance(node, BinaryOpNode) and node.op in ("+", "-"):
        s_right = sign if node.op == "+" else -sign
        return (_additive_terms(node.left, sign)
                + _additive_terms(node.right, s_right))
    return [(sign, node)]


def gate_a_algebraic_validity(
    node: ASTNode,
    X: Mapping[str, np.ndarray],
    y_std: float | None = None,
) -> tuple[bool, str, np.ndarray | None]:
    """Gate A: Algebraic Validity check.

    Verifies:
      1. Expression evaluates without runtime error.
      2. Output values are strictly finite (no NaN, Inf).
      3. Dynamic range is bounded (|val| <= 1e12).
      4. Expression is non-constant across inputs when variable is expected.
    """
    try:
        val = node.evaluate(X)
    except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError) as exc:
        return False, f"Evaluation error: {exc}", None

    val = np.asarray(val, dtype=float)
    if val.size == 0:
        return False, "Evaluation yielded empty array", None

    if not np.all(np.isfinite(val)):
        return False, "Singular expression: output contains NaN or Inf", None

    if np.any(np.abs(val) > 1e12):
        return False, "Numerical overflow: output values exceed 1e12", None

    # Check non-constant when variable is expected
    if val.size > 1:
        val_std = float(np.std(val))
        # If target Y has variance and expression has none, expression cannot explain data
        if y_std is not None and y_std > 1e-6 and val_std < 1e-12:
            return False, "Trivial constant expression: zero variance across dataset", None
        elif y_std is None and val_std < 1e-12:
            return False, "Trivial constant expression: zero variance across dataset", None

    return True, "Algebraically valid", val


def gate_b_physical_validity(
    node: ASTNode,
    target_dimension: Dimension,
) -> tuple[bool, str]:
    """Gate B: Physical Validity check.

    Enforces that expression output dimension strictly matches the target LHS dimension.
    """
    if node.dimension != target_dimension:
        return (
            False,
            f"Physical dimension mismatch: expected [{target_dimension}], got [{node.dimension}]",
        )
    return True, f"Output dimension [{node.dimension}] matches target LHS"


def check_residual_whiteness(resid: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
    """Check lag-1 autocorrelation |rho1| < threshold on residuals."""
    arr = np.asarray(resid, dtype=float).ravel()
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return False, 1.0
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return False, 1.0
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    return bool(abs(rho1) < threshold), rho1


def fit_and_score_candidate(
    node: ASTNode,
    X_tr: Mapping[str, np.ndarray],
    y_tr: np.ndarray,
    X_ho: Mapping[str, np.ndarray],
    y_ho: np.ndarray,
    target_name: str,
    noise_sigma: float | None = None,
    data_origin: str = DataOrigin.SIMULATION,
    fit_intercept: bool = False,
) -> ScoredCandidate:
    """Fit AST candidate parameters on train split and score on held-out split (Gate C)."""
    try:
        phi_tr = np.asarray(node.evaluate(X_tr), dtype=float)
        phi_ho = np.asarray(node.evaluate(X_ho), dtype=float)
    except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
        cand_dyn = CandidateDynamics(
            lhs=target_name,
            terms=(Term(node.to_str(), "theta_1"),),
            candidate_id=node.to_str(),
            metadata={"family": "open_discovery", "complexity": node.complexity, "ast": node},
        )
        return ScoredCandidate(
            cand_dyn,
            train_loss=float("inf"),
            holdout_rmse=float("inf"),
            provenance={
                "benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY,
                "data_origin": data_origin,
                "gate4_status": {"passed": False, "detail": "evaluation failed during scoring"},
            },
        )

    if not (np.all(np.isfinite(phi_tr)) and np.all(np.isfinite(phi_ho))):
        cand_dyn = CandidateDynamics(
            lhs=target_name,
            terms=(Term(node.to_str(), "theta_1"),),
            candidate_id=node.to_str(),
            metadata={"family": "open_discovery", "complexity": node.complexity, "ast": node},
        )
        return ScoredCandidate(
            cand_dyn,
            train_loss=float("inf"),
            holdout_rmse=float("inf"),
            provenance={
                "benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY,
                "data_origin": data_origin,
                "gate4_status": {"passed": False, "detail": "non-finite values in evaluation"},
            },
        )

    # Per-term design matrix: one OLS column per additive term so each
    # summand gets its own coefficient. Single-scale fitting (one theta
    # for the whole expression) collapses multi-term shapes (measured
    # 5000x RMSE penalty); lstsq absorbs any collinearity (e.g. constant
    # term vs intercept) via the minimum-norm solution.
    cols_tr, cols_ho, terms = [], [], []
    for sub in [s for _, s in _additive_terms(node)]:
        try:
            ctr = np.asarray(sub.evaluate(X_tr), dtype=float)
            cho = np.asarray(sub.evaluate(X_ho), dtype=float)
        except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
            continue
        if not (np.all(np.isfinite(ctr)) and np.all(np.isfinite(cho))):
            continue
        cols_tr.append(ctr)
        cols_ho.append(cho)
        terms.append(Term(sub.to_str(), f"theta_{len(terms)}"))
    if not cols_tr:
        # Degenerate: nothing evaluable; single zero column so OLS still runs.
        cols_tr, cols_ho = [np.zeros_like(y_tr)], [np.zeros_like(y_ho)]
        terms = [Term(node.to_str(), "theta_0")]
    if fit_intercept:
        M_tr = np.column_stack([np.ones_like(y_tr), *cols_tr])
        M_ho = np.column_stack([np.ones_like(y_ho), *cols_ho])
        terms = (Term("one", "theta_0"), *tuple(terms))
    else:
        M_tr = np.column_stack(cols_tr)
        M_ho = np.column_stack(cols_ho)

    # OLS parameter fitting on train split
    try:
        w, *_ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    except np.linalg.LinAlgError:
        w = np.zeros(M_tr.shape[1], dtype=float)

    pred_tr = M_tr @ w
    loss_tr = float(np.mean((y_tr - pred_tr) ** 2))

    # Holdout RMSE evaluation
    pred_ho = M_ho @ w
    resid_ho = y_ho - pred_ho
    rmse_ho = float(np.sqrt(np.mean(resid_ho**2)))

    # Residual whiteness check
    is_white, rho1 = check_residual_whiteness(resid_ho, threshold=0.5)

    obs_scale = observation_scale(y_ho)
    obs_rms = float(obs_scale.get("obs_rms", np.sqrt(np.mean(y_ho**2))))
    if noise_sigma is not None:
        noise_est = float(noise_sigma)
    else:
        obs_noise = float(obs_scale.get("noise_sigma", 0.0))
        noise_est = obs_noise if obs_noise > 0.0 else (float(np.std(resid_ho)) if len(resid_ho) > 1 else 0.01)

    # Gate 4 evidential gate: verify holdout adequacy
    ss_tot = float(np.sum((y_ho - np.mean(y_ho)) ** 2))
    ss_res = float(np.sum(resid_ho**2))
    if ss_tot > 1e-14:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 1.0 if rmse_ho < 1e-6 else 0.0

    # Evidential pass criteria: finite RMSE, variance explained or near-zero error, and within observation scale
    passed_evidential = bool(
        np.isfinite(rmse_ho)
        and (r2 >= 0.20 or rmse_ho <= 1e-5)
        and (rmse_ho <= 1.5 * obs_rms if obs_rms > 0 else True)
    )

    detail = (
        "holdout evidence acceptable"
        if passed_evidential
        else f"inadequate predictive fit (R2={r2:.2f}, RMSE={rmse_ho:.3g})"
    )

    param_vals = {t.parameter: float(w[i]) for i, t in enumerate(terms)}
    fit_obj = FitResult(param_values=param_vals, train_loss=loss_tr, nfev=1, success=True, message="OLS fit")

    cand_dyn = CandidateDynamics(
        lhs=target_name,
        terms=terms,
        candidate_id=node.to_str(),
        metadata={
            "family": "open_discovery",
            "complexity": node.complexity,
            "ast": node,
            "canonical_expr": node.canonical_str(),
            "param_values": param_vals,
        },
    )

    return ScoredCandidate(
        cand_dyn,
        fit=fit_obj,
        train_loss=loss_tr,
        holdout_rmse=rmse_ho,
        provenance={
            "benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY,
            "data_origin": data_origin,
            "gate4_status": {
                "passed": passed_evidential,
                "holdout_obs_rms": obs_rms,
                "holdout_noise_sigma": noise_est,
                "holdout_n_samples": len(y_ho),
                "rmse": rmse_ho,
                "holdout_residuals": resid_ho.tolist(),
                "holdout_sq_error": (resid_ho**2).tolist(),
                "r2": r2,
                "rho1": rho1,
                "is_white": is_white,
                "detail": detail,
            },
        },
    )


def select_pareto_candidates(
    candidates: list[ScoredCandidate],
    tau: float = 0.05,
    min_improvement: float = 1e-3,
) -> list[ScoredCandidate]:
    """Select Pareto frontier candidates minimizing (complexity, holdout_rmse).

    Rules:
      1. For any complexity k, keep the best candidate and any near-identical rival
         at that SAME complexity within resolution floor tau (e.g. collinear rivals).
      2. A higher complexity k > j is admitted to the Pareto frontier IF AND ONLY IF
         its holdout RMSE strictly improves upon all lower complexities by at least
         min_improvement. Otherwise, the higher complexity is Pareto-dominated by
         the simpler model.
    """
    if not candidates:
        return []

    valid = [
        c
        for c in candidates
        if c.holdout_rmse is not None
        and np.isfinite(c.holdout_rmse)
        and c.provenance.get("gate4_status", {}).get("passed", False)
    ]
    if not valid:
        valid = [c for c in candidates if c.holdout_rmse is not None and np.isfinite(c.holdout_rmse)]
    if not valid:
        return []

    # Group by complexity
    by_complexity: dict[int, list[ScoredCandidate]] = {}
    for c in valid:
        comp = int(c.candidate.metadata.get("complexity", 1))
        by_complexity.setdefault(comp, []).append(c)

    # For each complexity, sort by holdout_rmse
    for comp in by_complexity:
        by_complexity[comp].sort(
            key=lambda c: c.holdout_rmse if c.holdout_rmse is not None else float("inf")
        )

    sorted_complexities = sorted(by_complexity.keys())
    survivors: list[ScoredCandidate] = []
    min_rmse_so_far = float("inf")

    for comp in sorted_complexities:
        cands_at_comp = by_complexity[comp]
        best_at_comp = cands_at_comp[0]
        best_rmse = (
            best_at_comp.holdout_rmse
            if best_at_comp.holdout_rmse is not None
            else float("inf")
        )

        # Higher complexity is only admitted if it strictly improves upon simpler models
        if best_rmse < min_rmse_so_far - min_improvement:
            # Retain best candidate and any competitive rival at this SAME complexity
            for c in cands_at_comp:
                c_rmse = (
                    c.holdout_rmse if c.holdout_rmse is not None else float("inf")
                )
                if c_rmse <= best_rmse + tau:
                    survivors.append(c)
            min_rmse_so_far = best_rmse

    return survivors
