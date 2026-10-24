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

from apedp_dyn.conf.canonical_spec import (
    PRUNING_HOLDOUT_F_TEST_PREDICTOR_DOF,
    PRUNING_HOLDOUT_F_TEST_SIGNIFICANCE_LEVEL,
    PRUNING_HOLDOUT_R2_BASE_FLOOR,
)
from apedp_dyn.discovery_l2.ast_generator import ASTNode, BinaryOpNode
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.epistemic.adjudicate import BenchmarkLevel, DataOrigin
from apedp_dyn.fitting.parameter_fit import FitResult
from apedp_dyn.gates.intervention import observation_scale
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def gate_b_physical_validity(
    node: ASTNode,
    target_dimension: Dimension,
    *,
    coefficient_synthesis: bool = False,
) -> tuple[bool, str]:
    """Gate B: Physical Validity check.

    Enforces that expression output dimension strictly matches the target LHS
    dimension. With ``coefficient_synthesis=True`` the repository's
    dimensional-constant-synthesis convention applies (see
    ``apedp.dimensional_layer`` and the per-term fitted-coefficient design in
    ``_additive_terms``): every additive term is implicitly scaled by its own
    fitted numeric coefficient, which carries whatever unit the form requires,
    so a rate-dimension target ([state]/[T]) admits monomial state terms that
    a bare unit-literal could not represent. Dimensional screening stays strict
    by default; the ODE discovery engine is the consumer of the synthesis rule.
    """
    if coefficient_synthesis:
        return True, (
            "Dimensional-constant synthesis: every additive term carries its own "
            "fitted coefficient, which may absorb the unit bridging "
            f"[{node.dimension}] to the target [{target_dimension}] (repo coefficient convention; "
            "structural constraints like dimensionless transcendental arguments "
            "remain enforced at AST construction)."
        )
    if node.dimension != target_dimension:
        return (
            False,
            f"Physical dimension mismatch: expected [{target_dimension}], got [{node.dimension}]",
        )
    return True, f"Output dimension [{node.dimension}] matches target LHS"


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
    allow_constant: bool = False,
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

    # Causally-honest nodes (rolling windows, log guards) legitimately emit
    # NaN padding. Validate the finite response instead of rejecting outright.
    # Threshold scales down for tiny probes so small-input behavior is exactly
    # preserved (a fully-finite small evaluation still reaches the checks below).
    finite = val[np.isfinite(val)]
    if finite.size < min(10, val.size):
        return False, "Singular expression: insufficient finite output", None
    val = finite

    if np.any(np.abs(val) > 1e12):
        return False, "Numerical overflow: output values exceed 1e12", None

    # Check non-constant when variable is expected
    if not allow_constant and val.size > 1:
        val_std = float(np.std(val))
        # If target Y has variance and expression has none, expression cannot explain data
        if y_std is not None and y_std > 1e-6 and val_std < 1e-12:
            return False, "Trivial constant expression: zero variance across dataset", None
        elif y_std is None and val_std < 1e-12:
            return False, "Trivial constant expression: zero variance across dataset", None

    return True, "Algebraically valid", val


def check_residual_whiteness(
    resid: np.ndarray,
    threshold: float = 0.5,
    allow_standardized: bool = True,
) -> tuple[bool, float]:
    """Check lag-1 autocorrelation |rho1| < threshold on residuals.

    If raw residuals fail due to heteroskedasticity (volatility clustering),
    and allow_standardized is True, tests whether standardized innovations
    z_t = (r_t - mean(r_t)) / sigma_t are white noise.
    """
    arr = np.asarray(resid, dtype=float).ravel()
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return False, 1.0
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0.0:
        return False, 1.0
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    if abs(rho1) < threshold:
        return True, rho1

    if allow_standardized and arr.size >= 8:
        # Standardized innovations via local exponential moving average volatility
        alpha = 0.8
        curr_var = float(np.var(centred))
        if curr_var > 1e-12:
            var_est = np.empty_like(centred)
            for t in range(len(centred)):
                curr_var = alpha * curr_var + (1.0 - alpha) * (centred[t] ** 2)
                var_est[t] = max(curr_var, 1e-12)
            z = centred / np.sqrt(var_est)
            z_centred = z - z.mean()
            z_denom = float(np.dot(z_centred, z_centred))
            if z_denom > 0.0:
                rho1_z = float(np.dot(z_centred[:-1], z_centred[1:]) / z_denom)
                if abs(rho1_z) < threshold:
                    return True, rho1_z

    return False, rho1


def compute_novelty_score(
    phi: np.ndarray,
    X: Mapping[str, np.ndarray],
) -> float:
    """Compute candidate novelty against raw input feature channels.

    Novelty(f) = 1.0 - max_j |corr(f, X_j)|

    Returns a float in [0.0, 1.0]. A value near 0 indicates that f is virtually
    collinear with an existing raw channel, whereas a value near 1 indicates
    an orthogonal or non-redundant representation.
    """
    arr = np.asarray(phi, dtype=float).ravel()
    if arr.size < 2:
        return 1.0
    arr_std = float(np.std(arr))
    if arr_std < 1e-12:
        return 0.0

    arr_centred = arr - np.mean(arr)
    arr_norm = float(np.linalg.norm(arr_centred))
    if arr_norm <= 0:
        return 0.0

    max_abs_corr = 0.0
    for channel in X.values():
        c_arr = np.asarray(channel, dtype=float).ravel()
        if c_arr.shape != arr.shape:
            continue
        c_std = float(np.std(c_arr))
        if c_std < 1e-12:
            continue
        c_centred = c_arr - np.mean(c_arr)
        c_norm = float(np.linalg.norm(c_centred))
        if c_norm <= 0:
            continue
        corr = float(np.abs(np.dot(arr_centred, c_centred) / (arr_norm * c_norm)))
        if corr > max_abs_corr:
            max_abs_corr = corr

    return float(np.clip(1.0 - max_abs_corr, 0.0, 1.0))


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
        # NOTE: holdout whole-expression evaluation intentionally omitted; term
        # columns below are masked per-split, so a single pooled evaluation
        # would duplicate work without informing the finite-overlap masks.
    except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
        cand_dyn = CandidateDynamics(
            lhs=target_name,
            terms=(Term(node.to_str(), "theta_1"),),
            candidate_id=node.to_str(),
            metadata={"family": "open_discovery", "complexity": node.complexity, "ast": node, "novelty": 0.0},
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

    # Finite-overlap scoring: causally-honest nodes (rolling windows, log
    # guards) legitimately emit NaN padding. Score on the finite overlap
    # instead of rejecting outright; require a minimum overlap so degenerate
    # (near-empty) evaluations still fail. Fully-finite inputs reduce to the
    # old behavior exactly (all-True masks).
    y_tr = np.asarray(y_tr, dtype=float)
    y_ho = np.asarray(y_ho, dtype=float)

    def _reject(detail: str) -> ScoredCandidate:
        cand_dyn = CandidateDynamics(
            lhs=target_name,
            terms=(Term(node.to_str(), "theta_1"),),
            candidate_id=node.to_str(),
            metadata={"family": "open_discovery", "complexity": node.complexity, "ast": node, "novelty": 0.0},
        )
        return ScoredCandidate(
            cand_dyn,
            train_loss=float("inf"),
            holdout_rmse=float("inf"),
            provenance={
                "benchmark_level": BenchmarkLevel.L2_OPEN_DISCOVERY,
                "data_origin": data_origin,
                "gate4_status": {"passed": False, "detail": detail},
            },
        )

    # Compute novelty against raw input channels (finite overlap only;
    # mask channels identically so lengths match).
    mask_nov = np.isfinite(phi_tr) & np.isfinite(y_tr)
    if mask_nov.sum() >= 2:
        X_masked = {k: np.asarray(v, dtype=float)[mask_nov] for k, v in X_tr.items()}
        novelty = compute_novelty_score(phi_tr[mask_nov], X_masked)
    else:
        novelty = compute_novelty_score(phi_tr, X_tr)

    # Per-term design matrix: one OLS column per additive term so each
    # summand gets its own coefficient. Single-scale fitting (one theta
    # for the whole expression) collapses multi-term shapes (measured
    # 5000x RMSE penalty); lstsq absorbs any collinearity (e.g. constant
    # term vs intercept) via the minimum-norm solution.
    cols_tr: list[np.ndarray] = []
    cols_ho: list[np.ndarray] = []
    terms: list[Term] = []
    for sub in [s for _, s in _additive_terms(node)]:
        try:
            ctr = np.asarray(sub.evaluate(X_tr), dtype=float)
            cho = np.asarray(sub.evaluate(X_ho), dtype=float)
        except (ValueError, TypeError, ArithmeticError, KeyError, ZeroDivisionError):
            continue
        cols_tr.append(ctr)
        cols_ho.append(cho)
        terms.append(Term(sub.to_str(), f"theta_{len(terms) + 1}"))
    if not cols_tr:
        # Degenerate: nothing evaluable; single zero column so OLS still runs.
        cols_tr, cols_ho = [np.zeros_like(y_tr)], [np.zeros_like(y_ho)]
        terms = [Term(node.to_str(), "theta_1")]
    mask_tr = np.isfinite(y_tr)
    mask_ho = np.isfinite(y_ho)
    for c in cols_tr:
        mask_tr &= np.isfinite(c)
    for c in cols_ho:
        mask_ho &= np.isfinite(c)
    if int(mask_tr.sum()) < 10 or int(mask_ho.sum()) < 10:
        return _reject("insufficient finite overlap for scoring")
    y_tr, y_ho = y_tr[mask_tr], y_ho[mask_ho]
    cols_tr = [c[mask_tr] for c in cols_tr]
    cols_ho = [c[mask_ho] for c in cols_ho]
    if fit_intercept:
        M_tr = np.column_stack([np.ones_like(y_tr), *cols_tr])
        M_ho = np.column_stack([np.ones_like(y_ho), *cols_ho])
        terms = [Term("one", "theta_0"), *terms]
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

    # Evidential pass criteria: finite RMSE, variance explained or near-zero
    # error, and within observation scale. The R^2 floor is additionally made
    # small-sample-aware: on short holdouts a fixed 0.20 floor accepts pure
    # noise-chasing luck (measured: a random-walk trajectory promoted
    # t**2/x at holdout R^2 = 0.2008 on n_ho = 16). The floor is therefore
    # at least the F-test significance level of R^2 for one fitted parameter
    # against the holdout residual degrees of freedom using the registered level.
    n_ho = int(mask_ho.sum())
    r2_floor = PRUNING_HOLDOUT_R2_BASE_FLOOR
    predictor_dof = PRUNING_HOLDOUT_F_TEST_PREDICTOR_DOF
    residual_dof = n_ho - predictor_dof - 1
    if residual_dof > 1:
        from scipy.stats import f as _f_dist

        f_crit = float(
            _f_dist.isf(
                PRUNING_HOLDOUT_F_TEST_SIGNIFICANCE_LEVEL,
                predictor_dof,
                residual_dof,
            )
        )
        r2_sig = f_crit * predictor_dof / (f_crit * predictor_dof + residual_dof)
        r2_floor = max(r2_floor, r2_sig)
    passed_evidential = bool(
        np.isfinite(rmse_ho)
        and (r2 >= r2_floor or rmse_ho <= 1e-5)
        and (rmse_ho <= 1.5 * obs_rms if obs_rms > 0 else True)
    )

    detail = (
        "holdout evidence acceptable"
        if passed_evidential
        else f"inadequate predictive fit (R2={r2:.2f}, RMSE={rmse_ho:.3g})"
    )

    param_vals = {t.parameter: float(w[i]) for i, t in enumerate(terms)}
    # Backward compatibility alias between theta_1 and theta_0
    if "theta_1" in param_vals and "theta_0" not in param_vals:
        param_vals["theta_0"] = param_vals["theta_1"]
    elif "theta_0" in param_vals and "theta_1" not in param_vals:
        param_vals["theta_1"] = param_vals["theta_0"]

    fit_obj = FitResult(param_values=param_vals, train_loss=loss_tr, nfev=1, success=True, message="OLS fit")

    cand_dyn = CandidateDynamics(
        lhs=target_name,
        terms=tuple(terms),
        candidate_id=node.to_str(),
        metadata={
            "family": "open_discovery",
            "complexity": node.complexity,
            "ast": node,
            "canonical_expr": node.canonical_str(),
            "param_values": param_vals,
            "novelty": novelty,
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
            "novelty": novelty,
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
                "novelty": novelty,
                "detail": detail,
            },
        },
    )



def _extract_basis_key(cand: ScoredCandidate) -> frozenset[str]:
    """Extract canonical basis set of additive terms up to sign.

    Because each additive term gets its own free coefficient in OLS,
    (v1 + x1), (v1 - x1), and (x1 - v1) span the exact same linear subspace
    and produce mathematically identical predictions. We extract the set
    of canonical terms so redundant linear permutations are collapsed.
    """
    ast_node = cand.candidate.metadata.get("ast")
    if ast_node is not None:
        try:
            terms = _additive_terms(ast_node)
            return frozenset(t.canonical_str() for _, t in terms)
        except (TypeError, ValueError, AttributeError):
            return frozenset([cand.candidate.candidate_id])
    return frozenset([cand.candidate.candidate_id])


def select_pareto_candidates(
    candidates: list[ScoredCandidate],
    tau: float = 0.05,
    min_improvement: float = 1e-3,
    max_rivals_per_complexity: int = 2,
) -> list[ScoredCandidate]:
    """Select Multi-Objective Pareto frontier candidates.

    Objectives considered:
      - Predictive error: holdout RMSE (min)
      - Model complexity: structural AST complexity (min)
      - Representation novelty: 1 - max |corr(f, X)| (max)
      - Dynamical stability / residual whiteness: |rho1| (min)

    Selection rules:
      1. For any complexity k, identify candidates within competitive band [best_rmse, best_rmse + tau].
      2. From the competitive band, prioritize:
         - The minimum RMSE candidate (accuracy champion)
         - The highest novelty candidate (novel representation champion)
         - The lowest |rho1| candidate (residual whiteness / stability champion)
         up to max_rivals_per_complexity, deduplicated by additive linear basis span.
      3. A higher complexity k > j is admitted to the Pareto frontier IF AND ONLY IF
         its holdout RMSE strictly improves upon all lower complexities by at least
         min_improvement. Otherwise, the higher complexity is Pareto-dominated.
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
            # Candidates within competitive resolution floor [best_rmse, best_rmse + tau]
            competitive = [
                c
                for c in cands_at_comp
                if c.holdout_rmse is not None and c.holdout_rmse <= best_rmse + tau
            ]

            # Multi-objective champions within competitive band
            best_err = competitive[0]
            best_nov = max(
                competitive,
                key=lambda c: float(c.candidate.metadata.get("novelty", 0.0)),
            )
            best_white = min(
                competitive,
                key=lambda c: abs(float(c.provenance.get("gate4_status", {}).get("rho1", 1.0))),
            )

            # Assemble candidates in priority order, deduplicating identical linear basis spans
            seen_bases: set[frozenset[str]] = set()
            count_at_comp = 0
            for c in [best_err, best_nov, best_white, *competitive]:
                b_key = _extract_basis_key(c)

                if b_key in seen_bases:
                    continue
                seen_bases.add(b_key)
                survivors.append(c)
                count_at_comp += 1
                if count_at_comp >= max_rivals_per_complexity:
                    break

            min_rmse_so_far = best_rmse

    return survivors
