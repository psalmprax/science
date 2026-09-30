"""APEDP Rust Numerical & Validation Kernel Bridge.

Provides a unified interface with:
1. Rust Kernel (`apedp_kernel`) as the primary first-line numerical engine.
2. Pure Python reference engine as the transparent fallback.
3. Dual-backend differential cross-validation (`dual_validate=True`).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger("apedp.rust_bridge")

_RUST_AVAILABLE = False


class _RustKernel(Protocol):
    """Surface of the pyo3 `apedp_kernel` extension used by this bridge.

    The real module is a compiled extension with no stubs; this protocol gives
    mypy an accurate view of the handful of entry points we call.
    """

    def compile(self, expr: str, variables: list[str]) -> Any: ...
    def evaluate_batch(self, compiled: Any, data: list[np.ndarray]) -> np.ndarray: ...
    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, float, float, float, bool]: ...
    def score_candidates_parallel(
        self, compiled: list[Any], data: list[np.ndarray], target: np.ndarray
    ) -> list[Any]: ...
    def probe_singularity(
        self, compiled: Any, bounds: list[Any], num_grid_points: int
    ) -> tuple[bool, bool, float | None, float, str]: ...
    def compute_matrix_conditioning(
        self, cols: list[np.ndarray], rcond_threshold: float
    ) -> tuple[float, float, float, int, bool, list[float]]: ...
    def solve_linear_coefficients(
        self, cols: list[np.ndarray], target: np.ndarray, lambda_reg: float
    ) -> tuple[list[float], float, float, float, float, bool]: ...
    def evaluate_precision_boundary(
        self, separation_delta: float, condition_number: float, precision_bits: int
    ) -> tuple[str, float, float, float, bool]: ...


try:
    import apedp_kernel

    _RUST_AVAILABLE = True
except ImportError:
    apedp_kernel = cast(ModuleType, None)  # only read behind use_rust/is_rust_kernel_available()


def is_rust_kernel_available() -> bool:
    """Returns True if the native apedp_kernel extension is loaded and functional."""
    return _RUST_AVAILABLE


class RustKernelBridge:
    """High-performance bridge between APEDP Python orchestration and Rust numerical kernel."""

    def __init__(self, force_fallback: bool = False, dual_validate: bool = False):
        self.force_fallback = force_fallback
        self.dual_validate = dual_validate
        self.use_rust = _RUST_AVAILABLE and not force_fallback

    def evaluate_expression(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Evaluates expression over variable arrays using Rust first-line, Python fallback.

        Parameters
        ----------
        expr_str : str
            Expression string (e.g. 'x0**2 + sin(x1)').
        variable_names : Sequence[str]
            List of variable names matching the order in data.
        data : Sequence[np.ndarray]
            List of 1D float64 numpy arrays, one per variable.

        Returns
        -------
        np.ndarray
            1D float64 array of evaluated predictions.
        """
        data_f64 = [np.ascontiguousarray(d, dtype=np.float64) for d in data]

        kernel: _RustKernel | ModuleType = apedp_kernel
        if self.use_rust:
            try:
                compiled = kernel.compile(expr_str, list(variable_names))
                rust_out = kernel.evaluate_batch(compiled, data_f64)

                if self.dual_validate:
                    py_out = self._fallback_eval(expr_str, variable_names, data_f64)
                    diff = np.max(np.abs(rust_out - py_out))
                    if diff > 1e-10:
                        logger.warning(
                            "Dual-validation divergence for '%s': max diff = %e", expr_str, diff
                        )

                return rust_out
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust kernel eval failed for '%s' (%s), falling back to Python.", expr_str, e)
                return self._fallback_eval(expr_str, variable_names, data_f64)

        return self._fallback_eval(expr_str, variable_names, data_f64)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        """Computes MSE, MAE, R^2, and max residual with Rust first-line."""
        yt = np.ascontiguousarray(y_true, dtype=np.float64)
        yp = np.ascontiguousarray(y_pred, dtype=np.float64)

        if self.use_rust:
            try:
                mse, mae, r2, max_res, has_nan = cast(Any, apedp_kernel).compute_metrics(yt, yp)
                return {
                    "mse": mse,
                    "mae": mae,
                    "r_squared": r2,
                    "max_residual": max_res,
                    "has_nan_or_inf": has_nan,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust compute_metrics failed (%s), falling back to Python.", e)

        return self._fallback_metrics(yt, yp)

    def score_candidates_batch(
        self,
        candidate_exprs: Sequence[str],
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
        target: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Scores multiple candidate expressions in parallel across CPU cores using Rayon."""
        target_f64 = np.ascontiguousarray(target, dtype=np.float64)
        data_f64 = [np.ascontiguousarray(d, dtype=np.float64) for d in data]

        if self.use_rust:
            try:
                compiled_list = []
                valid_indices = []
                for idx, expr_str in enumerate(candidate_exprs):
                    try:
                        c = cast(Any, apedp_kernel).compile(expr_str, list(variable_names))
                        compiled_list.append(c)
                        valid_indices.append(idx)
                    except Exception:  # noqa: BLE001 - Class B, deliberately
                        # NOT a verdict. The Rust kernel refusing to compile one
                        # expression is a fallback to "this candidate is not
                        # evaluable by the fast path", and the remaining
                        # compiled_list is still scored normally. Nothing here
                        # derives a pass/fail from the failure: the caller
                        # scores what compiled and cannot see a candidate that
                        # was dropped as though it had scored well. The gap
                        # this leaves is observability -- a run cannot prove
                        # which path executed -- and the fix for that is a
                        # counter, not a log line.
                        #
                        # The same file's verdict-deriving paths are Class A and
                        # were fixed; see physics_constraints.py and
                        # singularity_probing.py for the _unverified() shape
                        # used where a check does decide something.
                        logger.debug("Skipping uncompilable candidate %r", expr_str, exc_info=True)

                if compiled_list:
                    raw_scores = cast(Any, apedp_kernel).score_candidates_parallel(
                        compiled_list, data_f64, target_f64
                    )
                    score_map = {}
                    for cid, mse, mae, r2, max_res, has_nan in raw_scores:
                        orig_idx = valid_indices[cid]
                        score_map[orig_idx] = {
                            "candidate_id": orig_idx,
                            "expression": candidate_exprs[orig_idx],
                            "mse": mse,
                            "mae": mae,
                            "r_squared": r2,
                            "max_residual": max_res,
                            "has_nan_or_inf": has_nan,
                        }

                    results = []
                    for idx, expr_str in enumerate(candidate_exprs):
                        if idx in score_map:
                            results.append(score_map[idx])
                        else:
                            # Fallback for this individual expression
                            py_eval = self._fallback_eval(expr_str, variable_names, data_f64)
                            metrics = self._fallback_metrics(target_f64, py_eval)
                            metrics["candidate_id"] = idx
                            metrics["expression"] = expr_str
                            results.append(metrics)
                    return results
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust parallel batch scoring failed (%s), falling back.", e)

        # Fallback scoring sequentially in Python
        results = []
        for idx, expr_str in enumerate(candidate_exprs):
            py_eval = self._fallback_eval(expr_str, variable_names, data_f64)
            metrics = self._fallback_metrics(target_f64, py_eval)
            metrics["candidate_id"] = idx
            metrics["expression"] = expr_str
            results.append(metrics)
        return results

    def probe_singularity(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        domain_bounds: Sequence[tuple[float, float]],
        num_grid_points: int = 2000,
    ) -> dict[str, Any]:
        """Probes for poles and asymptotic blow-ups across domain bounds."""
        if self.use_rust:
            try:
                compiled = cast(Any, apedp_kernel).compile(expr_str, list(variable_names))
                is_sound, has_pole, pole_coord, max_deriv, risk_type = (
                    cast(Any, apedp_kernel).probe_singularity(compiled, list(domain_bounds), num_grid_points)
                )
                return {
                    "is_sound": is_sound,
                    "has_pole": has_pole,
                    "pole_coordinate": pole_coord,
                    "max_derivative": max_deriv,
                    "risk_type": risk_type,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust probe_singularity failed (%s), falling back to Python.", e)

        return self._fallback_singularity_probe(expr_str, variable_names, domain_bounds, num_grid_points)

    def compute_matrix_conditioning(
        self,
        cols: Sequence[np.ndarray],
        rcond_threshold: float = 1e-12,
    ) -> dict[str, Any]:
        """Computes condition number kappa = sigma_max / sigma_min for design matrix columns."""
        col_slices = [np.ascontiguousarray(c, dtype=np.float64) for c in cols]
        if self.use_rust:
            try:
                res = cast(Any, apedp_kernel).compute_matrix_conditioning(
                    col_slices, rcond_threshold
                )
                cond, s_max, s_min, rank, is_well = res[0], res[1], res[2], res[3], res[4]
                s_list = list(res[5]) if len(res) > 5 else [s_max, s_min]
                return {
                    "condition_number": cond,
                    "max_singular_value": s_max,
                    "min_singular_value": s_min,
                    "effective_rank": rank,
                    "is_well_conditioned": is_well,
                    "singular_values": s_list,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust compute_matrix_conditioning failed (%s), falling back.", e)

        return self._fallback_conditioning(col_slices, rcond_threshold)

    def solve_linear_coefficients(
        self,
        basis_cols: Sequence[np.ndarray],
        target: np.ndarray,
        lambda_reg: float = 1e-9,
    ) -> dict[str, Any]:
        """Solves linear parameter least squares min_c ||A c - y||^2 + lambda ||c||^2 via QR."""
        col_slices = [np.ascontiguousarray(c, dtype=np.float64) for c in basis_cols]
        target_f64 = np.ascontiguousarray(target, dtype=np.float64)

        if self.use_rust:
            try:
                coeffs, mse, mae, r2, cond, is_well = cast(Any, apedp_kernel).solve_linear_coefficients(
                    col_slices, target_f64, lambda_reg
                )
                return {
                    "coefficients": np.array(coeffs, dtype=np.float64),
                    "mse": mse,
                    "mae": mae,
                    "r_squared": r2,
                    "condition_number": cond,
                    "is_well_conditioned": is_well,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust solve_linear_coefficients failed (%s), falling back.", e)

        return self._fallback_linear_solver(col_slices, target_f64, lambda_reg)

    def evaluate_precision_boundary(
        self,
        separation_delta: float,
        condition_number: float = 1.0,
        precision_bits: int = 64,
    ) -> dict[str, Any]:
        """Evaluates identifiability boundary under machine precision scaling."""
        if self.use_rust:
            try:
                p_type, eps_mach, eps_star, obs, resolved = (
                    cast(Any, apedp_kernel).evaluate_precision_boundary(separation_delta, condition_number, precision_bits)
                )
                return {
                    "precision_type": p_type,
                    "machine_epsilon": eps_mach,
                    "theoretical_threshold": eps_star,
                    "empirical_transition_observed": obs,
                    "resolved": resolved,
                }
            except Exception as e:  # noqa: BLE001 - best-effort probe must not crash
                logger.debug("Rust evaluate_precision_boundary failed (%s), falling back.", e)

        return self._fallback_precision_boundary(separation_delta, condition_number, precision_bits)

    # ------------------ Fallback Pure-Python Implementations ------------------

    def _fallback_eval(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        data: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Pure-Python / NumPy reference evaluator.

        The expression string is untrusted: it arrives from candidate search,
        CSV ingestion, and the REST layer, so it is parsed with the same
        allowlist policy as every other string-in path (:func:`safe_parse_expr`)
        and compiled with sympy's ``lambdify`` -- never handed to ``eval``.

        Passing ``{"__builtins__": {}}`` to ``eval`` is not a sandbox. Attribute
        traversal still reaches every class in the interpreter, so
        ``(1).__class__.__mro__[1].__subclasses__()`` executes and yields the
        live class table. The allowlist parser rejects that shape up front, and
        ``lambdify`` then emits code containing only the numeric calls the
        expression actually references.
        """
        shape = data[0].shape
        nan_result = lambda: np.full(shape, np.nan, dtype=np.float64)  # noqa: E731

        try:
            expr = safe_parse_expr(expr_str)
        except (TypeError, ValueError) as e:
            logger.debug("Rejected expression %r in Python fallback (%s).", expr_str, e)
            return nan_result()

        # A variable named in the data shadows a same-named mathematical
        # constant, matching the previous hand-built environment.
        env: dict[str, Any] = {"pi": np.pi, "e": np.e}
        for name, arr in zip(variable_names, data, strict=False):
            env[name] = arr

        try:
            argnames = tuple(sorted(str(s) for s in expr.free_symbols))
            func = sp.lambdify(
                argnames,
                expr,
                modules="numpy",
                cse=False,
                docstring_limit=0,
            )
            # ``lambdify`` builds a signature from the expression's free symbols
            # only, so pass exactly those; ``pi``/``e`` are already substituted
            # into the expression body by the parser.
            res = func(*(env[name] for name in argnames))
            if np.isscalar(res):
                return np.full(shape, float(cast(Any, res)), dtype=np.float64)
            arr = np.asarray(res, dtype=np.float64)
            # ``lambdify`` broadcasts constants; broadcast to the caller's shape
            # so downstream ``invalid_mask`` arithmetic stays aligned.
            if arr.shape != shape:
                arr = np.broadcast_to(arr, shape)
            return np.ascontiguousarray(arr)
        except Exception:  # noqa: BLE001 - best-effort probe must not crash
            return nan_result()

    def _fallback_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """Pure-Python / NumPy reference metrics."""
        if len(y_true) == 0 or len(y_true) != len(y_pred):
            return {
                "mse": float("nan"),
                "mae": float("nan"),
                "r_squared": float("nan"),
                "max_residual": float("nan"),
                "has_nan_or_inf": True,
            }

        residuals = y_true - y_pred
        has_nan = bool(np.isnan(residuals).any() or np.isinf(residuals).any())
        if has_nan:
            return {
                "mse": float("nan"),
                "mae": float("nan"),
                "r_squared": float("nan"),
                "max_residual": float("nan"),
                "has_nan_or_inf": True,
            }

        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))
        max_res = float(np.max(np.abs(residuals)))

        var_y = float(np.var(y_true))
        r2 = 1.0 - (mse / var_y) if var_y > 1e-15 else (1.0 if mse < 1e-15 else 0.0)

        return {
            "mse": mse,
            "mae": mae,
            "r_squared": r2,
            "max_residual": max_res,
            "has_nan_or_inf": False,
        }

    def _fallback_singularity_probe(
        self,
        expr_str: str,
        variable_names: Sequence[str],
        domain_bounds: Sequence[tuple[float, float]],
        num_grid_points: int,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy reference singularity probe."""
        if not domain_bounds:
            return {
                "is_sound": True,
                "has_pole": False,
                "pole_coordinate": None,
                "max_derivative": 0.0,
                "risk_type": "NONE",
            }

        min_x, max_x = domain_bounds[0]
        grid = np.linspace(min_x, max_x, num_grid_points + 1, dtype=np.float64)
        step = (max_x - min_x) / max(num_grid_points, 1)

        data = [grid]
        for b in domain_bounds[1:]:
            mid = (b[0] + b[1]) / 2.0
            data.append(np.full_like(grid, mid))

        y = self._fallback_eval(expr_str, variable_names, data)
        invalid_mask = ~np.isfinite(y) | (np.abs(y) > 1e6)

        has_pole = False
        pole_coord = None
        if np.any(invalid_mask):
            has_pole = True
            first_idx = np.where(invalid_mask)[0][0]
            pole_coord = float(grid[first_idx])

        diffs = np.abs(np.diff(y)) / step
        finite_diffs = diffs[np.isfinite(diffs)]
        max_deriv = float(np.max(finite_diffs)) if len(finite_diffs) > 0 else 0.0

        if not has_pole:
            for i in range(len(grid) - 1):
                d = diffs[i]
                if np.isfinite(d) and d > 1e4:
                    sign_flip = (y[i] * y[i + 1] < 0.0) and (abs(y[i]) > 50.0 or abs(y[i + 1]) > 50.0)
                    if sign_flip or abs(y[i]) > 500.0 or abs(y[i + 1]) > 500.0:
                        has_pole = True
                        pole_coord = float(grid[i])
                        break

        is_sound = not has_pole
        risk_type = "POLE_IN_DOMAIN" if has_pole else "NONE"

        return {
            "is_sound": is_sound,
            "has_pole": has_pole,
            "pole_coordinate": pole_coord,
            "max_derivative": max_deriv,
            "risk_type": risk_type,
        }

    def _fallback_conditioning(
        self,
        cols: Sequence[np.ndarray],
        rcond_threshold: float,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy SVD condition number fallback.

        A conditioning probe must never crash: SVD non-convergence (or a
        non-finite design matrix, e.g. NaN Jacobian columns from sqrt of
        negative inputs) IS the degeneracy signal, reported as ill-conditioned.
        """
        if not cols:
            return {
                "condition_number": float("inf"),
                "max_singular_value": 0.0,
                "min_singular_value": 0.0,
                "effective_rank": 0,
                "is_well_conditioned": False,
            }
        try:
            mat = np.column_stack(cols)
            if not np.all(np.isfinite(mat)):
                raise np.linalg.LinAlgError("non-finite design matrix")
            s = np.linalg.svd(mat, compute_uv=False)
        except (np.linalg.LinAlgError, ArithmeticError, ValueError):
            logger.debug("SVD conditioning probe degenerate; reporting ill-conditioned", exc_info=True)
            return {
                "condition_number": float("inf"),
                "max_singular_value": 0.0,
                "min_singular_value": 0.0,
                "effective_rank": 0,
                "is_well_conditioned": False,
            }
        sigma_max = float(s[0]) if len(s) > 0 else 0.0
        sigma_min = float(s[-1]) if len(s) > 0 else 0.0
        cond = float(sigma_max / sigma_min) if sigma_min > 1e-15 else float("inf")
        rank = int(np.sum(s > sigma_max * rcond_threshold))
        is_well = cond < (1.0 / max(rcond_threshold, 1e-15))
        return {
            "condition_number": cond,
            "max_singular_value": sigma_max,
            "min_singular_value": sigma_min,
            "effective_rank": rank,
            "is_well_conditioned": is_well,
            "singular_values": [float(v) for v in s],
        }

    def _fallback_linear_solver(
        self,
        basis_cols: Sequence[np.ndarray],
        target: np.ndarray,
        lambda_reg: float,
    ) -> dict[str, Any]:
        """Pure-Python / NumPy least squares solver fallback."""
        mat = np.column_stack(basis_cols)
        k = mat.shape[1]
        if lambda_reg > 1e-15:
            # Regularized normal equations, with SVD-based downgrade when the
            # Cholesky/solve path fails on a degenerate system.
            ata = mat.T @ mat + lambda_reg * np.eye(k)
            aty = mat.T @ target
            try:
                coeffs = np.linalg.solve(ata, aty)
            except np.linalg.LinAlgError:
                logger.debug("Normal-equations solve failed; downgrading to lstsq", exc_info=True)
                coeffs, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(mat, target, rcond=None)

        pred = mat @ coeffs
        residuals = target - pred
        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))
        var_y = float(np.var(target))
        r2 = 1.0 - (mse / var_y) if var_y > 1e-15 else (1.0 if mse < 1e-15 else 0.0)
        cond_rep = self._fallback_conditioning(basis_cols, 1e-12)

        return {
            "coefficients": coeffs,
            "mse": mse,
            "mae": mae,
            "r_squared": r2,
            "condition_number": cond_rep["condition_number"],
            "is_well_conditioned": cond_rep["is_well_conditioned"],
        }

    def _fallback_precision_boundary(
        self,
        separation_delta: float,
        condition_number: float,
        precision_bits: int,
    ) -> dict[str, Any]:
        """Pure-Python precision boundary calculation fallback."""
        if precision_bits == 32:
            eps_mach = float(np.finfo(np.float32).eps)
            p_name = "Float32 (Single Precision)"
        elif precision_bits == 64:
            eps_mach = float(np.finfo(np.float64).eps)
            p_name = "Float64 (Double Precision)"
        else:
            eps_mach = float(2.0 ** (-(precision_bits - 1)))
            p_name = f"CustomPrecision ({precision_bits}-bit)"

        eps_star = float(np.sqrt(2.0 * eps_mach))
        effective_noise = eps_mach * max(condition_number, 1.0)
        resolved = bool(separation_delta > effective_noise and separation_delta > eps_star)

        return {
            "precision_type": p_name,
            "machine_epsilon": eps_mach,
            "theoretical_threshold": eps_star,
            "empirical_transition_observed": resolved,
            "resolved": resolved,
        }


# Global default kernel instance
default_kernel = RustKernelBridge()
