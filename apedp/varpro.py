"""Variable Projection (Golub-Pereyra VarPro) for Separable Nonlinear Models.

Separates linear parameter estimation (analytically projected via pseudoinverse)
from nonlinear parameter search (L-BFGS-B on projected residuals):
    y(x) = sum_j beta_j * phi_j(x; alpha)

Reduces the dimensionality of the non-linear optimization problem from (p + q)
to q, substantially mitigating local-optima risks while retaining rigorous
basis identifiability diagnostics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt
import sympy as sp

from apedp.rust_bridge import RustKernelBridge

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentifiabilityReport:
    rank: int
    num_bases: int
    cond_number: float
    singular_values: list[float]
    status: str  # IDENTIFIABLE, ILL_CONDITIONED, RANK_DEFICIENT
    note: str


@dataclass(frozen=True)
class VarProResult:
    expression: sp.Expr
    linear_params: dict[sp.Symbol, float]
    nonlinear_params: dict[sp.Symbol, float]
    all_params: dict[sp.Symbol, float]
    mse: float
    r2: float
    bic: float
    aic: float
    converged: bool
    iterations: int
    identifiability: IdentifiabilityReport


class VarProOptimizer:
    """Solves separable nonlinear least-squares using the Variable Projection algorithm."""

    def __init__(self, max_iter: int = 150, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _diagnose_identifiability(Phi: np.ndarray, tol: float = 1e-10) -> IdentifiabilityReport:
        """Inspect singular value spectrum of basis design matrix Phi for collinearity/degeneracy."""
        n_rows, n_cols = Phi.shape
        if n_cols == 0 or n_rows == 0:
            return IdentifiabilityReport(
                rank=0,
                num_bases=0,
                cond_number=1.0,
                singular_values=[],
                status="RANK_DEFICIENT",
                note="Empty design matrix",
            )

        try:
            bridge = RustKernelBridge()
            cols = [Phi[:, j] for j in range(n_cols)]
            cond_rep = bridge.compute_matrix_conditioning(cols, rcond_threshold=tol)
            rank = cond_rep["effective_rank"]
            cond = cond_rep["condition_number"]
            s_list = cond_rep.get("singular_values", [])

            if rank < n_cols:
                status = "RANK_DEFICIENT"
                note = f"Rank deficiency: rank {rank} < {n_cols} basis functions (collinear basis)"
            elif cond > 1e7:
                status = "ILL_CONDITIONED"
                note = f"Poor conditioning (kappa={cond:.2e} > 1e7); parameters sensitive to noise"
            else:
                status = "IDENTIFIABLE"
                note = f"Well-conditioned basis (kappa={cond:.2e})"

            return IdentifiabilityReport(
                rank=rank,
                num_bases=n_cols,
                cond_number=round(cond, 2),
                singular_values=s_list,
                status=status,
                note=note,
            )
        except Exception:
            logger.debug("Identifiability QR fallback to SVD", exc_info=True)
            try:
                s = np.linalg.svd(Phi, compute_uv=False)
                s_list = [float(v) for v in s]
                max_s = s[0]
                min_s = s[-1]
                rank = int(np.sum(s > tol * max_s))
                cond = float(max_s / max(min_s, 1e-15))

                if rank < n_cols:
                    status = "RANK_DEFICIENT"
                    note = f"Rank deficiency: rank {rank} < {n_cols} basis functions (collinear basis)"
                elif cond > 1e7:
                    status = "ILL_CONDITIONED"
                    note = f"Poor conditioning (kappa={cond:.2e} > 1e7); parameters sensitive to noise"
                else:
                    status = "IDENTIFIABLE"
                    note = f"Well-conditioned basis (kappa={cond:.2e})"

                return IdentifiabilityReport(
                    rank=rank,
                    num_bases=n_cols,
                    cond_number=round(cond, 2),
                    singular_values=s_list,
                    status=status,
                    note=note,
                )
            except np.linalg.LinAlgError as exc:
                return IdentifiabilityReport(
                    rank=0,
                    num_bases=n_cols,
                    cond_number=float("inf"),
                    singular_values=[],
                    status="RANK_DEFICIENT",
                    note=f"SVD failure: {exc}",
                )

    @staticmethod
    def partition_parameters(
        skeleton: sp.Expr, params: list[sp.Symbol]
    ) -> tuple[list[sp.Symbol], list[sp.Symbol], list[sp.Expr]]:
        """
        Decomposes skeleton into sum_j beta_j * phi_j(x; alpha).
        Returns:
            (linear_params, nonlinear_params, basis_functions)
        """
        expanded = sp.expand(skeleton)
        terms = sp.Add.make_args(expanded)

        linear_params: list[sp.Symbol] = []
        basis_funcs: list[sp.Expr] = []
        assigned_params = set()

        for term in terms:
            for p in params:
                if p in term.free_symbols and p not in assigned_params:
                    d1 = sp.diff(term, p)
                    if p not in d1.free_symbols and term.subs(p, 0) == 0:
                        linear_params.append(p)
                        basis_funcs.append(d1)
                        assigned_params.add(p)
                        break

        nonlinear_params = [p for p in params if p not in assigned_params]

        if not linear_params:
            return [], params, [skeleton]

        return linear_params, nonlinear_params, basis_funcs

    def fit(
        self,
        skeleton: sp.Expr,
        params: list[sp.Symbol],
        variables: list[sp.Symbol],
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        alpha_bounds: dict[sp.Symbol, tuple[float, float]] | None = None,
    ) -> VarProResult:
        """Fit skeleton parameters using Variable Projection with identifiability reporting."""
        n = len(y_data)
        y = np.asarray(y_data, dtype=float).ravel()
        var_arrays = [x_data[str(v)].ravel() for v in variables]

        lin_params, nonlin_params, bases = self.partition_parameters(skeleton, params)

        # Case 1: Purely linear model
        if lin_params and not nonlin_params:
            Phi = np.zeros((n, len(bases)), dtype=float)
            for j, basis in enumerate(bases):
                f_lamb = sp.lambdify(variables, basis, modules=["numpy"])
                val = f_lamb(*var_arrays)
                Phi[:, j] = np.ones(n) * np.asarray(val, dtype=float) if np.isscalar(val) else np.asarray(val, dtype=float)

            ident = self._diagnose_identifiability(Phi)

            # Solve linear least squares
            try:
                # Tier 2: Optimized LAPACK lstsq
                beta_star, _residuals, _rank, _ = np.linalg.lstsq(Phi, y, rcond=1e-6)
                y_pred = Phi @ beta_star
                mse = float(np.mean((y - y_pred) ** 2))
                y_var = float(np.var(y))
                r2 = float(1.0 - mse / (y_var + 1e-12))
                k = len(lin_params)
                bic = float(n * np.log(max(mse, 1e-12)) + k * np.log(n))
                aic = float(n * np.log(max(mse, 1e-12)) + 2 * k)

                lin_dict = {p: float(beta_star[j]) for j, p in enumerate(lin_params)}
                fitted_expr = skeleton.xreplace({p: sp.Float(lin_dict[p]) for p in lin_params})

                return VarProResult(
                    expression=fitted_expr,
                    linear_params=lin_dict,
                    nonlinear_params={},
                    all_params=lin_dict,
                    mse=mse,
                    r2=r2,
                    bic=bic,
                    aic=aic,
                    converged=True,
                    iterations=1,
                    identifiability=ident,
                )
            except np.linalg.LinAlgError:
                pass

        # Case 2: Purely nonlinear or separable model
        all_eval_symbols = variables + nonlin_params
        basis_funcs = [sp.lambdify(all_eval_symbols, b, modules=["numpy"]) for b in bases]

        def _evaluate_phi(alpha_vals: np.ndarray) -> np.ndarray:
            Phi_mat = np.zeros((n, len(bases)), dtype=float)
            eval_args = var_arrays + list(alpha_vals)
            for j, f in enumerate(basis_funcs):
                val = f(*eval_args)
                Phi_mat[:, j] = (np.ones(n) * np.asarray(val, dtype=float) if np.isscalar(val)
                                 else np.asarray(val, dtype=float))
            return Phi_mat

        def _objective(alpha_vals: np.ndarray) -> float:
            Phi_mat = _evaluate_phi(alpha_vals)
            if not np.all(np.isfinite(Phi_mat)):
                return 1e12
            try:
                # Orthogonal projection: residual = y - Phi * (Phi^+ y)
                Q, _R = np.linalg.qr(Phi_mat, mode="reduced")
                y_proj = Q @ (Q.T @ y)
                res = y - y_proj
                return float(np.sum(res**2))
            except np.linalg.LinAlgError:
                return 1e12

        # Initial guess for alpha
        q = len(nonlin_params)
        alpha_0 = np.ones(q, dtype=float)
        bounds_list = []
        for p in nonlin_params:
            if alpha_bounds and p in alpha_bounds:
                bounds_list.append(alpha_bounds[p])
            else:
                bounds_list.append((-50.0, 50.0))

        res_opt = opt.minimize(
            _objective,
            alpha_0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        alpha_opt = res_opt.x
        Phi_opt = _evaluate_phi(alpha_opt)
        ident = self._diagnose_identifiability(Phi_opt)

        # Tier 2: Optimized LAPACK lstsq
        beta_opt, _, _, _ = np.linalg.lstsq(Phi_opt, y, rcond=1e-6)

        y_pred = Phi_opt @ beta_opt
        mse = float(np.mean((y - y_pred) ** 2))
        y_var = float(np.var(y))
        r2 = float(1.0 - mse / (y_var + 1e-12))
        k = len(lin_params) + len(nonlin_params)
        bic = float(n * np.log(max(mse, 1e-12)) + k * np.log(n))
        aic = float(n * np.log(max(mse, 1e-12)) + 2 * k)

        lin_dict = {p: float(beta_opt[j]) for j, p in enumerate(lin_params)}
        nonlin_dict = {p: float(alpha_opt[j]) for j, p in enumerate(nonlin_params)}
        all_dict = {**lin_dict, **nonlin_dict}

        fitted_expr = skeleton.xreplace({p: sp.Float(all_dict[p]) for p in all_dict})

        return VarProResult(
            expression=fitted_expr,
            linear_params=lin_dict,
            nonlinear_params=nonlin_dict,
            all_params=all_dict,
            mse=mse,
            r2=r2,
            bic=bic,
            aic=aic,
            converged=bool(res_opt.success),
            iterations=int(res_opt.nit),
            identifiability=ident,
        )


__all__ = [
    "IdentifiabilityReport",
    "VarProOptimizer",
    "VarProResult",
]
