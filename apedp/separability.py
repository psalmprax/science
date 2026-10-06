"""Numerical and symbolic separability analysis for multi-variable discovery.

Decomposes high-dimensional discovery problems into lower-dimensional sub-problems
by testing for additive and multiplicative coordinate separability:
- Additive separability: d^2 f / (dx_i dx_j) == 0
- Multiplicative separability: d^2 ln|f| / (dx_i dx_j) == 0

Uses controlled multivariate second-order Taylor expansion to prevent confounding
variance from other coordinates when K >= 3, with Ridge regularization and epistemic
domain checks on log-transform validity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SeparabilityType(str, Enum):
    NONE = "NONE"
    ADDITIVE = "ADDITIVE"
    MULTIPLICATIVE = "MULTIPLICATIVE"


@dataclass(frozen=True)
class MultiplicativeDomainCheck:
    status: str  # VALID, INVALID, UNEVALUATED
    zero_fraction: float
    sign_changes: int
    log_domain_coverage: float
    reason: str


@dataclass(frozen=True)
class SeparabilityPairResult:
    var1: str
    var2: str
    score_additive: float
    score_multiplicative: float
    verdict: SeparabilityType
    multiplicative_check: MultiplicativeDomainCheck
    details: dict[str, float]


@dataclass
class VariableDecomposition:
    additive_groups: list[list[str]]
    multiplicative_groups: list[list[str]]
    is_fully_additive: bool
    is_fully_multiplicative: bool
    pair_results: list[SeparabilityPairResult]


class SeparabilityAnalyzer:
    """Estimates scale-invariant mixed partial derivatives to detect variable separability."""

    def __init__(
        self,
        tolerance: float = 0.35,
        min_points: int = 15,
        ridge_alpha: float = 1e-3,
    ):
        self.tolerance = tolerance
        self.min_points = min_points
        self.ridge_alpha = ridge_alpha

    def _estimate_multivariate_mixed_curvatures(
        self, X: np.ndarray, y: np.ndarray, k_neighbors: int | None = None
    ) -> np.ndarray:
        """
        Estimate mixed partial derivatives d^2 y_norm / (du_i du_j) controlled for all other coordinates.
        Returns a (K, K) symmetric matrix of mixed partial root-mean-squared values.
        """
        n, k_vars = X.shape
        if n < 12:
            return np.ones((k_vars, k_vars))

        s_y = float(np.std(y)) + 1e-12
        y_norm = (y - np.mean(y)) / s_y

        # Normalize each coordinate to [0, 1]
        x_min = np.min(X, axis=0)
        x_range = np.max(X, axis=0) - x_min + 1e-12
        U = (X - x_min) / x_range

        # Number of basis functions: 1 (intercept) + k (linear) + k (quadratic) + k*(k-1)//2 (cross)
        num_cross = k_vars * (k_vars - 1) // 2
        p_basis = 1 + 2 * k_vars + num_cross

        k_nn = k_neighbors or max(p_basis + 6, min(n - 1, 30))
        k_nn = min(k_nn, n)

        # Mapping from (i, j) to column index in design matrix
        pair_to_col: dict[tuple[int, int], int] = {}
        col_idx = 1 + 2 * k_vars
        for i in range(k_vars):
            for j in range(i + 1, k_vars):
                pair_to_col[(i, j)] = col_idx
                pair_to_col[(j, i)] = col_idx
                col_idx += 1

        sample_indices = np.linspace(0, n - 1, min(n, 50), dtype=int)
        d2_accum: dict[tuple[int, int], list[float]] = {
            (i, j): [] for i in range(k_vars) for j in range(i + 1, k_vars)
        }

        eye_reg = self.ridge_alpha * np.eye(p_basis)
        eye_reg[0, 0] = 0.0

        for idx in sample_indices:
            pt = U[idx]
            dists = np.sum((U - pt) ** 2, axis=1)
            nbrs = np.argsort(dists)[:k_nn]
            if len(nbrs) < p_basis:
                continue

            du = U[nbrs] - pt
            cols = [np.ones(len(nbrs))]
            # Linear terms
            for i in range(k_vars):
                cols.append(du[:, i])
            # Quadratic terms
            for i in range(k_vars):
                cols.append(du[:, i] ** 2)
            # Cross terms
            for i in range(k_vars):
                for j in range(i + 1, k_vars):
                    cols.append(du[:, i] * du[:, j])

            A = np.column_stack(cols)
            try:
                AtA = A.T @ A + eye_reg
                Aty = A.T @ y_norm[nbrs]
                coeffs = np.linalg.solve(AtA, Aty)
                for (i, j), c_idx in pair_to_col.items():
                    if i < j:
                        d2_accum[(i, j)].append(float(coeffs[c_idx] ** 2))
            except (np.linalg.LinAlgError, ValueError):
                continue

        mat = np.zeros((k_vars, k_vars))
        for i in range(k_vars):
            for j in range(i + 1, k_vars):
                vals = d2_accum[(i, j)]
                rms = float(np.sqrt(np.mean(vals))) if vals else 1.0
                mat[i, j] = rms
                mat[j, i] = rms
        return mat

    @staticmethod
    def inspect_multiplicative_domain(y: np.ndarray) -> MultiplicativeDomainCheck:
        """Inspect whether data y can be validly subjected to logarithmic transformation."""
        n = len(y)
        if n == 0:
            return MultiplicativeDomainCheck(
                status="UNEVALUATED",
                zero_fraction=1.0,
                sign_changes=0,
                log_domain_coverage=0.0,
                reason="EMPTY_DATASET",
            )

        near_zero_mask = np.abs(y) < 1e-10
        zero_frac = float(np.mean(near_zero_mask))

        valid_signs = np.sign(y[~near_zero_mask])
        sign_changes = 0
        if len(valid_signs) > 1:
            sign_changes = int(np.sum(valid_signs[:-1] * valid_signs[1:] < 0))

        strictly_positive = np.all(y > 1e-10)
        strictly_negative = np.all(y < -1e-10)

        if strictly_positive or strictly_negative:
            return MultiplicativeDomainCheck(
                status="VALID",
                zero_fraction=zero_frac,
                sign_changes=sign_changes,
                log_domain_coverage=1.0 - zero_frac,
                reason="STRICT_SINGLE_SIGN_DOMAIN",
            )

        if zero_frac > 0.05 or sign_changes > 0:
            return MultiplicativeDomainCheck(
                status="INVALID",
                zero_fraction=zero_frac,
                sign_changes=sign_changes,
                log_domain_coverage=1.0 - zero_frac,
                reason=f"ZERO_CROSSINGS_OR_SIGN_FLIPS (zeros: {zero_frac:.1%}, sign flips: {sign_changes})",
            )

        return MultiplicativeDomainCheck(
            status="UNEVALUATED",
            zero_fraction=zero_frac,
            sign_changes=sign_changes,
            log_domain_coverage=1.0 - zero_frac,
            reason="AMBIGUOUS_DOMAIN",
        )

    def analyze_pair(
        self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray, var1: str, var2: str
    ) -> SeparabilityPairResult:
        """Evaluate additive and multiplicative separability between two variables in isolation."""
        X_2d = np.column_stack([x1, x2])
        curv_mat = self._estimate_multivariate_mixed_curvatures(X_2d, y)
        curv_add = float(curv_mat[0, 1])

        mult_check = self.inspect_multiplicative_domain(y)
        curv_mult = 10.0
        if mult_check.status == "VALID":
            ln_y = np.log(np.abs(y))
            curv_mat_mult = self._estimate_multivariate_mixed_curvatures(X_2d, ln_y)
            curv_mult = float(curv_mat_mult[0, 1])

        verdict = SeparabilityType.NONE
        if curv_add < self.tolerance and curv_add <= curv_mult:
            verdict = SeparabilityType.ADDITIVE
        elif mult_check.status == "VALID" and curv_mult < self.tolerance:
            verdict = SeparabilityType.MULTIPLICATIVE

        return SeparabilityPairResult(
            var1=var1,
            var2=var2,
            score_additive=round(curv_add, 4),
            score_multiplicative=round(curv_mult, 4),
            verdict=verdict,
            multiplicative_check=mult_check,
            details={
                "additive_curvature": curv_add,
                "multiplicative_curvature": curv_mult,
            },
        )

    def decompose(
        self, x_data: dict[str, np.ndarray], y_data: np.ndarray
    ) -> VariableDecomposition:
        """
        Decompose all variables into additive or multiplicative partition blocks
        using controlled joint multivariate regression to eliminate inter-variable confounding.
        """
        var_names = sorted(x_data.keys())
        n_vars = len(var_names)
        pair_results = []

        X_full = np.column_stack([x_data[v] for v in var_names])
        add_mat = self._estimate_multivariate_mixed_curvatures(X_full, y_data)

        mult_check = self.inspect_multiplicative_domain(y_data)
        mult_mat = np.full((n_vars, n_vars), 10.0)
        if mult_check.status == "VALID":
            ln_y = np.log(np.abs(y_data))
            mult_mat = self._estimate_multivariate_mixed_curvatures(X_full, ln_y)

        adj_add: dict[str, set[str]] = {v: set() for v in var_names}
        adj_mult: dict[str, set[str]] = {v: set() for v in var_names}

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                v1, v2 = var_names[i], var_names[j]
                curv_add = float(add_mat[i, j])
                curv_mult = float(mult_mat[i, j])

                verdict = SeparabilityType.NONE
                if curv_add < self.tolerance and curv_add <= curv_mult:
                    verdict = SeparabilityType.ADDITIVE
                elif mult_check.status == "VALID" and curv_mult < self.tolerance:
                    verdict = SeparabilityType.MULTIPLICATIVE

                pair_res = SeparabilityPairResult(
                    var1=v1,
                    var2=v2,
                    score_additive=round(curv_add, 4),
                    score_multiplicative=round(curv_mult, 4),
                    verdict=verdict,
                    multiplicative_check=mult_check,
                    details={
                        "additive_curvature": curv_add,
                        "multiplicative_curvature": curv_mult,
                    },
                )
                pair_results.append(pair_res)

                if verdict != SeparabilityType.ADDITIVE:
                    adj_add[v1].add(v2)
                    adj_add[v2].add(v1)

                if verdict != SeparabilityType.MULTIPLICATIVE:
                    adj_mult[v1].add(v2)
                    adj_mult[v2].add(v1)

        def _connected_components(adj: dict[str, set[str]]) -> list[list[str]]:
            visited = set()
            comps = []
            for v in var_names:
                if v not in visited:
                    comp = []
                    q = [v]
                    visited.add(v)
                    while q:
                        curr = q.pop()
                        comp.append(curr)
                        for nbr in adj[curr]:
                            if nbr not in visited:
                                visited.add(nbr)
                                q.append(nbr)
                    comps.append(sorted(comp))
            return sorted(comps, key=lambda c: (len(c), c))

        add_groups = _connected_components(adj_add)
        mult_groups = _connected_components(adj_mult)

        is_fully_add = len(add_groups) == n_vars and n_vars > 1
        is_fully_mult = len(mult_groups) == n_vars and n_vars > 1

        return VariableDecomposition(
            additive_groups=add_groups,
            multiplicative_groups=mult_groups,
            is_fully_additive=is_fully_add,
            is_fully_multiplicative=is_fully_mult,
            pair_results=pair_results,
        )


__all__ = [
    "MultiplicativeDomainCheck",
    "SeparabilityAnalyzer",
    "SeparabilityPairResult",
    "SeparabilityType",
    "VariableDecomposition",
]
