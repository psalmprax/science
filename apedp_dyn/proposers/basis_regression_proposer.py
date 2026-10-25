"""Basis function regression proposer (RBF, polynomials, splines).

Fast screening step - can guide symbolic search.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class BasisRegressionProposer(ProposerBase):
    def __init__(self, basis: str = "rbf", n_basis: int = 20, degree: int = 3):
        self.basis = basis
        self.n_basis = n_basis
        self.degree = degree

    @property
    def proposer_id(self) -> str:
        return f"basis_{self.basis}"

    def build_basis(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        n_samples, n_vars = X.shape
        Phi = [np.ones(n_samples)]
        meta = {"type": self.basis}
        if self.basis == "poly":
            for d in range(1, self.degree + 1):
                for i in range(n_vars):
                    Phi.append(X[:, i] ** d)
            if n_vars >= 2:
                for i in range(n_vars):
                    for j in range(i, n_vars):
                        Phi.append(X[:, i] * X[:, j])
        elif self.basis == "rbf":
            rng = np.random.RandomState(42)
            centers = rng.randn(self.n_basis, n_vars)
            for k in range(self.n_basis):
                diff = X - centers[k]
                r2 = np.sum(diff ** 2, axis=1)
                Phi.append(np.exp(-r2 / 2.0))
        else:
            for i in range(n_vars):
                Phi.append(X[:, i])
        Phi = np.column_stack(Phi)
        return Phi, meta

    def fit_and_propose(self, X: np.ndarray, y: np.ndarray | None = None, dX: np.ndarray | None = None, variable_names: list[str] | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        X = np.asarray(X, dtype=float)
        if y is None and dX is not None:
            y = dX
        elif y is None:
            y = np.zeros((X.shape[0], 1)) if X.ndim == 2 else np.zeros(X.shape[0])
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        Phi, meta = self.build_basis(X)
        # ridge regression
        try:
            coef = np.linalg.pinv(Phi.T @ Phi + 1e-6 * np.eye(Phi.shape[1])) @ Phi.T @ y
        except Exception:
            coef = np.linalg.lstsq(Phi, y, rcond=None)[0]
        formula = f"BasisRegression({self.basis}, n_basis={self.n_basis}, degree={self.degree})"
        def make_score(f=formula):
            def score(df):
                try:
                    return np.ones(len(df))
                except Exception:
                    return np.zeros(len(df))
            return score
        proposals = [CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"basis_{self.basis}_{hash(formula) % 10**8}",
            formula=formula,
            strategy_archetype=StrategyArchetype.DYNAMICAL_SYSTEM,
            declared_scale_invariant=False,
            declared_nominal_dependencies=tuple(variable_names),
            compute_score=make_score(formula),
            parameters={"coef_shape": coef.shape, "basis_meta": meta},
            metadata={"method": "basis_regression", "basis": self.basis},
        )]
        return proposals
