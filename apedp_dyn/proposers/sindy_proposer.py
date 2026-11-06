"""SINDy-style sparse identification proposer.

Implements sparse regression over a library of candidate terms to discover
parsimonious governing equations (ODEs/PDEs). Rust-accelerated where possible
with Python fallbacks.
"""
from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import apedp_kernel
    RUST_AVAILABLE = hasattr(apedp_kernel, 'compute_metrics')
except Exception:
    apedp_kernel = None
    RUST_AVAILABLE = False

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class SINDyProposer(ProposerBase):
    """Sparse Identification of Nonlinear Dynamics proposer."""

    def __init__(self, degree: int = 3, threshold: float = 1e-2, max_iter: int = 10):
        self.degree = degree
        self.threshold = threshold
        self.max_iter = max_iter

    @property
    def proposer_id(self) -> str:
        return "sindy_sparse"

    def build_library(self, X: np.ndarray, poly_order: int = 3) -> tuple[np.ndarray, list[str]]:
        """Build library of candidate functions."""
        n_samples, n_vars = X.shape
        lib = [np.ones(n_samples)]
        names = ["1"]
        # Linear terms
        for i in range(n_vars):
            lib.append(X[:, i])
            names.append(f"x{i}")
        # Polynomial terms
        if poly_order >= 2:
            for i in range(n_vars):
                for j in range(i, n_vars):
                    lib.append(X[:, i] * X[:, j])
                    names.append(f"x{i}*x{j}")
        if poly_order >= 3:
            for i in range(n_vars):
                for j in range(i, n_vars):
                    for k in range(j, n_vars):
                        lib.append(X[:, i] * X[:, j] * X[:, k])
                        names.append(f"x{i}*x{j}*x{k}")
        # Trig terms
        for i in range(n_vars):
            lib.append(np.sin(X[:, i]))
            names.append(f"sin(x{i})")
            lib.append(np.cos(X[:, i]))
            names.append(f"cos(x{i})")
        Theta = np.column_stack(lib)
        return Theta, names

    def sparse_regression(self, Theta: np.ndarray, dX: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Sequential thresholded least squares."""
        # Try Rust acceleration if available
        if RUST_AVAILABLE and apedp_kernel is not None and hasattr(apedp_kernel, 'sindy_sparse_regression'):
            try:
                Xi, active = apedp_kernel.sindy_sparse_regression(
                    np.ascontiguousarray(Theta, dtype=np.float64),
                    np.ascontiguousarray(dX, dtype=np.float64),
                    float(self.threshold),
                    int(self.max_iter),
                )
                Xi = np.asarray(Xi)
                if isinstance(active, np.ndarray):
                    active_list = [int(x) for x in active.tolist()]
                else:
                    active_list = list(active)
                return Xi, active_list
            except Exception as exc:
                logger.warning("SINDy Rust sparse regression failed; falling back to NumPy: %s", exc)
        Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
        for _ in range(self.max_iter):
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0
            for i in range(Xi.shape[1]):
                big = ~small[:, i]
                if np.sum(big) == 0:
                    continue
                Xi[big, i] = np.linalg.lstsq(Theta[:, big], dX[:, i], rcond=None)[0]
        active = []
        for i in range(Xi.shape[0]):
            if np.any(np.abs(Xi[i, :]) > 1e-12):
                active.append(i)
        return Xi, active

    def fit_and_propose(self, X: np.ndarray, dX: np.ndarray | None = None, variable_names: list[str] | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        X = np.asarray(X, dtype=float)
        if dX is None:
            # compute numerical derivative if not provided
            dX = np.gradient(X, axis=0)
        dX = np.asarray(dX, dtype=float)
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(X.shape[1])]
        Theta, names = self.build_library(X, self.degree)
        Xi, active = self.sparse_regression(Theta, dX)
        proposals = []
        for j in range(X.shape[1]):
            terms = []
            coeffs = []
            for i in active:
                if abs(Xi[i, j]) > 1e-12:
                    terms.append(names[i])
                    coeffs.append(float(Xi[i, j]))
            if not terms or terms == ["1"]:
                continue
            formula_parts = []
            from apedp_dyn.grammar.term_format import atom
            for c, t in zip(coeffs, terms):
                if t == "1":
                    formula_parts.append(f"{c:.6g}")
                elif t.startswith("x"):
                    formula_parts.append(f"{c:.6g}*{atom(t)}")
                else:
                    formula_parts.append(f"{c:.6g}*{atom(t)}")
            formula = " + ".join(formula_parts) if formula_parts else "0"

            def make_score(terms=tuple(terms), coeffs=tuple(coeffs)):
                # Bind this candidate's own (term, coefficient) pairs at build
                # time. The factory used to close over the loop's library names
                # and use their positions to index the shorter coefficient list,
                # so a later candidate scored an earlier one's terms -- or, when
                # the first term was not in the library's first column, raised.
                pairs = tuple((t, c) for t, c in zip(terms, coeffs) if t != "1")
                constants = tuple(c for t, c in zip(terms, coeffs) if t == "1")

                def score(df):
                    rows = len(df)
                    if not pairs:
                        return np.full(rows, float(sum(constants)) if constants
                                       else 0.0)
                    pred = np.zeros(rows, dtype=float)
                    for name, weight in pairs:
                        pred += weight * np.asarray(df[name], dtype=float).ravel()
                    return pred

                return score
            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"sindy_{zlib.crc32(formula.encode()) % 10**8}",
                formula=formula,
                strategy_archetype=StrategyArchetype.DYNAMICAL_SYSTEM,
                declared_scale_invariant=False,
                declared_nominal_dependencies=tuple(variable_names),
                compute_score=make_score(),
                parameters={"coefficients": coeffs, "terms": terms},
                metadata={"method": "sindy", "active_terms": len(terms)},
            ))
        return proposals


# Adapter for CandidateProposer interface
try:

    class SINDyGrammarAdapter:
        """Adapter to make SINDyProposer compatible with grammar CandidateProposer interface."""
        def __init__(self, degree: int = 3, threshold: float = 1e-2):
            self.proposer_id = "sindy-grammar-adapter"
            self._impl = SINDyProposer(degree=degree, threshold=threshold)

        def propose(self, problem) -> list:
            # Try to extract data from problem
            try:
                t = getattr(problem, 'train', {}).get('truth_t')
                Y = getattr(problem, 'train', {}).get('truth_y')
                if t is None or Y is None:
                    return []
                t = np.asarray(t)
                Y = np.asarray(Y)
                # compute derivatives
                dY = np.gradient(Y, t, axis=0)
                props = self._impl.fit_and_propose(Y, dY)
                results = []
                for p in props[:10]:  # limit
                    try:
                        # try to parse simple formula
                        results.append(p.formula)
                    except Exception:
                        continue
                return results
            except Exception:
                return []
except Exception:
    pass
