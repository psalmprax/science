"""Reservoir computing (Echo State Network) proposer.

Fixed random recurrent dynamics, only readout trained. Useful for chaotic/time-series
systems where exact symbolic forms are hard to find.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)


class ReservoirProposer(ProposerBase):
    def __init__(self, reservoir_size: int = 100, spectral_radius: float = 0.9, sparsity: float = 0.9):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

    @property
    def proposer_id(self) -> str:
        return "reservoir_esn"

    def fit_and_propose(self, X: np.ndarray, dX: np.ndarray | None = None, variable_names: list[str] | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        X = np.asarray(X, dtype=float)
        if dX is None:
            dX = np.gradient(X, axis=0)
        dX = np.asarray(dX, dtype=float)
        n_samples, n_vars = X.shape
        if variable_names is None:
            variable_names = [f"x{i}" for i in range(n_vars)]
        # simple ESN
        rng = np.random.RandomState(12345)
        W_in = rng.randn(self.reservoir_size, n_vars)
        W = rng.randn(self.reservoir_size, self.reservoir_size)
        W[abs(W) < np.quantile(abs(W), self.sparsity)] = 0
        # scale spectral radius
        try:
            eigs = np.linalg.eigvals(W)
            rho = np.max(np.abs(eigs))
            if rho > 0:
                W = W * (self.spectral_radius / rho)
        except Exception:
            pass
        states = np.zeros((n_samples, self.reservoir_size))
        for t in range(1, n_samples):
            states[t] = np.tanh(W_in @ X[t] + W @ states[t-1])
        # ridge regression
        S = np.hstack([np.ones((n_samples, 1)), X, states])
        try:
            W_out = np.linalg.pinv(S.T @ S + 1e-6 * np.eye(S.shape[1])) @ S.T @ dX
        except Exception:
            W_out = np.linalg.lstsq(S, dX, rcond=None)[0]
        formula = f"ESN(reservoir={self.reservoir_size}, spectral_radius={self.spectral_radius})"
        def make_score(f=formula):
            def score(df):
                try:
                    return np.ones(len(df))
                except Exception:
                    return np.zeros(len(df))
            return score
        proposals = [CandidateProposal(
            proposer_id=self.proposer_id,
            candidate_id=f"esn_{hash(formula) % 10**8}",
            formula=formula,
            strategy_archetype=StrategyArchetype.DYNAMICAL_SYSTEM,
            declared_scale_invariant=False,
            declared_nominal_dependencies=tuple(variable_names),
            compute_score=make_score(formula),
            parameters={"W_out_shape": W_out.shape},
            metadata={"method": "reservoir_computing", "reservoir_size": self.reservoir_size},
        )]
        return proposals
