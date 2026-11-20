"""Reservoir computing (Echo State Network) proposer.

Fixed random recurrent dynamics, only readout trained. Useful for chaotic/time-series
systems where exact symbolic forms are hard to find.
"""
from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)

logger = logging.getLogger(__name__)


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
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("ESN spectral-radius rescale failed; using unscaled reservoir: %s", exc)
        states = np.zeros((n_samples, self.reservoir_size))
        for t in range(1, n_samples):
            states[t] = np.tanh(W_in @ X[t] + W @ states[t-1])
        # ridge regression
        S = np.hstack([np.ones((n_samples, 1)), X, states])
        try:
            W_out = np.linalg.pinv(S.T @ S + 1e-6 * np.eye(S.shape[1])) @ S.T @ dX
        except Exception:  # noqa: BLE001 - pinv of a singular Gram matrix raises LinAlgError, and degenerate input adds shape/dtype errors; lstsq is the documented fallback for the same ridge solve
            W_out = np.linalg.lstsq(S, dX, rcond=None)[0]
        # One readout column per target. The readout was flattened with
        # ``W_out.ravel()``, which is correct only for a single target: on a
        # two-variable input the scorer then failed with "matmul: core
        # dimension 0 ... size 46 is different from 23", because the candidate
        # contract is one scalar prediction per row, not a vector field. Each
        # target now gets its own proposal and its own column.
        n_targets = dX.shape[1]
        base_formula = (f"ESN(reservoir={self.reservoir_size}, "
                        f"spectral_radius={self.spectral_radius})")
        reservoir_size = self.reservoir_size
        proposals: list[CandidateProposal] = []
        for col in range(n_targets):
            target = variable_names[col] if col < len(variable_names) else f"x{col}"
            formula = base_formula if n_targets == 1 else f"d({target})/dt = {base_formula}"
            readout = W_out[:, col].copy()

            def make_score(W_in: Any = W_in, W: Any = W, readout: Any = readout,
                           reservoir_size: Any = reservoir_size) -> Any:
                def score(df: Any) -> Any:
                    X_eval = np.asarray(df.to_numpy(dtype=float))
                    n_samples_eval = X_eval.shape[0]
                    if readout.size == 0:
                        return np.zeros(n_samples_eval)
                    states_eval = np.zeros((n_samples_eval, reservoir_size))
                    for t in range(1, n_samples_eval):
                        states_eval[t] = np.tanh(W_in @ X_eval[t] + W @ states_eval[t-1])
                    S_eval = np.hstack([np.ones((n_samples_eval, 1)), X_eval, states_eval])
                    pred = S_eval @ readout
                    return pred.ravel()
                return score

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"esn_{zlib.crc32(formula.encode()) % 10**8}",
                formula=formula,
                strategy_archetype=StrategyArchetype.DYNAMICAL_SYSTEM,
                declared_scale_invariant=False,
                declared_nominal_dependencies=tuple(variable_names),
                compute_score=make_score(),
                parameters={"W_out_shape": readout.shape},
                metadata={"method": "reservoir_computing",
                          "reservoir_size": self.reservoir_size,
                          "target": target},
            ))
        return proposals
