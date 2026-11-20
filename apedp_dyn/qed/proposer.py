"""The QED entry point into the competition: propose, never adjudicate.

Status: experimental

This adapter turns one search arm into a standard
:class:`~apedp_dyn.grammar.proposer_interface.CandidateProposal`, so a
quantum-assisted equation can compete against the existing proposers on the same
data and be adjudicated by the same referee (AGENTS.md 1.1). It fits on the rows
it is handed, emits a closed-form equation with its coefficients, and supplies a
``compute_score`` that re-evaluates *the same library* on whatever frame it is
given -- so an out-of-sample check is a real causal evaluation, not a replay of
an in-sample vector.

It has no access to holdout data, no acceptance logic, and no thresholds. If the
arm cannot search, it returns no proposals rather than a worse equation.
"""
from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from apedp_dyn.grammar.proposer_interface import (
    CandidateProposal,
    ProposerBase,
    StrategyArchetype,
)
from apedp_dyn.qed.arms import ABSTAINED, ARMS, ArmResult, run_arm
from apedp_dyn.qed.library import TermLibrary


class QuantumEquationProposer(ProposerBase):
    """Discover one governing equation by relational search over a term library."""

    def __init__(
        self,
        *,
        arm: str = "qaoa_surrogate",
        library: TermLibrary,
        penalty: float = 0.002,
        budget: int = 256,
        seed: int = 11,
        arm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; known arms are {sorted(ARMS)}")
        self.arm = arm
        self.library = library
        self.penalty = float(penalty)
        self.budget = int(budget)
        self.seed = int(seed)
        self.arm_kwargs = dict(arm_kwargs or {})

    @property
    def proposer_id(self) -> str:
        return f"qed_{self.arm}"

    def fit_and_propose(
        self,
        X: np.ndarray,
        dX: np.ndarray | None = None,
        variable_names: list[str] | None = None,
        *,
        target_column: int = 0,
        **kwargs: Any,
    ) -> Sequence[CandidateProposal]:
        states = np.asarray(X, dtype=float)
        if states.ndim != 2:
            raise ValueError(f"states must be 2-D, got {states.shape}")
        if variable_names is None:
            variable_names = [f"x{index}" for index in range(states.shape[1])]
        if dX is None:
            target = np.gradient(states[:, target_column])
        else:
            derivatives = np.asarray(dX, dtype=float)
            target = derivatives if derivatives.ndim == 1 else derivatives[:, target_column]
        if target.shape != (states.shape[0],):
            raise ValueError("target must have one row per state")

        class _Task:
            """The minimum an arm needs, so the adapter does not leak the task."""

            def __init__(self, library: Any, states: Any, target: Any, penalty: Any, budget: Any) -> None:
                self.library = library
                self.dev_states = states
                self.dev_target = target
                self.penalty = penalty
                self.budget = budget

            def objective(self) -> Any:
                from apedp_dyn.qed.objective import SupportObjective

                return SupportObjective(
                    self.library,
                    self.library.design_matrix(self.dev_states),
                    self.dev_target,
                    penalty=self.penalty,
                    budget=self.budget,
                )

        result: ArmResult = run_arm(
            self.arm,
            _Task(self.library, states, target, self.penalty, self.budget),
            seed=self.seed,
        )
        if result.status == ABSTAINED or result.best is None:
            return ()

        best = result.best
        names = self.library.names
        formula = self.library.render(
            best.support, best.coefficients, lhs=f"d({variable_names[target_column]})/dt"
        )
        support_names = [names[index] for index in best.support]
        coefficients = [float(value) for value in best.coefficients]
        library = self.library

        def make_score(keep: list[int], values: list[float], library:Any=library) -> Any:
            def score(df: pd.DataFrame) -> np.ndarray:
                frame = np.asarray(df.to_numpy(dtype=float), dtype=float)
                if frame.ndim != 2:
                    raise ValueError("compute_score expects a 2-D frame")
                design = library.design_matrix(frame)
                _out: np.ndarray = library.select(design, keep) @ np.asarray(values, dtype=float)
                return _out

            return score

        return [
            CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"qed_{zlib.crc32(formula.encode()) % 10**8}",
                formula=formula,
                strategy_archetype=StrategyArchetype.DYNAMICAL_SYSTEM,
                declared_scale_invariant=False,
                declared_nominal_dependencies=tuple(variable_names),
                compute_score=make_score(list(best.support), coefficients),
                parameters={
                    "support": support_names,
                    "coefficients": coefficients,
                    "objective": float(best.objective),
                    "residual": float(best.residual),
                },
                metadata={
                    "method": "qed_selection_search",
                    "arm": self.arm,
                    "evaluations": result.evaluations,
                    "budget": result.budget,
                    "library": list(names),
                    "penalty": self.penalty,
                },
            )
        ]


__all__ = ["QuantumEquationProposer"]
