"""The search objective, with the evaluation budget as a measured resource.

Status: experimental

A candidate equation ``dy/dt = sum_j z_j beta_j phi_j(state)`` is scored by

    J(z) = (1/N) * sum_i (y_i - sum_j z_j beta_j phi_j(s_i))^2 + lambda * |z|

with ``beta`` re-fitted by ordinary least squares for every selection ``z``, so
the coefficient estimate is never part of what a proposer searches. The first
term is prediction error, the second the parsimony penalty that makes an
over-complete library resolvable at all.

The budget is the resource the experiment controls: every *distinct* selection
costs exactly one unit, a repeated selection is free because it is already
known, and :meth:`SupportObjective.evaluate` raises :class:`BudgetExhausted`
rather than quietly spending more. That is what makes six proposers comparable:
they all pay the same currency for the same fits.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from apedp_dyn.qed.library import LibraryRefusal, TermLibrary


class BudgetExhausted(RuntimeError):
    """Raised when a proposer asks for a fit it no longer has budget for."""


@dataclass(frozen=True)
class Fit:
    """One scored selection: which terms, their coefficients, and the two costs."""

    support: tuple[int, ...]
    coefficients: tuple[float, ...]
    objective: float
    residual: float

    @property
    def complexity(self) -> int:
        return len(self.support)

    def as_dict(self, names: Sequence[str]) -> dict:
        return {
            "support": [names[index] for index in self.support],
            "coefficients": [float(value) for value in self.coefficients],
            "objective": float(self.objective),
            "residual": float(self.residual),
            "complexity": self.complexity,
        }


class SupportObjective:
    """``J(z)`` over a fixed design matrix, with a hard budget on distinct fits."""

    def __init__(
        self,
        library: TermLibrary,
        design: np.ndarray,
        target: np.ndarray,
        *,
        penalty: float,
        budget: int,
    ) -> None:
        design = np.asarray(design, dtype=float)
        target = np.asarray(target, dtype=float).ravel()
        if design.ndim != 2 or design.shape[1] != len(library):
            raise LibraryRefusal(
                f"design matrix {design.shape} does not match a {len(library)}-term library"
            )
        if target.shape != (design.shape[0],):
            raise LibraryRefusal(
                f"target has {target.size} rows but the design has {design.shape[0]}"
            )
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
            raise LibraryRefusal("design matrix and target must be finite")
        if not np.isfinite(penalty) or penalty < 0:
            raise LibraryRefusal(f"penalty must be finite and non-negative, got {penalty}")
        if budget < 1:
            raise LibraryRefusal(f"budget must be at least one fit, got {budget}")

        self.library = library
        self.design = design
        self.target = target
        self.penalty = float(penalty)
        self.budget = int(budget)
        self.evaluations = 0
        self._cache: dict[tuple[int, ...], Fit] = {}
        self._best: Fit | None = None

    @property
    def m(self) -> int:
        return len(self.library)

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    @property
    def best(self) -> Fit | None:
        return self._best

    @property
    def size(self) -> int:
        return 2 ** len(self.library)

    def cached(self, support: Sequence[int]) -> Fit | None:
        return self._cache.get(tuple(sorted(int(index) for index in support)))

    def evaluate(self, support: Sequence[int]) -> Fit:
        key = tuple(sorted(int(index) for index in support))
        known = self._cache.get(key)
        if known is not None:
            return known
        if self.evaluations >= self.budget:
            raise BudgetExhausted(
                f"{self.evaluations} fits already spent of a {self.budget}-fit budget"
            )
        fit = self._fit(key)
        self._cache[key] = fit
        self.evaluations += 1
        if self._best is None or fit.objective < self._best.objective:
            self._best = fit
        return fit

    def _fit(self, support: tuple[int, ...]) -> Fit:
        target = self.target
        count = target.size
        if not support:
            residual = float(target @ target / count)
            return Fit(support=(), coefficients=(), objective=residual, residual=residual)
        columns = self.library.select(self.design, support)
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(columns, target, rcond=None)
        off = target - columns @ coefficients
        residual = float(off @ off / count)
        penalty = self.penalty * len(support)
        return Fit(
            support=support,
            coefficients=tuple(float(value) for value in coefficients),
            objective=residual + penalty,
            residual=residual,
        )


__all__ = ["BudgetExhausted", "Fit", "SupportObjective"]
