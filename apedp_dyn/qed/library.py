"""The mathematical vocabulary a discovered equation is allowed to use.

Status: experimental

A library is a fixed, named list of scalar functions of the observed state, so
that "which terms are in the governing equation" becomes a subset-selection
problem instead of an unconstrained expression search. The library is the whole
of the hypothesis space: a term absent here can never be proposed, and a term
present here can only be rejected on grounds the data supplies.

Nothing in this module decides anything. It supplies columns; the objective in
:mod:`apedp_dyn.qed.objective` scores a selection and the referee in
:mod:`apedp_dyn.qed.referee` adjudicates a candidate.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# A column builder maps an (N, k) array of observed states to an (N,) column.
ColumnBuilder = Callable[[np.ndarray], np.ndarray]


class LibraryRefusal(ValueError):
    """Raised when a library or a design matrix cannot be used as declared."""


@dataclass(frozen=True)
class Term:
    """One named candidate term, e.g. ``x`` or ``sin(v)``."""

    name: str
    column: ColumnBuilder

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(self.column(np.asarray(states, dtype=float)), dtype=float).ravel()
        if values.shape != (states.shape[0],):
            raise LibraryRefusal(
                f"term {self.name!r} returned shape {values.shape} for {states.shape[0]} rows"
            )
        if not np.all(np.isfinite(values)):
            raise LibraryRefusal(f"term {self.name!r} produced a non-finite column")
        return values


@dataclass(frozen=True)
class TermLibrary:
    """An ordered, named set of terms.

    The order is the bit order of a selection: term ``j`` is selected when bit
    ``j`` of a support mask is set.
    """

    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise LibraryRefusal("a term library needs at least one term")
        names = [term.name for term in self.terms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise LibraryRefusal(f"duplicate term names: {duplicates}")

    @classmethod
    def from_specs(cls, specs: Iterable[tuple[str, ColumnBuilder]]) -> TermLibrary:
        return cls(terms=tuple(Term(name=name, column=fn) for name, fn in specs))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def index_of(self, name: str) -> int:
        for position, term in enumerate(self.terms):
            if term.name == name:
                return position
        raise LibraryRefusal(f"no term named {name!r} in this library")

    def support_of(self, names: Sequence[str]) -> tuple[int, ...]:
        return tuple(sorted(self.index_of(name) for name in names))

    def design_matrix(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2:
            raise LibraryRefusal(f"states must be 2-D, got shape {states.shape}")
        columns = [term.evaluate(states) for term in self.terms]
        return np.column_stack(columns)

    def select(self, design: np.ndarray, support: Sequence[int]) -> np.ndarray:
        support = tuple(sorted(int(index) for index in support))
        if len(set(support)) != len(support):
            raise LibraryRefusal(f"support repeats an index: {support}")
        if any(index < 0 or index >= len(self.terms) for index in support):
            raise LibraryRefusal(f"support {support} is out of range for {len(self.terms)} terms")
        return design[:, list(support)] if support else np.empty((design.shape[0], 0))

    def render(
        self,
        support: Sequence[int],
        coefficients: Sequence[float],
        *,
        lhs: str = "dy/dt",
    ) -> str:
        """Render ``lhs = c0*term0 + ...`` with explicit signs."""
        if len(support) != len(coefficients):
            raise LibraryRefusal(
                f"support {len(support)} and coefficients {len(coefficients)} disagree"
            )
        if not support:
            return f"{lhs} = 0"
        parts = []
        for index, coefficient in zip(support, coefficients, strict=True):
            name = self.terms[index].name
            body = f"{float(coefficient):+.6g}" if name == "1" else f"{float(coefficient):+.6g}*{name}"
            parts.append(body)
        return f"{lhs} = " + " ".join(parts)


# The library the recovery experiment uses: a damped harmonic oscillator's
# acceleration written in the observed state ``(x, v)``. It is fixed here, not
# per-task, so that "the truth support is {x, v}" cannot be tuned per seed.
OSCILLATOR_TERMS: tuple[tuple[str, ColumnBuilder], ...] = (
    ("1", lambda s: np.ones(s.shape[0])),
    ("x", lambda s: s[:, 0]),
    ("v", lambda s: s[:, 1]),
    ("x^2", lambda s: s[:, 0] ** 2),
    ("v^2", lambda s: s[:, 1] ** 2),
    ("xv", lambda s: s[:, 0] * s[:, 1]),
    ("x^3", lambda s: s[:, 0] ** 3),
    ("v^3", lambda s: s[:, 1] ** 3),
    ("sin(x)", lambda s: np.sin(s[:, 0])),
    ("sin(v)", lambda s: np.sin(s[:, 1])),
)


def oscillator_library() -> TermLibrary:
    """The ten-term library of :data:`OSCILLATOR_TERMS`."""
    return TermLibrary.from_specs(OSCILLATOR_TERMS)


__all__ = [
    "OSCILLATOR_TERMS",
    "ColumnBuilder",
    "LibraryRefusal",
    "Term",
    "TermLibrary",
    "oscillator_library",
]
