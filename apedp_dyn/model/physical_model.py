"""D1 physical model: an executable first-order dynamical law."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from apedp_dyn.model.equations import DynamicalEquation
from apedp_dyn.model.variables import (
    TIME_SYMBOL,
    InputVariable,
    Parameter,
    StateVariable,
)


@dataclass(frozen=True)
class PhysicalModel:
    """An executable first-order dynamical law (D1 contract)."""

    states: tuple[StateVariable, ...]
    inputs: tuple[InputVariable, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    equations: tuple[DynamicalEquation, ...] = ()
    t_units: str = "second"
    # DAE provenance. The materializer knows which DAE mode it applied and
    # which parameters carry constraint/timescale semantics; the model used to
    # drop this, so gate 1.5 had no mode to dispatch on and its constrained and
    # multi-timescale branches were unreachable.
    dae_mode: str = "none"
    dae_constraint_params: tuple[str, ...] = ()
    dae_timescales: tuple[str, ...] = ()

    def symbol_units(self) -> dict[str, str]:
        """Map every known symbol (incl. time) to its unit string."""
        out = {s.name: s.units for s in self.states}
        out.update({u.name: u.units for u in self.inputs})
        out.update({p.name: p.units for p in self.parameters})
        out[TIME_SYMBOL] = self.t_units
        return out

    def initial_vector(self) -> np.ndarray:
        return np.array([s.initial for s in self.states], dtype=float)

    def param_values(self) -> dict[str, float]:
        missing = [p.name for p in self.parameters if p.value is None]
        if missing:
            raise ValueError(f"unidentified parameters: {missing}")
        return {p.name: float(p.value) for p in self.parameters}  # type: ignore[arg-type]

    def rhs(self, t: float, x: np.ndarray, u_of_t) -> np.ndarray:
        """Right-hand side f(t, x, u). u_of_t: callable t -> array-like per input."""
        order = [s.name for s in self.states]
        eq_by_state = {e.state: e.sympy_expr for e in self.equations}
        u_vec = np.atleast_1d(np.asarray(u_of_t(t), dtype=float))
        subs: dict = {sp.Symbol(TIME_SYMBOL): t}
        subs.update({sp.Symbol(n): v for n, v in zip(order, x, strict=True)})
        inames = [u.name for u in self.inputs]
        if inames:
            # Checked rather than truncated: an arity mismatch must be an error,
            # not a silently dropped input.
            if len(u_vec) != len(inames):
                raise ValueError(
                    f"input arity mismatch: expected {len(inames)} input(s) "
                    f"{inames}, got {len(u_vec)}")
            subs.update({sp.Symbol(n): v
                         for n, v in zip(inames, u_vec, strict=True)})
        subs.update({sp.Symbol(k): v for k, v in self.param_values().items()})
        out = np.empty(len(order))
        for i, name in enumerate(order):
            val = complex(eq_by_state[name].subs(subs).evalf())
            if val.imag != 0:
                raise ValueError(f"non-real RHS for state {name!r}")
            out[i] = val.real
        return out
