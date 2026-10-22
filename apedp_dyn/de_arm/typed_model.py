"""DE arm, layer A: typed mathematical semantics (external-review step 1).

The recap's central correction: AD, ODE systems and PDE systems are NOT
three more node kinds — they are three KINDS of typed mathematical model
built FROM the shared expression vocabulary. This module declares the
typing layer every DE discovery candidate must carry, independent of any
evaluator:

  STATIC_EXPRESSION  — subject-free relation Y = f(X): no states, no
                       independent variable handling beyond evaluation.
  ODE_SYSTEM         — state variables (unique, non-empty), a declared
                       independent variable, a positive step, and equations
                       per state. Validation refuses incompleteness.
  PDE_SYSTEM         — field variables, independent variables (at least one
                       spatial), a positive step, and a declared boundary
                       treatment (``none``/``periodic``/``dirichlet``).

Typed semantics gate what evaluation may be attempted: a spatial derivative must
not be applied to a scalar constant, an ODE integrator must not accept an
expression without a declared state and independent variable — the typed
model is what makes those checks possible BEFORE a search runs.

Status: experimental (DE-discovery arm phase 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ModelKind",
    "OdeMathModel",
    "PdeMathModel",
    "StaticExpressionModel",
    "TypedMathModel",
]


class ModelKind(str, Enum):
    """Kind of typed mathematical model (the reviewer's central distinction)."""

    STATIC_EXPRESSION = "static_expression"
    ODE_SYSTEM = "ode_system"
    PDE_SYSTEM = "pde_system"


@dataclass(frozen=True)
class StaticExpressionModel:
    """A subject-free relation Y = f(X) — no dynamics attached."""

    kind: str = "static_expression"
    target: str = ""
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind != ModelKind.STATIC_EXPRESSION.value:
            raise ValueError("StaticExpressionModel must declare kind 'static_expression'")
        if not self.target:
            raise ValueError(
                "static expression model without a declared response variable — "
                "typing is what later checks (units, dependencies) stand on")
        if not self.features:
            raise ValueError(
                "static expression model without declared feature variables — "
                "the dimensional checks cannot run on an undeclared domain")


@dataclass(frozen=True)
class OdeMathModel:
    """A declared state equation system: dx/dt = F(x), with x(0) declared."""

    kind: str = "ode_system"
    states: tuple[str, ...] = ()
    independent_variable: str = "t"
    dt: float = 0.0
    initial_conditions: dict[str, float] = field(default_factory=dict)
    observable_limit: int = 0  # rows actually observed for the declared model
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != ModelKind.ODE_SYSTEM.value:
            raise ValueError("OdeMathModel must declare kind 'ode_system'")
        if not self.states:
            raise ValueError(
                "ODE system model must declare at least one state variable — an "
                "integrator without a declared state cannot be typed")
        if len(set(self.states)) != len(self.states):
            raise ValueError("ODE state variables must be unique")
        if any(not isinstance(s, str) or not s for s in self.states):
            raise ValueError("ODE state names must be non-empty strings")
        if not self.independent_variable:
            raise ValueError("ODE system model must declare its independent variable")
        if not (self.dt > 0):
            raise ValueError("ODE system model requires a positive declared step dt")
        for s in self.states:
            ic = self.initial_conditions.get(s)
            if ic is None or not (_is_finite_float(ic)):
                raise ValueError(
                    f"ODE state '{s}' has no declared finite initial condition — "
                    "trajectory evaluation cannot begin")


@dataclass(frozen=True)
class PdeMathModel:
    """A declared field-equation system on an independent-variable domain."""

    kind: str = "pde_system"
    fields: tuple[str, ...] = ()
    independent_variables: tuple[str, ...] = ()
    dx: dict[str, float] = field(default_factory=dict)
    dt: float = 0.0
    boundary: str = "none"  # none | periodic | dirichlet
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != ModelKind.PDE_SYSTEM.value:
            raise ValueError("PdeMathModel must declare kind 'pde_system'")
        if not self.fields:
            raise ValueError("PDE system model must declare at least one field variable")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("PDE field variables must be unique")
        if len(set(self.independent_variables)) != len(self.independent_variables):
            raise ValueError("PDE independent variables must be unique")
        if len(self.independent_variables) < 2:
            raise ValueError(
                "PDE system model needs at least one temporal and one spatial "
                "independent variable declared — a field without a domain is "
                "not a typed PDE")
        for iv in self.independent_variables:
            step = self.dx.get(iv)
            if iv != self.independent_variables[0]:
                # the FIRST declared independent variable is the temporal one
                if step is None or not (isinstance(step, float) and step > 0):
                    raise ValueError(
                        f"PDE spatial axis '{iv}' has no positive declared step dx — "
                        "spatial operators refuse to run typelessly")
        if not (self.dt > 0):
            raise ValueError("PDE system model requires a positive declared step dt")
        if self.boundary not in ("none", "periodic", "dirichlet"):
            raise ValueError(
                f"unknown boundary treatment {self.boundary!r}: PDE evaluation on "
                "an undeclared boundary fabricates residual values")


def _is_finite_float(v: Any) -> bool:
    try:
        return float(v) == float(v) and abs(float(v)) != float("inf")
    except (TypeError, ValueError):
        return False


TypedMathModel = StaticExpressionModel | OdeMathModel | PdeMathModel
