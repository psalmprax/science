"""D1 model variables: states, exogenous inputs, constant parameters."""
from __future__ import annotations

from dataclasses import dataclass

TIME_SYMBOL = "t"


@dataclass(frozen=True)
class StateVariable:
    """A dynamical state with initial condition. Units of the quantity itself."""

    name: str
    units: str
    initial: float


@dataclass(frozen=True)
class InputVariable:
    """An exogenous input u(t) prescribed by a signal, not solved for."""

    name: str
    units: str


@dataclass(frozen=True)
class Parameter:
    """A constant parameter.

    value=None means unidentified (admissibility gate must fail).
    bounds=None means unbounded (admissible for hand-specified models);
    grammar-materialized candidates always carry finite bounds, which the
    admissibility gate verifies and the fitter respects.
    """

    name: str
    units: str
    value: float | None
    bounds: tuple[float, float] | None = None
    # "rate" multiplies a grammar term; "scale" is an interior reference level
    # (e.g. a saturation half-saturation constant) that the fitter initializes
    # from the data scale rather than differentiating the equation for it.
    role: str = "rate"
