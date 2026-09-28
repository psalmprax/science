"""APEDP-Dyn D1: Core type system for dynamical systems discovery.

Introduces the foundational types that distinguish physical modeling from
static symbolic regression:

* **StateVariable** — observable or latent system state
* **InputVariable** — externally controlled forcing
* **Parameter** — approximately constant physical property
* **DynamicalEquation** — a single ODE: d(lhs)/dt = rhs
* **Observation** — maps measured quantities to states
* **PhysicalModel** — complete executable model object

Design principles:
    1. Variable classification (observable/latent/input/parameter) is
       first-class rather than inferred post-hoc.
    2. PhysicalModel is mutable during refinement but each validation
       result is frozen (immutable evidence).
    3. Epistemic status is a vector, not a scalar — structural,
       parameter, dynamic, and physical confidence are tracked separately.
    4. All result types extend DictCompatibleResult for backwards
       compatibility with dict-consuming code.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.discovery_types import DictCompatibleResult

logger = logging.getLogger("apedp.dyn.types")


# ── Variable taxonomy ─────────────────────────────────────────────────────────

class VariableKind(str, Enum):
    """Taxonomic classification of a physical variable.

    A generic symbolic regressor sees everything as "variables."
    APEDP distinguishes:
      OBSERVABLE  — directly measured by sensors
      LATENT      — not directly measured, needed for dynamics
      INPUT       — externally controlled by the experimenter
      PARAMETER   — approximately constant during an experiment
    """
    OBSERVABLE = "observable"
    LATENT = "latent"
    INPUT = "input"
    PARAMETER = "parameter"


# ── Core variable types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateVariable:
    """A dynamical state variable (observable or latent).

    Attributes:
        name:        Human-readable identifier, e.g. ``"x"``, ``"theta"``.
        symbol:      SymPy symbol used in equations.
        unit:        Pint-compatible unit string, e.g. ``"meter"``, ``"radian"``.
        kind:        Whether this state is directly observed or latent.
        description: Optional free-text description.
        bounds:      Physical domain constraints, e.g. ``(0.0, float('inf'))``
                     for a concentration that must be non-negative.
    """
    name: str
    symbol: sp.Symbol
    unit: str
    kind: VariableKind = VariableKind.OBSERVABLE
    description: str = ""
    bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class InputVariable:
    """An externally controlled input / forcing signal.

    Inputs are manipulated by the experimenter and are not evolved by the
    system dynamics.  Examples: applied force F(t), voltage V(t), flow rate.
    """
    name: str
    symbol: sp.Symbol
    unit: str
    description: str = ""


@dataclass(frozen=True)
class Parameter:
    """A physical parameter — approximately constant during an experiment.

    Parameters are estimated from data, not evolved in time.
    Examples: mass ``m``, spring constant ``k``, damping coefficient ``c``.

    Attributes:
        value:        Identified numerical value (None until estimated).
        bounds:       Search bounds for parameter estimation.
        identifiable: Set to True/False after identifiability analysis;
                      None means unchecked.
    """
    name: str
    symbol: sp.Symbol
    unit: str
    value: float | None = None
    bounds: tuple[float, float] = (-1e6, 1e6)
    description: str = ""
    identifiable: bool | None = None


# ── Observation model ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """Maps a measured quantity to an internal state variable.

    Real experiments often give y(t) = h(x(t)) + ε rather than x(t) directly.
    For example, a sensor may measure voltage but the internal state is charge.

    Attributes:
        name:           Sensor/measurement channel name.
        state_variable: Which state this observation measures.
        unit:           Unit of the *measurement* (may differ from state unit).
        transform:      Observation function h(x); ``None`` means identity.
        noise_std:      Estimated measurement noise standard deviation.
    """
    name: str
    state_variable: str
    unit: str
    transform: sp.Expr | None = None
    noise_std: float | None = None


# ── Dynamical equation ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DynamicalEquation:
    """A single ODE component: d^n(lhs)/dt^n = rhs.

    For first-order systems: dx/dt = f(x, u, p, t).
    For second-order systems (e.g. Newton's law): d²x/dt² = g(x, dx/dt, u, p, t).

    Attributes:
        lhs:   The state symbol being differentiated.
        rhs:   The right-hand-side expression.
        order: Derivative order (1 or 2).
    """
    lhs: sp.Symbol
    rhs: sp.Expr
    order: int = 1


# ── Trajectory data ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrajectoryData:
    """A recorded or simulated trajectory.

    Attributes:
        t:        Time points array.
        states:   Mapping from state name to values array.
        inputs:   Mapping from input name to values array (empty if autonomous).
        metadata: Arbitrary metadata (solver info, provenance, etc.).
    """
    t: np.ndarray
    states: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Intervention test ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InterventionTest:
    """Specifies a counterfactual intervention for model validation.

    Instead of only asking "Does it reproduce the measured data?", ask:
    "What does it predict if I double the initial displacement?"
    "What happens if the forcing frequency changes?"

    Attributes:
        name:                 Human-readable test description.
        modified_params:      Changed parameter values.
        modified_ic:          Changed initial conditions.
        modified_inputs:      Changed input functions (callables).
        reference_trajectory: Ground-truth trajectory for comparison.
    """
    name: str
    modified_params: dict[str, float] = field(default_factory=dict)
    modified_ic: dict[str, float] = field(default_factory=dict)
    modified_inputs: dict[str, Any] = field(default_factory=dict)
    reference_trajectory: TrajectoryData | None = None


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationResult(DictCompatibleResult):
    """Result of numerical ODE integration.

    Attributes:
        trajectory:  The computed trajectory (None on failure).
        success:     Whether integration completed without error.
        error:       Error message on failure.
        solver_info: Solver diagnostics (step count, function evaluations, etc.).
    """
    trajectory: TrajectoryData | None = None
    success: bool = False
    error: str | None = None
    solver_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult(DictCompatibleResult):
    """Result of trajectory or intervention validation.

    Attributes:
        mse:            Mean squared error across all states.
        r_squared:      Coefficient of determination.
        max_error:      Maximum absolute error.
        relative_error: Mean relative error.
        passed:         Whether the validation passed its thresholds.
        test_type:      Type of test: "trajectory", "intervention", "conservation".
        details:        Per-state or per-invariant detail metrics.
        note:           Free-text annotation.
    """
    mse: float = float("inf")
    r_squared: float = 0.0
    max_error: float = float("inf")
    relative_error: float = float("inf")
    passed: bool = False
    test_type: str = "trajectory"
    details: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class EpistemicStatus(DictCompatibleResult):
    """Multi-dimensional epistemic status vector.

    Unlike a single confidence score, this separates:
      structural  — is the functional form supported?
      parameter   — are the parameters identifiable?
      dynamic     — does the model simulate correctly?
      physical    — are physics constraints satisfied?
      ood         — does the model generalize?
      overall     — combined assessment
    """
    structural: str = "UNVERIFIED"
    parameter: str = "UNESTIMATED"
    dynamic: str = "UNSIMULATED"
    physical: str = "UNCHECKED"
    ood: str = "UNTESTED"
    overall: str = "PENDING"


# ── Physical Model ────────────────────────────────────────────────────────────

@dataclass
class PhysicalModel(DictCompatibleResult):
    """Complete physical system model: states + dynamics + parameters + constraints.

    This is the central model object for APEDP-Dyn.  It contains more than
    an equation — it represents an executable physical system that can be
    simulated, validated, and falsified.

    A complete model includes:
        - What are the states?         → states, inputs
        - What governs evolution?      → equations
        - What are the parameters?     → parameters
        - What constraints must hold?  → constraints, invariants
        - Does model predict new data? → via simulation + validation

    Mutable during refinement (not frozen), but evidence objects
    (SimulationResult, ValidationResult) are frozen.
    """
    name: str = "unnamed"
    states: list[StateVariable] = field(default_factory=list)
    inputs: list[InputVariable] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    equations: list[DynamicalEquation] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    # Constraints and invariants
    constraints: list[sp.Expr] = field(default_factory=list)   # inequality >= 0
    invariants: list[sp.Expr] = field(default_factory=list)    # conservation = 0

    # Metadata
    domain: str | None = None
    validity_region: dict[str, tuple[float, float]] = field(default_factory=dict)
    initial_conditions: dict[str, float] = field(default_factory=dict)

    # Epistemic status
    structural_status: str = "UNVERIFIED"
    parameter_status: str = "UNESTIMATED"
    dynamic_status: str = "UNSIMULATED"
    proof_status: str = "P0"

    # ── Derived properties ────────────────────────────────────────────────

    def state_symbols(self) -> list[sp.Symbol]:
        """Ordered list of state symbols."""
        return [s.symbol for s in self.states]

    def param_symbols(self) -> list[sp.Symbol]:
        """Ordered list of parameter symbols."""
        return [p.symbol for p in self.parameters]

    def param_values(self) -> dict[sp.Symbol, float]:
        """Mapping from parameter symbol to identified value.

        Only includes parameters whose ``value`` is not None.
        """
        return {
            p.symbol: p.value
            for p in self.parameters
            if p.value is not None
        }

    def n_states(self) -> int:
        """Number of state variables (including latent)."""
        return len(self.states)

    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def is_autonomous(self) -> bool:
        """Whether the system has no explicit time dependence or external inputs.

        A system is autonomous if:
          1. There are no InputVariables.
          2. No equation RHS contains the symbol ``t``.
        """
        if self.inputs:
            return False
        t_sym = sp.Symbol("t")
        for eq in self.equations:
            if t_sym in eq.rhs.free_symbols:
                return False
        return True

    def get_epistemic_status(self) -> EpistemicStatus:
        """Build multi-dimensional epistemic status vector."""
        return EpistemicStatus(
            structural=self.structural_status,
            parameter=self.parameter_status,
            dynamic=self.dynamic_status,
        )

    def to_first_order(self) -> PhysicalModel:
        """Convert all second-order equations to first-order form.

        For each DynamicalEquation with ``order == 2`` for state ``x``,
        introduces a latent velocity state ``v_x`` with:
          - dx/dt = v_x
          - dv_x/dt = original RHS (with dx/dt replaced by v_x)

        Returns a new PhysicalModel; does not mutate ``self``.
        """
        has_second_order = any(eq.order > 1 for eq in self.equations)
        if not has_second_order:
            return self  # already first-order

        new_states = list(self.states)
        new_equations = []
        velocity_map: dict[sp.Symbol, sp.Symbol] = {}

        for eq in self.equations:
            if eq.order == 1:
                new_equations.append(eq)
            elif eq.order == 2:
                # Find the state this equation governs
                state = next(
                    (s for s in self.states if s.symbol == eq.lhs),
                    None,
                )
                if state is None:
                    raise ValueError(
                        f"Second-order equation references unknown state "
                        f"'{eq.lhs}'"
                    )
                # Reuse an existing velocity state if one is already declared
                # (the natural modelling style when the identifier knows the
                # system is second-order).  Creating a duplicate state would
                # produce a duplicate lambdify argument later and break
                # compilation with "duplicate argument 'v_x'".
                v_name = f"v_{state.name}"
                existing = next(
                    (s for s in self.states if s.name == v_name),
                    None,
                )
                if existing is not None:
                    v_sym = existing.symbol
                else:
                    v_sym = sp.Symbol(v_name)
                    # Determine velocity unit from state unit
                    v_unit = f"{state.unit}/second"
                    v_state = StateVariable(
                        name=v_name,
                        symbol=v_sym,
                        unit=v_unit,
                        kind=VariableKind.LATENT,
                        description=f"Velocity of {state.name}",
                    )
                    new_states.append(v_state)
                velocity_map[eq.lhs] = v_sym

                # dx/dt = v_x
                new_equations.append(DynamicalEquation(
                    lhs=eq.lhs,
                    rhs=v_sym,
                    order=1,
                ))

                # dv_x/dt = rhs with Derivative(x, t) replaced by v_x (per
                # the docstring contract).  Leaving an unevaluated
                # Derivative in the RHS would break lambdify.
                rhs = eq.rhs.subs(sp.Derivative(eq.lhs, sp.Symbol("t")), v_sym)
                new_equations.append(DynamicalEquation(
                    lhs=v_sym,
                    rhs=rhs,
                    order=1,
                ))
            else:
                raise ValueError(
                    f"Equations of order {eq.order} are not supported "
                    f"(only 1 and 2)"
                )

        result = copy.copy(self)
        result.states = new_states
        result.equations = new_equations
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for JSON export."""
        return {
            "name": self.name,
            "n_states": self.n_states(),
            "n_params": self.n_params(),
            "autonomous": self.is_autonomous(),
            "states": [s.name for s in self.states],
            "parameters": [
                {"name": p.name, "value": p.value, "unit": p.unit}
                for p in self.parameters
            ],
            "equations": [
                {"lhs": str(eq.lhs), "rhs": str(eq.rhs), "order": eq.order}
                for eq in self.equations
            ],
            "structural_status": self.structural_status,
            "parameter_status": self.parameter_status,
            "dynamic_status": self.dynamic_status,
            "proof_status": self.proof_status,
            "domain": self.domain,
        }

    def __repr__(self) -> str:
        eqs = "; ".join(
            f"d{eq.lhs}/dt = {eq.rhs}" if eq.order == 1
            else f"d²{eq.lhs}/dt² = {eq.rhs}"
            for eq in self.equations
        )
        return (
            f"PhysicalModel('{self.name}', "
            f"states={self.n_states()}, "
            f"params={self.n_params()}, "
            f"eqs=[{eqs}])"
        )
