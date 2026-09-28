"""APEDP-Dyn D2: Physical model simulator.

Compiles a PhysicalModel into a numerical ODE system and integrates it
using SciPy's adaptive solvers.

Architecture:
    symbolic model → physical IR → numerical compiler → ODE solver → trajectory

Design principles:
    1. The compile step converts symbolic equations to fast lambdified functions.
    2. Second-order systems are automatically reduced to first-order form.
    3. Integration failures produce structured error reports, never exceptions.
    4. Stiff system detection triggers automatic solver fallback.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import sympy as sp

from apedp.dyn.types import (  # noqa: F401  (re-export surface for dyn users)
    DynamicalEquation,
    PhysicalModel,
    SimulationResult,
    StateVariable,
    TrajectoryData,
    VariableKind,
)

logger = logging.getLogger("apedp.dyn.simulator")


class PhysicalModelSimulator:
    """Compiles a PhysicalModel into a numerical ODE system and integrates it.

    Usage::

        sim = PhysicalModelSimulator(model)
        sim.compile()
        result = sim.simulate(
            t_span=(0, 10),
            initial_conditions={"x": 1.0, "v": 0.0},
            param_values={"k": 4.0, "m": 1.0},
        )

    The simulator converts second-order equations to first-order automatically,
    builds a lambdified RHS function, and integrates with scipy's ``solve_ivp``.
    """

    def __init__(
        self,
        model: PhysicalModel,
        solver: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> None:
        self.model = model
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        self._compiled = False
        self._fo_model: PhysicalModel | None = None
        self._state_order: list[str] = []
        self._lambdified_rhs: Callable | None = None
        self._param_names: list[str] = []
        self._input_names: list[str] = []

    def compile(self) -> None:
        """Convert symbolic equations to a fast numerical RHS function.

        Steps:
            1. Convert model to first-order form if any equation has order > 1.
            2. Order states consistently.
            3. Build a single vectorised lambdified RHS.
            4. Store compiled state for ``simulate()``.

        Raises:
            ValueError: If a state has no governing equation.
        """
        # Convert to first order
        fo = self.model.to_first_order()
        self._fo_model = fo
        self._state_order = [s.name for s in fo.states]
        self._param_names = [p.name for p in fo.parameters]
        self._input_names = [inp.name for inp in fo.inputs]

        state_syms = [s.symbol for s in fo.states]
        param_syms = [p.symbol for p in fo.parameters]
        input_syms = [inp.symbol for inp in fo.inputs]
        t_sym = sp.Symbol("t")

        # Build ordered RHS expressions
        rhs_exprs: list[sp.Expr] = []
        for state in fo.states:
            eq = next(
                (e for e in fo.equations if e.lhs == state.symbol and e.order == 1),
                None,
            )
            if eq is None:
                raise ValueError(
                    f"No first-order equation found for state '{state.name}' "
                    f"(symbol: {state.symbol}). Available equations govern: "
                    f"{[str(e.lhs) for e in fo.equations]}"
                )
            rhs_exprs.append(eq.rhs)

        # Build a single vectorised lambda: f(t, y_vec, p_vec, u_vec) -> dydt_vec
        all_args = [t_sym, *state_syms, *param_syms, *input_syms]
        rhs_matrix = sp.Matrix(rhs_exprs)

        self._lambdified_rhs = sp.lambdify(
            all_args, rhs_matrix, modules=["numpy"]
        )
        self._compiled = True
        logger.info(
            "Compiled %s: %d states, %d params, %d inputs",
            fo.name,
            len(state_syms),
            len(param_syms),
            len(input_syms),
        )

    def simulate(
        self,
        t_span: tuple[float, float],
        initial_conditions: dict[str, float],
        param_values: dict[str, float] | None = None,
        input_funcs: dict[str, Callable[[float], float]] | None = None,
        t_eval: np.ndarray | None = None,
        max_step: float = 0.01,
    ) -> SimulationResult:
        """Integrate the ODE system.

        Args:
            t_span: ``(t0, tf)`` integration interval.
            initial_conditions: ``{state_name: value}`` for every state.
            param_values: ``{param_name: value}``; uses model defaults if None.
            input_funcs: ``{input_name: callable(t) -> value}``.
            t_eval: Specific time points for output.
            max_step: Maximum integration step size.

        Returns:
            SimulationResult with trajectory data on success, or with
            ``success=False`` and an error message on failure.
        """
        if not self._compiled or self._fo_model is None:
            return SimulationResult(
                success=False,
                error="Model not compiled. Call compile() first.",
            )

        fo = self._fo_model
        if param_values is None:
            param_values = {}

        if input_funcs is None:
            input_funcs = {}

        # Build initial state vector
        try:
            y0 = np.array(
                [
                    initial_conditions[name]
                    for name in self._state_order
                ],
                dtype=float,
            )
        except KeyError:
            missing = set(self._state_order) - set(initial_conditions)
            return SimulationResult(
                success=False,
                error=(
                    f"Missing initial conditions for states: {missing}. "
                    f"Required: {self._state_order}"
                ),
            )

        # Build parameter vector
        p_vec = []
        for pname in self._param_names:
            if pname in param_values:
                p_vec.append(float(param_values[pname]))
            else:
                # Use model default
                param = next(
                    (p for p in fo.parameters if p.name == pname), None
                )
                if param is not None and param.value is not None:
                    p_vec.append(float(param.value))
                else:
                    return SimulationResult(
                        success=False,
                        error=(
                            f"No value for parameter '{pname}'. "
                            f"Provide it in param_values or set a default "
                            f"in the model."
                        ),
                    )
        p_arr = np.array(p_vec, dtype=float)

        # Build RHS wrapper for solve_ivp.  compile() must have run first:
        # simulate() guards on _compiled above, so the callable is set.
        assert self._lambdified_rhs is not None
        lambdified = self._lambdified_rhs
        input_names = self._input_names

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            # Evaluate input functions at time t
            u_vals = [
                float(input_funcs[name](t)) if name in input_funcs else 0.0
                for name in input_names
            ]
            # Build full argument list: [t, y0, y1, ..., p0, p1, ..., u0, u1, ...]
            args = [t, *list(y), *list(p_arr), *u_vals]
            result = lambdified(*args)
            # lambdify with Matrix returns a (n, 1) matrix
            out = np.asarray(result, dtype=float).ravel()
            # Check for NaN/Inf
            if not np.all(np.isfinite(out)):
                raise ValueError(
                    f"Non-finite derivative at t={t:.6f}: "
                    f"dydt={out}, y={y}"
                )
            return out

        # Integrate
        try:
            from scipy.integrate import solve_ivp

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_ivp(
                    rhs,
                    t_span,
                    y0,
                    method=self.solver,
                    t_eval=t_eval,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_step=max_step,
                    dense_output=True,
                )

            if not sol.success:
                # Try stiff solver fallback
                if self.solver != "Radau":
                    logger.info(
                        "Primary solver %s failed: %s. "
                        "Falling back to Radau (stiff solver).",
                        self.solver,
                        sol.message,
                    )
                    sol = solve_ivp(
                        rhs,
                        t_span,
                        y0,
                        method="Radau",
                        t_eval=t_eval,
                        rtol=self.rtol,
                        atol=self.atol,
                        max_step=max_step,
                    )

            if not sol.success:
                return SimulationResult(
                    success=False,
                    error=f"Integration failed: {sol.message}",
                    solver_info={"method": self.solver, "nfev": sol.nfev},
                )

            # Build trajectory data
            states_dict: dict[str, np.ndarray] = {}
            for i, name in enumerate(self._state_order):
                states_dict[name] = sol.y[i]

            inputs_dict: dict[str, np.ndarray] = {}
            for name in input_names:
                if name in input_funcs:
                    inputs_dict[name] = np.array(
                        [input_funcs[name](t) for t in sol.t]
                    )

            trajectory = TrajectoryData(
                t=sol.t,
                states=states_dict,
                inputs=inputs_dict,
                metadata={
                    "solver": self.solver,
                    "nfev": int(sol.nfev),
                    "t_span": t_span,
                },
            )

            return SimulationResult(
                trajectory=trajectory,
                success=True,
                solver_info={
                    "method": self.solver,
                    "nfev": int(sol.nfev),
                    "message": sol.message,
                },
            )

        except ValueError as e:
            return SimulationResult(
                success=False,
                error=f"Numerical error during integration: {e}",
            )
        except Exception as e:  # noqa: BLE001 -- structured error report, never raise
            return SimulationResult(
                success=False,
                error=f"Unexpected error: {type(e).__name__}: {e}",
            )
