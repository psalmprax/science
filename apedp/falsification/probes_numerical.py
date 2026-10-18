"""Numerical axis: NaN/Inf injection, overflow and conditioning collapse."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp

from apedp.evidence import EvidenceTier
from apedp.safety import quiet_float_warnings

from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis


class _NumericalProbesMixin:
    """Numerical axis: NaN/Inf injection, overflow and conditioning collapse."""

    @quiet_float_warnings()
    def _probe_numerical_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]]
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Scans dense logarithmic and linear grids for floating point overflow, underflow, and gradient explosions."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {
            "nan_count": 0,
            "inf_count": 0,
            "grad_explosions": 0,
            "probe_errors": [],
        }

        syms = list(expr.free_symbols)
        if not syms:
            # A model with no free variable has no conditioning to measure: the
            # whole axis is unevaluable, which is not the same as clean.
            self._record_probe_error(
                metrics,
                "NumericalStabilityProbe",
                "NoFreeVariables",
                "constant model: no conditioning is measurable across a domain",
            )
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
            return ces, metrics

        fn = sp.lambdify(syms, expr, modules=["numpy", "math"])

        # Grid testing
        num_samples = 40
        grids: dict[str, np.ndarray] = {}
        for s in syms:
            s_name = str(s)
            low, high = domains.get(s_name, (0.01, 10.0))
            if low > 0 and high / low > 100:
                grids[s_name] = np.geomspace(low, high, num_samples)
            else:
                grids[s_name] = np.linspace(low, high, num_samples)

        # Evaluate across grid combinations
        primary_sym = syms[0]
        primary_grid = grids[str(primary_sym)]
        fixed_args = [np.median(grids[str(s)]) for s in syms[1:]]

        y_vals: list[float] = []
        for x_val in primary_grid:
            try:
                args = [x_val, *fixed_args]
                res = float(fn(*args))
                if math.isnan(res):
                    metrics["nan_count"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="NaNInjectionProbe", failure_class=FailureClass.NON_FINITE_VALUE,
                        counterexample_point={str(primary_sym): float(x_val)},
                        violation_description=f"NaN encountered at {primary_sym!s}={x_val:.4e}.",
                        severity="FATAL"
                    ))
                    break
                if math.isinf(res):
                    metrics["inf_count"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="InfinityOverflowProbe", failure_class=FailureClass.NON_FINITE_VALUE,
                        counterexample_point={str(primary_sym): float(x_val)},
                        violation_description=f"Numerical overflow (+/- inf) at {primary_sym!s}={x_val:.4e}.",
                        severity="FATAL"
                    ))
                    break
                y_vals.append(res)
            except (ValueError, TypeError, ArithmeticError, AttributeError) as e:
                ces.append(Counterexample(
                    axis=FalsificationAxis.NUMERICAL,
                    probe_name="EvaluationExceptionProbe", failure_class=FailureClass.EVALUATION_FAILURE,
                    counterexample_point={str(primary_sym): float(x_val)},
                    violation_description=f"Evaluation crashed at {primary_sym!s}={x_val:.4e}: {e}",
                    severity="FATAL"
                ))
                break

        # Check normalized condition & gradient stability
        if len(y_vals) >= 5:
            low_x, high_x = domains.get(str(primary_sym), (primary_grid[0], primary_grid[-1]))
            # For multidecade variables (spanning >2 decades) with strictly positive outputs,
            # use scale-invariant logarithmic elasticity |d ln y / d ln x|
            if low_x > 0 and (high_x / low_x) > 100.0 and all(y > 0 for y in y_vals):
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_x = np.log(np.array(primary_grid[:len(y_vals)], dtype=float))
                    log_y = np.log(np.array(y_vals, dtype=float))
                    d_log_x = np.diff(log_x)
                    d_log_y = np.diff(log_y)
                    elasticity = np.abs(d_log_y / (d_log_x + 1e-15))
                max_elasticity = float(np.nanmax(elasticity)) if len(elasticity) > 0 else 0.0
                metrics["max_elasticity"] = max_elasticity
                if max_elasticity > 50.0 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                    metrics["grad_explosions"] += 1
                    ces.append(Counterexample(
                        axis=FalsificationAxis.NUMERICAL,
                        probe_name="GradientExplosionProbe",
                        counterexample_point={str(primary_sym): float(primary_grid[np.nanargmax(elasticity)])},
                        failure_class=FailureClass.ILL_CONDITIONED,
                        violation_description=f"Numerical elasticity exploded (|d ln y / d ln x| = {max_elasticity:.2e}), indicating unphysical discontinuity.",
                        severity="HIGH",
                        metric_value=max_elasticity
                    ))
            else:
                y_range = max(y_vals) - min(y_vals)
                x_range = float(primary_grid[len(y_vals) - 1] - primary_grid[0])
                if y_range > 1e-12 and x_range > 1e-12:
                    dy_norm = np.diff(y_vals) / y_range
                    dx_norm = np.diff(primary_grid[:len(y_vals)]) / x_range
                    with np.errstate(divide="ignore", invalid="ignore"):
                        norm_gradients = np.abs(dy_norm / (dx_norm + 1e-15))
                    max_norm_grad = float(np.nanmax(norm_gradients)) if len(norm_gradients) > 0 else 0.0
                    metrics["max_normalized_gradient"] = max_norm_grad

                    # Check for derivative gradient explosion (normalized slope > 1e4)
                    d_exploded = False
                    try:
                        d_expr = sp.diff(expr, primary_sym)
                        d_fn = sp.lambdify(syms, d_expr, modules=["numpy", "math"])
                        for x_pt in [primary_grid[0], primary_grid[-1], 0.0, float(np.median(primary_grid))]:
                            if low_x <= x_pt <= high_x:
                                test_args = [x_pt, *fixed_args]
                                d_val = abs(float(d_fn(*test_args)))
                                norm_d = d_val * x_range / (y_range + 1e-15)
                                if norm_d > 1e4 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                                    d_exploded = True
                                    metrics["grad_explosions"] += 1
                                    ces.append(Counterexample(
                                        axis=FalsificationAxis.NUMERICAL,
                                        probe_name="GradientExplosionProbe",
                                        counterexample_point={str(primary_sym): float(x_pt)},
                                        failure_class=FailureClass.ILL_CONDITIONED,
                                        violation_description=f"Numerical gradient exploded (normalized slope {norm_d:.2e} > 1e4), indicating extreme stiffness or numerical discontinuity.",
                                        severity="HIGH",
                                        metric_value=norm_d
                                    ))
                                    break
                    except (
                        AttributeError,
                        ArithmeticError,
                        NotImplementedError,
                        TypeError,
                        ValueError,                        ) as exc:
                            # The gradient is still measured, by the finite-difference
                            # path immediately below: this is a degradation of the
                            # probe, not a hole in the axis, so it is recorded
                            # without voiding a clean verdict.
                            self._record_probe_error(
                                metrics,
                                "GradientExplosionProbe",
                                exc,
                                f"symbolic derivative of {expr}",
                                measured_alternative=True,
                            )

                    if not d_exploded and max_norm_grad > 1e4 and not any(c.axis == FalsificationAxis.BOUNDARY for c in ces):
                        metrics["grad_explosions"] += 1
                        ces.append(Counterexample(
                            axis=FalsificationAxis.NUMERICAL,
                            probe_name="GradientExplosionProbe",
                            counterexample_point={str(primary_sym): float(primary_grid[np.nanargmax(norm_gradients)])},
                            failure_class=FailureClass.ILL_CONDITIONED,
                            violation_description=f"Numerical normalized gradient exploded ({max_norm_grad:.2e}x scale factor), indicating extreme stiffness or numerical discontinuity.",
                            severity="HIGH",
                            metric_value=max_norm_grad
                        ))
                else:
                    # A model with no variation along the swept symbol has no
                    # conditioning to measure, but that is a null result rather than
                    # a crash: it is published as a metric, not as a probe error.
                    metrics["degenerate_sweep"] = True
                    metrics["sweep_output_range"] = y_range

        metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.NUMERICAL].value
        return ces, metrics
