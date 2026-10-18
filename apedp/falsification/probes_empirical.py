"""Empirical axis: residual minimisation against reference observations."""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from apedp.evidence import EvidenceTier
from apedp.safety import quiet_float_warnings

from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis


class _EmpiricalProbesMixin:
    """Empirical axis: residual minimisation against reference observations."""

    @quiet_float_warnings()
    def _probe_empirical_fidelity(
        self,
        expr: sp.Expr,
        empirical_data: dict[str, Any]
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Evaluates residual error and R^2 goodness-of-fit against empirical reference observations."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {"probe_errors": []}

        x_data = np.asarray(empirical_data["x"], dtype=float)
        y_obs = np.asarray(empirical_data["y"], dtype=float)

        syms = list(expr.free_symbols)
        if len(syms) != 1:
            # Skipping the axis is not passing it: a multi-variable model gets no
            # empirical verdict, and the report says why.
            self._record_probe_error(
                metrics,
                "EmpiricalResidualProbe",
                "ModelNotUnivariate",
                f"empirical comparison is implemented for 1 free variable, model has {len(syms)}",
            )
            metrics["evidence_tier"] = EvidenceTier.UNRESOLVED.value
            return ces, metrics

        metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.EMPIRICAL].value

        fn = sp.lambdify(syms[0], expr, modules=["numpy", "math"])
        try:
            y_pred = np.array([float(fn(x)) for x in x_data])
            residuals = y_obs - y_pred
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
            r_squared = 1.0 - (ss_res / (ss_tot + 1e-15))
            nrmse = float(np.sqrt(np.mean(residuals ** 2)) / (np.std(y_obs) + 1e-15))

            metrics["r_squared"] = r_squared
            metrics["nrmse"] = nrmse

            if r_squared < 0.70 or nrmse > 0.35:
                ces.append(Counterexample(
                    axis=FalsificationAxis.EMPIRICAL,
                    probe_name="EmpiricalResidualProbe", failure_class=FailureClass.EMPIRICAL_FIT_REJECTED,
                    counterexample_point={"worst_x": float(x_data[np.argmax(np.abs(residuals))])},
                    violation_description=f"Empirical fit rejected: R^2={r_squared:.3f} < 0.70 (NRMSE={nrmse:.1%}).",
                    severity="HIGH",
                    metric_value=r_squared
                ))
        except (TypeError, ValueError, ArithmeticError, AttributeError, NotImplementedError) as e:
            # Already surfaced as a failure rather than swallowed: a model that
            # cannot be evaluated at the reference observations fails the
            # comparison. An exception type outside this list is a defect in APEDP
            # and propagates.
            ces.append(Counterexample(
                axis=FalsificationAxis.EMPIRICAL,
                probe_name="EmpiricalEvaluationProbe", failure_class=FailureClass.EMPIRICAL_EVALUATION_FAILURE,
                counterexample_point={},
                violation_description=f"Empirical evaluation failed on test data: {e}",
                severity="HIGH"
            ))

        return ces, metrics
