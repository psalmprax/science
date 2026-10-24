"""Boundary axis: divergences and directional limits at declared boundaries."""
from __future__ import annotations

import logging
import math
from typing import Any

import sympy as sp

from apedp.safety import LimitTimeout, limit_with_timeout, quiet_float_warnings

from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis

logger = logging.getLogger("apedp.falsification")

class _BoundaryProbesMixin:
    """Boundary axis: divergences and directional limits at declared boundaries."""

    @quiet_float_warnings()
    def _probe_boundary_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        critical_boundaries: dict[str, list[float]] | None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Evaluates directional asymptotic limits as variables approach critical points, 0, and infinity."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {
            "infinite_limits": 0,
            "limit_mismatches": 0,
            "probe_errors": [],
        }

        targets: dict[str, list[float]] = {}
        for s_name, (low, high) in domains.items():
            targets[s_name] = []
            if low <= 0.0 <= high:
                targets[s_name].append(0.0)

        if critical_boundaries:
            for k, vals in critical_boundaries.items():
                if k not in targets:
                    targets[k] = []
                targets[k].extend(vals)

        for sym in expr.free_symbols:
            s_name = str(sym)
            pts = set(targets.get(s_name, []))
            if not pts:
                continue

            # Substitute other variables with domain median to keep limits univariate, fast and stable
            other_syms = [s for s in expr.free_symbols if s != sym]
            sub_dict = {
                s: float(domains.get(str(s), (1.0, 1.0))[0] + domains.get(str(s), (1.0, 1.0))[1]) / 2.0
                for s in other_syms
            }
            eval_expr = expr.subs(sub_dict)

            for pt in pts:
                try:
                    # Fast-path: Check direct substitution first
                    direct_val = None
                    try:
                        direct_subs = eval_expr.subs(sym, pt)
                        if direct_subs.is_number and not direct_subs.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                            direct_val = float(direct_subs)
                    except (TypeError, ValueError, ArithmeticError, AttributeError) as exc:
                        # Direct substitution is only a fast path; the limits below
                        # are the actual measurement, so this is not swallowed.
                        logger.debug("Direct substitution failed at %s=%s: %s", s_name, pt, exc)
                        direct_val = None

                    if direct_val is not None and math.isfinite(direct_val):
                        # Regular and finite at pt, no singularity
                        continue

                    # Evaluate right limit (bounded; a timeout is an unknown, so it is
                    # recorded rather than treated as divergent *or* as clean)
                    try:
                        lim_right = limit_with_timeout(eval_expr, sym, pt, dir="+", timeout=2.0)
                    except LimitTimeout as exc:
                        self._record_probe_error(
                            metrics, "AsymptoticDivergenceProbe", exc, f"limit {s_name} -> {pt}^+ unresolved"
                        )
                        continue
                    is_div_right = lim_right.has(sp.oo) or lim_right.has(-sp.oo) or lim_right.has(sp.zoo)
                    if is_div_right:
                        metrics["infinite_limits"] += 1
                        ces.append(Counterexample(
                            axis=FalsificationAxis.BOUNDARY,
                            probe_name="AsymptoticDivergenceProbe",
                            counterexample_point={s_name: pt, "direction": "+"},
                            failure_class=FailureClass.BOUNDARY_DIVERGENCE,
                            violation_description=f"Formula diverges to {lim_right} as {s_name} -> {pt}^+.",
                            severity="FATAL"
                        ))
                    # Evaluate left limit if not at boundary min (bounded)
                    if pt > domains.get(s_name, (0, 0))[0]:
                        try:
                            lim_left = limit_with_timeout(eval_expr, sym, pt, dir="-", timeout=2.0)
                        except LimitTimeout as exc:
                            self._record_probe_error(
                                metrics,
                                "AsymptoticDivergenceProbe",
                                exc,
                                f"limit {s_name} -> {pt}^- unresolved",
                            )
                            continue
                        is_div_left = lim_left.has(sp.oo) or lim_left.has(-sp.oo) or lim_left.has(sp.zoo)
                        if is_div_left:
                            metrics["infinite_limits"] += 1
                            ces.append(Counterexample(
                                axis=FalsificationAxis.BOUNDARY,
                                probe_name="AsymptoticDivergenceProbe",
                                counterexample_point={s_name: pt, "direction": "-"},
                                failure_class=FailureClass.BOUNDARY_DIVERGENCE,
                                violation_description=f"Formula diverges to {lim_left} as {s_name} -> {pt}^-.",
                                severity="FATAL"
                            ))
                        elif not is_div_right and not is_div_left and lim_right.is_number and lim_left.is_number:
                            if abs(float(lim_right) - float(lim_left)) > 1e-4:
                                metrics["limit_mismatches"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.BOUNDARY,
                                    probe_name="DirectionalLimitMismatchProbe",
                                    counterexample_point={s_name: pt},
                                    failure_class=FailureClass.DIRECTIONAL_LIMIT_MISMATCH,
                                    violation_description=f"Jump discontinuity at {s_name}={pt}: lim+={float(lim_right):.4g} != lim-={float(lim_left):.4g}.",
                                    severity="HIGH"
                                ))
                except (
                    AttributeError,
                    ArithmeticError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as exc:
                    # Nothing was decided about the behaviour at this point.
                    self._record_probe_error(
                        metrics, "BoundaryAsymptoteProbe", exc, f"point {s_name}={pt}"
                    )

        # No declared or discovered boundary point means nothing to examine; the
        # axis stays silent rather than claiming a result.
        if any(targets.values()):
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.BOUNDARY].value
        return ces, metrics
