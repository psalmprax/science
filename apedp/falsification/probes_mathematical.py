"""Mathematical axis: denominator roots, branch cuts, exact poles."""
from __future__ import annotations

from typing import Any

import sympy as sp

from apedp.safety import quiet_float_warnings

from .counterexamples import Counterexample
from .taxonomy import AXIS_EVIDENCE_TIER, FailureClass, FalsificationAxis


class _MathematicalProbesMixin:
    """Mathematical axis: denominator roots, branch cuts, exact poles."""

    @quiet_float_warnings()
    def _probe_mathematical_counterexamples(
        self,
        expr: sp.Expr,
        domains: dict[str, tuple[float, float]],
        critical_boundaries: dict[str, list[float]] | None = None
    ) -> tuple[list[Counterexample], dict[str, Any]]:
        """Searches for denominator zeros, complex branch cuts, and unremovable singularities."""
        ces: list[Counterexample] = []
        metrics: dict[str, Any] = {"poles_found": 0, "complex_branches": 0, "probe_errors": []}

        # Gather all denominators to check: top-level denominator + negative powers in subexpressions
        denoms_to_check: list[sp.Expr] = []
        _numer, denom = expr.as_numer_denom()
        if denom != 1:
            denoms_to_check.append(denom)
        for pow_atom in expr.atoms(sp.Pow):
            if pow_atom.exp.is_number and float(pow_atom.exp) < 0:
                denoms_to_check.append(pow_atom.base)

        seen_poles: set[tuple[str, float]] = set()
        for d_expr in denoms_to_check:
            for sym in d_expr.free_symbols:
                s_name = str(sym)
                low, high = domains.get(s_name, (-10.0, 10.0))

                # Skip pure exponentials (exp(u) != 0 for any real u)
                if d_expr.func == sp.exp or (d_expr.is_Pow and d_expr.base == sp.E):
                    continue

                try:
                    # Substitute other symbols with domain medians to keep solving univariate and fast
                    other_syms = [s for s in d_expr.free_symbols if s != sym]
                    sub_dict = {
                        s: float(domains.get(str(s), (1.0, 1.0))[0] + domains.get(str(s), (1.0, 1.0))[1]) / 2.0
                        for s in other_syms
                    }
                    eval_denom = d_expr.subs(sub_dict)

                    # Search for real roots of denominator inside domain
                    roots = []
                    if eval_denom.is_polynomial(sym):
                        roots = sp.real_roots(eval_denom)
                    elif not eval_denom.has(sp.exp):
                        roots = sp.solve(eval_denom, sym)

                    crit_pts = critical_boundaries.get(s_name, []) if critical_boundaries else []

                    for r in roots:
                        if r.is_real and r.is_number:
                            r_val = float(r)
                            tol_low = 1e-5 * max(abs(low), 1e-12)
                            tol_high = 1e-5 * max(abs(high), 1e-12)
                            in_domain = (low - tol_low) <= r_val <= (high + tol_high)
                            in_crit = any(abs(r_val - cb) <= 1e-5 * max(abs(cb), 1e-6) for cb in crit_pts)
                            pole_key = (s_name, round(r_val, 6))
                            if (in_domain or in_crit) and pole_key not in seen_poles:
                                seen_poles.add(pole_key)
                                metrics["poles_found"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.MATHEMATICAL,
                                    probe_name="DenominatorRootProbe", failure_class=FailureClass.DENOMINATOR_POLE,
                                    counterexample_point={s_name: r_val},
                                    violation_description=f"Denominator pole detected at {s_name}={r_val:.5g} causing division by zero.",
                                    severity="FATAL",
                                    metric_value=r_val
                                ))
                except (
                    AttributeError,
                    ArithmeticError,
                    NotImplementedError,
                    TypeError,
                    ValueError,  # includes sympy.SympifyError / PolynomialError
                ) as exc:
                    # The pole search could not run for this symbol: an unknown, so
                    # it is reported rather than silently dropping the probe.
                    self._record_probe_error(
                        metrics, "DenominatorRootProbe", exc, f"symbol '{s_name}' denominator {d_expr}"
                    )

        # Complex branch cut check (e.g. square roots of negative values)
        for sub in expr.atoms(sp.Pow):
            base, exp = sub.as_base_exp()
            if exp.is_number and float(exp) < 1.0 and float(exp) > 0.0:  # e.g. sqrt
                for sym in base.free_symbols:
                    s_name = str(sym)
                    low, _ = domains.get(s_name, (-1.0, 1.0))
                    if low < 0:
                        try:
                            # Test evaluate at low
                            test_sub = dict.fromkeys(base.free_symbols, 1.0)
                            test_sub[sym] = low
                            val = complex(base.subs(test_sub).evalf())
                            if val.real < 0:
                                metrics["complex_branches"] += 1
                                ces.append(Counterexample(
                                    axis=FalsificationAxis.MATHEMATICAL,
                                    probe_name="ComplexBranchCutProbe", failure_class=FailureClass.COMPLEX_BRANCH_CUT,
                                    counterexample_point={s_name: low},
                                    violation_description=f"Radicand {base} becomes negative ({val.real:.3f}) at {s_name}={low}, producing unphysical imaginary values.",
                                    severity="FATAL"
                                ))
                        except (
                            AttributeError,
                            ArithmeticError,
                            NotImplementedError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            # Radicand evaluation failed for this sub-expression: the
                            # branch-cut behaviour is unknown, not clean.
                            self._record_probe_error(
                                metrics, "ComplexBranchCutProbe", exc, f"radicand {base}"
                            )

        # A clearance is only as strong as the methods behind it. An axis with
        # nothing to analyse (no denominator, no root or fractional power) is
        # silent rather than a hole, so it publishes no tier at all.
        if denoms_to_check or expr.atoms(sp.Pow):
            metrics["evidence_tier"] = AXIS_EVIDENCE_TIER[FalsificationAxis.MATHEMATICAL].value
        return ces, metrics
