"""Dedicated Physics Constraint Layer for APEDP.

Evaluates and certifies physical soundness across:
1. Dimensional consistency: [LHS] = [RHS] via DimensionalAssuranceEngine
2. Multi-variable boundary conditions: f(x_0, t_0, ...) = y_0
3. Asymptotic behavior: far-field limits (x -> oo) and origin limits (x -> 0+)
4. Positivity & physical state constraints: y(x) >= 0
5. Parity & transformation symmetry: f(-x) = +/- f(x), coordinate exchange, scale invariance
6. Conservation laws: time-invariance (df/dt = 0), flux divergence
7. Sign-definite parameter constraints: m > 0, T > 0, k > 0
8. Out-of-distribution (OOD) extrapolation bounds
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from apedp.dimensional_layer import check_units
from apedp.discovery_types import DictCompatibleResult
from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)


def _unverified(check: str, exc: Exception) -> str:
    """Wording for a check that raised rather than returned a verdict.

    A compliance report concludes with ``is_sound = not violations`` and
    ``score = passed_checks / 7``. A handler that only logs therefore does not
    merely lose a message: the flag it was guarding stays ``True``, nothing is
    appended, and a check that never ran is indistinguishable from a check that
    passed -- in the verdict, in the score, and to every consumer downstream.
    An expression that diverges at the origin could be certified physically
    sound on the strength of an ``sp.limit`` call that raised.

    These evaluators raise on ordinary input, not exotic input: ``sp.limit``
    on an expression with no limit, ``lambdify`` on a piecewise or a piecewise-
    differentiated form, division by a symbolic zero. So this is an expected
    path, and the honest report is "could not be evaluated", scored as a
    failure, rather than a silent pass.
    """
    return f"{check} could not be evaluated ({type(exc).__name__}: {exc})"


@dataclass(frozen=True)
class PhysicsSpecification(DictCompatibleResult):
    """Specification of known physical constraints for a scientific problem."""

    variable_units: dict[str, str] = field(default_factory=dict)
    expected_units: dict[str, str] = field(default_factory=dict)  # backwards-compatible alias
    target_unit: str | None = None
    origin_limit: float | None = None
    far_field_limit: float | None = None
    positivity: bool = False
    monotonicity: str | None = None      # "increasing", "decreasing", or None
    parity: str | None = None            # "even", "odd", or None
    symmetries: list[str] = field(default_factory=list)  # e.g. ["parity_x", "exchange_x_y", "scale_invariance"]
    conservation_laws: list[str] = field(default_factory=list)  # e.g. ["time_independent", "energy_conservation"]
    boundary_conditions: list[tuple[dict[str, float], float]] = field(default_factory=list)
    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    ood_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_asymptotic_growth: str | None = None  # "bounded", "extinction", None
    rejection_policy: str = "penalized"       # "strict" or "penalized"
    tolerance: float = 1e-4

    def get_effective_variable_units(self) -> dict[str, str]:
        """Merges variable_units and expected_units."""
        res = dict(self.expected_units)
        res.update(self.variable_units)
        return res


@dataclass(frozen=True)
class PhysicsComplianceReport(DictCompatibleResult):
    """Fine-grained certification of compliance with physical principles."""

    is_physically_sound: bool
    dimensional_valid: bool
    boundary_valid: bool
    asymptotics_valid: bool
    positivity_valid: bool
    parity_valid: bool
    symmetry_valid: bool = True
    conservation_valid: bool = True
    violations: list[str] = field(default_factory=list)
    compliance_score: float = 1.0  # [0.0, 1.0]

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


class PhysicsConstraintEngine:
    """Certifies physical admissibility and boundary invariants of synthesized candidates."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        expr_str: str,
        spec: PhysicsSpecification | None = None,
        x_domain: tuple[float, float] = (0.1, 10.0),
        parameters: dict[str, float] | None = None,
    ) -> PhysicsComplianceReport:
        if spec is None:
            return PhysicsComplianceReport(
                is_physically_sound=True,
                dimensional_valid=True,
                boundary_valid=True,
                asymptotics_valid=True,
                positivity_valid=True,
                parity_valid=True,
                symmetry_valid=True,
                conservation_valid=True,
                compliance_score=1.0,
            )

        violations: list[str] = []
        dim_valid = True
        boundary_valid = True
        asymp_valid = True
        pos_valid = True
        parity_valid = True
        sym_valid = True
        cons_valid = True

        # Parse expression to SymPy
        try:
            sym_expr = safe_parse_expr(expr_str)
        except Exception as e:
            violations.append(f"Failed to parse symbolic expression: {e}")
            return PhysicsComplianceReport(
                is_physically_sound=False,
                dimensional_valid=False,
                boundary_valid=False,
                asymptotics_valid=False,
                positivity_valid=False,
                parity_valid=False,
                symmetry_valid=False,
                conservation_valid=False,
                violations=violations,
                compliance_score=0.0,
            )

        # Substitute any known fixed parameters if passed
        if parameters:
            sym_expr = sym_expr.subs({sp.Symbol(k): sp.Float(v) for k, v in parameters.items() if sp.Symbol(k) in sym_expr.free_symbols})

        free_symbols = sorted(sym_expr.free_symbols, key=lambda s: str(s))
        # 1. Dimensional Consistency Enforcement
        units_dict = spec.get_effective_variable_units()
        if units_dict:
            try:
                verdict = check_units(sym_expr, units_dict, target_unit=spec.target_unit)
                if verdict.status == "FAIL":
                    dim_valid = False
                    violations.append(f"Dimensional consistency violation: {verdict.reason}")
            except Exception as e:
                dim_valid = False
                violations.append(_unverified("Dimensional consistency check", e))

        # 2. Multi-Variable Boundary Conditions
        if spec.boundary_conditions:
            for pt_dict, expected_val in spec.boundary_conditions:
                try:
                    val_sub = sym_expr.subs({sp.Symbol(k): sp.Float(v) for k, v in pt_dict.items()})
                    val_float = float(val_sub.evalf())
                    if abs(val_float - expected_val) > spec.tolerance:
                        boundary_valid = False
                        violations.append(
                            f"Boundary condition violation at {pt_dict}: got {val_float:.4g}, expected {expected_val:.4g}"
                        )
                except Exception as e:
                    boundary_valid = False
                    violations.append(f"Boundary condition evaluation failed at {pt_dict}: {e}")

        # 3. 1D Origin Limit (x -> 0+)
        x_sym = next((s for s in free_symbols if s.name in {"x", "r", "d", "t"}), free_symbols[0] if free_symbols else sp.Symbol("x"))
        if spec.origin_limit is not None and x_sym in free_symbols:
            try:
                lim_0 = sp.limit(sym_expr, x_sym, 0, dir="+")
                if lim_0.has(x_sym) or not lim_0.is_number:
                    f_num_0 = sp.lambdify(x_sym, sym_expr, modules=["scipy", "numpy"])
                    lim_0_val = float(f_num_0(1e-7))
                else:
                    lim_0_val = float(lim_0) if lim_0.is_finite else float("inf")

                if not np.isfinite(lim_0_val):
                    violations.append(
                        f"Singular origin violation: lim_{{{x_sym}->0+}} f = diverges, "
                        f"expected finite {spec.origin_limit:.4g}"
                    )
                    boundary_valid = False
                elif abs(lim_0_val - spec.origin_limit) > spec.tolerance:
                    violations.append(
                        f"Origin limit violation: lim_{{{x_sym}->0+}} f = {lim_0_val:.4g}, "
                        f"expected {spec.origin_limit:.4g}"
                    )
                    boundary_valid = False
            except Exception as e:
                boundary_valid = False
                violations.append(_unverified("Origin limit check", e))

        # 4. Asymptotic Far-Field Limit (x -> oo)
        if spec.far_field_limit is not None and x_sym in free_symbols:
            try:
                lim_inf = sp.limit(sym_expr, x_sym, sp.oo)
                if lim_inf.has(x_sym) or not lim_inf.is_number:
                    f_num_inf = sp.lambdify(x_sym, sym_expr, modules=["scipy", "numpy"])
                    lim_inf_val = float(f_num_inf(1e7))
                else:
                    lim_inf_val = float(lim_inf) if lim_inf.is_finite else float("inf")

                if not np.isfinite(lim_inf_val):
                    violations.append(
                        f"Unphysical asymptotic divergence: lim_{{{x_sym}->oo}} f diverges, "
                        f"expected {spec.far_field_limit:.4g}"
                    )
                    asymp_valid = False
                elif abs(lim_inf_val - spec.far_field_limit) > spec.tolerance:
                    violations.append(
                        f"Far-field limit violation: lim_{{{x_sym}->oo}} f = {lim_inf_val:.4g}, "
                        f"expected {spec.far_field_limit:.4g}"
                    )
                    asymp_valid = False
            except Exception as e:
                asymp_valid = False
                violations.append(_unverified("Far-field limit check", e))

        # 5. Positivity, Monotonicity, and Parity across Domain
        try:
            if len(free_symbols) == 1:
                f_num = sp.lambdify(free_symbols[0], sym_expr, modules=["scipy", "numpy"])
                grid = np.linspace(x_domain[0], x_domain[1], 100)
                y_eval = np.asarray(f_num(grid), dtype=float)

                if spec.positivity:
                    min_val = float(np.min(y_eval))
                    if min_val < -spec.tolerance:
                        violations.append(
                            f"Positivity violation: minimum value on domain is {min_val:.4g} < 0"
                        )
                        pos_valid = False

                if spec.monotonicity == "increasing":
                    diffs = np.diff(y_eval)
                    if np.any(diffs < -spec.tolerance):
                        violations.append("Monotonicity violation: function decreases on domain")
                elif spec.monotonicity == "decreasing":
                    diffs = np.diff(y_eval)
                    if np.any(diffs > spec.tolerance):
                        violations.append("Monotonicity violation: function increases on domain")

                if spec.parity == "even":
                    sym_grid = np.linspace(0.1, min(x_domain[1], 5.0), 30)
                    y_pos = np.asarray(f_num(sym_grid), dtype=float)
                    y_neg = np.asarray(f_num(-sym_grid), dtype=float)
                    diff = np.max(np.abs(y_pos - y_neg))
                    if diff > spec.tolerance * 10:
                        violations.append(f"Even parity violation: max |f(x) - f(-x)| = {diff:.4g}")
                        parity_valid = False
                elif spec.parity == "odd":
                    sym_grid = np.linspace(0.1, min(x_domain[1], 5.0), 30)
                    y_pos = np.asarray(f_num(sym_grid), dtype=float)
                    y_neg = np.asarray(f_num(-sym_grid), dtype=float)
                    diff = np.max(np.abs(y_pos + y_neg))
                    if diff > spec.tolerance * 10:
                        violations.append(f"Odd parity violation: max |f(x) + f(-x)| = {diff:.4g}")
                        parity_valid = False
            elif len(free_symbols) > 1 and spec.positivity:
                # Multi-variable positivity check on random grid
                rng = np.random.default_rng(42)
                pts = {s: rng.uniform(x_domain[0], x_domain[1], 50) for s in free_symbols}
                f_mv = sp.lambdify(free_symbols, sym_expr, modules=["scipy", "numpy"])
                y_mv = np.asarray(f_mv(*[pts[s] for s in free_symbols]), dtype=float)
                if np.any(y_mv < -spec.tolerance):
                    pos_valid = False
                    violations.append(f"Multi-variable positivity violation: minimum {float(np.min(y_mv)):.4g} < 0")
        except Exception as e:
            pos_valid = False
            parity_valid = False
            violations.append(_unverified("Domain numerical check (positivity/parity)", e))

        # 6. Symmetries & Noether Connections
        if spec.symmetries:
            for sym_desc in spec.symmetries:
                if sym_desc.startswith("parity_"):
                    v_name = sym_desc.split("_", 1)[1]
                    v_sym = sp.Symbol(v_name)
                    if v_sym in free_symbols:
                        # Check spatial parity
                        sub_neg = sym_expr.subs(v_sym, -v_sym)
                        diff_even = sp.simplify(sym_expr - sub_neg)
                        diff_odd = sp.simplify(sym_expr + sub_neg)
                        if diff_even != 0 and diff_odd != 0:
                            sym_valid = False
                            violations.append(f"Symmetry violation: {v_name} is neither even nor odd under parity inversion")
                elif sym_desc.startswith("exchange_"):
                    parts = sym_desc.split("_")[1:]
                    if len(parts) == 2:
                        v1, v2 = sp.Symbol(parts[0]), sp.Symbol(parts[1])
                        if v1 in free_symbols and v2 in free_symbols:
                            dummy = sp.Dummy("exchange_dummy")
                            sub_ex = sym_expr.subs(v1, dummy).subs(v2, v1).subs(dummy, v2)
                            if sp.simplify(sym_expr - sub_ex) != 0:
                                sym_valid = False
                                violations.append(f"Permutation symmetry violation: f is not symmetric under exchange of {parts[0]} and {parts[1]}")
                elif sym_desc == "scale_invariance":
                    # Homogeneity: f(lambda * x) = lambda^k * f(x)
                    lam = sp.Symbol("lam", positive=True)
                    sub_scale = sym_expr.subs({s: lam * s for s in free_symbols})
                    ratio = sp.simplify(sub_scale / sym_expr)
                    if ratio.has(*free_symbols):
                        sym_valid = False
                        violations.append("Scale invariance violation: f(lambda * x) does not factor out homogeneously")

        # 7. Conservation Laws
        if spec.conservation_laws:
            for cons in spec.conservation_laws:
                if cons in {"time_independent", "time_invariance", "energy_conservation"}:
                    t_sym = next((s for s in free_symbols if s.name in {"t", "time"}), None)
                    if t_sym is not None:
                        df_dt = sp.diff(sym_expr, t_sym)
                        if sp.simplify(df_dt) != 0:
                            cons_valid = False
                            violations.append(f"Conservation violation: dynamic quantity depends explicitly on time ({t_sym})")

        # 8. OOD Extrapolation Bounds
        if spec.ood_bounds and spec.max_asymptotic_growth:
            try:
                for v_name, (lo, hi) in spec.ood_bounds.items():
                    v_sym = sp.Symbol(v_name)
                    if v_sym in free_symbols:
                        f_ood = sp.lambdify(v_sym, sym_expr.subs({s: 1.0 for s in free_symbols if s != v_sym}), modules=["scipy", "numpy"])
                        ood_pts = np.linspace(lo, hi, 20)
                        y_ood = np.asarray(f_ood(ood_pts), dtype=float)
                        if spec.max_asymptotic_growth == "extinction":
                            far_val = abs(float(y_ood[-1]))
                            if far_val > spec.tolerance * 100:
                                asymp_valid = False
                                violations.append(f"OOD extinction violation: value at {hi} is {far_val:.4g} > 0")
                        elif spec.max_asymptotic_growth == "bounded":
                            max_val = float(np.max(np.abs(y_ood)))
                            if max_val > 1e6 or not np.all(np.isfinite(y_ood)):
                                asymp_valid = False
                                violations.append(f"OOD bounded growth violation: value exceeds bound ({max_val:.4g})")
            except Exception as e:
                asymp_valid = False
                violations.append(_unverified("Out-of-domain bounds check", e))

        # 9. Parameter Domain Constraints
        if parameters and spec.parameter_bounds:
            for p_name, (p_min, p_max) in spec.parameter_bounds.items():
                if p_name in parameters:
                    val = parameters[p_name]
                    if val < p_min or val > p_max:
                        violations.append(
                            f"Parameter domain violation: {p_name}={val:.4g} outside [{p_min}, {p_max}]"
                        )

        total_checks = 7
        passed_checks = sum([dim_valid, boundary_valid, asymp_valid, pos_valid, parity_valid, sym_valid, cons_valid])
        score = passed_checks / total_checks

        is_sound = len(violations) == 0
        if spec.rejection_policy == "strict" and not is_sound:
            score = 0.0

        return PhysicsComplianceReport(
            is_physically_sound=is_sound,
            dimensional_valid=dim_valid,
            boundary_valid=boundary_valid,
            asymptotics_valid=asymp_valid,
            positivity_valid=pos_valid,
            parity_valid=parity_valid,
            symmetry_valid=sym_valid,
            conservation_valid=cons_valid,
            violations=violations,
            compliance_score=score,
        )


__all__ = [
    "PhysicsComplianceReport",
    "PhysicsConstraintEngine",
    "PhysicsSpecification",
]
