"""APEDP Repair: Boundary Audits, Physical Singularity Repair, and Agnostic Regularization Engine.

Security: all string-in entry points use :func:`apedp.safety.safe_parse_expr`
(allowlist, length cap, no attribute access) and
:func:`apedp.safety.limit_with_timeout` (wall-clock bounded ``sp.limit``).
See ``apedp/safety.py`` for the threat model.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pint
import sympy as sp

from apedp.core import PHYSICAL_CONSTANTS, ClosedLoopAssuranceSystem, DimensionalUnit, ureg
from apedp.safety import LimitTimeout, limit_with_timeout, quiet_float_warnings, safe_parse_expr

logger = logging.getLogger(__name__)

# Exceptions a best-effort symbolic/numeric probe is allowed to swallow and treat
# as "unknown". SymPy reports malformed input as SympifyError (a ValueError) and
# unsupported algebra as NotImplementedError / PolynomialError; numerically
# evaluating a lambdified expression adds ArithmeticError (ZeroDivisionError /
# OverflowError / FloatingPointError) and RecursionError; a caller-supplied
# object that is not a SymPy expression adds TypeError / AttributeError, and
# registry lookups add KeyError. Anything outside this set signals a defect in
# APEDP itself and must propagate instead of being silently absorbed.
PROBE_SWALLOWED_ERRORS: tuple[type[Exception], ...] = (
    ArithmeticError,
    AttributeError,
    KeyError,
    NotImplementedError,
    RecursionError,
    TypeError,
    ValueError,  # includes sympy.SympifyError
    sp.PolynomialError,
)


class PhysicalAnomalyRepairEngine:
    """Specialized physical anomaly auditor and regularizer for classical, relativistic, and statistical models."""

    def __init__(self, assurance_system: ClosedLoopAssuranceSystem):
        self.system = assurance_system

    def _ensure_registered(self, name: str, unit_str: str) -> sp.Symbol:
        sym = sp.Symbol(name)
        if sym not in self.system.engine.symbol_units:
            return self.system.engine.register_variable(name, unit_str)
        return sym

    def audit_and_repair_gravity(self, r_symbol_name: str = 'r_dist') -> str:
        """Audits Newtonian gravity for r -> 0 singularity and deploys softened gravitational model."""
        r = self._ensure_registered(r_symbol_name, 'meter')
        eps = self._ensure_registered('epsilon_soft', 'meter')
        m_phys = self._ensure_registered('m_phys', 'kilogram')
        m_target = self._ensure_registered('m_target', 'kilogram')
        # Softened Plummer-type gravitational force: F = G * m1 * m2 / (r^2 + eps^2)
        soft_expr = (
            sp.Symbol('G_grav')
            * m_phys
            * m_target
            / (r**2 + eps**2)
        )
        model_name = "Newton_Gravity_Softened"
        self.system.register_model_spec(
            model_name,
            soft_expr,
            "newton",
            metadata={'regularization': 'Plummer softening', 'valid_domain': 'r >= 0'}
        )
        return model_name

    def audit_and_repair_relativity(
        self,
        v_symbol_name: str = 'v_phys',
        c_symbol_name: str = 'c_light',
        m_symbol_name: str = 'm_phys'
    ) -> str:
        """Audits relativistic kinetic energy for v >= c imaginary outputs and deploys 4th/6th order Taylor expansion."""
        v = self._ensure_registered(v_symbol_name, 'meter / second')
        c = self._ensure_registered(c_symbol_name, 'meter / second')
        m = self._ensure_registered(m_symbol_name, 'kilogram')

        # 4th-order Taylor series approximation: E_k = m*c^2 * (0.5 * (v/c)^2 + 3/8 * (v/c)^4 + 5/16 * (v/c)^6)
        beta_sq = (v**2) / (c**2)
        taylor_expr = m * (c**2) * (
            sp.Rational(1, 2) * beta_sq
            + sp.Rational(3, 8) * (beta_sq**2)
            + sp.Rational(5, 16) * (beta_sq**3)
        )
        model_name = "Relativistic_KE_Taylor"
        self.system.register_model_spec(
            model_name,
            taylor_expr,
            "joule",
            metadata={'regularization': 'Taylor expansion', 'valid_domain': 'all velocities (guaranteed real)'}
        )
        return model_name

    def audit_and_repair_bayes_factor(self) -> str:
        """Audits Bayes Factor matrix for Z_j -> 0 evidence divergence and deploys safe regularizer."""
        Z_i = self._ensure_registered('Z_i', 'dimensionless')
        Z_j = self._ensure_registered('Z_j', 'dimensionless')
        beta_reg = self._ensure_registered('beta_reg', 'dimensionless')

        safe_bf_expr = Z_i / (Z_j + beta_reg)
        model_name = "Bayes_Factor_Safe"
        self.system.register_model_spec(
            model_name,
            safe_bf_expr,
            "dimensionless",
            metadata={'regularization': 'Tikhonov evidence regularizer', 'valid_domain': 'Z_j >= 0'}
        )
        return model_name

    def audit_and_repair_resonance(self) -> str:
        """Audits damped harmonic oscillator amplitude for omega -> omega_0, zeta -> 0 singularity."""
        w = self._ensure_registered('omega', '1 / second')
        w0 = self._ensure_registered('omega_0', '1 / second')
        zeta = self._ensure_registered('zeta', 'dimensionless')
        F0 = self._ensure_registered('F_0', 'newton')
        m = self._ensure_registered('m_phys', 'kilogram')
        eps = self._ensure_registered('epsilon_res', '1 / second**2')

        # Amplitude A(w) = (F0 / m) / sqrt((w0^2 - w^2)^2 + (2*zeta*w0*w)^2 + eps^2)
        denom = sp.sqrt((w0**2 - w**2)**2 + (2 * zeta * w0 * w)**2 + eps**2)
        safe_amp_expr = (F0 / m) / denom
        model_name = "Harmonic_Resonance_Softened"
        self.system.register_model_spec(
            model_name,
            safe_amp_expr,
            "meter",
            metadata={'regularization': 'Resonance damping regularizer', 'valid_domain': 'all omega'}
        )
        return model_name

    def audit_and_repair_coulomb(self) -> str:
        """Audits Coulomb point-charge field divergence and applies Debye plasma screening."""
        q1 = self._ensure_registered('q_source', 'coulomb')
        q2 = self._ensure_registered('q_test', 'coulomb')
        lambda_d = self._ensure_registered('lambda_debye', 'meter')
        r = self._ensure_registered('r_dist', 'meter')
        eps_soft = self._ensure_registered('epsilon_soft', 'meter')
        eps_0 = sp.Symbol('eps_0')

        # Shielded regularized Coulomb force: F = (q1*q2 / (4*pi*eps_0*(r^2 + eps_soft^2))) * exp(-r / lambda_d)
        shielded_expr = (q1 * q2 / (4 * sp.pi * eps_0 * (r**2 + eps_soft**2))) * sp.exp(-r / lambda_d)
        model_name = "Coulomb_Debye_Shielded"
        self.system.register_model_spec(
            model_name,
            shielded_expr,
            "newton",
            metadata={'regularization': 'Debye screening + Core softening'}
        )
        return model_name

    def audit_and_repair_logarithm(self, x_symbol_name: str, expected_unit_str: str = "dimensionless") -> str:
        """Audits logarithmic potential/entropy singularity at x -> 0 and applies smooth offset regularization."""
        x = self._ensure_registered(x_symbol_name, expected_unit_str)
        x_unit = self.system.engine.symbol_units.get(x, self.system.engine.registry(expected_unit_str))
        eps_name = f"eps_{x_symbol_name}"
        eps = self._ensure_registered(eps_name, str(x_unit))

        if x_unit.dimensionality != self.system.engine.registry.dimensionless.dimensionality:
            x_ref_name = f"{x_symbol_name}_0"
            x_ref = self._ensure_registered(x_ref_name, str(x_unit))
            argument = ((x + sp.sqrt(x**2 + eps**2)) / 2 + eps) / x_ref
        else:
            argument = (x + sp.sqrt(x**2 + eps**2)) / 2 + eps

        soft_log_expr = sp.log(argument)
        model_name = f"Log_{x_symbol_name}_Regularized"
        self.system.register_model_spec(
            model_name,
            soft_log_expr,
            "dimensionless",
            metadata={'regularization': 'Logarithmic smooth positive boundary softening'}
        )
        return model_name

    def audit_and_repair_square_root(self, x_symbol_name: str, expected_unit_str: str = "meter**0.5") -> str:
        """Audits square-root branch cut for non-positive values and applies smooth non-negative envelope."""
        x = self._ensure_registered(x_symbol_name, 'meter')
        eps_name = f"eps_{x_symbol_name}"
        x_unit = str(self.system.engine.symbol_units.get(x, 'meter'))
        eps = self._ensure_registered(eps_name, x_unit)
        soft_sqrt_expr = sp.sqrt((x + sp.sqrt(x**2 + eps**2)) / 2)
        model_name = f"Sqrt_{x_symbol_name}_Regularized"
        self.system.register_model_spec(
            model_name,
            soft_sqrt_expr,
            expected_unit_str,
            metadata={'regularization': 'Square root positive envelope regularizer'}
        )
        return model_name


class AgnosticClosedLoopAssuranceSystem:
    """Agnostic pipeline: dynamic multi-domain tokenized unit inference, boundary anomaly stress-testing, and automated repair."""

    def __init__(self, registry: pint.UnitRegistry | None = None):
        self.ureg = registry or ureg
        # Register custom non-SI units if not present
        for def_str in ['dollar = [currency]', 'euro = [currency] = eur', 'bit = [data]', 'byte = 8 * bit']:
            try:
                self.ureg.define(def_str)
            except (ValueError, AttributeError, pint.PintError) as e:
                logger.warning("Could not define custom unit: %s (%s)", def_str, e)
        self.default_unit_map: dict[str, str] = {
            'm': 'kilogram', 'mass': 'kilogram', 'kg': 'kilogram',
            'r': 'meter', 'dist': 'meter', 'x': 'meter', 'y': 'meter', 'z': 'meter',
            'h': 'meter', 'd': 'meter', 'length': 'meter', 'radius': 'meter', 'height': 'meter',
            'v': 'meter / second', 'vel': 'meter / second', 'velocity': 'meter / second', 'u': 'meter / second',
            'a': 'meter / second**2', 'acc': 'meter / second**2',
            't': 'second', 'time': 'second', 'tau': 'second', 'period': 'second',
            'e': 'joule', 'energy': 'joule', 'work': 'joule', 'ke': 'joule', 'pe': 'joule',
            'f': 'newton', 'force': 'newton',
            'p': 'pascal', 'press': 'pascal', 'pressure': 'pascal',
            'temp': 'kelvin', 'temperature': 'kelvin',
            'q': 'coulomb', 'charge': 'coulomb',
            'i': 'ampere', 'current': 'ampere',
            'beta': 'dimensionless', 'eta': 'dimensionless',
            'rho': 'kilogram / meter**3',
            'g': 'meter**3 / (kilogram * second**2)',
            'c': 'meter / second',
            'h_bar': 'joule * second',
            'k_b': 'joule / kelvin',
            'sigma': 'watt / (meter**2 * kelvin**4)',
            'eps': 'meter',
            # Computer Science / Information Theory
            'bit': 'bit', 'byte': 'byte', 'packet': 'dimensionless', 'latency': 'second',
            'bps': 'bit / second', 'flops': '1 / second',
            # Economics / Finance
            'usd': 'dollar', 'dollar': 'dollar', 'eur': 'euro', 'price': 'dollar',
            'cost': 'dollar', 'revenue': 'dollar', 'profit': 'dollar', 'qty': 'dimensionless',
            'rate': '1 / second'
        }

    def register_unit_mapping(self, token: str, unit_str: str) -> None:
        """Dynamically registers a domain token mapping (e.g. for custom non-SI units)."""
        self.default_unit_map[token.lower()] = unit_str

    def infer_units_for_expression(
        self,
        expr: sp.Expr,
        variable_units: dict[str, str] | None = None,
        strict: bool = False,
    ) -> dict[sp.Symbol, DimensionalUnit]:
        """Maps free symbols in an expression to units using tokenized heuristic and multi-domain vocabulary.

        ``variable_units`` (name -> unit string, e.g. ``{'Q_dot': 'watt'}``)
        takes precedence over the heuristic. Supply it whenever the true
        units are known: bare-token guessing cannot distinguish ``Q_dot``
        (heat rate, watt) from ``q`` (charge, coulomb), ``E0`` (volt) from
        ``e`` (joule), or ``H_h`` (joule) from ``h`` (meter).

        When ``strict`` is True, symbols that would fall back to the
        dimensionless guess raise ``ValueError`` instead, forcing callers
        to supply explicit units for ambiguous bare tokens.
        """
        inferred: dict[sp.Symbol, DimensionalUnit] = {}
        overrides = dict(variable_units or {})
        for sym in expr.free_symbols:
            name = str(sym)
            if name in PHYSICAL_CONSTANTS:
                inferred[sym] = PHYSICAL_CONSTANTS[name].unit
                continue
            if name in overrides:
                try:
                    inferred[sym] = self.ureg(overrides[name])
                    continue
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning("Bad override unit '%s' for '%s' (%s), falling back to heuristic",
                                   overrides[name], name, e)
            lname = name.lower()
            ignore_modifiers = {
                'total', 'avg', 'average', 'min', 'minimum', 'max', 'maximum',
                'delta', 'd', 'target', 'phys', 'source', 'err', 'error', 'count',
                'ref', 'norm', 'rel', 'nominal', 'measured', 'true'
            }
            tokens = [t for t in re.split(r'[_0-9]+', lname) if t]

            matched = False
            # 1. Exact match on full name
            if lname in self.default_unit_map:
                try:
                    inferred[sym] = self.ureg(self.default_unit_map[lname])
                    matched = True
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning("Could not infer unit for '%s' (%s)", name, e)
            if not matched:
                filtered_tokens = [t for t in tokens if t not in ignore_modifiers] or tokens
                for t in reversed(filtered_tokens):
                    if t in self.default_unit_map:
                        try:
                            inferred[sym] = self.ureg(self.default_unit_map[t])
                            matched = True
                            break
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.warning("Could not infer unit for '%s' (%s)", t, e)
                            break
            if not matched and tokens:
                for prefix, u_str in self.default_unit_map.items():
                    if tokens[0].startswith(prefix) and len(tokens[0]) <= len(prefix) + 2:
                        try:
                            inferred[sym] = self.ureg(u_str)
                            matched = True
                            break
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.warning("Unit inference failed for prefix '%s' (%s)", prefix, e)

            if not matched:
                if strict and name not in overrides:
                    raise ValueError(
                        f"Ambiguous units for symbol '{name}': supply explicit "
                        f"variable_units instead of relying on bare-token guessing"
                    )
                logger.warning(
                    "Unit for '%s' is ambiguous; defaulting to dimensionless. "
                    "Supply variable_units={'%s': '<unit>'} or use strict_units=True to refuse.",
                    name, name,
                )
                inferred[sym] = self.ureg.dimensionless
        return inferred

    def compute_expression_dimensionality(
        self, expr: sp.Expr, variable_units: dict[sp.Symbol, DimensionalUnit]
    ) -> DimensionalUnit:
        """Recursively computes Pint Unit for any parsed SymPy expression."""
        if isinstance(expr, sp.Symbol):
            return variable_units.get(expr, self.ureg.dimensionless)
        if isinstance(expr, (int, float, sp.Integer, sp.Float, sp.Rational)):
            return self.ureg.dimensionless
        if isinstance(expr, sp.Add):
            # A bare numeric literal is a unitless offset: ``3*x + 1`` means the
            # intercept carries x's units. Comparing literals against dimensioned
            # terms would reject every affine law, so they adopt the units of the
            # terms around them. Dimensions still have to agree term-to-term.
            first: DimensionalUnit | None = None
            for term in expr.args:
                if term.is_number:
                    continue
                term_u = self.compute_expression_dimensionality(term, variable_units)
                if first is None:
                    first = term_u
                elif first.dimensionality != term_u.dimensionality:
                    raise TypeError(f"Agnostic Inconsistency: Cannot add {first} and {term_u} in '{expr}'")
            if first is None:
                return self.ureg.dimensionless
            return first
        if isinstance(expr, sp.Mul):
            res = self.compute_expression_dimensionality(expr.args[0], variable_units)
            for arg in expr.args[1:]:
                res = res * self.compute_expression_dimensionality(arg, variable_units)
            return res
        if isinstance(expr, sp.Pow):
            base, exp = expr.args
            base_u = self.compute_expression_dimensionality(base, variable_units)
            try:
                exp_val = float(exp)
                return base_u ** exp_val
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.warning("Could not parse exponent %r, returning dimensionless (%s)", exp, e)
                return self.ureg.dimensionless
        if hasattr(expr, 'func'):
            if expr.func == sp.sqrt:
                return self.compute_expression_dimensionality(expr.args[0], variable_units) ** 0.5
            transcendental = {
                sp.exp, sp.log, sp.sin, sp.cos, sp.tan,
                sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan,
                sp.erf, sp.erfc,
            }
            if expr.func in transcendental:
                self.compute_expression_dimensionality(expr.args[0], variable_units)
                return self.ureg.dimensionless
            if expr.func in (sp.Abs, sp.re, sp.im, sp.conjugate):
                return self.compute_expression_dimensionality(expr.args[0], variable_units)
            if expr.func in (sp.sign, sp.Heaviside):
                self.compute_expression_dimensionality(expr.args[0], variable_units)
                return self.ureg.dimensionless
            if expr.func in (sp.Min, sp.Max):
                args = expr.args
                first = self.compute_expression_dimensionality(args[0], variable_units)
                for a in args[1:]:
                    a_u = self.compute_expression_dimensionality(a, variable_units)
                    if first.dimensionality != a_u.dimensionality:
                        raise TypeError(f"Agnostic Inconsistency: Incompatible units in {expr.func.__name__}: {first} vs {a_u}")
                return first
            return self.ureg.dimensionless
        return self.ureg.dimensionless

    # Symbols that must never be treated as varying boundary parameters:
    # universal physical constants plus our own eps_* regularizers.
    FIXED_SYMBOLS: ClassVar[frozenset] = frozenset({
        'c_light', 'G_grav', 'h_bar', 'k_B', 'e_charge', 'eps_0', 'mu_0',
        'sigma_sb', 'R_gas', 'N_A', 'g_earth',
    })

    def _regularizer(
        self,
        var_units: dict[sp.Symbol, DimensionalUnit],
        sym: sp.Symbol,
        x: sp.Expr,
        fallback_unit: DimensionalUnit,
    ) -> sp.Symbol:
        """Returns an eps regularizer sized to ``x``'s own units.

        Reuses ``eps_<sym>`` when its assigned units already match; otherwise
        mints a suffixed symbol (``eps_<sym>_2``, ...). Sharing one symbol
        across nodes with different units silently corrupts the final
        homogeneity check, which is what rejected laws like Torricelli's
        ``sqrt(2*g*h)`` (outer ``m/s`` node vs inner ``m^2/s^2`` node).
        """
        try:
            need = self.compute_expression_dimensionality(x, var_units)
        except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
            logger.warning("Could not size regularizer, using fallback (%s)", e)
            need = None
        candidate = sp.Symbol(f"eps_{sym}")
        if candidate not in var_units:
            var_units[candidate] = need if need is not None else fallback_unit
            return candidate
        if need is not None and var_units[candidate].dimensionality == need.dimensionality:
            return candidate
        i = 2
        while True:
            candidate = sp.Symbol(f"eps_{sym}_{i}")
            if candidate not in var_units:
                var_units[candidate] = need if need is not None else fallback_unit
                return candidate
            if need is not None and var_units[candidate].dimensionality == need.dimensionality:
                return candidate
            i += 1

    def autonomous_stress_test_and_repair(
        self,
        formula_str: str,
        variable_units: dict[str, str] | None = None,
        critical_values: dict[str, list[float]] | None = None,
        limit_timeout: float = 2.0,
        strict_units: bool = False,
    ) -> dict[str, Any]:
        """Autonomously parses, audits for boundary singularities, and synthesizes regularized equations.

        ``variable_units`` (name -> unit string) is forwarded to the unit
        inference and takes precedence over token guessing. Pass the true
        units whenever they are known. When ``strict_units`` is True,
        ambiguous bare-token guesses raise instead of silently defaulting
        to dimensionless.

        ``critical_values`` maps symbol name -> extra finite boundary points
        to probe in addition to 0 (e.g. ``{'v': [299792458.0]}`` for ``v→c``
        or ``{'M': [1.0]}``). Mach-like symbols additionally probe ±1 and
        velocity-like symbols probe ``c_light`` when present; ±oo is not
        probed (unbounded growth there is not a defect).

        All parsing uses :func:`apedp.safety.safe_parse_expr` and all
        ``sp.limit`` calls are bounded by ``limit_timeout`` seconds.
        Repaired results include Tier-3 asymptotic distortion
        (``max_distortion`` on an operational grid), the ``eps_*`` symbols
        introduced, and a ``validity_domain`` statement. Callers must supply
        numeric values for ``eps_*`` symbols when evaluating repaired models.

        PDE/ODE residuals (``Derivative``/``Integral`` forms) are not repaired:
        they are template-matched via :mod:`apedp.pde_templates` and returned
        with ``status="OUT_OF_SCOPE"`` naming the shape (or unknown).
        """
        if "Derivative" in formula_str or "Integral(" in formula_str:
            from apedp.pde_templates import TEMPLATES, recognize_pde
            from apedp.safety import safe_parse_pde

            try:
                pde_expr = safe_parse_pde(formula_str)
            except (TypeError, ValueError) as e:
                logger.warning("Stress test rejected PDE formula '%s': %s", formula_str[:80], e)
                return {"status": "REJECTED", "error": str(e)}
            hits = recognize_pde(pde_expr)
            if hits:
                primary = hits[0]
                return {
                    "status": "OUT_OF_SCOPE",
                    "anomalies": [],
                    "original_expression": pde_expr,
                    "repaired_expression": pde_expr,
                    "inferred_unit": "unknown",
                    "eps_symbols": [],
                    "distortion": {"max_distortion": None, "mean_distortion": None},
                    "validity_domain": (
                        f"PDE template {primary}: {TEMPLATES[primary]}. "
                        "Class-level shape only; solve with eval_extended "
                        "(residual/manufactured-solution or finite-difference)."
                    ),
                    "probe_errors": [],
                    "probe_error_count": 0,
                    "unresolved_probe_count": 0,
                }
            return {
                "status": "OUT_OF_SCOPE",
                "anomalies": [],
                "original_expression": pde_expr,
                "repaired_expression": pde_expr,
                "inferred_unit": "unknown",
                "eps_symbols": [],
                "distortion": {"max_distortion": None, "mean_distortion": None},
                "validity_domain": "PDE shape matched no known template; see eval_extended.",
                "probe_errors": [],
                "probe_error_count": 0,
                "unresolved_probe_count": 0,
            }
        try:
            expr = safe_parse_expr(formula_str)
            var_units = self.infer_units_for_expression(expr, variable_units, strict=strict_units)
            original_unit = self.compute_expression_dimensionality(expr, var_units)

            # Every probe that cannot answer its own question is recorded here.
            # A swallowed error used to be indistinguishable from a clean result:
            # the boundary audit assumed "not singular" and the report said
            # APPROVED, so a crashed probe read as a passed check. Each entry says
            # whether it left its question *unanswered*; only unanswered probes can
            # block a clean verdict, so a crash with a measured fallback stays
            # informational.
            probe_errors: list[dict[str, Any]] = []

            def _record_probe_error(
                probe: str,
                stage: str,
                detail: str,
                symbol: str | None = None,
                point: str | None = None,
            ) -> int:
                probe_errors.append(
                    {
                        "probe": probe,
                        "stage": stage,
                        "detail": detail,
                        "symbol": symbol,
                        "point": point,
                        "unresolved": True,
                    }
                )
                logger.debug("Repair probe '%s' (%s) failed: %s", probe, stage, detail)
                return len(probe_errors) - 1

            def _resolve(error_index: int | None, how: str) -> None:
                """Mark a crashed probe's question as answered by other means."""
                if error_index is None:
                    return
                probe_errors[error_index]["resolution"] = how
                probe_errors[error_index]["unresolved"] = False

            def _bounded_limit(
                e: sp.Expr, s: sp.Symbol, v: Any, stage: str
            ) -> tuple[sp.Expr | None, int | None]:
                """``sp.limit`` under a wall-clock bound -> ``(limit, error_index)``.

                ``error_index`` indexes the recorded probe error when the bound
                expired or the limit raised, so the caller can either resolve it
                through another measurement or let it block a clean verdict.
                """
                try:
                    return limit_with_timeout(e, s, v, timeout=limit_timeout), None
                except LimitTimeout:
                    return None, _record_probe_error(
                        "LimitProbe",
                        stage,
                        f"limit timed out after {limit_timeout}s",
                        symbol=str(s),
                        point=str(v),
                    )
                except PROBE_SWALLOWED_ERRORS as exc:
                    return None, _record_probe_error(
                        "LimitProbe",
                        stage,
                        f"{type(exc).__name__}: {exc}",
                        symbol=str(s),
                        point=str(v),
                    )

            # Audit boundary behavior
            anomalies: list[tuple[sp.Symbol, str]] = []
            num, den = sp.fraction(expr)
            for sym in expr.free_symbols:
                if str(sym) in self.FIXED_SYMBOLS or str(sym).startswith("eps_"):
                    continue
                # 1. Denominator / division by zero at 0, at caller-supplied
                # critical values (v→c, M→1, ...), and at transonic M=±1.
                # Probing only sym→0 stamped relativistic KE at v=c
                # APPROVED, so the defaults must cover the common physical
                # boundaries even when the caller supplies nothing. ±oo is
                # NOT probed by default: every unbounded law (x+y, x**2)
                # tends to oo there, which is growth, not a defect.
                probe_points: list[Any] = [0]
                probe_points.extend((critical_values or {}).get(str(sym), []))
                _lname = str(sym).lower()
                if _lname in ("m", "mach", "mach_number", "machnumber"):
                    for _mp in (1.0, -1.0):
                        if _mp not in probe_points:
                            probe_points.append(_mp)
                # Auto-detect physical constants as critical values: if the
                # expression contains e.g. c_light alongside a velocity-like
                # symbol, probe v→c.
                for const_name in ("c_light",):
                    if const_name in [str(s) for s in expr.free_symbols] and _lname in (
                        "v", "v_phys", "velocity", "u", "vel", "speed",
                    ):
                        # An absent registry entry yields None and is skipped via
                        # the AttributeError below, as is a non-numeric value.
                        constant = PHYSICAL_CONSTANTS.get(const_name)
                        try:
                            probe_points.append(float(constant.value))
                        except PROBE_SWALLOWED_ERRORS as exc:
                            # The critical point is simply not probed, so this is
                            # a gap rather than a fallback: leave it blocking.
                            _record_probe_error(
                                "PhysicalConstantProbe",
                                "probe-point-selection",
                                f"{type(exc).__name__}: {exc}",
                                symbol=str(sym),
                                point=const_name,
                            )
                for pt in probe_points:
                    lim, lim_error = _bounded_limit(expr, sym, pt, "singularity")
                    den_lim, den_error = (
                        _bounded_limit(den, sym, pt, "denominator")
                        if den.has(sym)
                        else (None, None)
                    )
                    is_singular = False
                    if lim is not None:
                        is_singular = (
                            lim.has(sp.oo)
                            or lim.has(sp.zoo)
                            or lim.has(sp.nan)
                            or lim in [sp.oo, -sp.oo, sp.zoo, sp.nan]
                        )
                    if den_lim is not None and den_lim == 0:
                        is_singular = True
                    if lim is None and den.has(sym):
                        try:
                            if den.subs(sym, pt) == 0:
                                is_singular = True
                        except PROBE_SWALLOWED_ERRORS as exc:
                            _record_probe_error(
                                "DenominatorSubstitutionProbe",
                                "singularity",
                                f"{type(exc).__name__}: {exc}",
                                symbol=str(sym),
                                point=str(pt),
                            )
                        else:
                            # The vanishing-denominator question was answered
                            # directly at this point, so the crashed limit left no
                            # gap; the boundary decision is still a measurement.
                            _resolve(lim_error, "denominator evaluated by substitution")
                            _resolve(den_error, "denominator evaluated by substitution")
                    elif den_lim is not None:
                        _resolve(den_error, "denominator limit evaluated")
                    if is_singular:
                        label = (
                            "Division-by-zero singularity at 0"
                            if pt == 0
                            else f"Boundary singularity at {sym}={pt}"
                        )
                        if (sym, label) not in anomalies:
                            anomalies.append((sym, label))
                # 1b. Relativistic-pattern check: sqrt(1 - (v/c)^2) style
                # denominators vanish at v=c even when sympy limits hang.
                for sub in list(expr.atoms(sp.Pow)) + ([expr] if isinstance(expr, sp.Pow) else []):
                    if len(sub.args) == 2 and sub.args[1] in (sp.Rational(-1, 2), -1, -2, sp.Integer(-1)):
                        try:
                            base = sub.args[0]
                            if base.has(sym):
                                for const_name in ("c_light",):
                                    if const_name in [str(s) for s in base.free_symbols]:
                                        tag = f"Relativistic-style singularity near {sym}={const_name}"
                                        if not any(s == sym and t == tag for s, t in anomalies):
                                            # Confirm numerically before flagging.
                                            try:
                                                f = sp.lambdify(sym, sub, modules=["numpy"])
                                                cval = float(PHYSICAL_CONSTANTS[const_name].value)
                                                import math as _math

                                                probe = [cval * 0.999999, cval * 0.9999999]
                                                vals = []
                                                for p in probe:
                                                    try:
                                                        # Probing just below v=c on
                                                        # purpose: the non-finite
                                                        # value is the finding.
                                                        with quiet_float_warnings():
                                                            vals.append(abs(float(f(p))))
                                                    except PROBE_SWALLOWED_ERRORS as exc:
                                                        # Recorded *and* resolved: the
                                                        # fallback counts the point as
                                                        # divergent, which flags an
                                                        # anomaly rather than clearing
                                                        # one, so it cannot read as a pass.
                                                        _resolve(
                                                            _record_probe_error(
                                                                "RelativisticNumericProbe",
                                                                "relativistic",
                                                                f"{type(exc).__name__}: {exc}",
                                                                symbol=str(sym),
                                                                point=str(p),
                                                            ),
                                                            "counted as divergent (conservative)",
                                                        )
                                                        vals.append(float("inf"))
                                                if any((not _math.isfinite(v)) or v > 1e6 for v in vals):
                                                    anomalies.append((sym, tag))
                                            except PROBE_SWALLOWED_ERRORS as exc:
                                                # The tag is not added, so a real
                                                # relativistic pole can go unreported:
                                                # leave it unresolved.
                                                _record_probe_error(
                                                    "RelativisticConfirmationProbe",
                                                    "relativistic",
                                                    f"{type(exc).__name__}: {exc}",
                                                    symbol=str(sym),
                                                    point=str(sub),
                                                )
                        except PROBE_SWALLOWED_ERRORS as exc:
                            _record_probe_error(
                                "BoundaryPowProbe",
                                "relativistic",
                                f"{type(exc).__name__}: {exc}",
                                symbol=str(sym),
                                point=str(sub),
                            )
                            continue

                # 2. Logarithmic singularity at boundary x -> 0 or negative
                for sub in expr.atoms(sp.log):
                    arg = sub.args[0]
                    if arg.has(sym):
                        lim_log, _log_error = _bounded_limit(arg, sym, 0, "logarithmic")
                        if lim_log is not None and (
                            lim_log == 0 or lim_log.has(sp.oo) or lim_log.has(-sp.oo)
                        ):
                            anomalies.append((sym, "Logarithmic boundary singularity at 0"))

                # 3. Square root branch-cut for non-positive values
                all_pow_atoms = list(expr.atoms(sp.Pow))
                if isinstance(expr, sp.Pow):
                    all_pow_atoms.append(expr)
                for sub in all_pow_atoms:
                    if len(sub.args) == 2 and sub.args[1] == sp.Rational(1, 2):
                        base = sub.args[0]
                        if base.has(sym):
                            anomalies.append((sym, "Square-root branch-cut for non-positive inputs"))

            repaired_expr = expr
            eps_introduced: list[str] = []
            if anomalies:
                # 1. Robust denominator extraction & softening (also covers
                # generic "Boundary singularity" and relativistic tags).
                for sym, err_type in anomalies:
                    if "Division-by-zero" in err_type or "Boundary singularity" in err_type or "Relativistic" in err_type:
                        reg = sp.Symbol(f"eps_{sym}")
                        var_units[reg] = var_units.get(sym, self.ureg.meter)
                        if str(reg) not in eps_introduced:
                            eps_introduced.append(str(reg))
                        num, den = sp.fraction(repaired_expr)
                        try:
                            den_lim = limit_with_timeout(den, sym, 0, timeout=limit_timeout)
                        except LimitTimeout:
                            den_lim = None
                            _record_probe_error(
                                "DenominatorLimitProbe",
                                "softening",
                                f"limit timed out after {limit_timeout}s",
                                symbol=str(sym),
                                point="0",
                            )
                        except PROBE_SWALLOWED_ERRORS as exc:
                            # A crash here picks the non-softening repair branch,
                            # so the choice of regularization is no longer backed
                            # by a measurement. Recorded as a gap in the report.
                            _record_probe_error(
                                "DenominatorLimitProbe",
                                "softening",
                                f"{type(exc).__name__}: {exc}",
                                symbol=str(sym),
                                point="0",
                            )
                            den_lim = None
                        if den_lim == 0:
                            # Substitute sym -> sqrt(sym^2 + reg^2) to guarantee den > 0 on all of R
                            # and strictly preserve dimensional units for all powers and prefactors
                            soft_sym = sp.sqrt(sym**2 + reg**2)
                            den = den.subs(sym, soft_sym)
                            repaired_expr = num / den
                        else:
                            soft_sym = sp.sqrt(sym**2 + reg**2)
                            repaired_expr = repaired_expr.subs(sym, soft_sym)

                # 2. Regularize logarithms: log(x) -> log((x + sqrt(x^2 + eps^2))/2 + eps)
                # eps carries x's own units (not sym's): x is often a compound
                # expression such as L/g whose units differ from any one symbol.
                # Nodes already containing an eps_* regularizer are skipped:
                # eps-guarded arguments are strictly positive sums, so inner
                # envelope roots can never go singular (re-softening them only
                # nests envelopes and breaks finite-at-zero evaluation).
                for sym, err_type in anomalies:
                    if "Logarithmic" in err_type:
                        log_nodes = [node for node in repaired_expr.atoms(sp.log) if node.args[0].has(sym)]
                        for node in log_nodes:
                            x = node.args[0]
                            if any(str(s).startswith("eps_") for s in x.free_symbols):
                                continue
                            reg = self._regularizer(var_units, sym, x, self.ureg.dimensionless)
                            if str(reg) not in eps_introduced:
                                eps_introduced.append(str(reg))
                            soft_log = sp.log((x + sp.Pow(x**2 + reg**2, sp.Rational(1, 2))) / 2 + reg)
                            repaired_expr = repaired_expr.subs(node, soft_log)

                # 3. Regularize square roots: sqrt(x) -> sqrt((x + sqrt(x^2 + eps^2))/2)
                # eps carries x's own units for the same reason as above.
                for sym, err_type in anomalies:
                    if "Square-root" in err_type:
                        sqrt_nodes = list({
                            node for node in repaired_expr.atoms(sp.Pow)
                            if len(node.args) == 2 and node.args[1] == sp.Rational(1, 2) and node.args[0].has(sym)
                        })
                        for node in sqrt_nodes:
                            x = node.args[0]
                            if any(str(s).startswith("eps_") for s in x.free_symbols):
                                continue
                            reg = self._regularizer(var_units, sym, x, self.ureg.meter)
                            if str(reg) not in eps_introduced:
                                eps_introduced.append(str(reg))
                            soft_sqrt = sp.Pow((x + sp.Pow(x**2 + reg**2, sp.Rational(1, 2))) / 2, sp.Rational(1, 2))
                            repaired_expr = repaired_expr.subs(node, soft_sqrt)

                # Re-verify repaired expression
                repaired_unit = self.compute_expression_dimensionality(repaired_expr, var_units)
                if repaired_unit.dimensionality != original_unit.dimensionality:
                    raise TypeError(
                        f"Regularization broke dimensional homogeneity: {original_unit} -> {repaired_unit}"
                    )
            else:
                repaired_unit = original_unit

            # Tier-3 distortion quantification + validity domain. Previously
            # nothing quantified the approximation error the softening
            # introduces; now every REPAIRED result carries it.
            distortion: dict[str, Any] = {"max_distortion": None, "mean_distortion": None}
            validity_domain = "all probed boundaries finite" if not anomalies else (
                "regularized; re-validate for |x| >> eps and re-fit eps_* on data"
            )
            if anomalies:
                try:
                    from apedp.repair import RegularizationAuditor as _RA

                    free = [s for s in expr.free_symbols if not str(s).startswith("eps_")]
                    if len(free) == 1:
                        sym0 = free[0]
                        f_class = sp.lambdify(sym0, expr, modules=["numpy"])
                        f_reg = sp.lambdify(
                            sym0,
                            repaired_expr.subs({sp.Symbol(e): 1e-6 for e in eps_introduced}),
                            modules=["numpy"],
                        )
                        grid = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
                        t3 = _RA.audit_tier3_asymptotic_distortion(f_class, f_reg, grid)
                        distortion = {
                            "max_distortion": t3["max_distortion"],
                            "mean_distortion": t3["mean_distortion"],
                        }
                except (ValueError, TypeError, ArithmeticError) as e:
                    logger.warning("Distortion quantification failed (%s)", e)

            unresolved = [e for e in probe_errors if e["unresolved"]]
            if anomalies:
                status = "REPAIRED"
            elif unresolved:
                # Nothing was found *and* a boundary question went unanswered.
                # That is not a clean bill of health, so it must not read as one:
                # before this, the swallowed probe was simply assumed to mean
                # "no singularity here" and the result said APPROVED.
                status = "UNDETERMINED"
                validity_domain = "not established: " + "; ".join(
                    f"{e['probe']} at {e['symbol']}={e['point']}" for e in unresolved
                )
            else:
                status = "APPROVED"
            return {
                "status": status,
                "anomalies": anomalies,
                "original_expression": expr,
                "repaired_expression": repaired_expr,
                "inferred_unit": str(repaired_unit.units if hasattr(repaired_unit, 'units') else repaired_unit),
                "eps_symbols": eps_introduced,
                "distortion": distortion,
                "validity_domain": validity_domain,
                "probe_errors": probe_errors,
                "probe_error_count": len(probe_errors),
                "unresolved_probe_count": len(unresolved),
            }
        except (ValueError, TypeError) as e:
            logger.warning("Stress test rejected formula '%s': %s", formula_str, e)
            return {"status": "REJECTED", "error": str(e)}


class RegularizationAuditor:
    """Four-Tier Assurance Engine & Parameter Sweep Optimizer.

    Formalizes regularization testing into four distinct diagnostic tiers:
    1. Tier 1: Mathematical Verification — Limit existence, pole removal, derivative continuity.
    2. Tier 2: Numerical Stability — Float overflow/underflow, NaN checks, conditioning across bounds.
    3. Tier 3: Asymptotic Verification — Quantifies pre-singularity distortion delta(x) = |f_reg - f_class| / |f_class|.
    4. Tier 4: Empirical Sweep & Calibration — Optimizes regularization parameters (e.g. eps, dc, z0)
       to minimize operational distortion while enforcing boundary bounds.
    """

    @staticmethod
    def audit_tier1_mathematical_limit(
        expr: sp.Expr,
        sym: sp.Symbol,
        boundary_val: float,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Verify symbolic limit existence and pole elimination at the boundary."""
        try:
            lim = limit_with_timeout(expr, sym, boundary_val, timeout=timeout)
            is_finite = not (lim.has(sp.oo) or lim.has(-sp.oo) or lim.has(sp.zoo) or lim.has(sp.nan))
            return {
                "limit_value": float(lim) if is_finite and lim.is_number else str(lim),
                "is_finite": is_finite,
                "singularity_removed": is_finite,
            }
        except LimitTimeout as e:
            return {"limit_value": None, "is_finite": False, "singularity_removed": False, "error": str(e)}
        except (ValueError, TypeError, ArithmeticError) as e:
            return {"limit_value": None, "is_finite": False, "singularity_removed": False, "error": str(e)}

    @staticmethod
    def audit_tier2_numerical_stability(
        func: Callable[[float], float],
        test_grid: list[float] | np.ndarray
    ) -> dict[str, Any]:
        """Verify numerical stability: immunity to NaNs, overflows, and large condition numbers."""
        values: list[float] = []
        has_nan = False
        has_inf = False
        for x in test_grid:
            try:
                v = float(func(float(x)))
                if math.isnan(v):
                    has_nan = True
                elif math.isinf(v):
                    has_inf = True
                values.append(v)
            except (ValueError, TypeError, ArithmeticError) as e:
                logger.debug("Stability probe failed at x=%r (%s)", x, e)
                has_inf = True
                values.append(float('inf'))

        finite_vals = [v for v in values if math.isfinite(v)]
        condition_spread = (max(finite_vals) / (min(finite_vals) + 1e-15)) if finite_vals else float('inf')
        return {
            "num_points_tested": len(test_grid),
            "nan_detected": has_nan,
            "inf_detected": has_inf,
            "numerically_stable": (not has_nan) and (not has_inf),
            "max_value": max(finite_vals) if finite_vals else float('nan'),
            "min_value": min(finite_vals) if finite_vals else float('nan'),
            "condition_spread": condition_spread,
        }

    @staticmethod
    @quiet_float_warnings()
    def audit_tier3_asymptotic_distortion(
        classical_func: Callable[[float], float],
        regularized_func: Callable[[float], float],
        operational_grid: list[float] | np.ndarray,
        distortion_threshold: float = 0.05
    ) -> dict[str, Any]:
        """Quantify distortion in the operational regime: delta(x) = |f_reg(x) - f_class(x)| / |f_class(x)|."""
        distortions: dict[float, float] = {}
        for x in operational_grid:
            x_f = float(x)
            f_c = float(classical_func(x_f))
            f_r = float(regularized_func(x_f))
            if abs(f_c) > 1e-15:
                err = abs(f_r - f_c) / abs(f_c)
            else:
                err = abs(f_r - f_c)
            distortions[x_f] = err

        max_err = max(distortions.values()) if distortions else 0.0
        mean_err = float(np.mean(list(distortions.values()))) if distortions else 0.0
        return {
            "max_distortion": max_err,
            "mean_distortion": mean_err,
            "within_tolerance": max_err <= distortion_threshold,
            "distortion_profile": distortions,
        }

    @classmethod
    def audit_tier4_parameter_sweep(
        cls,
        regularized_factory: Callable[[float], Callable[[float], float]],
        classical_func: Callable[[float], float],
        param_candidates: list[float],
        operational_grid: list[float] | np.ndarray,
        boundary_point: float,
        max_boundary_allowance: float | None = None
    ) -> dict[str, Any]:
        """Sweep candidate regularization parameters to find optimal tradeoff between distortion and boundary relief."""
        results: list[dict[str, Any]] = []
        for param in param_candidates:
            reg_f = regularized_factory(param)
            t3 = cls.audit_tier3_asymptotic_distortion(classical_func, reg_f, operational_grid)
            b_val = float(reg_f(boundary_point))
            is_valid = math.isfinite(b_val) and (max_boundary_allowance is None or abs(b_val) <= max_boundary_allowance)
            results.append({
                "param_value": param,
                "max_operational_distortion": t3["max_distortion"],
                "boundary_value": b_val,
                "admissible": is_valid,
            })

        admissible = [r for r in results if r["admissible"]]
        best = min(admissible, key=lambda r: r["max_operational_distortion"]) if admissible else None
        return {
            "sweep_results": results,
            "optimal_parameter": best["param_value"] if best else None,
            "optimal_distortion": best["max_operational_distortion"] if best else None,
            "optimal_boundary_value": best["boundary_value"] if best else None,
        }
