"""Contradictory-constraint engine: simultaneous laws that cannot both hold."""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

from .counterexamples import (
    InfeasibilityCertificate,
)


class ContradictoryConstraintEngine:
    """Symbolic infeasibility engine for declared constraint sets.

    The engine parses each constraint's mathematical content (the text after the
    ``label:``) into typed claims, then attempts to *derive* a contradiction
    between two claims by exact symbolic computation (SymPy limits, integration,
    inequality algebra). No verdict is selected by matching prose keywords: a
    certificate exists only when a pairwise derivation actually succeeds, and the
    certificate text states every premise used (continuity at the contact point,
    declared or assumed parameter signs, the integration interval).

    Recognized claim forms (math part after ``label:``):

    * ``f(a) = v`` -- pin: the model takes value ``v`` at numeric point ``a``.
    * ``lim_{x->a+} f(x) = v`` -- one-sided limit claim at a contact point.
    * ``f(x) >= g(x)`` / ``f(x) <= g(x)`` -- pointwise envelope bound.
    * ``v <= c`` / ``v >= alpha*c`` -- scalar bound on a named quantity.
    * ``dE/dt = 0`` / ``dE/dt >= P_0`` -- signed rate claim (chains like
      ``dE/dt >= P_0 > 0`` also register ``P_0 > 0`` as a sign declaration).
    * ``f(x) = 1/x`` or a bare expression like ``1/x`` -- closed-form claim.
    * ``Im(f) = 0`` / ``real_output`` -- the model must be real-valued.
    * ``int_1_inf_f_finite`` / ``bounded_integral`` -- the model's integral over
      ``[1, oo)`` must be finite.
    * ``k > 0`` / ``alpha > 1`` -- declared-sign assumptions (premises, never
      contradictions on their own).

    Derived contradictions, each computed symbolically at decision time:

    1. envelope bound vs limit/pin: the envelope's exact one-sided limit at the
       contact point exceeds the pinned value, so continuity fails.
    2. envelope bound vs envelope bound / scalar vs scalar: the admitted ranges
       are disjoint (exact difference of envelopes with declared/assumed signs).
    3. rate vs rate: the rate ranges are disjoint (e.g. ``= 0`` vs ``>= P_0``
       with ``P_0 > 0``).
    4. form vs real_output: the closed form's imaginary part is provably nonzero.
    5. form vs integrable: the closed form's exact integral over ``[1, oo)``
       diverges.

    Where a derivation needs the sign of an undeclared parameter (e.g. ``k`` in
    ``f(x) >= k/x**2``), the parameter is taken strictly positive *and the
    assumption is disclosed in the certificate text* -- it is a premise of the
    proof, not a hidden keyword. Unparseable constraints are skipped (refusal to
    interpret), never treated as satisfied.
    """

    _PROSE_TAIL_RE = re.compile(r"\s+for\s+(?:all|every|any)\b.*$|\s+whenever\b.*$", re.IGNORECASE)

    @staticmethod
    def _split_label(constraint: str) -> tuple[str, str]:
        """Split ``"label: math"`` into ``(label, math)``; no colon means both are the text."""
        text = constraint.strip().rstrip(".")
        if ":" in text:
            label, math_part = text.split(":", 1)
            return label.strip(), math_part.strip()
        return text, text

    @staticmethod
    def _strip_prose(math_text: str) -> str:
        """Drop trailing qualifier prose (``for all x > 0``) from a math snippet."""
        return ContradictoryConstraintEngine._PROSE_TAIL_RE.sub("", math_text).strip()

    @staticmethod
    def _parse_symbols(text: str) -> list[str]:
        """Identifiers in a math snippet, longest first (so ``x_1`` wins over ``x``)."""
        return sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)), key=len, reverse=True)

    @staticmethod
    def _make_symbol(name: str, declared: dict[str, bool]) -> sp.Symbol:
        """Build a SymPy symbol honouring declared-sign assumptions when available."""
        if name in declared:
            return sp.Symbol(name, positive=True) if declared[name] else sp.Symbol(name, negative=True)
        return sp.Symbol(name, real=True)

    @classmethod
    def _sub_symbols(cls, text: str, declared: dict[str, bool]) -> dict[str, sp.Symbol]:
        # Math-function names must stay resolvable to functions, not be shadowed
        # by plain Symbols (which makes ``sqrt(...)`` uncallable).
        return {
            name: cls._make_symbol(name, declared)
            for name in cls._parse_symbols(text)
            if name not in ("sqrt", "exp", "log", "Abs", "abs", "pi")
        }

    @staticmethod
    def _safe_sympify(math_text: str, local_dict: dict[str, sp.Symbol]) -> sp.Expr | None:
        """Parse a math snippet with no attribute access and no execution surface.

        Returns ``None`` when the snippet is not a closed symbolic expression
        (prose, relational chains, attribute access, unknown call syntax). This is
        a *refusal to interpret*, not a mismatch verdict: uninterpreted text is
        skipped by the pairwise rules, never silently treated as satisfied.
        """
        cleaned = math_text.strip().rstrip(".")
        cleaned = cleaned.replace("^", "**")  # in constraint math ^ means power, not XOR
        if not cleaned:
            return None
        # Attribute access is refused outright (same policy as safety.safe_parse_expr):
        # remove digit.digit float literals first, any other dot is suspicious.
        if "." in re.sub(r"\d+\.\d+", "", cleaned):
            return None
        if any(tok in cleaned for tok in ("__", ";", "`", "\n")):
            return None
        # Only model symbols f/g/h/u and a small math-function allowlist may be
        # called; unknown call syntax is refused rather than resolved against
        # arbitrary globals by sympify.
        for name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", cleaned):
            if name not in ("f", "g", "h", "u", "sqrt", "exp", "log", "Abs", "abs"):
                return None
        # A tiny math-function locals map, so ``sqrt(...)`` parses as the function
        # rather than an uncallable Symbol (merged under the caller's symbols).
        merged: dict[str, sp.Basic] = {
            "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log,
            "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
        }
        merged.update(local_dict)
        # Implicit multiplication for coefficient literals: ``1.5c`` -> ``1.5*c``,
        # ``2x`` -> ``2*x`` (leaves ``x**2`` alone: digit must PRECEDE the name).
        cleaned = re.sub(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*([A-Za-z_])", r"\1*\2", cleaned)
        try:
            expr = sp.sympify(cleaned, locals=merged, rational=True)
        except (sp.SympifyError, SyntaxError, TypeError, ValueError):
            return None
        if not isinstance(expr, sp.Expr) or isinstance(expr, sp.core.relational.Relational):
            return None
        return expr

    @classmethod
    def _rebuild_positive(cls, expr: sp.Expr, keep: set[str] | None = None) -> tuple[sp.Expr, list[str]]:
        """Rebuild an expression with all free symbols positive (except ``keep``).

        Returns the rebuilt expression and the names whose sign was *assumed*
        rather than declared, so the certificate can disclose the premise.
        """
        keep = keep or set()
        assumed: list[str] = []
        subs_map: dict[sp.Symbol, sp.Symbol] = {}
        for sym in expr.free_symbols:
            if sym.name in keep:
                continue
            if not sym.is_positive:
                subs_map[sym] = sp.Symbol(sym.name, positive=True)
                assumed.append(sym.name)
        return expr.subs(subs_map), sorted(assumed)

    # ------------------------------------------------------------------
    # Claim types
    # ------------------------------------------------------------------

    @dataclass
    class _BoundClaim:
        """A pointwise envelope claim: ``fn(var) rel expr`` with ``rel`` in {>=, <=, >, <}."""

        label: str
        fn: str
        var: str
        direction: str
        envelope: sp.Expr

        @property
        def is_lower(self) -> bool:
            return self.direction in (">=", ">")

    @classmethod
    def _parse_bound(cls, label: str, math_text: str, declared: dict[str, bool]) -> ContradictoryConstraintEngine._BoundClaim | None:
        m = re.match(
            r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*(>=|<=|>|<)\s*(.+?)\s*$",
            math_text,
        )
        if not m:
            return None
        fn, var, direction, env_text = m.groups()
        env_text = cls._strip_prose(env_text)
        syms = cls._sub_symbols(env_text, declared)
        syms[var] = sp.Symbol(var, real=True)
        envelope = cls._safe_sympify(env_text, syms)
        if envelope is None:
            return None
        return cls._BoundClaim(label=label, fn=fn, var=var, direction=direction, envelope=envelope)

    @dataclass
    class _LimitClaim:
        """``lim_{var->a+} fn(var) = v``, or a pin ``fn(a) = v`` at a numeric point."""

        label: str
        fn: str
        var: str  # "" for a plain pin (continuity assumed to identify value and limit)
        point: sp.Expr
        side: str  # "+", "-", or ""
        value: sp.Expr

    @classmethod
    def _parse_limit(
        cls, label: str, math_text: str, declared: dict[str, bool]
    ) -> ContradictoryConstraintEngine._LimitClaim | None:
        # lim_{x->0+} f(x) = 0   |   lim_{x->0+} f = 0
        m = re.match(
            r"^\s*lim_?\s*\{?\s*([A-Za-z_]\w*)\s*->\s*([^\s}]+?)\s*(\+|-)?\s*\}?\s*"
            r"([A-Za-z_]\w*)\s*(?:\(\s*[A-Za-z_]\w*\s*\))?\s*=\s*(.+?)\s*$",
            math_text,
        )
        if m:
            var, point_text, side, fn, value_text = m.groups()
            syms = cls._sub_symbols(f"{point_text} {value_text}", declared)
            syms[var] = sp.Symbol(var, real=True)
            point = cls._safe_sympify(point_text, syms)
            value = cls._safe_sympify(value_text, syms)
            if point is None or value is None:
                return None
            return cls._LimitClaim(label=label, fn=fn, var=var, point=point, side=side or "", value=value)
        # Pin form: f(0) = 0, f(0, t) = 0 -- first argument is the (numeric) point.
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(\s*([^,()]+?)\s*(?:,[^()]*)?\)\s*=\s*(.+?)\s*$", math_text)
        if m and m.group(1) != "lim":
            fn, point_text, value_text = m.groups()
            point_text = cls._strip_prose(point_text)
            value_text = cls._strip_prose(value_text)
            syms = cls._sub_symbols(f"{point_text} {value_text}", declared)
            point = cls._safe_sympify(point_text, syms)
            value = cls._safe_sympify(value_text, syms)
            if point is None or value is None or not point.is_number:
                return None  # a symbolic point is a form claim, handled elsewhere
            return cls._LimitClaim(label=label, fn=fn, var="", point=point, side="", value=value)
        return None

    @staticmethod
    def _limit_of(expr: sp.Expr, var: sp.Symbol, point: sp.Expr, side: str) -> sp.Expr | None:
        """Exact one-sided limit, or None when SymPy cannot decide."""
        try:
            if side in ("+", "-"):
                return sp.limit(expr, var, point, dir=side)
            return sp.limit(expr, var, point)
        except (
            sp.PoleError, NotImplementedError, ZeroDivisionError, ValueError, TypeError, ArithmeticError
        ):
            return None

    @dataclass
    class _RateClaim:
        """A signed rate claim on a named quantity: ``dE/dt = 0`` or ``dE/dt >= P_0``."""

        label: str
        name: str
        direction: str
        value: sp.Expr

    @classmethod
    def _parse_rate(cls, label: str, math_text: str, declared: dict[str, bool]) -> ContradictoryConstraintEngine._RateClaim | None:
        m = re.match(
            r"^\s*d([A-Za-z_]\w*)/d[A-Za-z_]\w*\s*(>=|<=|=|>|<)\s*(.+?)\s*$",
            math_text,
        )
        if not m:
            return None
        name, direction, value_text = m.groups()
        # Strip a chained relational tail (``dE/dt >= P_0 > 0`` -> ``P_0``); the
        # chain member itself was already registered as a declaration in pass 0.
        value_text = re.split(r"\s*(?:>=|<=|>|<)\s*", value_text)[0].strip()
        value_text = cls._strip_prose(value_text)
        syms = cls._sub_symbols(value_text, declared)
        value = cls._safe_sympify(value_text, syms)
        if value is None:
            return None
        return cls._RateClaim(label=label, name=name, direction=direction, value=value)

    @dataclass
    class _ScalarClaim:
        """A scalar bound on a named quantity with no parens: ``v <= c``."""

        label: str
        name: str
        direction: str
        bound: sp.Expr

    @classmethod
    def _parse_scalar(cls, label: str, math_text: str, declared: dict[str, bool]) -> ContradictoryConstraintEngine._ScalarClaim | None:
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*(>=|<=|>|<)\s*(.+?)\s*$", math_text)
        if not m:
            return None
        name, direction, bound_text = m.groups()
        bound_text = cls._strip_prose(bound_text)
        syms = cls._sub_symbols(bound_text, declared)
        bound = cls._safe_sympify(bound_text, syms)
        if bound is None:
            return None
        return cls._ScalarClaim(label=label, name=name, direction=direction, bound=bound)

    @dataclass
    class _FormClaim:
        """The model equals a closed-form expression: ``f(x) = 1/x`` (or bare ``1/x``)."""

        label: str
        fn: str
        expr: sp.Expr

    @classmethod
    def _parse_form(cls, label: str, math_text: str, declared: dict[str, bool]) -> ContradictoryConstraintEngine._FormClaim | None:
        math_text = cls._strip_prose(math_text)
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*=\s*(.+?)\s*$", math_text)
        if m and m.group(1) != "lim":
            fn, var, expr_text = m.groups()
            syms = cls._sub_symbols(expr_text, declared)
            syms[var] = sp.Symbol(var, real=True)
            expr = cls._safe_sympify(expr_text, syms)
            if expr is None or not expr.has(syms[var]):
                return None  # constant RHS is a pin, handled by _parse_limit
            return cls._FormClaim(label=label, fn=fn, expr=expr)
        # Bare-expression form (e.g. ``sqrt(-1 - x**2)``, ``1/x``): the constraint
        # asserts the model has exactly this closed form. Model-symbol calls
        # (f(x) ...) without '=' are not closed forms; math-function calls are.
        call_names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", math_text)
        if "=" in math_text or any(name in ("f", "g", "h", "u") for name in call_names):
            return None
        syms = cls._sub_symbols(math_text, declared)
        expr = cls._safe_sympify(math_text, syms)
        if expr is None or not expr.free_symbols:
            return None
        if len(expr.free_symbols) != 1:
            return None
        var = next(iter(expr.free_symbols))
        return cls._FormClaim(label=label, fn=f"f({var.name})", expr=expr)

    # ------------------------------------------------------------------
    # Pairwise derivation rules
    # ------------------------------------------------------------------

    @classmethod
    def _bound_vs_limit(
        cls, bound: ContradictoryConstraintEngine._BoundClaim, lim: ContradictoryConstraintEngine._LimitClaim
    ) -> InfeasibilityCertificate | None:
        """Lower envelope's exact limit at the contact point exceeds the pinned value."""
        if bound.fn != lim.fn or not bound.is_lower:
            return None
        if lim.var:
            # Explicit limit claim: same variable, one-sided at the contact point.
            if lim.var != bound.var or lim.side != "+":
                return None
        elif lim.side or not lim.point.is_number:
            return None  # a pin at a non-numeric point is not a contact claim
        if not lim.value.is_number:
            return None
        var_name = lim.var or bound.var
        rebuilt, assumed = cls._rebuild_positive(bound.envelope, keep={bound.var})
        envelope_lim = cls._limit_of(rebuilt, sp.Symbol(var_name, real=True), lim.point, "+")
        if envelope_lim is None or not envelope_lim.is_infinite:
            return None
        if not (envelope_lim > lim.value):
            return None
        premise = (
            f" Premise: envelope parameter(s) {', '.join(assumed)} taken strictly positive "
            "(undeclared; the bound is only a positive lower envelope under this sign)."
            if assumed else ""
        )
        return InfeasibilityCertificate(
            is_infeasible=True,
            conflicting_constraints=[
                f"{lim.label}: lim_{{{var_name}->{lim.point}+}} {lim.fn} = {lim.value}",
                f"{bound.label}: {bound.fn}({bound.var}) >= {bound.envelope}",
            ],
            incompatibility_proof=(
                "CONTRADICTORY_CONSTRAINTS_PROOF: Constraint "
                f"'{bound.label}' requires {bound.fn}({bound.var}) >= {bound.envelope} for all "
                f"{bound.var} > 0. Taking the exact one-sided limit (SymPy): "
                f"lim_{{{bound.var}->{lim.point}+}} {bound.envelope} = {envelope_lim}. "
                f"Any function satisfying the bound and continuous at "
                f"{bound.var} = {lim.point} then has "
                f"lim_{{{bound.var}->{lim.point}+}} {bound.fn} >= {envelope_lim}, which contradicts "
                f"constraint '{lim.label}' requiring the limit equal {lim.value}. "
                "The system of constraints is mutually exclusive under continuity at the "
                "contact point; the admissible function space is empty."
                + premise
            ),
            mathematical_argument=(
                f"lim_{{{bound.var}->{lim.point}+}} {bound.fn}({bound.var}) >= "
                f"lim_{{{bound.var}->{lim.point}+}} {bound.envelope} = {envelope_lim} != {lim.value}"
            ),
        )

    @classmethod
    def _envelopes_disjoint(
        cls,
        lower_label: str, lower_expr: sp.Expr,
        upper_label: str, upper_expr: sp.Expr,
        subject: str,
    ) -> InfeasibilityCertificate | None:
        """Exact gap between a lower and an upper envelope is provably negative."""
        rebuilt, assumed = cls._rebuild_positive(upper_expr - lower_expr)
        gap = sp.simplify(rebuilt)
        if gap.is_negative is not True:
            return None
        premise = (
            f" Premise: parameter sign(s) of {', '.join(assumed)} taken positive (undeclared)."
            if assumed else ""
        )
        return InfeasibilityCertificate(
            is_infeasible=True,
            conflicting_constraints=[
                f"{lower_label}: {subject} >= {lower_expr}",
                f"{upper_label}: {subject} <= {upper_expr}",
            ],
            incompatibility_proof=(
                "CONTRADICTORY_CONSTRAINTS_PROOF: Constraint "
                f"'{lower_label}' requires {subject} >= {lower_expr} while constraint "
                f"'{upper_label}' requires {subject} <= {upper_expr}. Their exact difference "
                f"is {gap} < 0, so no value satisfies both bounds simultaneously; the "
                "admissible set is empty."
                + premise
            ),
            mathematical_argument=(
                f"{upper_expr} - {lower_expr} = {gap} < 0 => "
                f"{lower_expr} > {upper_expr} (Contradiction)"
            ),
        )

    @classmethod
    def _rate_vs_rate(
        cls, r1: ContradictoryConstraintEngine._RateClaim, r2: ContradictoryConstraintEngine._RateClaim
    ) -> InfeasibilityCertificate | None:
        """Disjoint ranges on the same rate quantity (``= 0`` vs ``>= P_0``, P_0 > 0)."""
        if r1.name != r2.name:
            return None
        for eq, ineq in ((r1, r2), (r2, r1)):
            if eq.direction != "=" or ineq.direction == "=":
                continue
            # Substituting the pinned rate into the inequality must be provably false.
            lhs_minus_rhs = sp.simplify(eq.value - ineq.value)
            violated = (
                lhs_minus_rhs.is_negative if ineq.direction == ">="
                else lhs_minus_rhs.is_nonpositive if ineq.direction == ">"
                else lhs_minus_rhs.is_positive if ineq.direction == "<="
                else lhs_minus_rhs.is_nonnegative  # "<"
            )
            if violated is not True:
                continue  # not derivable: the pair is merely uninformative, not contradicted
            return InfeasibilityCertificate(
                is_infeasible=True,
                conflicting_constraints=[
                    f"{eq.label}: d{eq.name}/dt = {eq.value}",
                    f"{ineq.label}: d{ineq.name}/dt {ineq.direction} {ineq.value}",
                ],
                incompatibility_proof=(
                    "CONTRADICTORY_CONSTRAINTS_PROOF: Constraint "
                    f"'{eq.label}' pins d{eq.name}/dt = {eq.value} exactly, while constraint "
                    f"'{ineq.label}' requires d{ineq.name}/dt {ineq.direction} {ineq.value}. "
                    f"Substituting the pinned rate gives {eq.value} {ineq.direction} {ineq.value}, "
                    f"i.e. {lhs_minus_rhs} {ineq.direction} 0, which is false under the declared "
                    "parameter signs. The admissible set is empty; no regularization can "
                    "reconcile the two."
                ),
                mathematical_argument=(
                    f"{eq.value} = d{eq.name}/dt {ineq.direction} {ineq.value} => "
                    f"{eq.value} {ineq.direction} {ineq.value} => "
                    f"{lhs_minus_rhs} {ineq.direction} 0 (Contradiction)"
                ),
            )
        return None

    @classmethod
    def _scalar_vs_scalar(
        cls, s1: ContradictoryConstraintEngine._ScalarClaim, s2: ContradictoryConstraintEngine._ScalarClaim
    ) -> InfeasibilityCertificate | None:
        """Disjoint scalar ranges on the same named quantity (``v <= c`` vs ``v >= 1.5c``)."""
        if s1.name != s2.name:
            return None
        lower, upper = (
            (s1, s2) if s1.direction in (">=", ">") else (s2, s1)
        )
        if lower.direction not in (">=", ">") or upper.direction not in ("<=", "<"):
            return None
        return cls._envelopes_disjoint(
            lower.label, lower.bound, upper.label, upper.bound, subject=lower.name,
        )

    @classmethod
    def _form_vs_real_output(
        cls, form: ContradictoryConstraintEngine._FormClaim, label: str
    ) -> InfeasibilityCertificate | None:
        """Closed form whose imaginary part is provably nonzero cannot be real-valued."""
        rebuilt, _ = cls._rebuild_positive(form.expr)
        im = sp.simplify(sp.im(rebuilt))
        if im.is_positive is not True:
            return None
        return InfeasibilityCertificate(
            is_infeasible=True,
            conflicting_constraints=[
                f"{label}: Im({form.fn}) = 0",
                f"{form.label}: {form.fn} = {form.expr}",
            ],
            incompatibility_proof=(
                "CONTRADICTORY_CONSTRAINTS_PROOF: Constraint "
                f"'{form.label}' fixes {form.fn} = {form.expr}. Its imaginary part, computed "
                f"exactly (SymPy im/simplify), is Im({form.expr}) = {im}, strictly positive for "
                "every real x (the radicand is strictly negative on R). Constraint "
                f"'{label}' requires Im({form.fn}) = 0. Both cannot hold; the admissible "
                "function space is empty."
            ),
            mathematical_argument=f"Im({form.expr}) = {im} != 0 on R",
        )

    @classmethod
    def _form_vs_integrable(
        cls, form: ContradictoryConstraintEngine._FormClaim, label: str
    ) -> InfeasibilityCertificate | None:
        """Closed form whose exact integral over [1, oo) diverges cannot be integrable."""
        if len(form.expr.free_symbols) != 1:
            return None
        sym = next(iter(form.expr.free_symbols))
        try:
            integral = sp.integrate(form.expr, (sym, 1, sp.oo))
        except (
            NotImplementedError, ValueError, TypeError, ArithmeticError, ZeroDivisionError
        ):
            return None
        if integral not in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            return None
        return InfeasibilityCertificate(
            is_infeasible=True,
            conflicting_constraints=[
                f"{form.label}: {form.fn} = {form.expr}",
                f"{label}: integral of the model over [1, inf) is finite",
            ],
            incompatibility_proof=(
                "CONTRADICTORY_CONSTRAINTS_PROOF: Constraint "
                f"'{form.label}' fixes {form.fn} = {form.expr}. Its exact integral over "
                f"[1, oo) (SymPy integrate) is {integral}. Constraint '{label}' requires a "
                "finite total integral on [1, oo). Both cannot hold without changing the form "
                "itself; the admissible function space is empty."
            ),
            mathematical_argument=f"\\int_1^\\infty {form.expr} d{sym.name} = {integral} (diverges)",
        )

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    #: ``decl``-style sign assumptions, standalone or chained (``k > 0``,
    #: ``alpha > 1``, ``... dE/dt >= P_0 > 0``). Numeric right-hand side only.
    _SIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<)\s*(\d+\.?\d*)")

    @classmethod
    def check_incompatibility(cls, constraints: list[str]) -> InfeasibilityCertificate | None:
        """Analyzes declared constraints for mutual contradiction by symbolic derivation.

        Returns a certificate only when a pairwise symbolic rule *derives* the
        contradiction (exact SymPy limit / integrate / inequality algebra);
        uninterpreted or merely uninformative constraint sets return ``None`` --
        no verdict, never a silent pass. Every premise used (declared or assumed
        parameter signs, continuity at the contact point, integration interval)
        is stated in the certificate text.
        """
        if not constraints:
            return None

        # Pass 0: collect declared-sign assumptions, from standalone decls and
        # from chained relations inside compound constraints. Numeric RHS only.
        declared: dict[str, bool] = {}
        for constraint in constraints:
            _, math_text = cls._split_label(constraint)
            for name, rel, rhs_text in cls._SIGN_RE.findall(math_text):
                rhs = float(rhs_text)
                # A decl "P_0 > 0" states the *variable* is positive, so a strict
                # relation against zero already implies a positive sign; a
                # non-strict one does not (>= 0 allows equality), and "name >= c"
                # implies positivity only when c itself is positive.
                if (rel == ">" and rhs >= 0) or (rel == ">=" and rhs > 0):
                    declared.setdefault(name, True)
                elif (rel == "<" and rhs <= 0) or (rel == "<=" and rhs < 0):
                    declared.setdefault(name, False)

        # Pass 1: parse every constraint into typed claims.
        bounds: list[ContradictoryConstraintEngine._BoundClaim] = []
        limits: list[ContradictoryConstraintEngine._LimitClaim] = []
        rates: list[ContradictoryConstraintEngine._RateClaim] = []
        scalars: list[ContradictoryConstraintEngine._ScalarClaim] = []
        forms: list[ContradictoryConstraintEngine._FormClaim] = []
        real_output: list[str] = []
        integrable: list[str] = []
        for constraint in constraints:
            label, math_text = cls._split_label(constraint)
            compact = re.sub(r"\s+", "", math_text.lower())
            if compact in ("real_output", "im(f)=0", "im(f(x))=0") or (
                "im(" in compact and "=0" in compact
            ):
                real_output.append(label)
                continue
            if compact in ("int_1_inf_f_finite", "bounded_integral") or re.match(
                r"^int[_0-9a-z]*fin(?:ite)?$", compact
            ):
                integrable.append(label)
                continue
            # Parse in specificity order: rate, limit/pin, envelope bound,
            # scalar bound, closed form.
            claim = (
                cls._parse_rate(label, math_text, declared)
                or cls._parse_limit(label, math_text, declared)
                or cls._parse_bound(label, math_text, declared)
                or cls._parse_scalar(label, math_text, declared)
                or cls._parse_form(label, math_text, declared)
            )
            if isinstance(claim, ContradictoryConstraintEngine._RateClaim):
                rates.append(claim)
            elif isinstance(claim, ContradictoryConstraintEngine._LimitClaim):
                limits.append(claim)
            elif isinstance(claim, ContradictoryConstraintEngine._BoundClaim):
                bounds.append(claim)
            elif isinstance(claim, ContradictoryConstraintEngine._ScalarClaim):
                scalars.append(claim)
            elif isinstance(claim, ContradictoryConstraintEngine._FormClaim):
                forms.append(claim)

        # Pass 2: pairwise symbolic derivations. First derivation wins.
        for bound in bounds:
            for lim in limits:
                cert = cls._bound_vs_limit(bound, lim)
                if cert is not None:
                    return cert
        for i, b1 in enumerate(bounds):
            for b2 in bounds[i + 1:]:
                if b1.is_lower == b2.is_lower:
                    continue
                lower, upper = (b1, b2) if b1.is_lower else (b2, b1)
                cert = cls._envelopes_disjoint(
                    lower.label, lower.envelope, upper.label, upper.envelope,
                    subject=f"{lower.fn}({lower.var})",
                )
                if cert is not None:
                    return cert
        for i, r1 in enumerate(rates):
            for r2 in rates[i + 1:]:
                cert = cls._rate_vs_rate(r1, r2)
                if cert is not None:
                    return cert
        for i, s1 in enumerate(scalars):
            for s2 in scalars[i + 1:]:
                cert = cls._scalar_vs_scalar(s1, s2)
                if cert is not None:
                    return cert
        for form in forms:
            for label in real_output:
                cert = cls._form_vs_real_output(form, label)
                if cert is not None:
                    return cert
            for label in integrable:
                cert = cls._form_vs_integrable(form, label)
                if cert is not None:
                    return cert
        return None

    check_infeasibility = check_incompatibility
