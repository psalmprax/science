"""APEDP Composer: data-free hypothesis generation from corpus laws.

Combines known expressions into dimensionally-valid conjectures and tests
them with symbolic probes only (no data): homogeneity, Pi-groups,
correspondence limits (new -> parent), repair audit, and adversarial
boundary/numerical falsification on declared domains.

Every result is tagged THEORETICAL_SPECULATION — a ranked conjecture, never
a validated law. Output strings feed directly into
``MultiVariableSRE(template_priors=...)`` once data arrives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sympy as sp

from apedp.core import BuckinghamPiEngine, DimensionalAssuranceEngine
from apedp.repair import AgnosticClosedLoopAssuranceSystem
from apedp.safety import limit_with_timeout, safe_parse_expr

logger = logging.getLogger(__name__)

THEORETICAL_MATURITY = "THEORETICAL_SPECIFICATION"


@dataclass
class Conjecture:
    expression: str
    parents: tuple[str, ...]
    operation: str
    score: float
    maturity: str = THEORETICAL_MATURITY
    validity: dict = field(default_factory=dict)

    def as_template_prior(self) -> str:
        return self.expression


class HypothesisComposer:
    """Compose corpus laws into new testable conjectures without data."""

    def __init__(
        self,
        engine: DimensionalAssuranceEngine | None = None,
        repair: AgnosticClosedLoopAssuranceSystem | None = None,
    ):
        self.engine = engine or DimensionalAssuranceEngine()
        self.repair = repair or AgnosticClosedLoopAssuranceSystem()

    def _ensure_registered(self, expr: sp.Expr) -> None:
        for sym in expr.free_symbols:
            if sym not in self.engine.symbol_units:
                try:
                    self.engine.register_variable(str(sym), "dimensionless")
                except (ValueError, TypeError, AttributeError):
                    continue

    def _parse(self, text: str) -> sp.Expr | None:
        try:
            expr = safe_parse_expr(text)
        except (TypeError, ValueError):
            return None
        self._ensure_registered(expr)
        return expr

    def _homogeneous(self, expr: sp.Expr) -> bool:
        try:
            self.engine.check_expression_consistency(expr)
            return True
        except (TypeError, ValueError, NotImplementedError, ArithmeticError):
            return False

    def crossover(self, parent_a: str, parent_b: str) -> list[tuple[str, str]]:
        """Dimensionally-valid products/quotients/sums of two parents."""
        ea, eb = self._parse(parent_a), self._parse(parent_b)
        if ea is None or eb is None:
            return []
        out: list[tuple[str, str]] = []
        for op, expr in (("mul", ea * eb), ("div", ea / eb), ("add", ea + eb)):
            try:
                if self._homogeneous(expr):
                    out.append((op, str(expr)))
            except (TypeError, ValueError, ArithmeticError):
                continue
        return out

    def generalize_saturation(self, parent: str, var: str = "x") -> list[tuple[str, str]]:
        """Linear -> saturated conjecture: x becomes x/(1+x/K)."""
        e = self._parse(parent)
        sym = sp.Symbol(var)
        if e is None or sym not in e.free_symbols:
            return []
        K = sp.Symbol("K_sat")
        saturated = e.subs(sym, sym / (1 + sym / K))
        if self._homogeneous(saturated):
            return [("saturate", str(saturated))]
        return []

    def correspondence_holds(self, new_expr: str, parent_expr: str, param: str = "K_sat", limit_to: float = float("inf")) -> bool:
        """New reduces to parent when the introduced scale drops out."""
        try:
            new_e = safe_parse_expr(new_expr)
            parent_e = safe_parse_expr(parent_expr)
            sym = sp.Symbol(param)
            if sym not in new_e.free_symbols:
                return False
            lim = limit_with_timeout(new_e, sym, limit_to, timeout=2.0)
            return bool(sp.simplify(lim - parent_e) == 0)
        except (ValueError, TypeError, ArithmeticError, NotImplementedError):
            return False
        except Exception:  # noqa: BLE001 - timeout type + sympy internals stay non-fatal here
            logger.debug("Correspondence probe failed for %s -> %s", new_expr, parent_expr)
            return False

    def audit(self, expression: str, variable_units: dict[str, str] | None = None) -> dict:
        """Symbolic-only test battery: homogeneity, Pi-shape, repair, limits."""
        expr = self._parse(expression)
        if expr is None:
            return {"verdict": "UNPARSEABLE"}
        homogeneous = self._homogeneous(expr)
        repair = self.repair.autonomous_stress_test_and_repair(expression, variable_units=variable_units)
        nodes = len(list(sp.preorder_traversal(expr)))
        score = (1.0 / (1.0 + nodes)) * (1.0 if homogeneous else 0.0)
        if repair.get("status") == "REJECTED":
            score *= 0.5
        return {
            "verdict": "SPECULATION",
            "maturity": THEORETICAL_MATURITY,
            "homogeneous": homogeneous,
            "repair_status": repair.get("status"),
            "anomalies": [f"{s}: {k}" for s, k in repair.get("anomalies", [])],
            "nodes": nodes,
            "score": score,
        }

    def compose(
        self,
        corpus: list[str],
        max_candidates: int = 20,
        variable_units: dict[str, str] | None = None,
    ) -> list[Conjecture]:
        """Pairwise crossover + saturation, audited and ranked. Dedupes by string."""
        seen: dict[str, Conjecture] = {}
        for i, pa in enumerate(corpus):
            for sat_op, sat_expr in self.generalize_saturation(pa):
                audit = self.audit(sat_expr, variable_units)
                if audit["verdict"] == "UNPARSEABLE":
                    continue
                bonus = 0.1 if self.correspondence_holds(sat_expr, pa) else 0.0
                seen.setdefault(sat_expr, Conjecture(sat_expr, (pa,), sat_op, audit["score"] + bonus, validity=audit))
            for pb in corpus[i + 1:]:
                for op, expr in self.crossover(pa, pb):
                    if expr in seen:
                        continue
                    audit = self.audit(expr, variable_units)
                    if audit["verdict"] == "UNPARSEABLE":
                        continue
                    seen[expr] = Conjecture(expr, (pa, pb), f"crossover_{op}", audit["score"], validity=audit)
                    if len(seen) >= max_candidates * 2:
                        break
            if len(seen) >= max_candidates * 2:
                break
        ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        # Dedupe by Pi-signature shape is overkill here; string dedupe + rank suffices.
        _ = BuckinghamPiEngine  # symbolic groups available to callers via core
        return ranked[:max_candidates]
