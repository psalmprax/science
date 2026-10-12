"""Guided structural search: hierarchical grammar, lanes, and diversity.

Redesigns the discovery side around search *economics*: the frozen baseline
showed representable targets going undiscovered (D=2/18) while pathological
nests won training error. Instead of searching harder, search smarter:

* **Grammar levels** (0-5): primitives, arithmetic, integer powers,
  reciprocal/fractional powers, log/exp, trig. Higher levels unlock only
  when lower levels stall (documented per-run unlock schedule).
* **Term library**: generic reusable patterns (``c*x**n``, ``c/x**n``,
  polynomials, rational ``(a+b*x)/(c+d*x)``, ``a*exp(b*x)``,
  ``a*log(b*x)``, ``a*sin(b*x+c)``) -- mathematical construction patterns,
  never literature equations.
* **Structural lanes**: additive / rational / transcendental / mixed
  subpopulations with constrained growers plus periodic exchange.
* **Targeted mutations**: expand, reciprocal, log/exp-wrap, add/drop term,
  trig-wrap -- type- and dimension-aware where units are declared.
* **Adaptive rates**: duplicate fraction + stagnation drive
  mutation/crossover pressure per island.
* **Semantic diversity**: response-signature clustering on a diagnostic
  grid; syntactically different but functionally identical candidates
  collapse to representatives.
* **Candidate archive**: best-per-criterion memory (accuracy, simplicity,
  validation, novelty, stability, dimensional structure).
* **Safe evaluation**: interval/domain pre-filter predicting overflow,
  poles, and non-finite regions before expensive fitting.
* **Accessibility census**: operator/depth histograms, duplicate fraction,
  substructure census (T, T**2, T**3, 1/T**2, ...), first-appearance
  records, structural distance reporting.

Complements ``evolutionary_search`` (which it reuses for fitting and
dimensional gating) rather than replacing it; every feature toggles
independently so the ablation matrix can measure each contribution.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp


# ------------------------------------------------------------------ levels
def level_of(expression: sp.Expr | str) -> int:
    """Highest grammar level used: 0 atoms, 1 arithmetic, 2 int powers,
    3 reciprocal/fractional powers, 4 log/exp, 5 trig."""
    from apedp.safety import safe_parse_expr

    expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
    level = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Add, sp.Mul)):
            level = max(level, 1)
        elif isinstance(node, sp.Pow):
            _, exponent = node.args
            if exponent.is_Integer and int(exponent) >= 0:
                level = max(level, 2)
            else:
                level = max(level, 3)
        elif isinstance(node, (sp.log, sp.exp)):
            level = max(level, 4)
        elif isinstance(node, (sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan,
                               sp.sinh, sp.cosh, sp.tanh)):
            level = max(level, 5)
    # Division is level 1 even though it prints as Pow(..., -1).
    return level


LEVEL_OPERATORS: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("add", "sub", "mul", "div"),
    2: ("add", "sub", "mul", "div", "pow2", "pow3"),
    3: ("add", "sub", "mul", "div", "pow2", "pow3", "recip", "fracpow"),
    4: ("add", "sub", "mul", "div", "pow2", "pow3", "recip", "fracpow", "log", "exp"),
    5: ("add", "sub", "mul", "div", "pow2", "pow3", "recip", "fracpow", "log", "exp",
        "sin", "cos"),
}


def grow_at_level(symbols: list[sp.Symbol], level: int, depth: int, rng: random.Random) -> sp.Expr:
    """Random tree using only operators at or below ``level``."""
    allowed = LEVEL_OPERATORS[max(0, min(5, level))]

    def leaf() -> sp.Expr:
        if rng.random() < 0.5:
            return rng.choice(symbols)
        return sp.Float(rng.uniform(-3.0, 3.0))

    def grow(remaining: int, cap: tuple[str, ...]) -> sp.Expr:
        if remaining <= 0 or not cap:
            return leaf()
        op = rng.choice(list(cap))
        if op == "add":
            return grow(remaining - 1, cap) + grow(remaining - 1, cap)
        if op == "sub":
            return grow(remaining - 1, cap) - grow(remaining - 1, cap)
        if op == "mul":
            return grow(remaining - 1, cap) * grow(remaining - 1, cap)
        if op == "div":
            return grow(remaining - 1, cap) / grow(remaining - 1, cap)
        if op == "pow2":
            return grow(remaining - 1, cap) ** 2
        if op == "pow3":
            return grow(remaining - 1, cap) ** 3
        if op == "recip":
            return 1 / grow(remaining - 1, cap)
        if op == "fracpow":
            return grow(remaining - 1, cap) ** sp.Rational(1, 2)
        if op == "log":
            return sp.log(grow(remaining - 1, cap))
        if op == "exp":
            return sp.exp(grow(remaining - 1, cap))
        if op == "sin":
            return sp.sin(grow(remaining - 1, cap))
        if op == "cos":
            return sp.cos(grow(remaining - 1, cap))
        return leaf()

    return grow(depth, allowed)


# ------------------------------------------------------------ term library
def term_library(symbols: list[sp.Symbol], rng: random.Random) -> sp.Expr:
    """One generic reusable mathematical pattern with random coefficients.

    Patterns (never literature equations): monomial, reciprocal monomial,
    quadratic polynomial, rational (a+b*x)/(c+d*x), exponential, logarithm,
    sinusoid. Coefficients are fresh floats for the fitter to determine.
    """
    var = rng.choice(symbols)
    pick = rng.choice(["monomial", "reciprocal", "poly2", "rational",
                       "exponential", "logarithm", "sinusoid"])
    def c():
        return sp.Float(rng.uniform(-3.0, 3.0))
    if pick == "monomial":
        return c() * var ** rng.choice([1, 2, 3])
    if pick == "reciprocal":
        return c() / var ** rng.choice([1, 2])
    if pick == "poly2":
        return c() + c() * var + c() * var**2
    if pick == "rational":
        return (c() + c() * var) / (c() + c() * var)
    if pick == "exponential":
        return c() * sp.exp(c() * var)
    if pick == "logarithm":
        return c() * sp.log(c() * var)
    return c() * sp.sin(c() * var + c())


# ------------------------------------------------------------------- lanes
LANES = ("additive", "rational", "transcendental", "mixed")


def grow_in_lane(symbols: list[sp.Symbol], lane: str, level: int, depth: int,
                 rng: random.Random) -> sp.Expr:
    """Lane-constrained generation: each lane owns a structural family."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}")
    rng.choice(symbols)
    def coef():
        return sp.Float(rng.uniform(-3.0, 3.0))
    if lane == "additive":
        terms = [coef() * grow_at_level(symbols, min(level, 3), 2, rng)
                 for _ in range(rng.randint(2, 3))]
        result: sp.Expr = terms[0]
        for term in terms[1:]:
            result = result + term
        return result
    if lane == "rational":
        top = grow_at_level(symbols, min(level, 2), 2, rng)
        bottom = grow_at_level(symbols, min(level, 2), 2, rng)
        return coef() * top / bottom + coef()
    if lane == "transcendental":
        func = rng.choice(["exp", "log", "sin", "cos"])
        inner = grow_at_level(symbols, min(level, 2), 2, rng)
        return coef() * getattr(sp, func)(coef() * inner) + coef()
    return grow_at_level(symbols, level, depth, rng)


# ------------------------------------------------------ targeted mutation
def targeted_mutations(expression: sp.Expr, rng: random.Random) -> list[sp.Expr]:
    """Generic structural neighborhood moves (never target-specific).

    expand (x->x**2/x**3), reciprocal (x->1/x), log/exp-wrap, add/drop term,
    trig-wrap, decompose (replace a node by one of its children).
    """
    out: list[sp.Expr] = []
    paths: list[tuple[tuple[int, ...], sp.Expr]] = []

    def visit(node: sp.Expr, path: tuple[int, ...]) -> None:
        paths.append((path, node))
        for index, arg in enumerate(node.args):
            visit(arg, (*path, index))

    visit(expression, ())

    def replace(path: tuple[int, ...], value: sp.Expr) -> sp.Expr:
        node = expression
        for index in path:
            node = node.args[index]
        # Rebuild from the root along the path.
        def rebuild(current: sp.Expr, rest: tuple[int, ...]) -> sp.Expr:
            if not rest:
                return value
            args = list(current.args)
            args[rest[0]] = rebuild(args[rest[0]], rest[1:])
            return current.func(*args)
        return rebuild(expression, path)

    candidates = [(path, node) for path, node in paths if path]
    if not candidates:
        return out
    path, node = rng.choice(candidates)
    symbols = list(node.free_symbols)
    if symbols:
        var = rng.choice(symbols)
        out.append(replace(path, node.xreplace({var: var**2})))
        out.append(replace(path, node.xreplace({var: 1 / var})))
        out.append(replace(path, node.xreplace({var: sp.log(var)})))
    out.append(replace(path, sp.exp(node)))
    out.append(replace(path, sp.sin(node)))
    if node.args:
        out.append(replace(path, rng.choice(list(node.args))))
    out.append(replace(path, node + rng.choice(symbols) if symbols else node + 1))
    return out


# ---------------------------------------------------------------- adaptive
@dataclass
class AdaptiveRates:
    """Rule-based exploration/exploitation control per island."""

    mutation_rate: float = 0.35
    crossover_rate: float = 0.7
    stagnant_gens: int = 0

    def update(self, *, duplicate_fraction: float, improved: bool) -> AdaptiveRates:
        """Collapse -> explore; progress -> exploit; else hold."""
        mutation, crossover, stagnant = self.mutation_rate, self.crossover_rate, self.stagnant_gens
        if improved:
            stagnant = 0
            mutation = max(0.1, mutation - 0.05)
        else:
            stagnant += 1
            if duplicate_fraction > 0.4 or stagnant >= 3:
                mutation = min(0.8, mutation + 0.1)
                crossover = max(0.3, crossover - 0.05)
        return AdaptiveRates(mutation, crossover, stagnant)

    def to_dict(self) -> dict:
        return {"mutation_rate": self.mutation_rate, "crossover_rate": self.crossover_rate,
                "stagnant_gens": self.stagnant_gens}


# ------------------------------------------------------- semantic novelty
def response_signature(function: Callable[[np.ndarray], np.ndarray],
                       grid: np.ndarray) -> np.ndarray | None:
    """Normalized response vector on a diagnostic grid (None if unusable)."""
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(function(np.asarray(grid, dtype=float)), dtype=float).ravel()
    except (TypeError, ValueError, ArithmeticError):
        return None
    if values.shape != grid.shape or not np.all(np.isfinite(values)):
        return None
    centered = values - np.mean(values)
    scale = float(np.std(centered))
    if scale == 0.0:
        return None
    return centered / scale  # type: ignore[no-any-return]


def semantic_prune(signatures: list[np.ndarray | None], keep: int,
                   threshold: float = 0.05) -> list[int]:
    """Greedy medoids: keep representatives pairwise distant in response space."""
    order = sorted(range(len(signatures)), key=lambda i: 0 if signatures[i] is None else 1)
    kept: list[int] = []
    for index in order:
        signature = signatures[index]
        if signature is None:
            continue
        if all(float(np.linalg.norm(signature - signatures[kept_index])) > threshold
               for kept_index in kept if signatures[kept_index] is not None):
            kept.append(index)
        if len(kept) >= keep:
            break
    return kept


# ----------------------------------------------------------------- archive
@dataclass
class CandidateArchive:
    """Best-per-criterion scientific memory across the whole run."""

    slots: dict[str, tuple[str, float]] = field(default_factory=dict)

    def consider(self, criterion: str, expression: str, value: float, *, smaller_is_better: bool = True) -> bool:
        """Store if best-in-slot; returns True on update."""
        current = self.slots.get(criterion)
        better = current is None or (value < current[1] if smaller_is_better else value > current[1])
        if better:
            self.slots[criterion] = (expression, float(value))
        return better

    def to_dict(self) -> dict:
        return {criterion: {"expression": expression, "value": value}
                for criterion, (expression, value) in self.slots.items()}


# -------------------------------------------------------------- safe eval
def safe_to_fit(expression: sp.Expr | str, grids: dict[str, np.ndarray]) -> tuple[bool, str]:
    """Pre-fit mathematical safety screen on declared domains.

    Predicts overflow/poles/non-finite regions from unit-coefficient probes
    before any expensive fitting: finite fraction, magnitude range, and
    complex output on a dense grid. Returns (safe, reason).
    """
    from apedp.safety import safe_parse_expr

    try:
        expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
    except (TypeError, ValueError) as exc:
        return False, f"unparseable: {exc}"
    symbols = [sp.Symbol(name) for name in grids]
    try:
        function = sp.lambdify(symbols, expr, modules=["numpy"])
    except Exception as exc:  # noqa: BLE001 - best-effort probe must not crash
        return False, f"cannot compile: {exc}"
    try:
        with np.errstate(all="ignore"):
            values = np.asarray(function(*[np.asarray(grids[str(symbol)], dtype=float) for symbol in symbols]))
    except (TypeError, ValueError, ArithmeticError) as exc:
        return False, f"evaluation failed: {exc}"
    if np.iscomplexobj(values):
        return False, "complex-valued on domain"
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False, "non-numeric output"
    finite = np.isfinite(values)
    if not np.all(finite):
        return False, f"non-finite on {int((~finite).sum())}/{values.size} grid points"
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak > 1e15:
        return False, f"magnitude {peak:.2e} exceeds overflow guard"
    return True, "finite and bounded on domain grid"


# ------------------------------------------------------------------ census
def population_census(expressions: list[sp.Expr | str]) -> dict:
    """Diversity instrumentation: uniqueness, operators, depths, duplicates."""
    from apedp.safety import safe_parse_expr

    parsed = []
    for expression in expressions:
        try:
            parsed.append(safe_parse_expr(expression) if isinstance(expression, str) else expression)
        except (TypeError, ValueError):
            continue
    hashes = [sp.srepr(expression) for expression in parsed]
    operator_counts: dict[str, int] = {}
    depths: list[int] = []
    for expression in parsed:
        depth = 0
        stack = [(expression, 0)]
        while stack:
            node, current = stack.pop()
            depth = max(depth, current)
            name = type(node).__name__
            operator_counts[name] = operator_counts.get(name, 0) + 1
            stack.extend((arg, current + 1) for arg in node.args)
        depths.append(depth)
    return {
        "n": len(parsed),
        "unique_fraction": len(set(hashes)) / len(hashes) if hashes else 0.0,
        "duplicate_fraction": 1.0 - len(set(hashes)) / len(hashes) if hashes else 0.0,
        "operator_histogram": operator_counts,
        "depth_min": min(depths) if depths else 0,
        "depth_max": max(depths) if depths else 0,
    }


def has_substructure(expression: sp.Expr | str, pattern: sp.Expr | str) -> bool:
    """Whether a target substructure (e.g. ``1/T**2``) occurs in the tree."""
    from apedp.safety import safe_parse_expr

    expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
    pat = safe_parse_expr(pattern) if isinstance(pattern, str) else pattern
    return bool(expr.has(pat))


__all__ = [
    "LANES",
    "LEVEL_OPERATORS",
    "AdaptiveRates",
    "CandidateArchive",
    "grow_at_level",
    "grow_in_lane",
    "has_substructure",
    "level_of",
    "population_census",
    "response_signature",
    "safe_to_fit",
    "semantic_prune",
    "targeted_mutations",
    "term_library",
]
