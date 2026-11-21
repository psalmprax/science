"""Counterfeit laws: a different form that fits the same rows almost as well as the true law.

Status: experimental

A law here is a *recipe*: building blocks in one input each (the input itself, a
power, an exponential decay, a logarithm, a sine, a saturating ratio, a square
root, a tanh), joined left to right by sums, differences, products and
quotients. It is the grammar of ``scripts/build_invented_laws.py`` written so
that a law remembers how it was put together.

A counterfeit is the same recipe with one block exchanged for another kind (or
one summed block left out), and every constant then fitted to the true law
itself, without noise, over the whole domain. So it is the closest law of that
other form: ``distance`` is what separates the two at best, as a share of the
true law's spread. A search that returns approximations returns laws like
these; a referee that reports one whose distance is beyond the accuracy it
claims has reported a wrong law.

What this does not make: counterfeits that differ from the law only outside the
domain, or that use an input the law does not. Both exist (the second whenever
inputs move together) and neither can be told from the law by rows drawn where
the law was sampled.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

KINDS = ("linear", "square", "cube", "inverse", "decay", "log", "sine", "ratio", "root", "tanh")
WITH_CONSTANT = frozenset({"decay", "log", "sine", "ratio", "root", "tanh"})   # kinds with a constant inside
JOINS = ("add", "subtract", "multiply", "divide")
STARTS = (0.3, 1.3, 3.7)        # where the constant inside an exchanged block starts its fit
FIT_ROWS, CHECK_ROWS = 600, 2000


@dataclass(frozen=True)
class Step:
    kind: str
    symbol: int
    inner: float = 1.0          #: the constant inside the block, for the kinds that have one
    join: str = "start"         #: how the block joins what came before it
    scale: float = 1.0          #: multiplies the first block or a summed one; is added to the block under a quotient


Recipe = tuple[Step, ...]


@dataclass(frozen=True)
class Counterfeit:
    text: str
    change: str                 #: what was exchanged, e.g. ``step 1: ratio -> tanh``
    distance: float             #: RMS difference from the law over the domain, as a share of the law's spread


def _block(step: Step, x: sp.Symbol, constant: Callable[[float], sp.Expr]) -> sp.Expr:
    if step.kind == "linear":
        return x
    if step.kind == "square":
        return x ** 2
    if step.kind == "cube":
        return x ** 3
    if step.kind == "inverse":
        return 1 / x
    a = constant(step.inner)
    if step.kind == "decay":
        return sp.exp(-a * x)
    if step.kind == "log":
        return sp.log(1 + a * x)
    if step.kind == "sine":
        return sp.sin(a * x)
    if step.kind == "ratio":
        return x / (a + x)
    if step.kind == "root":
        return sp.sqrt(a + x)
    if step.kind == "tanh":
        return sp.tanh(a * x)
    raise ValueError(f"unknown kind {step.kind!r}")


def _assemble(recipe: Recipe, inputs: int, constant: Callable[[float], sp.Expr]) -> sp.Expr:
    symbols = [sp.Symbol(f"x{i}") for i in range(inputs)]
    first = recipe[0]
    expr = constant(first.scale) * _block(first, symbols[first.symbol], constant)
    for step in recipe[1:]:
        if step.join in ("add", "subtract"):
            scale = constant(step.scale)
            term = scale * _block(step, symbols[step.symbol], constant)
            expr = expr + term if step.join == "add" else expr - term
        elif step.join == "multiply":
            expr = expr * _block(step, symbols[step.symbol], constant)
        elif step.join == "divide":
            expr = expr / (constant(step.scale) + _block(step, symbols[step.symbol], constant))
        else:
            raise ValueError(f"unknown join {step.join!r}")
    return expr


def law_of(recipe: Recipe, inputs: int) -> sp.Expr:
    """The recipe as an expression in ``x0 ..``."""
    return _assemble(recipe, inputs, lambda value: sp.Float(float(value)))


def parametrised(recipe: Recipe, inputs: int) -> tuple[sp.Expr, list[sp.Symbol], list[float]]:
    """The recipe with every constant a parameter: the expression, the parameters, and their values."""
    params: list[sp.Symbol] = []
    values: list[float] = []

    def constant(value: float) -> sp.Expr:
        params.append(sp.Symbol(f"_k{len(params)}"))
        values.append(float(value))
        return params[-1]

    return _assemble(recipe, inputs, constant), params, values


def draw_recipe(rng: np.random.Generator, inputs: int, constants: Sequence[float]) -> Recipe:
    """A recipe that uses every input, drawn as the invented-law grammar draws."""
    extra = int(rng.integers(0, 2))
    order = [*range(inputs), *(int(rng.integers(0, inputs)) for _ in range(extra))]
    rng.shuffle(order)

    def step(symbol: int, join: str) -> Step:
        return Step(str(rng.choice(KINDS)), int(symbol), float(rng.choice(constants)), join, float(rng.choice(constants)))

    return (step(order[0], "start"), *(step(symbol, str(rng.choice(JOINS))) for symbol in order[1:]))


def _values(expr: sp.Expr, inputs: int, X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(inputs)], expr, "numpy")
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=complex) * np.ones(len(X))
    real: np.ndarray = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
    return real


def rounded(value: float, digits: int) -> float:
    return float(f"{float(value):.{digits}g}")


def fitted_to(recipe: Recipe, inputs: int, X: np.ndarray, target: np.ndarray, digits: int = 4) -> sp.Expr | None:
    """The recipe with every constant fitted to ``target`` on the rows ``X``, printed to ``digits`` figures."""
    expr, params, values = parametrised(recipe, inputs)
    fn = sp.lambdify([*(sp.Symbol(f"x{i}") for i in range(inputs)), *params], expr, "numpy")
    spread = float(np.std(target)) or 1.0

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*X.T, *theta), dtype=complex) * np.ones(len(X))
        ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
        out: np.ndarray = np.where(ok, (value.real - target) / spread, 1e3)
        return out

    try:
        fit = least_squares(resid, np.array(values, dtype=float), max_nfev=300)
    except (ValueError, TypeError, FloatingPointError):
        return None
    if not np.all(np.isfinite(fit.x)):
        return None
    return expr.xreplace({p: sp.Float(rounded(v, digits)) for p, v in zip(params, fit.x, strict=True)})


def distance(expr: sp.Expr, inputs: int, X: np.ndarray, target: np.ndarray) -> float:
    """RMS difference between ``expr`` and ``target`` on ``X``, as a share of the target's spread."""
    values = _values(expr, inputs, X)
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.sqrt(np.mean((values - target) ** 2)) / max(float(np.std(target)), 1e-300))


def counterfeits(recipe: Recipe, inputs: int, low: float, high: float, seed: int = 0, digits: int = 4,
                 nearest: float = 1e-3, farthest: float = 0.3) -> list[Counterfeit]:
    """Every one-block exchange of ``recipe`` whose best fit to the law is between ``nearest`` and ``farthest``.

    Sorted by distance. A counterfeit nearer than ``nearest`` is the law to within what the
    benchmarks call exact; one farther than ``farthest`` is nothing a search would return.
    """
    rng = np.random.default_rng(seed)
    X_fit, X_check = rng.uniform(low, high, (FIT_ROWS, inputs)), rng.uniform(low, high, (CHECK_ROWS, inputs))
    law = law_of(recipe, inputs)
    fit_target, check_target = _values(law, inputs, X_fit), _values(law, inputs, X_check)
    if not (np.all(np.isfinite(fit_target)) and np.all(np.isfinite(check_target))):
        raise ValueError("the law is not finite on the domain")
    found: dict[str, Counterfeit] = {}

    def consider(changed: Recipe, change: str) -> None:
        expr = fitted_to(changed, inputs, X_fit, fit_target, digits)
        if expr is None or expr.free_symbols != law.free_symbols:
            return
        gap = distance(expr, inputs, X_check, check_target)
        text = str(expr)
        if nearest <= gap <= farthest and (text not in found or gap < found[text].distance):
            found[text] = Counterfeit(text, change, gap)

    for index, step in enumerate(recipe):
        for kind in KINDS:
            if kind == step.kind:
                continue
            for start in (STARTS if kind in WITH_CONSTANT else (step.inner,)):
                changed = (*recipe[:index], replace(step, kind=kind, inner=start), *recipe[index + 1:])
                consider(changed, f"step {index}: {step.kind} -> {kind}")
        if index > 0 and step.join in ("add", "subtract"):
            consider((*recipe[:index], *recipe[index + 1:]), f"step {index} ({step.kind}) left out")
    best: dict[str, Counterfeit] = {}
    for fake in found.values():          # one counterfeit per exchange: the best of its starts
        if fake.change not in best or fake.distance < best[fake.change].distance:
            best[fake.change] = fake
    return sorted(best.values(), key=lambda fake: (fake.distance, fake.text))
