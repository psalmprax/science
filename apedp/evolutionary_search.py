# Databricks notebook source
"""Upgraded evolutionary search: projection, tuning, islands, physics priors.

Combines six capabilities in one engine so each can be ablated separately:

1. **Variable projection**: numeric literals are lifted to parameter symbols;
   parameters entering linearly are solved by exact least squares while only
   genuinely nonlinear parameters are searched. Collapses the search
   dimension on exactly the grammars APEDP uses most.
2. **Constant tuning**: nonlinear parameters get seeded coordinate descent
   (deterministic; no scipy dependency) instead of random-constant mutation.
3. **Islands + migration + threads**: ring-migration island model; fitness
   evaluation optionally fanned out over a thread pool.
4. **Physics-informed generation**: dimensionless seeding, dimensional gating
   at birth (structural violations only -- free coefficients are never
   punished), and declared-symmetry rejection.
5. **Pareto hall-of-fame with grouped selection**: nondominated front on
   (train error, complexity); the final pick minimizes *holdout* error along
   the front instead of training loss.
6. **ALPS age layering**: islands double as age layers; newcomers enter at
   layer 0 and age upward, preserving exploration late in the run.

All randomness flows from one seeded generator: identical seeds give
identical runs (thread count may affect evaluation order only through
pre-assigned work items, never through shared global RNG state).
"""
from __future__ import annotations

import logging
import random
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger("apedp.evolutionary_search")


_TRANSCENDENTAL = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.sqrt)


@dataclass
class SearchConfig:
    """Tunable search configuration; every upgrade is a flag."""

    islands: int = 4
    island_size: int = 30
    generations: int = 25
    migration_interval: int = 5
    max_depth: int = 4
    max_complexity: int = 30
    seed: int = 0
    use_projection: bool = True
    tune_constants: bool = True
    tune_restarts: int = 2
    tune_iters: int = 40
    alps: bool = True
    threads: int = 0
    normalize: bool = True
    pi_seed: bool = True
    symmetries: dict[str, str] = field(default_factory=dict)
    coef_penalty: float = 0.5
    tournament_size: int = 4
    mutation_rate: float = 0.35
    allow_exp: bool = True
    allow_trig: bool = True
    allow_log: bool = True
    allow_sqrt: bool = True
    allow_inv: bool = False


@dataclass
class SearchResult:
    """Outcome of one evolutionary run."""

    best_expression: str | None
    grouped_pick: str | None
    pareto_front: list[dict[str, Any]]
    generations_run: int
    n_evaluations: int
    history: list[dict[str, Any]]
    variable_scaling: dict[str, list[float]] = field(default_factory=dict)
    denormalized_best: str | None = None
    denormalized_pick: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_expression": self.best_expression,
            "grouped_pick": self.grouped_pick,
            "pareto_front": self.pareto_front,
            "generations_run": self.generations_run,
            "n_evaluations": self.n_evaluations,
            "history": self.history,
            "variable_scaling": self.variable_scaling,
            "denormalized_best": self.denormalized_best,
            "denormalized_pick": self.denormalized_pick,
        }


def denormalize_expression(
    expression: sp.Expr | str,
    scaling: dict[str, list[float]],
) -> sp.Expr:
    """Map a standardized-space expression back to original units.

    Input symbols map ``x -> (x - mu)/sigma`` (exact algebra; fitted
    constants absorb the scale). A ``"__target__"`` entry additionally wraps
    the whole form as ``sigma_y * (...) + mu_y``. Denormalized constants are
    therefore *not* directly comparable to literature constants -- the
    scaling record makes the mapping auditable.
    """
    expr = safe_parse_expr(expression) if isinstance(expression, str) else expression
    substitutions = {}
    for name, (mu, sigma) in scaling.items():
        if name == "__target__":
            continue
        if sigma != 0.0:
            substitutions[sp.Symbol(name)] = (sp.Symbol(name) - mu) / sigma
    result = sp.expand(expr.xreplace(substitutions))
    if "__target__" in scaling:
        target_mu, target_sigma = scaling["__target__"]
        result = sp.expand(target_sigma * result + target_mu)
    return result


def skeletonize(expression: sp.Expr) -> tuple[sp.Expr, list[sp.Symbol]]:
    """Lift every numeric literal to a fresh parameter symbol ``p_i``."""
    symbols: list[sp.Symbol] = []
    counter = [0]

    def replace(node: sp.Expr) -> sp.Expr:
        # Only Float literals are lifted: Integer/Rational exponents are
        # structural (x**2 must keep its 2), not fittable parameters.
        if isinstance(node, sp.Float):
            name = f"p_{counter[0]}"
            counter[0] += 1
            symbol = sp.Symbol(name)
            symbols.append(symbol)
            return symbol
        if node.args:
            rebuilt = node.func(*[replace(arg) for arg in node.args])
            return rebuilt
        return node

    return replace(expression), symbols


def partition_parameters(skeleton: sp.Expr, params: list[sp.Symbol]) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    """Split parameters into nonlinear / linear by structural test.

    A parameter is nonlinear when it occurs inside a transcendental (or any
    non-polynomial unary) or when the skeleton has total polynomial degree
    above 1 in the parameter set (products of parameters). Everything else
    is solved exactly by least squares.
    """
    inside: set[sp.Symbol] = set()

    def visit(node: sp.Expr, under_transcendental: bool) -> None:
        is_trans = isinstance(node, sp.Function) and not isinstance(node, (sp.Abs, sp.Min, sp.Max))
        for arg in node.args:
            if isinstance(arg, sp.Symbol) and arg in params and (under_transcendental or is_trans):
                inside.add(arg)
            visit(arg, under_transcendental or is_trans)

    visit(skeleton, False)
    try:
        poly_degree = sp.Poly(skeleton, params).total_degree() if params else 0
    except Exception as e:  # noqa: BLE001 - see rationale below; SymPy's failure modes here are version-dependent
        # Not polynomial in the parameters (transcendentals present):
        # only the transcendental-contained parameters are nonlinear.
        # SymPy raises a wide and version-dependent set here (PolynomialError,
        # GeneratorsNeeded, and its own internal types), so the broad catch is
        # deliberate; it is logged rather than swallowed so a genuine failure
        # in building the Poly is still visible.
        logger.debug("Polynomial degree probe failed for %s: %s", skeleton, e)
        return sorted(inside, key=str), [p for p in params if p not in inside]
    if poly_degree <= 1:
        return sorted(inside, key=str), [p for p in params if p not in inside]
    return list(params), []


def _lambdify_cached(cache: dict[str, Callable], expression: sp.Expr, symbols: list[sp.Symbol]) -> Callable:
    key = sp.srepr(expression)
    cached = cache.get(key)
    if cached is None:
        cached = sp.lambdify(symbols, expression, modules=["numpy"])
        cache[key] = cached
    return cached


def _broken_after_substitution(expression: sp.Expr) -> bool:
    """Detect infinities created by parameter substitution (e.g. ``c/0``).

    The birth gate cannot see these: they arise only once fitted values
    (often exactly 0.0 from least squares) replace the parameter symbols.
    """
    if not isinstance(expression, sp.Basic):
        return False
    return bool(expression.has(sp.zoo, sp.nan, sp.oo))


def _real_or_none(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """Convert a lambdified prediction to a real finite array, else None.

    Complex output (fractional powers of negative intermediates, log of
    negatives) marks the candidate broken *at these parameter values*;
    callers translate that into infinite error so the search moves on.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if np.iscomplexobj(array):
        return None
    try:
        array = np.asarray(array, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if array.shape != shape:
        try:
            array = np.broadcast_to(array, shape).copy()
        except ValueError:
            return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def evaluate_skeleton(
    skeleton: sp.Expr,
    params: list[sp.Symbol],
    nonlinear_values: dict[sp.Symbol, float],
    data_symbols: list[sp.Symbol],
    arrays: dict[str, np.ndarray],
    target: np.ndarray,
    cache: dict,
    *,
    use_projection: bool = True,
    cond_limit: float = 1.0e14,
    coef_limit: float = 1.0e12,
    dense_check: bool = False,
) -> tuple[float, dict[sp.Symbol, float]]:
    """Mean squared error with exact projection of the linear subset.

    Returns ``(mse, full_parameter_map)``. Non-finite predictions yield
    ``inf`` so broken candidates (division by zero, overflow) die quietly.
    Pathology guards (all documented, all fail-closed to ``inf``):

    * design condition number above ``cond_limit`` (near-singular basis);
    * fitted coefficient magnitude above ``coef_limit``;
    * ``dense_check``: for single-variable data, finiteness on a 4x denser
      grid over the training interval (catches poles between observations).
    """
    _nonlinear, linear = partition_parameters(skeleton, params) if use_projection else (params, [])
    fixed = dict(nonlinear_values)
    for name in linear:
        fixed.setdefault(name, 0.0)
    design_cols = []
    for name in linear:
        probe = dict(fixed)
        probe.update(dict.fromkeys(linear, 0.0))
        probe[name] = 1.0
        substituted = skeleton.xreplace(probe)
        if _broken_after_substitution(substituted):
            return float("inf"), fixed
        function = _lambdify_cached(cache, substituted, data_symbols)
        with np.errstate(all="ignore"):
            column = _real_or_none(function(*[arrays[str(symbol)] for symbol in data_symbols]), target.shape)
        if column is None:
            return float("inf"), fixed
        design_cols.append(column)
    if linear:
        design = np.column_stack(design_cols)
        try:
            with np.errstate(all="ignore"):
                condition = float(np.linalg.cond(design))
        except np.linalg.LinAlgError:
            return float("inf"), fixed
        if not np.isfinite(condition) or condition > cond_limit:
            return float("inf"), fixed
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        if not np.all(np.isfinite(coef)) or bool(np.any(np.abs(coef) > coef_limit)):
            return float("inf"), fixed
        for name, value in zip(linear, coef, strict=False):
            fixed[name] = float(value)
    substituted = skeleton.xreplace(fixed)
    if _broken_after_substitution(substituted):
        return float("inf"), fixed
    function = _lambdify_cached(cache, substituted, data_symbols)
    with np.errstate(all="ignore"):
        prediction = _real_or_none(function(*[arrays[str(symbol)] for symbol in data_symbols]), target.shape)
    if prediction is None:
        return float("inf"), fixed
    if dense_check and len(data_symbols) == 1:
        name = str(data_symbols[0])
        lo, hi = float(np.min(arrays[name])), float(np.max(arrays[name]))
        dense = np.linspace(lo, hi, 4 * len(target))
        with np.errstate(all="ignore"):
            dense_pred = _real_or_none(function(np.asarray(dense)), (len(dense),))
        if dense_pred is None:
            return float("inf"), fixed
    return float(np.mean((prediction - target) ** 2)), fixed


def tune_nonlinear(
    skeleton: sp.Expr,
    params: list[sp.Symbol],
    data_symbols: list[sp.Symbol],
    arrays: dict[str, np.ndarray],
    target: np.ndarray,
    cache: dict,
    *,
    restarts: int = 2,
    iters: int = 40,
    seed: int = 0,
    use_projection: bool = True,
) -> tuple[dict[sp.Symbol, float], float]:
    """Seeded coordinate descent over nonlinear parameters (deterministic).

    Falls back to the all-zero start when there is nothing nonlinear, which
    reduces exactly to projected least squares.
    """
    nonlinear, _ = partition_parameters(skeleton, params) if use_projection else (params, [])
    rng = random.Random(seed)
    best_map = dict.fromkeys(params, 0.0)
    best_mse, best_map = evaluate_skeleton(skeleton, params, best_map, data_symbols, arrays, target, cache, use_projection=use_projection)
    # Deterministic start grid: zeros, one-hot unit probes per nonlinear
    # parameter (both signs), then seeded random restarts. The grid matters:
    # pure random starts can miss narrow basins that unit probes bracket.
    starts = [dict.fromkeys(params, 0.0)]
    for name in nonlinear:
        for sign in (1.0, -1.0):
            probe_start = dict.fromkeys(params, 0.0)
            probe_start[name] = sign
            starts.append(probe_start)
    for _ in range(max(restarts - 1, 0)):
        starts.append({name: rng.uniform(-2.0, 2.0) for name in nonlinear} | {name: 0.0 for name in params if name not in nonlinear})
    for start in starts:
        current = dict(start)
        current_mse, current = evaluate_skeleton(skeleton, params, current, data_symbols, arrays, target, cache, use_projection=use_projection)
        step = 0.5
        for _ in range(iters):
            improved = False
            for name in nonlinear:
                for delta in (step, -step):
                    trial = dict(current)
                    trial[name] = trial[name] + delta
                    trial_mse, trial = evaluate_skeleton(skeleton, params, trial, data_symbols, arrays, target, cache, use_projection=use_projection)
                    if trial_mse < current_mse:
                        current, current_mse = trial, trial_mse
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-9:
                    break
        if current_mse < best_mse:
            best_mse, best_map = current_mse, current
    return best_map, best_mse


def _stable_seed(expression: sp.Expr) -> int:
    """Deterministic per-expression seed (``hash()`` is process-salted)."""
    return zlib.crc32(sp.srepr(expression).encode()) % (2**31)


def node_count(expression: sp.Expr) -> int:
    """Complexity: tree nodes plus nothing else; coefficients priced separately."""
    return sum(1 for _ in sp.preorder_traversal(expression))


def _complexity(expression: sp.Expr, params: list[sp.Symbol], coef_penalty: float) -> float:
    concrete = expression.xreplace(dict.fromkeys(params, 1.0))
    return float(node_count(concrete) + coef_penalty * len(params))


class _IslandSearch:
    """Single evolutionary run; the outer driver handles islands and ALPS."""

    def __init__(
        self,
        data_symbols: list[sp.Symbol],
        arrays: dict[str, np.ndarray],
        target: np.ndarray,
        variable_units: dict[str, str] | None,
        target_unit: str | None,
        symmetries: dict[str, str],
        config: SearchConfig,
        rng: random.Random,
        cache: dict,
        y_scale: float = 1.0,
    ) -> None:
        self.data_symbols = data_symbols
        self.arrays = arrays
        self.target = target
        self.variable_units = variable_units or {}
        self.target_unit = target_unit
        self.symmetries = symmetries
        self.config = config
        self.rng = rng
        self.cache = cache
        self.y_scale = y_scale
        self.unary = (
            (["exp"] if config.allow_exp else [])
            + (["sin", "cos"] if config.allow_trig else [])
            + (["log"] if config.allow_log else [])
            + (["sqrt"] if config.allow_sqrt else [])
            + (["inv"] if config.allow_inv else [])
        )

    # -- generation ------------------------------------------------------
    def _random_leaf(self) -> sp.Expr:
        if self.rng.random() < 0.5:
            return self.rng.choice(self.data_symbols)
        return sp.Float(self.rng.uniform(-3.0, 3.0))

    def _grow(self, depth: int) -> sp.Expr:
        if depth <= 0:
            return self._random_leaf()
        choice = self.rng.choice(["add", "sub", "mul", "div", "pow"] + (["unary"] if self.unary else []))
        if choice == "add":
            return self._grow(depth - 1) + self._grow(depth - 1)
        if choice == "sub":
            return self._grow(depth - 1) - self._grow(depth - 1)
        if choice == "mul":
            return self._grow(depth - 1) * self._grow(depth - 1)
        if choice == "div":
            return self._grow(depth - 1) / self._grow(depth - 1)
        if choice == "pow":
            base = self._grow(depth - 1)
            return base**self.rng.choice([2, 3, sp.Rational(1, 2)])
        func = self.rng.choice(self.unary)
        if func == "inv":
            return 1 / self._grow(depth - 1)
        return getattr(sp, func)(self._grow(depth - 1))

    def _admissible_at_birth(self, expression: sp.Expr) -> bool:
        # Eager sympy simplification can produce literal infinities
        # (e.g. 1/(x - x) -> zoo); these are never viable candidates.
        if expression.has(sp.zoo, sp.nan, sp.oo):
            return False
        skeleton, _ = skeletonize(expression)
        if self.variable_units:
            from apedp.dimensional_layer import check_units

            symbols = {str(symbol): None for symbol in skeleton.free_symbols}
            declared = {name: self.variable_units[name] for name in symbols if name in self.variable_units}
            verdict = check_units(skeleton, declared, self.target_unit)
            if verdict.status == "FAIL":
                return False
        for symbol, transform in self.symmetries.items():
            if sp.Symbol(symbol) not in skeleton.free_symbols:
                continue
            from apedp.scientific_layers import symmetry_check

            if symmetry_check(str(skeleton), symbol, transform) == "FAIL":
                return False
        return node_count(expression) <= self.config.max_complexity * 2

    def _pi_seeds(self) -> list[sp.Expr]:
        seeds: list[sp.Expr] = []
        names = [str(symbol) for symbol in self.data_symbols]
        units = self.variable_units
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                if units and units.get(first) == units.get(second):
                    first_symbol, second_symbol = sp.Symbol(first), sp.Symbol(second)
                    seeds.append(first_symbol / second_symbol)
                    seeds.append(first_symbol - second_symbol)
        if units:
            try:
                from apedp.scientific_layers import pi_groups

                groups = pi_groups({name: units[name] for name in names if name in units})
                for group in groups:
                    try:
                        candidate = safe_parse_expr(str(group))
                        if candidate.free_symbols <= set(self.data_symbols):
                            seeds.append(candidate)
                    except (TypeError, ValueError):
                        continue
            except (ImportError, TypeError, ValueError, AttributeError):
                pass
        return seeds

    # -- genetic operators -------------------------------------------------
    @staticmethod
    def _paths(expression: sp.Expr) -> list[tuple[int, ...]]:
        paths: list[tuple[int, ...]] = []

        def visit(node: sp.Expr, path: tuple[int, ...]) -> None:
            paths.append(path)
            for index, _ in enumerate(node.args):
                visit(node.args[index], (*path, index))

        visit(expression, ())
        return paths

    @staticmethod
    def _get(expression: sp.Expr, path: tuple[int, ...]) -> sp.Expr:
        node = expression
        for index in path:
            node = node.args[index]
        return node

    @staticmethod
    def _set(expression: sp.Expr, path: tuple[int, ...], value: sp.Expr) -> sp.Expr:
        if not path:
            return value
        args = list(expression.args)
        args[path[0]] = _IslandSearch._set(args[path[0]], path[1:], value)
        return expression.func(*args)

    def crossover(self, first: sp.Expr, second: sp.Expr) -> sp.Expr:
        paths_first = [path for path in self._paths(first) if path]
        paths_second = [path for path in self._paths(second) if path]
        if not paths_first or not paths_second:
            return first
        child = self._set(first, self.rng.choice(paths_first), self._get(second, self.rng.choice(paths_second)))
        return child

    def mutate(self, expression: sp.Expr) -> sp.Expr:
        roll = self.rng.random()
        if roll < 0.4:
            paths = [path for path in self._paths(expression) if path]
            if paths:
                return self._set(expression, self.rng.choice(paths), self._grow(2))
            return expression
        if roll < 0.7:
            floats = list(expression.atoms(sp.Float))
            if floats:
                target = self.rng.choice(floats)
                return expression.xreplace({target: sp.Float(float(target) * self.rng.uniform(0.5, 1.5))})
            return expression
        return self._grow(self.config.max_depth)

    # -- fitness ------------------------------------------------------------
    def tournament(self, pool: list[tuple[sp.Expr, float]]) -> sp.Expr:
        """Best of k random draws from a scored pool."""
        draws = [self.rng.choice(pool) for _ in range(min(self.config.tournament_size, len(pool)))]
        return min(draws, key=lambda pair: pair[1])[0]

    def fitness(self, expression: sp.Expr) -> tuple[float, float, dict[sp.Symbol, float]]:
        skeleton, params = skeletonize(expression)
        if self.config.tune_constants:
            param_map, mse = tune_nonlinear(
                skeleton, params, self.data_symbols, self.arrays, self.target, self.cache,
                restarts=self.config.tune_restarts, iters=self.config.tune_iters,
                seed=_stable_seed(expression), use_projection=self.config.use_projection,
            )
        else:
            mse, param_map = evaluate_skeleton(
                skeleton, params, dict.fromkeys(params, 0.0),
                self.data_symbols, self.arrays, self.target, self.cache,
                use_projection=self.config.use_projection,
            )
        complexity = _complexity(expression, params, self.config.coef_penalty)
        # Report physical-unit error: the search optimizes in normalized
        # space, but selection and reporting compare physical errors, so
        # results stay comparable across runs with different target scales.
        return mse * self.y_scale**2, complexity, param_map


def candidate_pathology(expression: sp.Expr | str) -> dict[str, object]:
    """Structural pathology metrics for one picked expression.

    Records depth, size, coefficient scale, nested powers, transcendental
    load, and oscillation proxies (derivative sign changes on [0,1] after
    affine rescaling is left to the caller; here we count the operators
    that enable them). Used to show P1/P2 constraints suppressing
    pathological picks -- and currently, to document their prevalence.
    """
    from apedp.safety import safe_parse_expr

    expr = safe_parse_expr(expression) if isinstance(expression, str) else expression

    def depth(node: sp.Expr) -> int:
        if not node.args:
            return 0
        return 1 + max(depth(arg) for arg in node.args)

    from apedp.dimensional_layer import _TRANSCENDENTAL_NAMES

    trans = [node for node in sp.preorder_traversal(expr)
             if isinstance(node, sp.Function) and type(node).__name__ in _TRANSCENDENTAL_NAMES]
    nested_pow = 0
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and any(isinstance(arg, sp.Pow) for arg in node.args):
            nested_pow += 1
    coefs = [abs(float(term)) for term in expr.atoms(sp.Float) if np.isfinite(float(term))]
    return {
        "depth": depth(expr),
        "nodes": node_count(expr),
        "max_coefficient": max(coefs, default=0.0),
        "n_coefficients": len(coefs),
        "transcendentals": len(trans),
        "nested_powers": nested_pow,
    }


def _pareto_front(records: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    front = []
    for candidate in records:
        dominated = any(other[1] <= candidate[1] and other[2] <= candidate[2] and (other[1], other[2]) != (candidate[1], candidate[2]) for other in records)
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item[1])


def evolve(
    x_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    x_test: dict[str, np.ndarray] | None = None,
    y_test: np.ndarray | None = None,
    *,
    variable_units: dict[str, str] | None = None,
    target_unit: str | None = None,
    config: SearchConfig | None = None,
    on_generation: Callable[[int, list[str]], None] | None = None,
) -> SearchResult:
    """Run the island evolutionary search with grouped final selection.

    The Pareto front trades train error against complexity; ``grouped_pick``
    is the front member minimizing *holdout* error when test data is
    supplied, else the lowest-train-error member. Every upgrade in
    :class:`SearchConfig` toggles independently for ablation.

    ``on_generation`` is an observability hook only (generation index plus
    stringified population); it cannot alter the search, so instrumented
    runs remain comparable with uninstrumented ones.
    """
    cfg = config or SearchConfig()
    rng = random.Random(cfg.seed)
    names = list(x_train.keys())
    data_symbols = [sp.Symbol(name) for name in names]
    raw_arrays = {name: np.asarray(x_train[name], dtype=float).ravel() for name in names}
    raw_target = np.asarray(y_train, dtype=float).ravel()
    # Normalize numerics for conditioning; symbols keep original units so
    # dimensional gating and reported expressions stay in physical variables.
    # Scale-only (no centering): mean shifts would move reciprocal-term
    # singularities into the data range (1/(x-mu) is not 1/x) and change the
    # spanned family, while pure rescaling preserves every span exactly.
    # Target normalization uses train-only statistics; all reported errors
    # are converted back to physical units inside fitness.
    scaling: dict[str, list[float]] = {}
    arrays = dict(raw_arrays)
    target = raw_target
    y_scale = 1.0
    if cfg.normalize:
        for name in names:
            sigma = float(np.std(raw_arrays[name]))
            if sigma == 0.0 or not np.isfinite(sigma):
                continue
            scaling[name] = [0.0, sigma]
            arrays[name] = raw_arrays[name] / sigma
        target_mu = float(np.mean(raw_target))
        target_sigma = float(np.std(raw_target))
        if target_sigma != 0.0 and np.isfinite(target_sigma):
            scaling["__target__"] = [target_mu, target_sigma]
            target = (raw_target - target_mu) / target_sigma
            y_scale = target_sigma
    cache: dict = {}
    islands: list[list[tuple[sp.Expr, int]]] = []
    for _ in range(cfg.islands):
        population: list[tuple[sp.Expr, int]] = []
        search = _IslandSearch(data_symbols, arrays, target, variable_units, target_unit, cfg.symmetries, cfg, rng, cache, y_scale)
        if cfg.pi_seed:
            for seed_expr in search._pi_seeds():
                if search._admissible_at_birth(seed_expr):
                    population.append((seed_expr, 0))
        attempts = 0
        while len(population) < cfg.island_size and attempts < cfg.island_size * 20:
            attempts += 1
            candidate = search._grow(cfg.max_depth)
            if search._admissible_at_birth(candidate):
                population.append((candidate, 0))
        islands.append(population)
    max_age = [4 * (2**index) for index in range(cfg.islands - 1)] + [10**9] if cfg.alps and cfg.islands > 1 else [10**9] * cfg.islands
    search = _IslandSearch(data_symbols, arrays, target, variable_units, target_unit, cfg.symmetries, cfg, rng, cache, y_scale)
    history: list[dict] = []
    evaluated = 0
    tuned: dict[str, tuple[float, dict[sp.Symbol, float]]] = {}

    def score_many(expressions: list[sp.Expr]) -> list[tuple[float, float, dict]]:
        def work(expression: sp.Expr) -> tuple[float, float, dict]:
            return search.fitness(expression)

        if cfg.threads and cfg.threads > 1:
            with ThreadPoolExecutor(max_workers=cfg.threads) as pool:
                results = list(pool.map(work, expressions))
        else:
            results = [work(expression) for expression in expressions]
        for expression, (mse, _, param_map) in zip(expressions, results, strict=False):
            key = str(expression)
            if key not in tuned or mse < tuned[key][0]:
                tuned[key] = (mse, param_map)
        return results

    for generation in range(cfg.generations):
        for island_index, population in enumerate(islands):
            exprs = [expression for expression, _ in population]
            scored = score_many(exprs)
            evaluated += len(exprs)
            scored_pool = [(population[index][0], scored[index][0]) for index in range(len(population))]
            scored_pool.sort(key=lambda pair: pair[1])
            keep = max(2, len(population) // 2)
            elite = scored_pool[:keep]
            survivors = [(expression, next(age for expr2, age in population if expr2 == expression) + 1) for expression, _ in elite]
            children: list[tuple[sp.Expr, int]] = []
            guard = 0
            while len(survivors) + len(children) < cfg.island_size and guard < cfg.island_size * 4:
                guard += 1
                first = search.tournament(elite)
                second = search.tournament(elite)
                child = search.crossover(first, second) if rng.random() < 0.7 else first
                if rng.random() < cfg.mutation_rate:
                    child = search.mutate(child)
                if search._admissible_at_birth(child):
                    ages = [age for expression, age in population if expression == first or expression == second]
                    child_age = (max(ages) + 1) if ages else 1
                    children.append((child, child_age))
                if len(children) > cfg.island_size * 3:
                    break
            merged = survivors + children[: max(0, cfg.island_size - len(survivors))]
            if cfg.alps:
                cap = max_age[island_index]
                aged = [(expression, age) for expression, age in merged if age <= cap]
                while len(aged) < cfg.island_size:
                    newcomer = search._grow(cfg.max_depth)
                    if search._admissible_at_birth(newcomer):
                        aged.append((newcomer, 0))
                    else:
                        break
                merged = aged[: cfg.island_size] if aged else merged[: cfg.island_size]
            islands[island_index] = merged if merged else population
        if cfg.migration_interval and (generation + 1) % cfg.migration_interval == 0 and len(islands) > 1:
            bests = []
            for population in islands:
                scored = score_many([expression for expression, _ in population])
                bests.append(min(zip(population, scored, strict=False), key=lambda pair: pair[1][0])[0][0])
            for index in range(len(islands)):
                if islands[(index + 1) % len(islands)]:
                    islands[(index + 1) % len(islands)][-1] = (bests[index], 0)
        if generation % 5 == 0 or generation == cfg.generations - 1:
            history.append({"generation": generation, "islands": len(islands)})
        if on_generation is not None:
            on_generation(generation, [str(expression) for population in islands for expression, _ in population])
    seen: dict[str, tuple[float, float]] = {}
    for population in islands:
        for expression, _ in population:
            key = str(expression)
            if key not in tuned:
                continue
            mse, _ = tuned[key]
            skeleton, params = skeletonize(expression)
            complexity = _complexity(expression, params, cfg.coef_penalty)
            if key not in seen or mse < seen[key][0]:
                seen[key] = (mse, complexity)
    front = _pareto_front([(key, mse, complexity) for key, (mse, complexity) in seen.items()])
    best = min(front, key=lambda item: item[1])[0] if front else None
    grouped_pick: str | None = None
    y_mu, y_scale_locals = scaling.get("__target__", [0.0, 1.0])
    if x_test is not None and y_test is not None and front:
        test_arrays = {name: np.asarray(x_test[name], dtype=float).ravel() for name in x_test}
        if scaling:
            # Grouped evaluation happens in the same standardized space the
            # front was fitted in; denormalization is presentation-only.
            for name, (mu, sigma) in scaling.items():
                if name in test_arrays:
                    test_arrays[name] = (test_arrays[name] - mu) / sigma
        test_target = np.asarray(y_test, dtype=float).ravel()
        scored_front = []
        for key, _, _ in front:
            expression = safe_parse_expr(key)
            skeleton, params = skeletonize(expression)
            param_map, _ = tune_nonlinear(
                skeleton, params, data_symbols, arrays, target, cache,
                restarts=1, iters=cfg.tune_iters, seed=cfg.seed, use_projection=cfg.use_projection,
            )
            substituted = skeleton.xreplace(param_map)
            if _broken_after_substitution(substituted):
                continue
            function = _lambdify_cached(cache, substituted, data_symbols)
            with np.errstate(all="ignore"):
                prediction = _real_or_none(
                    function(*[test_arrays[str(symbol)] for symbol in data_symbols]), test_target.shape
                )
            if prediction is not None:
                # Back to physical units: the front was fitted to the
                # train-normalized target, the holdout target is raw.
                scored_front.append((key, float(np.mean((prediction * y_scale_locals + y_mu - test_target) ** 2))))
        if scored_front:
            grouped_pick = min(scored_front, key=lambda pair: pair[1])[0]
    def _denormalized(expression_str: str | None) -> str | None:
        if expression_str is None or not scaling:
            return expression_str
        try:
            return str(denormalize_expression(expression_str, scaling))
        except (TypeError, ValueError, ArithmeticError):
            return None

    return SearchResult(
        best_expression=best,
        grouped_pick=grouped_pick,
        pareto_front=[{"expression": key, "train_mse": mse, "complexity": complexity} for key, mse, complexity in front],
        generations_run=cfg.generations,
        n_evaluations=evaluated,
        history=history,
        variable_scaling=scaling,
        denormalized_best=_denormalized(best),
        denormalized_pick=_denormalized(grouped_pick),
    )


__all__ = [
    "SearchConfig",
    "SearchResult",
    "_IslandSearch",
    "_pareto_front",
    "denormalize_expression",
    "evaluate_skeleton",
    "evolve",
    "node_count",
    "partition_parameters",
    "skeletonize",
    "tune_nonlinear",
]
