"""Constrained symbolic regression engine for APEDP (legacy standalone engine).

This module implements a unit-aware symbolic regression engine that:
- Uses SymPy expression trees over a configurable set of variables.
- Filters candidate models via DimensionalAssuranceEngine.
- Fits parameters to data using a simple evolutionary loop.

Division of labor: prefer ``apedp.discovery.MultiVariableSRE`` for all
dimensional discovery (guided seeding, BIC/AIC selection, degeneracy
refusal). This class is kept for backwards compatibility and small
dimensionless demos (tournament selection, crossover, ``sin``/``cos``/
``exp``). It does NOT implement BIC selection or degeneracy refusal.

It is designed to sit on top of the assurance core in
`apedp.core`.
"""

from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
from apedp.safety import quiet_float_warnings

logger = logging.getLogger(__name__)


@dataclass
class RegressionConfig:
    population_size: int = 64
    generations: int = 50
    max_depth: int = 4
    tournament_size: int = 4
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    allowed_functions: tuple[Callable[[sp.Expr], sp.Expr], ...] = (sp.sin, sp.cos, sp.exp)
    seed: int = 42


class SymbolicRegressor:
    """Unit-aware symbolic regression engine.

    Usage:
        engine = DimensionalAssuranceEngine()
        platform = ClosedLoopAssuranceSystem(engine)
        # register variables with units
        x = engine.register_variable('x', 'meter')
        y = engine.register_variable('y', 'second')
        target_unit_str = 'meter / second'  # example
        reg = SymbolicRegressor(engine, ['x', 'y'], target_unit_str)
        best_expr, history = reg.fit(X, y)
    """

    def __init__(
        self,
        assurance_engine: DimensionalAssuranceEngine,
        variable_names: list[str],
        target_unit_str: str,
        config: RegressionConfig | None = None,
    ) -> None:
        warnings.warn(
            "SymbolicRegressor is legacy; prefer apedp.discovery.MultiVariableSRE "
            "for dimensional discovery (guided seeding, BIC/AIC, degeneracy refusal).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.engine = assurance_engine
        self.variable_names = variable_names
        self.symbols = {name: sp.Symbol(name) for name in variable_names}
        self.target_unit = self.engine.registry(target_unit_str)
        self.config = config or RegressionConfig()
        self.rng = random.Random(self.config.seed)

    # ----------------------- Expression generation -----------------------
    # NOTE: randomness below drives evolutionary search, not cryptography;
    # stdlib random is the correct tool (secrets is non-seedable and slower).

    def _random_variable(self) -> sp.Symbol:
        return self.symbols[self.rng.choice(self.variable_names)]  # nosec B311

    def _random_constant(self) -> sp.Float:
        # small numeric constants
        return sp.Float(self.rng.uniform(-5.0, 5.0))  # nosec B311

    def _random_unary(self, expr: sp.Expr) -> sp.Expr:
        func = self.rng.choice(self.config.allowed_functions)  # nosec B311
        return func(expr)

    def _grow_tree(self, depth: int) -> sp.Expr:
        if depth == 0:
            # leaf: variable or constant
            if self.rng.random() < 0.5:  # nosec B311
                return self._random_variable()
            return self._random_constant()

        # choose operation
        valid_ops = ['add', 'sub', 'mul', 'div', 'pow']
        if self.config.allowed_functions:
            valid_ops.append('unary')
        op = self.rng.choice(valid_ops)  # nosec B311
        if op in ('add', 'sub', 'mul', 'div'):
            left = self._grow_tree(depth - 1)
            right = self._grow_tree(depth - 1)
            if op == 'add':
                return left + right
            if op == 'sub':
                return left - right
            if op == 'mul':
                return left * right
            if op == 'div':
                return left / right
        elif op == 'pow':
            base = self._grow_tree(depth - 1)
            # restrict exponent to small integers for stability
            exp = self.rng.choice([sp.Integer(2), sp.Integer(3), sp.Integer(-1)])  # nosec B311
            return base ** exp
        else:  # unary
            inner = self._grow_tree(depth - 1)
            return self._random_unary(inner)

    @staticmethod
    def _is_finite_expr(expr: sp.Expr) -> bool:
        """Reject oo/-oo/zoo/nan, which SymPy's lambdify printer cannot emit.

        These arise during generation (e.g. ``x - x`` then dividing by it) and
        pass the dimensional gatekeeper, since zoo has no arguments to check.
        """
        return not expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo)

    def _generate_population(self) -> list[sp.Expr]:
        pop: list[sp.Expr] = []
        max_attempts = self.config.population_size * 50
        attempts = 0
        while len(pop) < self.config.population_size and attempts < max_attempts:
            attempts += 1
            expr = self._grow_tree(self.config.max_depth)
            if not self._is_finite_expr(expr):
                continue
            try:
                unit = self.engine.check_expression_consistency(expr)
                if unit.dimensionality == self.target_unit.dimensionality:
                    pop.append(expr)
            except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                logger.debug("Discarding dimensionally invalid expression (%s)", e)
                continue

        # If unreachable, seed with variables matching target dimension or dimensionless fallback
        if not pop:
            for sym in self.symbols.values():
                try:
                    u = self.engine.symbol_units.get(sym)
                    if u and u.dimensionality == self.target_unit.dimensionality:
                        pop.append(sym)
                except (AttributeError, KeyError):
                    logger.debug("Symbol unit check failed for fallback", exc_info=True)
            if not pop and self.target_unit.dimensionality == self.engine.registry.dimensionless.dimensionality:
                pop.append(sp.Float(1.0))
            if not pop:
                raise ValueError(
                    f"Cannot generate dimensionally valid candidates for '{self.target_unit}' from {self.variable_names}"
                )

        while len(pop) < self.config.population_size:
            pop.append(self.rng.choice(pop))  # nosec B311
        return pop

    # ----------------------- Fitness evaluation -------------------------

    def _evaluate_expr(self, expr: sp.Expr, X: dict[str, np.ndarray]) -> np.ndarray:
        # order of arguments must match tuple(self.symbols.values())
        args = [X[name] for name in self.variable_names]
        try:
            # lambdify for performance; it raises for expressions it cannot
            # print (e.g. zoo), so it must stay inside the guard.
            f = sp.lambdify(tuple(self.symbols.values()), expr, modules=['numpy'])
            # Candidate expressions are arbitrary, so the search routinely
            # evaluates a form that is singular somewhere in the data grid
            # (``1/x`` with a zero sample, ``sqrt`` of a negative one, ``exp``
            # overflowing). numpy's warning about that is expected, and
            # ``_fitness`` turns the resulting non-finite prediction into an
            # outright rejection -- so the warning only buries the real ones.
            with quiet_float_warnings():
                y_pred = f(*args)
            arr = np.asarray(y_pred, dtype=float)
            if args and (arr.ndim == 0 or arr.shape != args[0].shape):
                arr = np.full(args[0].shape, float(arr), dtype=float)
            return arr
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.warning("Numerical evaluation failed, returning NaNs (%s)", e)
            return np.full_like(args[0], np.nan, dtype=float)

    def _fitness(self, expr: sp.Expr, X: dict[str, np.ndarray], y: np.ndarray) -> float:
        y_pred = self._evaluate_expr(expr, X)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf')
        # mean squared error
        mse = float(np.mean((y_pred - y) ** 2))
        # complexity penalty: number of nodes
        complexity = len(list(sp.preorder_traversal(expr)))
        return mse + 1e-3 * complexity

    # ----------------------- Genetic operators --------------------------

    def _tournament_select(self, population: list[sp.Expr], scores: dict[sp.Expr, float]) -> sp.Expr:
        contestants = self.rng.sample(population, k=min(self.config.tournament_size, len(population)))  # nosec B311
        best = min(contestants, key=lambda e: scores[e])
        return best

    def _mutate(self, expr: sp.Expr) -> sp.Expr:
        if self.rng.random() > self.config.mutation_rate:  # nosec B311
            return expr
        # simple mutation: replace a random subtree with a new random tree
        expr_tree = list(sp.preorder_traversal(expr))
        target = self.rng.choice(expr_tree)  # nosec B311
        new_subtree = self._grow_tree(depth=2)
        return expr.xreplace({target: new_subtree})

    def _crossover(self, parent1: sp.Expr, parent2: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        if self.rng.random() > self.config.crossover_rate:  # nosec B311
            return parent1, parent2
        tree1 = list(sp.preorder_traversal(parent1))
        tree2 = list(sp.preorder_traversal(parent2))
        if not tree1 or not tree2:
            return parent1, parent2
        cut1 = self.rng.choice(tree1)  # nosec B311
        cut2 = self.rng.choice(tree2)  # nosec B311
        child1 = parent1.xreplace({cut1: cut2})
        child2 = parent2.xreplace({cut2: cut1})
        return child1, child2

    # ----------------------- Main optimization loop ---------------------

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray) -> tuple[sp.Expr, list[dict[str, Any]]]:
        """Run symbolic regression and return best expression plus history.

        X: dict mapping variable name -> numpy array of values.
        y: numpy array of target values.
        """
        self.rng = random.Random(self.config.seed)
        population = self._generate_population()
        history: list[dict[str, Any]] = []

        scores: dict[sp.Expr, float] = {expr: self._fitness(expr, X, y) for expr in population}
        best = min(population, key=lambda e: scores[e])
        # Carried as a scalar: ``best`` outlives ``scores``, which is rebuilt
        # from the new population each generation.
        best_score = scores[best]

        for gen in range(self.config.generations):
            new_population: list[sp.Expr] = []
            max_breed_attempts = self.config.population_size * 20
            attempts = 0
            while len(new_population) < self.config.population_size and attempts < max_breed_attempts:
                attempts += 1
                parent1 = self._tournament_select(population, scores)
                parent2 = self._tournament_select(population, scores)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                for child in (child1, child2):
                    if not self._is_finite_expr(child):
                        continue
                    try:
                        unit = self.engine.check_expression_consistency(child)
                        if unit.dimensionality == self.target_unit.dimensionality:
                            new_population.append(child)
                    except (ValueError, TypeError, ArithmeticError, NotImplementedError) as e:
                        logger.debug("Discarding dimensionally invalid child (%s)", e)
                        continue
                    if len(new_population) >= self.config.population_size:
                        break

            while len(new_population) < self.config.population_size:
                new_population.append(self.rng.choice(population))  # nosec B311

            population = new_population
            scores = {expr: self._fitness(expr, X, y) for expr in population}
            gen_best = min(population, key=lambda e: scores[e])
            if scores[gen_best] < best_score:
                best, best_score = gen_best, scores[gen_best]

            history.append({
                'generation': gen,
                'best_score': best_score,
                'best_expr': best,
            })

        return best, history


if __name__ == '__main__':
    # Minimal demo: fit y ~ x (dimensionless example)
    engine = DimensionalAssuranceEngine()
    platform = ClosedLoopAssuranceSystem(engine)
    x = engine.register_variable('x', 'dimensionless')
    target_unit_str = 'dimensionless'
    reg = SymbolicRegressor(engine, ['x'], target_unit_str)

    X = {'x': np.linspace(-1, 1, 50)}
    y = X['x'] ** 3 + 0.1 * np.random.randn(50)  # cubic with noise

    best_expr, hist = reg.fit(X, y)
    print("Best expression:", best_expr)
    print("Best score:", hist[-1]['best_score'])
