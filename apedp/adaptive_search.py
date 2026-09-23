# Databricks notebook source
"""Adaptive evolutionary search orchestration with persistent search state."""
from __future__ import annotations

import contextlib
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from apedp.decision import LocalSearchController, SearchAction
from apedp.discovery import MultiVariableSRE
from apedp.gated_assurance import ParetoFrontierTracker, ScientificImprovementMetric

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSearchResult:
    best_candidate: sp.Expr | None
    best_score: float
    history: list[dict[str, Any]]
    actions: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str | None = None
    stateful: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate": str(self.best_candidate) if self.best_candidate is not None else None,
            "best_score": self.best_score,
            "history": self.history,
            "actions": self.actions,
            "stopped_reason": self.stopped_reason,
            "stateful": self.stateful,
        }


@contextlib.contextmanager
def _seeded_global_rng(seed: int | None) -> Iterator[None]:
    """Seed the process RNG streams for one search, then restore them.

    The symbolic-regression engine samples mutations from the *global*
    ``random`` / ``numpy`` generators, so reproducibility requires seeding them.
    Seeding without restoring would leak an order dependency into every other
    component that shares those generators -- whether an unrelated caller got
    the same numbers would depend on whether an adaptive search ran first.
    Seeding and restoring around the search keeps the run reproducible while
    leaving the global streams exactly as they were found.
    """
    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _initial_population(engine: MultiVariableSRE, pop_size: int) -> list[sp.Expr]:
    target_dim = engine.target_unit.dimensionality
    population = list(engine.template_exprs[:pop_size])
    seed_fn = engine._generate_guided_candidate if engine.guided else engine._generate_candidate
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 20:
        candidate = seed_fn(depth=0, max_depth=3)
        try:
            if (engine.system.engine.check_expression_consistency(
                    candidate).dimensionality == target_dim
                    and all(candidate != existing for existing in population)):
                population.append(candidate)
        except (ValueError, TypeError, ArithmeticError,
                NotImplementedError) as exc:
            # Candidate rejected: not dimensionally consistent, or the engine
            # could not evaluate it. Never counted as admitted to the seed set.
            logger.debug("seed candidate rejected: %s", exc)
        attempts += 1
    while len(population) < pop_size:
        population.append(engine._generate_candidate(depth=0, max_depth=2))
    return population


def _run_stateful_generation(engine, population, x_data, y_data, state):
    scored = [(engine.fitness(expr, x_data, y_data), expr) for expr in population]
    scored.sort(key=lambda item: item[0])
    generation = state["generation"]
    score, expr = scored[0]
    improved = score < state["best_score"]
    if improved:
        state["best_score"] = score
        state["best_candidate"], _ = engine.optimize_parameters(expr, x_data, y_data)
        state["improvements"] += 1
        state["without_improvement"] = 0
    else:
        state["without_improvement"] += 1

    for candidate_score, candidate_expr in scored:
        complexity = len(list(sp.preorder_traversal(candidate_expr)))
        metric = ScientificImprovementMetric(
            i_fit=1.0 / (1.0 + candidate_score),
            i_stability=1.0 / (1.0 + candidate_score),
            i_physicality=1.0 if engine._check_dimensional(candidate_expr) else 0.0,
            i_generality=0.0,
            i_complexity=1.0 / (1.0 + complexity * 0.1),
            i_falsifiability=1.0,
        )
        state["frontier"].consider_candidate(candidate_expr, metric)

    record = {
        "generation": generation,
        "best_score": state["best_score"],
        "best_expr": state["best_candidate"],
        "improved": improved,
        "generations_without_improvement": state["without_improvement"],
        "pareto_frontier_size": len(state["frontier"].frontier),
    }
    elites = [expr for _, expr in scored[:max(1, int(len(population) * 0.2))]]
    next_population = list(elites)
    elite_text = {str(expr) for expr in elites}
    for candidate in state["frontier"].get_frontier():
        if str(candidate) not in elite_text:
            next_population.append(candidate)
            elite_text.add(str(candidate))
    while len(next_population) < len(population):
        next_population.append(engine.mutate(random.choice(elites)))
    state["generation"] += 1
    return next_population[:len(population)], record


def run_adaptive_discovery(
    engine: MultiVariableSRE,
    x_data: dict[str, np.ndarray],
    y_data: np.ndarray,
    *,
    generations: int = 30,
    pop_size: int = 100,
    seed: int | None = 42,
    controller: LocalSearchController | None = None,
    chunk_size: int = 5,
) -> AdaptiveSearchResult:
    """Run evolution while preserving population and frontier across actions."""
    if generations < 1 or pop_size < 1 or chunk_size < 1:
        raise ValueError("generations, pop_size, and chunk_size must be positive")
    controller = controller or LocalSearchController()
    if not isinstance(engine, MultiVariableSRE):
        return _run_legacy_chunked(engine, x_data, y_data, generations, pop_size, seed, controller, chunk_size)

    with _seeded_global_rng(seed):
        engine._fitness_cache.clear()
        population = _initial_population(engine, pop_size)
        state: dict[str, Any] = {"generation": 0, "best_candidate": None, "best_score": float("inf"), "improvements": 0, "without_improvement": 0, "frontier": ParetoFrontierTracker()}
        history: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        stopped_reason = None

        for _ in range(generations):
            population, record = _run_stateful_generation(engine, population, x_data, y_data, state)
            history.append(record)
            if (state["generation"] % chunk_size) == 0 or state["generation"] == generations:
                variance = float(np.var(y_data))
                telemetry = dict(record)
                telemetry.update({
                    "final_normalized_score": state["best_score"] / variance if variance > 0 else float("inf"),
                    "improvements": state["improvements"],
                })
                decision = controller.decide(telemetry)
                action = decision.to_dict()
                action["telemetry"] = telemetry
                actions.append(action)
                for flag in ("allow_power_mutation", "allow_rational_mutation", "allow_compose_mutation", "allow_logcomp_mutation"):
                    if decision.parameters.get(flag) is True:
                        setattr(engine, flag, True)
                if decision.action in (SearchAction.STOP_STAGNATION, SearchAction.HUMAN_REVIEW):
                    stopped_reason = decision.reason
                    break
            if state["best_score"] < 1e-25:
                break

        return AdaptiveSearchResult(state["best_candidate"], state["best_score"], history, actions, stopped_reason, True)


def _run_legacy_chunked(engine, x_data, y_data, generations, pop_size, seed, controller, chunk_size):
    """Compatibility path for test doubles and non-MultiVariableSRE engines."""
    remaining, completed = generations, 0
    history, actions = [], []
    best_candidate, best_score, stopped_reason = None, float("inf"), None
    while remaining > 0:
        current = min(chunk_size, remaining)
        candidate, score, chunk_history = engine.run_discovery(x_data, y_data, generations=current, pop_size=pop_size, seed=None if seed is None else seed + completed)
        history.extend(chunk_history)
        if score < best_score:
            best_candidate, best_score = candidate, score
        if chunk_history:
            latest = chunk_history[-1]
            variance = float(np.var(y_data))
            telemetry = {"generation": completed + current - 1, "best_score": score, "final_normalized_score": score / variance if variance > 0 else float("inf"), "generations_without_improvement": int(engine.stagnation_report.get("generations_without_improvement", 0)), "pareto_frontier_size": int(latest.get("pareto_frontier_size", 0)), "improvements": int(engine.stagnation_report.get("improvements", 0))}
            decision = controller.decide(telemetry)
            action = decision.to_dict()
            action["telemetry"] = telemetry
            actions.append(action)
            if decision.action in (SearchAction.STOP_STAGNATION, SearchAction.HUMAN_REVIEW):
                stopped_reason = decision.reason
                break
        completed += current
        remaining -= current
    return AdaptiveSearchResult(best_candidate, best_score, history, actions, stopped_reason, False)


__all__ = ["AdaptiveSearchResult", "run_adaptive_discovery"]