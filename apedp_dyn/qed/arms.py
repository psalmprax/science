"""Six proposers that search the same selection space under the same budget.

Status: experimental

Each arm is an independent search mechanism over the binary selection vector
``z``. They all receive a fresh budgeted objective, they all pay one unit per
distinct selection, and none of them can see the holdout rows or the answer.
An arm that cannot run says so with a reason (:class:`ArmAbstained`) instead of
returning a worse answer: a proposer that did not search is not a proposer that
lost.

* ``random_restart`` -- first-improvement hill climbing from random starts.
* ``simulated_annealing`` -- Metropolis on single-bit flips, geometric cooling.
* ``genetic_algorithm`` -- bitstrings, tournament selection, uniform crossover.
* ``boolean_gp`` -- genetic programming over a set-building language
  (``and``/``or`` over term leaves), so the genotype is a variable-size tree and
  the phenotype is still a support the shared objective can score.
* ``rime_relaxed`` -- the RIME soft/hard operators applied to a relaxed
  selection vector in ``[0, 1]`` and hardened at 0.5. This is the RIME *operator
  set on a different search space*, not
  :class:`apedp_dyn.proposers.rime_proposer.RimeProposer`, which evolves
  financial feature mechanisms through the native Rust kernel; no native RIME
  code is used here.
* ``qaoa_surrogate`` -- QAOA over an explicitly approximate quadratic surrogate
  of ``J``, with the surrogate's cost paid from the same budget.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.qed.library import TermLibrary
from apedp_dyn.qed.objective import BudgetExhausted, Fit, SupportObjective
from apedp_dyn.qed.qaoa import (
    build_quadratic_surrogate,
    mask_to_support,
    run_qaoa,
)

SEARCHED = "searched"
ABSTAINED = "abstained"


class ArmAbstained(RuntimeError):
    """Raised by an arm that cannot run, carrying the reason in the message."""


@dataclass(frozen=True)
class ArmResult:
    """One arm's measured attempt: what it found, what it spent, and how long."""

    arm: str
    status: str
    reason: str
    best: Fit | None
    evaluations: int
    budget: int
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def searched(self) -> bool:
        return self.status == SEARCHED

    def as_dict(self, library: TermLibrary) -> dict[str, Any]:
        names = library.names
        return {
            "arm": self.arm,
            "status": self.status,
            "reason": self.reason,
            "evaluations": self.evaluations,
            "budget": self.budget,
            "seconds": round(float(self.seconds), 6),
            "best": None if self.best is None else self.best.as_dict(names),
            "detail": dict(self.detail),
        }


def _random_support(m: int, rng: np.random.Generator) -> tuple[int, ...]:
    bits = rng.random(m) < 0.35
    if not bits.any():
        bits[int(rng.integers(m))] = True
    return tuple(int(index) for index in np.flatnonzero(bits))


def _flip(support: Sequence[int], index: int) -> tuple[int, ...]:
    members = set(support)
    if index in members:
        members.remove(index)
    else:
        members.add(index)
    return tuple(sorted(members))


# --------------------------------------------------------------------------- #
# Classical selection search
# --------------------------------------------------------------------------- #


def random_restart(objective: SupportObjective, rng: np.random.Generator, *, restarts: int = 12) -> dict[str, Any]:
    m = objective.m
    accepted = 0
    starts = 0
    for _ in range(restarts):
        if objective.remaining <= 0:
            break
        starts += 1
        current = objective.evaluate(_random_support(m, rng))
        for _ in range(3 * m):
            if objective.remaining <= 0:
                break
            improved = False
            for index in rng.permutation(m):
                if objective.remaining <= 0:
                    break
                candidate = objective.evaluate(_flip(current.support, int(index)))
                if candidate.objective < current.objective - 1e-15:
                    current = candidate
                    improved = True
                    accepted += 1
                    break
            if not improved:
                break
    return {"starts": starts, "accepted_steps": accepted}


def simulated_annealing(
    objective: SupportObjective,
    rng: np.random.Generator,
    *,
    restarts: int = 4,
    start_temperature: float = 0.02,
    end_temperature: float = 1e-5,
) -> dict[str, Any]:
    m = objective.m
    moves = 0
    accepted = 0
    for attempt in range(restarts):
        if objective.remaining <= 0:
            break
        current = objective.evaluate(_random_support(m, rng))
        steps = max(1, objective.remaining // max(1, restarts - attempt))
        for step in range(steps):
            if objective.remaining <= 0:
                break
            fraction = step / max(1, steps - 1)
            temperature = start_temperature * (end_temperature / start_temperature) ** fraction
            candidate = objective.evaluate(_flip(current.support, int(rng.integers(m))))
            moves += 1
            delta = candidate.objective - current.objective
            if delta <= 0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                current = candidate
                accepted += 1
    return {"moves": moves, "accepted_moves": accepted, "restarts": restarts}


def genetic_algorithm(
    objective: SupportObjective,
    rng: np.random.Generator,
    *,
    population: int = 16,
    tournament: int = 3,
    mutation_rate: float = 0.15,
    max_generations: int = 400,
) -> dict[str, Any]:
    m = objective.m
    pool: dict[tuple[int, ...], Fit] = {}

    def remember(support: tuple[int, ...]) -> Fit:
        known = objective.cached(support)
        if known is not None:
            return known
        fit = objective.evaluate(support)
        pool[support] = fit
        return fit

    for _ in range(population):
        if objective.remaining <= 0:
            break
        remember(_random_support(m, rng))

    generations = 0
    # A repeated selection is free, so the budget alone cannot guarantee
    # termination: once a population recombines into selections it has already
    # scored, generations cost nothing. The cap is the terminator; the budget
    # stays the binding constraint on how much was searched.
    while objective.remaining > 0 and pool and generations < max_generations:
        generations += 1
        contenders = list(pool)
        parents = []
        for _ in range(2):
            picks = rng.choice(len(contenders), size=min(tournament, len(contenders)), replace=False)
            parents.append(min((contenders[int(pick)] for pick in picks), key=lambda key: pool[key].objective))
        mask = [1 if index in parents[0] else 0 for index in range(m)]
        other = {1 if index in parents[1] else 0 for index in range(m)}
        child = [int(value) if rng.random() < 0.5 else (1 if index in other else 0)
                 for index, value in enumerate(mask)]
        for index in range(m):
            if rng.random() < mutation_rate:
                child[index] = 1 - child[index]
        support = tuple(index for index, bit in enumerate(child) if bit)
        if objective.cached(support) is not None and m > 1:
            # A recombination that lands on an already scored selection spends
            # nothing and teaches nothing; nudge it once so the budget is used.
            flip = int(rng.integers(m))
            support = _flip(support, flip)
        fit = remember(support)
        worst = max(pool, key=lambda key: pool[key].objective)
        if fit.objective < pool[worst].objective:
            del pool[worst]
            pool[support] = fit
    return {
        "generations": generations,
        "generation_cap": max_generations,
        "final_population": len(pool),
        "population": population,
    }


# --------------------------------------------------------------------------- #
# Genetic programming over a set-building language
# --------------------------------------------------------------------------- #


GpNode = tuple


def _gp_leaf(m: int, rng: np.random.Generator) -> GpNode:
    return ("term", int(rng.integers(m)))


def _gp_random_tree(m: int, rng: np.random.Generator, depth: int, max_depth: int = 3) -> GpNode:
    if depth >= max_depth or rng.random() < 0.3:
        return _gp_leaf(m, rng)
    operator = "and" if rng.random() < 0.5 else "or"
    return (operator, _gp_random_tree(m, rng, depth + 1, max_depth), _gp_random_tree(m, rng, depth + 1, max_depth))


def _gp_support(node: GpNode) -> frozenset[int]:
    if node[0] == "term":
        return frozenset((node[1],))
    left = _gp_support(node[1])
    right = _gp_support(node[2])
    return left & right if node[0] == "and" else left | right


def _gp_paths(node: GpNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if node[0] != "term":
        paths.extend(_gp_paths(node[1], (*prefix, 1)))
        paths.extend(_gp_paths(node[2], (*prefix, 2)))
    return paths


def _gp_get(node: GpNode, path: tuple[int, ...]) -> GpNode:
    for step in path:
        node = node[step]
    return node


def _gp_replace(node: GpNode, path: tuple[int, ...], replacement: GpNode) -> GpNode:
    if not path:
        return replacement
    head, *tail = path
    children = list(node)
    children[head] = _gp_replace(children[head], tuple(tail), replacement)
    return tuple(children)


def _gp_render(node: GpNode, names: Sequence[str]) -> str:
    if node[0] == "term":
        _out: str = names[node[1]]
        return _out
    return f"({_gp_render(node[1], names)} {node[0]} {_gp_render(node[2], names)})"


def boolean_gp(
    objective: SupportObjective,
    rng: np.random.Generator,
    *,
    population: int = 20,
    tournament: int = 3,
    mutation_rate: float = 0.2,
    max_generations: int = 400,
) -> dict[str, Any]:
    m = objective.m
    names = objective.library.names
    individuals: dict[frozenset[int], tuple[GpNode, Fit]] = {}

    def score(node: GpNode) -> tuple[GpNode, Fit]:
        support = frozenset(_gp_support(node))
        known = individuals.get(support)
        if known is not None:
            return node, known[1]
        ordered = tuple(sorted(support))
        fit = objective.cached(ordered) or objective.evaluate(ordered)
        individuals[support] = (node, fit)
        return node, fit

    live: list[GpNode] = []
    for _ in range(population):
        if objective.remaining <= 0:
            break
        node = _gp_random_tree(m, rng, 0)
        live.append(score(node)[0])

    generations = 0
    while objective.remaining > 0 and live and generations < max_generations:
        generations += 1
        contender_indices = rng.choice(len(live), size=min(tournament, len(live)), replace=False)
        parent_a = live[min((int(index) for index in contender_indices),
                            key=lambda index: score(live[index])[1].objective)]
        index_b = int(rng.integers(len(live)))
        parent_b = live[index_b]

        if rng.random() < 0.7:
            path_a = _gp_paths(parent_a)[int(rng.integers(len(_gp_paths(parent_a))))]
            path_b = _gp_paths(parent_b)[int(rng.integers(len(_gp_paths(parent_b))))]
            child = _gp_replace(parent_a, path_a, _gp_get(parent_b, path_b))
        else:
            child = parent_a
        if rng.random() < mutation_rate:
            path = _gp_paths(child)[int(rng.integers(len(_gp_paths(child))))]
            child = _gp_replace(child, path, _gp_random_tree(m, rng, 0))
        if frozenset(_gp_support(child)) in individuals and m > 1:
            # Same rule as the bitstring arm: a duplicate structure costs
            # nothing, so re-seed one leaf to keep the budget doing work.
            path = _gp_paths(child)[int(rng.integers(len(_gp_paths(child))))]
            child = _gp_replace(child, path, _gp_leaf(m, rng))

        child_node, child_fit = score(child)
        worst_index = max(range(len(live)), key=lambda index: score(live[index])[1].objective)
        if child_fit.objective < score(live[worst_index])[1].objective:
            live[worst_index] = child_node
        else:
            live.append(child_node)
            if len(live) > population:
                del live[max(range(len(live)), key=lambda index: score(live[index])[1].objective)]

    best_node = min(live, key=lambda node: score(node)[1].objective) if live else None
    return {
        "generations": generations,
        "generation_cap": max_generations,
        "final_population": len(live),
        "distinct_structures": len(individuals),
        "best_expression": None if best_node is None else _gp_render(best_node, names),
    }


# --------------------------------------------------------------------------- #
# RIME operators on a relaxed selection vector
# --------------------------------------------------------------------------- #


def rime_relaxed(
    objective: SupportObjective,
    rng: np.random.Generator,
    *,
    population: int = 8,
    iterations: int = 60,
    reinitialise_every: int = 15,
) -> dict[str, Any]:
    """Soft and hard RIME on a relaxed selection vector in ``[0, 1]``.

    The objective is binary-valued, so the relaxation has to be hardened before
    it can be scored. Thresholding at 0.5 was measured to collapse the whole
    population onto one selection within a few steps (nine distinct selections
    in 480 attempts): every agent then re-scores a cached fit and the arm spends
    almost none of its budget. Hardening by sampling ``z_j ~ Bernoulli(r_j)``
    keeps the operator on the same continuous geometry while letting distinct
    selections be paid for, and stale agents are re-initialised so the soft-rime
    attraction does not end every trajectory at the same point.
    """
    m = objective.m
    agents = [rng.random(m) for _ in range(population)]

    best_vector = agents[0].copy()
    best_score = float("inf")
    steps = 0
    sampled = 0
    for step in range(iterations):
        if objective.remaining <= 0:
            break
        steps += 1
        cooling = np.sqrt((step + 1) / iterations)
        for index in range(population):
            if objective.remaining <= 0:
                break
            soft = best_vector + cooling * rng.random(m) * (best_vector - agents[index]) * np.abs(
                rng.standard_normal(m)
            )
            vector = np.clip(soft, 0.0, 1.0)
            if rng.random() < 0.5:
                vector = np.clip(np.where(rng.random(m) < 0.5, best_vector, vector), 0.0, 1.0)
            bits = rng.random(m) < vector
            support = tuple(int(position) for position in np.flatnonzero(bits))
            sampled += 1
            score = objective.evaluate(support).objective
            if score < best_score:
                best_vector, best_score = vector.copy(), score
            agents[index] = vector
        if reinitialise_every and (step + 1) % reinitialise_every == 0:
            agents[int(rng.integers(population))] = rng.random(m)
    return {
        "iterations": steps,
        "population": population,
        "sampled_selections": sampled,
        "best_relaxed_vector": [round(float(value), 6) for value in best_vector],
    }


# --------------------------------------------------------------------------- #
# QAOA over an approximate quadratic surrogate
# --------------------------------------------------------------------------- #


def qaoa_surrogate(
    objective: SupportObjective,
    rng: np.random.Generator,
    *,
    depth: int = 2,
    shots: int = 512,
    restarts: int = 3,
) -> dict[str, Any]:
    m = objective.m
    needed = m * (m + 1) // 2 + 1
    try:
        qubo, paid = build_quadratic_surrogate(objective.evaluate, m)
    except BudgetExhausted as exhausted:
        raise ArmAbstained(
            f"the quadratic surrogate needs {needed} exact fits of J; {exhausted}"
        ) from exhausted

    run = run_qaoa(qubo, depth=depth, shots=shots, rng=rng, restarts=restarts)
    before = objective.evaluations
    offered = 0
    answer = None
    for mask in run.candidates:
        if objective.remaining <= 0:
            break
        fit = objective.evaluate(mask_to_support(mask, m))
        offered += 1
        if answer is None or fit.objective < answer.objective:
            answer = fit
    # A candidate whose selection the surrogate build already scored is free, so
    # the budget was charged for fewer fits than there were candidates.
    purchased = objective.evaluations - before
    if answer is None:
        raise ArmAbstained(
            f"the surrogate spent the whole {objective.budget}-fit budget "
            f"({paid} fits) and no candidate could be scored"
        )

    surrogate_mask, surrogate_cost = qubo.argmin()
    names = objective.library.names
    return {
        # The answer is the best *scored candidate*, not the best selection the
        # surrogate build happened to fit. Fitting a surrogate evaluates every
        # singleton and every pair, so reporting the best of those would make
        # the arm's answer an exhaustive pair search and credit the circuit for
        # an enumeration it did not perform.
        "answer": answer,
        "surrogate_fits": paid,
        "surrogate_minimum": float(surrogate_cost),
        "surrogate_optimum_support": [names[index] for index in mask_to_support(surrogate_mask, m)],
        "depth": depth,
        "shots": shots,
        "angles": [float(angle) for angle in run.angles],
        "expected_cost": float(run.expected_cost),
        "optimal_cost": float(run.optimal_cost),
        "unique_sampled": len(run.samples),
        "candidates_offered": offered,
        "scored_new_fits": purchased,
        "probability_of_surrogate_optimum": run.probability_of(surrogate_mask),
        "rotations": run.rotations,
    }


ARMS: dict[str, Callable[..., dict[str, Any]]] = {
    "random_restart": random_restart,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
    "boolean_gp": boolean_gp,
    "rime_relaxed": rime_relaxed,
    "qaoa_surrogate": qaoa_surrogate,
}


def run_arm(name: str, task, *, seed: int) -> ArmResult:
    """Run one arm on a fresh budgeted objective and seal what it measured.

    The arm only ever receives ``task.objective()``, which is built from the
    task's *development* rows; it cannot reach the holdout trajectory or the
    generating equation through this call, because the object it is handed does
    not contain them.
    """
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known arms are {sorted(ARMS)}")
    objective = task.objective()
    rng = np.random.default_rng(seed)
    status = SEARCHED
    reason = ""
    detail: dict[str, Any] = {}
    declared = None
    started = time.perf_counter()
    try:
        detail = ARMS[name](objective, rng) or {}
        # An arm may report a specific answer (``detail["answer"]``) when part of
        # what it evaluated was model fitting rather than a proposed hypothesis.
        declared = detail.pop("answer", None)
    except ArmAbstained as refusal:
        status, reason = ABSTAINED, str(refusal)
    except BudgetExhausted as exhausted:
        reason = f"stopped when the budget ran out: {exhausted}"
    elapsed = time.perf_counter() - started
    if status == SEARCHED:
        best = declared if declared is not None else objective.best
    else:
        # An abstention carries no candidate: fits spent on a surrogate the arm
        # then refused to build are a cost, not an answer.
        best = None
    return ArmResult(
        arm=name,
        status=status,
        reason=reason,
        best=best,
        evaluations=objective.evaluations,
        budget=objective.budget,
        seconds=elapsed,
        detail=detail,
    )


__all__ = [
    "ABSTAINED",
    "ARMS",
    "SEARCHED",
    "ArmAbstained",
    "ArmResult",
    "boolean_gp",
    "genetic_algorithm",
    "qaoa_surrogate",
    "random_restart",
    "rime_relaxed",
    "run_arm",
    "simulated_annealing",
]
