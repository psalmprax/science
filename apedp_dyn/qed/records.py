"""The QED-001 record: what was measured, at which budget, and what it does not show.

Status: experimental

The record is organised as *conditions* -- the same task searched under different
per-arm budgets -- and every number in it can be recomputed from its own rows:

* the task's library, formulation, noise level and generating equation;
* an exhaustive sweep of the whole selection space, run by the harness outside
  every search budget, which establishes that the penalised optimum *is* the
  generating support and how far the runner-up is;
* one row per arm per seed, with the budget it was given, the fits it spent, the
  selection it returned and its adjudication;
* a cross-condition table, because a method that ties at one budget and diverges
  at another is a different finding from one that ties at both;
* an advantage block computed from those rows rather than written by hand.

An arm that abstains is recorded as an abstention with its reason, never as a
loss. A tie is recorded as a tie.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np

from apedp_dyn.qed.arms import ARMS, run_arm
from apedp_dyn.qed.objective import BudgetExhausted
from apedp_dyn.qed.referee import (
    ABSTAIN,
    ALTERNATIVE,
    HOLDOUT_R2_FLOOR,
    INVALID,
    PREREGISTRATION_NOTE,
    STABILITY_TOLERANCE,
    VALID,
    adjudicate,
    reference_scores,
)
from apedp_dyn.qed.tasks import DiscoveryTask

RECORD = "qed_001_recovery"
SCHEMA = "qed.recovery/1"

LIMITATIONS: tuple[str, ...] = (
    "One synthetic system, one ten-term library, one noise level and one time step. "
    "Nothing here calibrates difficulty or transfers to another library.",
    "The derivative is a centred finite difference of the observed velocity, so noise is "
    "amplified by differentiation. A smoothing or total-variation derivative estimator "
    "would change every number and was not measured.",
    "Only the acceleration equation is discovered. The kinematic identity dx/dt = v is "
    "supplied, which is why the companion coordinate is reported and not counted.",
    "The task is fair only in the benchmark sense: an exhaustive sweep confirms the "
    "penalised optimum is the generating support by a measured margin. A real discovery "
    "has no such oracle, so this is not evidence that discovery is easy in general.",
    "The QAOA arm optimises a quadratic surrogate of a non-quadratic objective and runs on "
    "a dense statevector simulator. No hardware, no sampling noise in the angles, no "
    "device calibration, and the surrogate is lossy by construction -- the record keeps "
    "both the surrogate's minimiser and the selection the arm actually scored.",
    "No quantum-advantage claim. The quantum arm's search space is ten bits and its circuit "
    "is simulated classically, so any comparison here concerns search quality at a fixed "
    "budget, never computational speed.",
    "Approach C of the research programme (a quality-diversity sampler over circuits) was "
    "not tested. Only the differentiable-circuit and combinatorial-search ideas were.",
    "RIME is used as an operator set on a relaxed selection vector, not through the native "
    "kernel in apedp_dyn.proposers.rime_proposer, and the GP arm evolves a set-building "
    "language rather than arithmetic expression trees. Both are stated in their modules.",
    "Three seeds per arm per condition. That is enough to see a broken arm and not enough "
    "to estimate a per-arm variance, so the record reports counts and medians, not "
    "significance tests.",
    "More than one budget is measured, but the space is the same; the record therefore says "
    "how the arms rank at those budgets and nothing about the budget where they rank "
    "stably.",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes(root: Path, paths: Sequence[str]) -> dict:
    return {path: _sha256(root / path) for path in paths}


def oracle_sweep(task: DiscoveryTask) -> dict:
    """Enumerate the whole selection space, outside every search budget.

    This is the harness's reference computation, not a competing proposer: it
    establishes whether the task is fair (the optimum is the generating support)
    and how far the runner-up is, which is what makes a missed optimum
    interpretable as a search failure rather than a badly posed problem.
    """
    objective = task.objective(budget=2 ** len(task.library))
    fits = []
    for mask in range(2 ** len(task.library)):
        support = tuple(index for index in range(len(task.library)) if (mask >> index) & 1)
        try:
            fits.append(objective.evaluate(support))
        except BudgetExhausted:  # pragma: no cover - the budget is the whole space
            break
    fits.sort(key=lambda fit: (fit.objective, fit.support))
    best, runner_up = fits[0], fits[1]
    margin = float((runner_up.objective - best.objective) / runner_up.objective)
    names = task.library.names
    truth_objective = next(
        fit.objective for fit in fits if set(fit.support) == set(task.truth_support)
    )
    return {
        "space": 2 ** len(task.library),
        "fits": objective.evaluations,
        "best": best.as_dict(names),
        "runner_up": runner_up.as_dict(names),
        "margin": margin,
        "agrees_with_truth": set(best.support) == set(task.truth_support),
        "truth_objective": float(truth_objective),
        "note": "harness reference, outside every search budget; not a proposer",
    }


def _task_block(task: DiscoveryTask) -> dict:
    return {
        "name": task.name,
        "derivation": task.derivation,
        "library": list(task.library.names),
        "dt": task.dt,
        "discovered_coordinate": task.discovered,
        "kinematic_pairs": [list(pair) for pair in task.kinematic_pairs],
        "development_rows": int(task.dev_states.shape[0]),
        "holdout_rows": int(task.holdout_states.shape[0]),
        "noise_level": task.noise_level,
        "seed": task.seed,
        "penalty": task.penalty,
        "budget_per_arm": task.budget,
        "truth_support": list(task.truth_names),
        "truth_coefficients": [float(value) for value in task.truth_coefficients],
        "development_ic": [float(value) for value in task.dev_states[0]],
        "holdout_ic": [float(value) for value in task.holdout_states[0]],
        "reference": reference_scores(task),
    }


def _arm_rows(task: DiscoveryTask, arms: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    rows = []
    for arm in arms:
        runs = []
        for seed in seeds:
            result = run_arm(arm, task, seed=seed)
            verdict = adjudicate(task, result, seed=seed)
            runs.append(
                {
                    "seed": seed,
                    "result": result.as_dict(task.library),
                    "adjudication": verdict.as_dict(),
                }
            )
        objectives = [
            row["result"]["best"]["objective"] for row in runs if row["result"]["best"] is not None
        ]
        r2s = [
            row["adjudication"]["holdout_r2"]
            for row in runs
            if row["adjudication"]["holdout_r2"] is not None
        ]
        statuses = dict.fromkeys((VALID, ALTERNATIVE, INVALID, ABSTAIN), 0)
        for row in runs:
            statuses[row["adjudication"]["status"]] += 1
        rows.append(
            {
                "arm": arm,
                "runs": runs,
                "searched_runs": sum(1 for row in runs if row["result"]["status"] == "searched"),
                "recovered_runs": sum(1 for row in runs if row["adjudication"]["recovered"]),
                "statuses": statuses,
                "mean_objective": float(np.mean(objectives)) if objectives else None,
                "best_objective": float(np.min(objectives)) if objectives else None,
                "mean_holdout_r2": float(np.mean(r2s)) if r2s else None,
                "evaluations_per_run": [row["result"]["evaluations"] for row in runs],
                "support_names": list(task.library.names),
            }
        )
    return rows


def _comparison(rows: Sequence[dict], oracle: dict) -> dict:
    oracle_objective = float(oracle["best"]["objective"])
    hits = {
        row["arm"]: sum(
            1
            for run in row["runs"]
            if run["result"]["best"] is not None
            and abs(run["result"]["best"]["objective"] - oracle_objective) <= 1e-12
        )
        for row in rows
    }
    classical = [row for row in rows if row["arm"] != "qaoa_surrogate"]
    quantum = next((row for row in rows if row["arm"] == "qaoa_surrogate"), None)
    best_classical = min(
        (row["best_objective"] for row in classical if row["best_objective"] is not None),
        default=None,
    )
    quantum_best = None if quantum is None else quantum["best_objective"]
    classical_recoveries = [row["recovered_runs"] for row in classical]
    best_classical_recoveries = max(classical_recoveries, default=None)
    if quantum_best is None or best_classical is None:
        advantage = {
            "claimed": False,
            "reason": "the quantum arm or every classical arm produced no candidate",
        }
    else:
        advantage = {
            "claimed": False,
            "quantum_best_objective": float(quantum_best),
            "best_classical_objective": float(best_classical),
            "quantum_minus_classical": float(quantum_best - best_classical),
            # Best-of-k hides a per-run difference, so the recovery counts are
            # reported beside it: an arm that is lucky once is not an arm that
            # works.
            "quantum_recovered_runs": quantum["recovered_runs"],
            "best_classical_recovered_runs": best_classical_recoveries,
            "runs_per_arm": len(quantum["runs"]),
            "reason": (
                "no advantage is claimed: the quantum arm's circuit is simulated classically "
                "on ten selection bits, so this is search quality at a fixed budget, not a "
                "computational speedup"
            ),
        }
    matching = sorted(arm for arm, count in hits.items() if count)
    tied = len(matching) > 1
    return {
        "oracle_best_objective": oracle_objective,
        "oracle_hits_by_arm": hits,
        "arms_matching_the_oracle": matching,
        "arms_tied_at_the_oracle": tied,
        "advantage": advantage,
    }


def circuit_probe(task: DiscoveryTask, seed: int, *, depth: int = 2, shots: int = 512) -> dict:
    """Re-run the quantum arm's circuit from its own seed and ask what it believed.

    The arm cannot answer this: it does not know the generating support. The
    harness can, because it does -- and the answer decides whether a recovered
    equation was found *by the circuit* or merely *among its samples*. The
    surrogate is rebuilt from the same ``1 + m + m(m-1)/2`` fits in the same
    order and the generator is seeded identically, so the distribution is the
    arm's own, not a lookalike. This probe spends no search budget: it is a
    diagnostic about a circuit, not a proposer.
    """
    from apedp_dyn.qed.qaoa import (
        build_quadratic_surrogate,
        mask_to_support,
        run_qaoa,
        support_to_mask,
    )

    objective = task.objective(budget=len(task.library) * (len(task.library) + 1) // 2 + 1)
    qubo, paid = build_quadratic_surrogate(objective.evaluate, objective.m)
    run = run_qaoa(qubo, depth=depth, shots=shots, rng=np.random.default_rng(seed))
    probabilities = np.asarray(run.probabilities, dtype=float)
    truth_mask = support_to_mask(task.truth_support, objective.m)
    top = int(np.argmax(probabilities))
    return {
        "seed": seed,
        "depth": run.depth,
        "shots": run.shots,
        "space": int(probabilities.size),
        "uniform_probability": float(1.0 / probabilities.size),
        "truth_probability": float(probabilities[truth_mask]),
        "truth_rank": int(np.sum(probabilities > probabilities[truth_mask])) + 1,
        "top_selection": [task.library.names[index] for index in mask_to_support(top, objective.m)],
        "top_probability": float(probabilities[top]),
        "effective_selections": float(1.0 / np.sum(probabilities**2)),
        "surrogate_minimum": float(qubo.argmin()[1]),
        "surrogate_fits": paid,
    }


def _surrogate_block(rows: Sequence[dict], task: DiscoveryTask) -> dict | None:
    """What the quantum arm's approximation actually optimised.

    The surrogate is quadratic and ``J`` is not, so the interesting question is
    not whether QAOA ran but whether the thing it minimised is the thing the
    experiment wanted. This block answers that by comparing the surrogate's own
    minimiser with the generating support -- and reports the case where they
    differ, which is a measurement about the surrogate rather than about the
    circuit.
    """
    row = next((entry for entry in rows if entry["arm"] == "qaoa_surrogate"), None)
    if row is None:
        return None
    truth = sorted(task.truth_names)
    runs = []
    for run in row["runs"]:
        detail = run["result"]["detail"] or {}
        if not detail:
            runs.append(
                {
                    "seed": run["seed"],
                    "status": run["result"]["status"],
                    "reason": run["result"]["reason"],
                }
            )
            continue
        optimum = sorted(detail["surrogate_optimum_support"])
        runs.append(
            {
                "seed": run["seed"],
                "surrogate_fits": detail["surrogate_fits"],
                "surrogate_optimum": optimum,
                "surrogate_minimum": detail["surrogate_minimum"],
                "surrogate_optimum_matches_truth": optimum == truth,
                "probability_of_surrogate_optimum": detail["probability_of_surrogate_optimum"],
                "candidates_offered": detail["candidates_offered"],
                "scored_new_fits": detail["scored_new_fits"],
                "unique_sampled": detail["unique_sampled"],
                "shots": detail["shots"],
                "depth": detail["depth"],
                "rotations_total": detail["rotations"]["rotations_total"],
                "best_scored_support": (
                    None if run["result"]["best"] is None else run["result"]["best"]["support"]
                ),
            }
        )
    probes = [circuit_probe(task, entry["seed"]) for entry in runs if entry.get("seed") is not None]
    return {
        "arm": "qaoa_surrogate",
        "surrogate_fits_per_run": [entry.get("surrogate_fits") for entry in runs],
        "runs": runs,
        "surrogate_optimum_matched_truth": sum(
            1 for entry in runs if entry.get("surrogate_optimum_matches_truth")
        ),
        "circuit": probes,
        "truth_probability_below_uniform_runs": sum(
            1
            for probe in probes
            if probe["truth_probability"] <= 2.0 * probe["uniform_probability"]
        ),
        "note": (
            "the surrogate is a quadratic approximation of a non-quadratic objective; its "
            "minimiser is reported separately from the best selection the arm scored, "
            "and the circuit probe records what probability the circuit itself put on the "
            "generating support, which the arm cannot compute because it does not know it"
        ),
    }


def build_condition(
    task: DiscoveryTask,
    *,
    arms: Sequence[str] | None = None,
    seeds: Sequence[int] = (11, 12, 13),
) -> dict:
    """One budget condition: the task, the fairness sweep, the rows and the checks."""
    arms = tuple(arms or ARMS)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise KeyError(f"unknown arms: {unknown}")

    oracle = oracle_sweep(task)
    rows = _arm_rows(task, arms, seeds)
    comparison = _comparison(rows, oracle)
    tag = f"budget_{task.budget}:"

    checks = [
        {
            "name": tag + "task_is_fair",
            "passed": bool(oracle["agrees_with_truth"]),
            "value": oracle["margin"],
            "requirement": "the penalised optimum over the whole space is the generating support",
        },
        {
            "name": tag + "oracle_margin_is_measured",
            "passed": oracle["margin"] > 0,
            "value": oracle["margin"],
            "requirement": "the runner-up is strictly worse, so a miss is a search failure",
        },
        {
            "name": tag + "search_is_required",
            "passed": task.budget < oracle["space"],
            "value": {"budget": task.budget, "space": oracle["space"]},
            "requirement": "the per-arm budget is smaller than the space it searches",
        },
        {
            "name": tag + "every_arm_respects_the_budget",
            "passed": all(
                evaluation <= task.budget
                for row in rows
                for evaluation in row["evaluations_per_run"]
            ),
            "value": {row["arm"]: row["evaluations_per_run"] for row in rows},
            "requirement": "no arm spent more than the declared budget",
        },
    ]
    return {
        "budget": int(task.budget),
        "space": int(oracle["space"]),
        "task": _task_block(task),
        "oracle": oracle,
        "arms": rows,
        "comparison": comparison,
        "surrogate": _surrogate_block(rows, task),
        "checks": checks,
    }


def cross_condition(conditions: Sequence[dict]) -> dict:
    """The same arms under different budgets, so a tie can be read against a harder one."""
    table: dict[str, list[dict]] = {}
    for condition in conditions:
        for row in condition["arms"]:
            table.setdefault(row["arm"], []).append(
                {
                    "budget": condition["budget"],
                    "best_objective": row["best_objective"],
                    "mean_objective": row["mean_objective"],
                    "recovered_runs": row["recovered_runs"],
                    "statuses": row["statuses"],
                    "evaluations_per_run": row["evaluations_per_run"],
                }
            )
    return {
        "conditions": [condition["budget"] for condition in conditions],
        "per_arm": table,
        "advantage_by_condition": [
            {
                "budget": condition["budget"],
                "quantum_minus_classical": condition["comparison"]["advantage"].get(
                    "quantum_minus_classical"
                ),
            }
            for condition in conditions
        ],
    }


def build_record(
    conditions: Sequence[dict],
    *,
    root: Path,
    script: str,
    modules: Sequence[str],
) -> dict:
    if not conditions:
        raise ValueError("a record needs at least one condition")
    cross = cross_condition(conditions)
    checks = [check for condition in conditions for check in condition["checks"]]
    checks.extend(
        [
            {
                "name": "no_advantage_claimed",
                "passed": all(
                    condition["comparison"]["advantage"]["claimed"] is False
                    for condition in conditions
                ),
                "value": cross["advantage_by_condition"],
                "requirement": "a simulated circuit at ten bits cannot support a speedup claim",
            },
            {
                "name": "holdout_is_never_used_by_a_proposer",
                "passed": True,
                "value": "structural: arms receive a SupportObjective over development rows only",
                "requirement": "the holdout trajectory is reachable only from the referee",
            },
        ]
    )
    return {
        "record": RECORD,
        "schema": SCHEMA,
        "generated_on": date.today().isoformat(),
        "purpose": (
            "Does a quantum-assisted proposer recover a governing equation from noisy "
            "observations as well as the classical proposers, when every arm is held to the "
            "same number of objective fits?"
        ),
        "thresholds": {
            "holdout_r2_floor": HOLDOUT_R2_FLOOR,
            "stability_tolerance": STABILITY_TOLERANCE,
            "note": PREREGISTRATION_NOTE,
        },
        "conditions": list(conditions),
        "cross_condition": cross,
        "checks": checks,
        "limitations": list(LIMITATIONS),
        "reproducibility": {
            "script": script,
            "script_sha256": _sha256(root / script),
            "modules": code_hashes(root, modules),
            "python": "deterministic: every arm is seeded and every fit is numpy lstsq",
        },
    }


def write_record(record: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "LIMITATIONS",
    "RECORD",
    "SCHEMA",
    "build_condition",
    "build_record",
    "code_hashes",
    "cross_condition",
    "oracle_sweep",
    "write_record",
]
