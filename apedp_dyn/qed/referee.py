"""Adjudication of a discovered equation, independent of the search that found it.

Status: experimental

The referee is the only part of this experiment that sees the holdout
trajectory, and it is the only part that is allowed to grade. It asks five
questions, in this order, and reports each on its own:

1. **Structural recovery** -- is the selected support the support the data were
   generated from?
2. **Forward prediction** -- integrating the candidate from the holdout initial
   condition, does it reproduce the holdout *discovered* coordinate
   (``R**2 >= HOLDOUT_R2_FLOOR``)? The companion coordinate is reported but not
   counted: ``dx/dt = v`` is supplied as a kinematic identity, so recovering
   ``x`` is not evidence about the discovered law.
3. **Parameters** -- does every true coefficient lie inside a residual-bootstrap
   interval computed at the candidate's own support?
4. **Parameter stability** -- do those coefficients survive refitting on
   subsamples?
5. **Ablation separability** -- is the candidate a local optimum under removal
   and addition of one term, measured on the holdout rollout and not on the
   search objective? A candidate whose term can be deleted at no cost is not
   separable from its own ablation, and that is reported rather than averaged
   away.

The status is deliberately two things at once: ``status`` says whether the
candidate predicts unseen data (``VALID`` / ``ALTERNATIVE`` / ``INVALID`` /
``ABSTAIN``) and ``recovered`` says whether the structure matched. An equation
that predicts the holdout as well as the truth but is written differently is an
``ALTERNATIVE`` -- a non-identifiability result, not a win and not a failure.

``HOLDOUT_R2_FLOOR`` and ``STABILITY_TOLERANCE`` are **experiment-local**
thresholds for QED-001. They are not in
``specifications/canonical_preregistered_spec_v1.json`` and must never be cited
as preregistered values: adding a quantity there is a specification change
(AGENTS.md 1.3), and this experiment has not made one.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.qed.library import TermLibrary
from apedp_dyn.qed.objective import Fit
from apedp_dyn.qed.tasks import DiscoveryTask

VALID = "VALID"
ALTERNATIVE = "ALTERNATIVE"
INVALID = "INVALID"
ABSTAIN = "ABSTAIN"

HOLDOUT_R2_FLOOR = 0.98
STABILITY_TOLERANCE = 0.2
SEPARATION_TOLERANCE = 1e-4
BOOTSTRAP_SAMPLES = 400
STABILITY_REPLICATES = 16
SUBSAMPLE_FRACTION = 0.8
PREREGISTRATION_NOTE = (
    "experiment-local thresholds; not part of the canonical preregistered specification"
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    value: object
    requirement: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "value": self.value,
            "requirement": self.requirement,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Adjudication:
    arm: str
    status: str
    recovered: bool
    support: tuple[str, ...]
    coefficients: tuple[float, ...]
    holdout_r2: float | None
    companion_r2: float | None
    checks: tuple[Check, ...]
    ablation: dict[str, Any]
    interval: dict[str, Any]
    stability: float | None
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.status in (VALID, ALTERNATIVE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "status": self.status,
            "recovered": bool(self.recovered),
            "support": list(self.support),
            "coefficients": [float(value) for value in self.coefficients],
            "holdout_r2": None if self.holdout_r2 is None else float(self.holdout_r2),
            "companion_r2": None if self.companion_r2 is None else float(self.companion_r2),
            "checks": [check.as_dict() for check in self.checks],
            "ablation": dict(self.ablation),
            "interval": dict(self.interval),
            "stability": None if self.stability is None else float(self.stability),
            "notes": list(self.notes),
        }


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    spread = float(np.sum((observed - observed.mean()) ** 2))
    if spread <= 0:
        return 0.0
    return 1.0 - residual / spread


def _fit(task: DiscoveryTask, states: np.ndarray, target: np.ndarray, support: Sequence[int]) -> Any:
    design = task.library.design_matrix(states)
    columns = task.library.select(design, support)
    if columns.shape[1] == 0:
        return np.zeros(0)
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(columns, target, rcond=None)
    return np.asarray(coefficients, dtype=float)


def rollout_scores(
    task: DiscoveryTask, support: Sequence[int], coefficients: Sequence[float]
) -> dict[str, Any]:
    """Integrate a candidate from the holdout initial condition and score it.

    A diverged rollout scores ``0.0`` and says so in ``diverged``: the number is
    kept finite for the record, and the flag is what a reader should believe.
    A wrong polynomial equation diverges far more often than it merely misses.
    """
    support = tuple(int(index) for index in support)
    run = task.rollout(
        task.holdout_states[0],
        support,
        tuple(float(value) for value in coefficients),
        steps=task.holdout_states.shape[0],
    )
    if run.diverged:
        return {
            "discovered_r2": 0.0,
            "companion_r2": None,
            "drifting": True,
            "diverged": True,
            "finite_rows": run.finite_rows,
            "rows": int(run.states.shape[0]),
        }
    states = run.states
    discovered = _r2(task.holdout_states[:, task.discovered], states[:, task.discovered])
    companion = [
        _r2(task.holdout_states[:, index], states[:, index])
        for index in range(task.width)
        if index != task.discovered
    ]
    return {
        "discovered_r2": float(discovered),
        "companion_r2": float(min(companion)) if companion else None,
        "drifting": bool(not np.all(np.isfinite(states)) or np.max(np.abs(states)) > 1e3),
        "diverged": False,
        "finite_rows": run.finite_rows,
        "rows": int(states.shape[0]),
    }


def bootstrap_interval(
    task: DiscoveryTask,
    support: Sequence[int],
    coefficients: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Residual bootstrap at the candidate's own support."""
    support = tuple(int(index) for index in support)
    if not support:
        return {"support": [], "level": level, "samples": samples, "bounds": []}
    design = task.library.design_matrix(task.dev_states)
    columns = task.library.select(design, support)
    fitted = columns @ np.asarray(coefficients, dtype=float)
    residual = task.dev_target - fitted
    rng = np.random.default_rng(seed)
    draws: np.ndarray = np.empty((samples, len(support)), dtype=float)
    for index in range(samples):
        synthetic = fitted + rng.choice(residual, size=residual.size, replace=True)
        drawn, _residuals, _rank, _singular = np.linalg.lstsq(columns, synthetic, rcond=None)
        draws[index] = np.asarray(drawn, dtype=float)
    tail = (1.0 - level) / 2.0
    bounds = [
        (float(np.quantile(draws[:, column], tail)), float(np.quantile(draws[:, column], 1 - tail)))
        for column in range(len(support))
    ]
    return {
        "support": [task.library.names[index] for index in support],
        "level": level,
        "samples": samples,
        "bounds": bounds,
    }


def parameter_stability(
    task: DiscoveryTask,
    support: Sequence[int],
    coefficients: Sequence[float],
    *,
    replicates: int = STABILITY_REPLICATES,
    fraction: float = SUBSAMPLE_FRACTION,
    seed: int = 0,
    tolerance: float = STABILITY_TOLERANCE,
) -> float:
    """Fraction of subsample refits that stay within ``tolerance`` relative error.

    Support *re-selection* on a subsample is the proposer's variance, not the
    referee's: the referee holds the proposed structure fixed and asks whether
    the numbers survive. A structure that needs a different selection on 20% of
    the data is a finding about the task, and the arms' spread reports it.
    """
    support = tuple(int(index) for index in support)
    if not support:
        return 0.0
    reference: np.ndarray = np.asarray(coefficients, dtype=float)
    design = task.library.design_matrix(task.dev_states)
    columns = task.library.select(design, support)
    rng = np.random.default_rng(seed)
    rows = task.dev_target.size
    keep = max(2, round(fraction * rows))
    stable = 0
    for _ in range(replicates):
        chosen = np.sort(rng.choice(rows, size=keep, replace=False))
        drawn, _residuals, _rank, _singular = np.linalg.lstsq(columns[chosen], task.dev_target[chosen], rcond=None)
        drawn = np.asarray(drawn, dtype=float)
        scale = np.maximum(np.abs(reference), 1e-12)
        if np.all(np.abs(drawn - reference) <= tolerance * scale):
            stable += 1
    return stable / replicates


def ablation(
    task: DiscoveryTask,
    support: Sequence[int],
    coefficients: Sequence[float],
    *,
    candidate_r2: float,
) -> dict[str, Any]:
    """Remove one term, add one term, and score each on the holdout rollout."""
    support = tuple(int(index) for index in support)
    removals = []
    for index in support:
        reduced = tuple(term for term in support if term != index)
        reduced_coefficients = _fit(task, task.dev_states, task.dev_target, reduced)
        scores = rollout_scores(task, reduced, reduced_coefficients)
        removals.append(
            {
                "removed": task.library.names[index],
                "support": [task.library.names[term] for term in reduced],
                "holdout_r2": scores["discovered_r2"],
                "gap": float(candidate_r2 - scores["discovered_r2"]),
            }
        )
    additions = []
    for index in range(len(task.library.terms)):
        if index in support:
            continue
        extended = tuple(sorted(set(support) | {index}))
        extended_coefficients = _fit(task, task.dev_states, task.dev_target, extended)
        scores = rollout_scores(task, extended, extended_coefficients)
        additions.append(
            {
                "added": task.library.names[index],
                "support": [task.library.names[term] for term in extended],
                "holdout_r2": scores["discovered_r2"],
                "gain": float(scores["discovered_r2"] - candidate_r2),
            }
        )
    smallest_gap = min((row["gap"] for row in removals), default=float("inf"))
    largest_gain = max((row["gain"] for row in additions), default=float("-inf"))
    return {
        "removals": removals,
        "additions": additions,
        "smallest_removal_gap": None if smallest_gap == float("inf") else float(smallest_gap),
        "largest_addition_gain": None if largest_gain == float("-inf") else float(largest_gain),
        "tolerance": SEPARATION_TOLERANCE,
        "separated": bool(
            smallest_gap > SEPARATION_TOLERANCE and largest_gain <= SEPARATION_TOLERANCE
        ),
    }


def reference_scores(task: DiscoveryTask) -> dict[str, Any]:
    """The truth's own holdout score, for the record and for the tests."""
    scores = rollout_scores(task, task.truth_support, task.truth_coefficients)
    return {
        "support": list(task.truth_names),
        "coefficients": [float(value) for value in task.truth_coefficients],
        **scores,
    }


def adjudicate(
    task: DiscoveryTask,
    result: Any,
    *,
    seed: int = 0,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    stability_replicates: int = STABILITY_REPLICATES,
) -> Adjudication:
    """Grade one arm's best selection, or record that it abstained."""
    library: TermLibrary = task.library
    if not result.searched or result.best is None:
        return Adjudication(
            arm=result.arm,
            status=ABSTAIN,
            recovered=False,
            support=(),
            coefficients=(),
            holdout_r2=None,
            companion_r2=None,
            checks=(
                Check(
                    name="searched",
                    passed=False,
                    value=result.evaluations,
                    requirement="the arm must spend at least one fit",
                    detail=result.reason,
                ),
            ),
            ablation={},
            interval={},
            stability=None,
            notes=(f"no candidate: {result.reason or 'the arm returned nothing'}",),
        )

    best: Fit = result.best
    support = tuple(best.support)
    coefficients = tuple(float(value) for value in best.coefficients)
    names = library.names
    scores = rollout_scores(task, support, coefficients)
    candidate_r2 = scores["discovered_r2"]

    recovered = set(support) == set(task.truth_support)
    holdout_passed = candidate_r2 >= HOLDOUT_R2_FLOOR
    status = VALID if (recovered and holdout_passed) else (ALTERNATIVE if holdout_passed else INVALID)

    interval = bootstrap_interval(
        task, support, coefficients, samples=bootstrap_samples, seed=seed
    )
    truth_by_name = dict(
        zip(
            (names[index] for index in task.truth_support),
            (float(value) for value in task.truth_coefficients),
            strict=True,
        )
    )
    interval_by_name = dict(
        zip(interval.get("support", []), interval.get("bounds", []), strict=True)
    )
    parameters_inside = {}
    for name, truth_value in truth_by_name.items():
        bound = interval_by_name.get(name)
        if bound is None:
            parameters_inside[name] = False
        else:
            parameters_inside[name] = bool(bound[0] <= truth_value <= bound[1])

    stability = parameter_stability(
        task, support, coefficients, replicates=stability_replicates, seed=seed
    )
    ablation_report = ablation(task, support, coefficients, candidate_r2=candidate_r2)

    checks = (
        Check(
            name="structural_recovery",
            passed=bool(recovered),
            value=[names[index] for index in support],
            requirement=f"the support must equal {list(task.truth_names)}",
        ),
        Check(
            name="forward_prediction",
            passed=bool(holdout_passed),
            value=float(candidate_r2),
            requirement=f"holdout R^2 >= {HOLDOUT_R2_FLOOR} on the discovered coordinate",
            detail=f"companion coordinate R^2 {scores['companion_r2']}; "
            f"dx/dt = v is supplied, so it is not counted",
        ),
        Check(
            name="parameters_inside_interval",
            passed=bool(parameters_inside) and all(parameters_inside.values()),
            value=parameters_inside,
            requirement="every true coefficient inside its bootstrap interval",
            detail=f"{round(interval.get('level', 0) * 100)}% residual bootstrap, "
            f"{interval.get('samples', 0)} draws",
        ),
        Check(
            name="parameter_stability",
            passed=bool(stability >= SUBSAMPLE_FRACTION),
            value=float(stability),
            requirement=f"at least {SUBSAMPLE_FRACTION:.0%} of subsample refits within "
            f"{STABILITY_TOLERANCE:.0%} of the full-data coefficients",
        ),
        Check(
            name="ablation_separability",
            passed=bool(ablation_report["separated"]),
            value={
                "smallest_removal_gap": ablation_report["smallest_removal_gap"],
                "largest_addition_gain": ablation_report["largest_addition_gain"],
            },
            requirement=(
                f"removing any term must cost more than {SEPARATION_TOLERANCE:g} of holdout "
                f"R^2, and adding any term must gain no more than that"
            ),
        ),
    )

    notes = []
    if not recovered:
        notes.append(
            "the support is not the generating support; forward prediction alone cannot "
            "distinguish this equation from the truth on this holdout"
        )
    if not ablation_report["separated"]:
        notes.append(
            "the candidate is not ablation-separable on the holdout: at least one term can be "
            "removed at no cost, or one added for nothing"
        )
    if scores.get("diverged"):
        notes.append(
            f"the rollout left the finite range after {scores.get('finite_rows')} of "
            f"{scores.get('rows')} rows; its R^2 is recorded as 0.0 and means nothing"
        )

    return Adjudication(
        arm=result.arm,
        status=status,
        recovered=bool(recovered),
        support=tuple(names[index] for index in support),
        coefficients=coefficients,
        holdout_r2=float(candidate_r2),
        companion_r2=scores["companion_r2"],
        checks=checks,
        ablation=ablation_report,
        interval=interval,
        stability=float(stability),
        notes=tuple(notes),
    )


__all__ = [
    "ABSTAIN",
    "ALTERNATIVE",
    "BOOTSTRAP_SAMPLES",
    "HOLDOUT_R2_FLOOR",
    "INVALID",
    "PREREGISTRATION_NOTE",
    "SEPARATION_TOLERANCE",
    "STABILITY_REPLICATES",
    "STABILITY_TOLERANCE",
    "SUBSAMPLE_FRACTION",
    "VALID",
    "Adjudication",
    "Check",
    "ablation",
    "adjudicate",
    "bootstrap_interval",
    "parameter_stability",
    "reference_scores",
    "rollout_scores",
]
