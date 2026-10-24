# Databricks notebook source
"""Ablation ladder: measure what each validation layer contributes.

A rung is a named candidate set produced under a cumulative layer stack
(e.g. raw search, +dimensions, +grouped holdout, +transfer, +identifiability).
The harness replays selection at every rung and records *what changed*,
turning "the validation stack helps" into per-layer measured deltas.

:func:`selection_report` (item 1) renders the fit-cost ledger: every
candidate's raw fit error alongside the constraints it satisfies, so the
price of admissibility -- best unconstrained error versus selected error --
is explicit rather than a hidden tradeoff.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from math import inf
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RungEvaluation:
    """One candidate's measured record at one rung."""

    expression: str
    train_error: float | None
    test_error: float | None
    constraints_satisfied: tuple[str, ...] = ()
    constraints_failed: tuple[str, ...] = ()
    refused: bool = False
    detail: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "train_error": self.train_error,
            "test_error": self.test_error,
            "constraints_satisfied": list(self.constraints_satisfied),
            "constraints_failed": list(self.constraints_failed),
            "refused": self.refused,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class RungResult:
    """Selection outcome at one rung of the ladder."""

    rung: str
    layers_active: tuple[str, ...]
    evaluations: tuple[RungEvaluation, ...]
    selected: str | None
    best_unconstrained_error: float | None
    selected_error: float | None
    admissibility_price: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "layers_active": list(self.layers_active),
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "selected": self.selected,
            "best_unconstrained_error": self.best_unconstrained_error,
            "selected_error": self.selected_error,
            "admissibility_price": self.admissibility_price,
        }


EvaluateFn = Callable[[str], RungEvaluation]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sort_key(value: Any) -> float:
    """Order candidates by error; missing or non-finite errors sort last.

    ``min`` needs a totally ordered key, but the error fields are optional.
    Callers filter non-finite entries out before ranking, so this only has to
    stay total rather than faithful for the dropped entries.
    """
    number = _finite(value)
    return number if number is not None else inf


def selection_report(evaluations: Iterable[RungEvaluation]) -> dict[str, Any]:
    """Fit-cost ledger: raw error per candidate plus constraint satisfaction.

    ``admissibility_price`` is the gap between the best raw test error among
    all evaluated candidates and the test error of the best *admitted*
    (non-refused) candidate. Positive means constraints cost fit quality;
    zero means the winner satisfies everything; ``None`` means no admission.
    """
    items = list(evaluations)
    test_errors = [(item.expression, item.test_error) for item in items if _finite(item.test_error) is not None]
    admitted = [(item.expression, item.test_error) for item in items if not item.refused and _finite(item.test_error) is not None]
    best_raw = min(test_errors, key=lambda pair: _sort_key(pair[1])) if test_errors else (None, None)
    best_adm = min(admitted, key=lambda pair: _sort_key(pair[1])) if admitted else (None, None)
    price = None
    if best_raw[1] is not None and best_adm[1] is not None:
        price = float(best_adm[1] - best_raw[1])
    return {
        "candidates": [item.to_dict() for item in items],
        "best_unconstrained": best_raw[0],
        "best_unconstrained_error": best_raw[1],
        "selected": best_adm[0],
        "selected_error": best_adm[1],
        "admissibility_price": price,
    }


def run_ablation_ladder(
    rungs: Iterable[tuple[str, tuple[str, ...], Iterable[str]] | tuple[str, tuple[str, ...], Iterable[str], str]],
    evaluate: EvaluateFn,
) -> dict[str, Any]:
    """Replay selection cumulatively: ``(rung_name, layers_active, candidates)``.

    An optional fourth element selects the ranking metric: ``"test"``
    (default; what a holdout layer would crown) or ``"train"`` (what raw
    search without a holdout crowns). The raw rung should use ``"train"``:
    that is precisely the overfitting trap the holdout layer exists to fix.

    The same ``evaluate`` callback runs at every rung; only the candidate set
    (shaped by the layers active at that rung) and the ranking metric change.
    Returns per-rung results plus the cross-rung deltas: which rung first
    changed the selection, and each layer stack's admissibility price.
    """
    results: list[RungResult] = []
    for rung in rungs:
        rung_name, layers_active, candidates = rung[0], rung[1], rung[2]
        select = rung[3] if len(rung) > 3 else "test"
        if select not in ("train", "test"):
            raise ValueError("rung selection rule must be 'train' or 'test'")
        evaluations = tuple(evaluate(expression) for expression in candidates)
        # Bind the metric as a default argument rather than closing over `select`:
        # a lambda capturing the loop variable would take the *final* rung's
        # metric for every rung once the closure escaped the iteration.
        def key(item: RungEvaluation, _select: str = select) -> float:
            return _sort_key(item.test_error if _select == "test" else item.train_error)

        scored = [(item.expression, key(item)) for item in evaluations if _finite(key(item)) is not None]
        selected = min(scored, key=lambda pair: pair[1])[0] if scored else None
        ledger = selection_report(evaluations)
        if selected is not None:
            chosen = next(item for item in evaluations if item.expression == selected)
            selected_error = _finite(key(chosen))
        else:
            selected_error = None
        # admissibility_price stays test-based (ledger) at every rung so the
        # price of constraints is comparable across rungs; selected follows
        # the rung's own ranking metric.
        results.append(
            RungResult(
                rung=rung_name,
                layers_active=tuple(layers_active),
                evaluations=evaluations,
                selected=selected,
                best_unconstrained_error=ledger["best_unconstrained_error"],
                selected_error=selected_error,
                admissibility_price=ledger["admissibility_price"],
            )
        )
    selections = [result.selected for result in results]
    first_change: str | None = None
    for index in range(1, len(selections)):
        if selections[index] != selections[0]:
            first_change = results[index].rung
            break
    return {
        "rungs": [result.to_dict() for result in results],
        "selections": selections,
        "first_selection_change": first_change,
        "layers_measured": sorted({layer for result in results for layer in result.layers_active}),
    }


__all__ = ["RungEvaluation", "RungResult", "run_ablation_ladder", "selection_report"]
