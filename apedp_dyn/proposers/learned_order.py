"""An order of candidate laws that is learned from the system's own problems.

Status: experimental

**Why.** A proposer that fits constants hands the referee its candidates best fit first (BIC). On
the curriculum that order is wrong in a particular way: the list holds the law, or a law as short,
but over-fitted laws stand ahead of it. On five first problems an accurate candidate was first on
every list, and the first one no longer than the true law stood at places 1, 24 and 38. The route
confirms from the top of the list, and the product hands the referee three candidates, so a law at
place 24 is as good as not proposed.

**What learns.** A logistic model of "the judge would count this candidate solved", from what a
candidate looks like beside the others on its list: how far its fit is from the best fit, on the
rows it was fitted on and on rows it never saw, how many bits it takes to write and how many more
than the shortest on the list, its constants (how many, how many near nothing, how many huge), its
operations. The labels come from the curriculum, whose true law is known: the system can make as
many labelled problems as it likes, and nothing here reads a held-back law or a benchmark's.
Candidates are then handed over most likely first.

A rule that needs no learning stands beside it as a reference (``simplest_within``): among the
candidates whose fit on unseen rows is within a tolerance of the best, the shortest first. It is
the one-standard-error rule of model selection in this setting.

**It decides nothing.** The order of a list is all it changes; every candidate is still fitted,
confirmed and checked by the path as before (AGENTS.md 1.1). A model trained on one curriculum has
learned that curriculum's laws: whether its order helps on laws of another kind is a measurement
to be taken there. **Not in force**, and nothing builds it but its study script and tests.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FEATURES = ("place", "fit_gap", "held_gap", "held_below_best", "law_bits", "bits_above_shortest", "constants",
            "small_constants", "large_constants", "operations", "steps", "held_minus_train")
FLOOR = 1e-12
LABEL = "solved"


def _digits(r2: Any) -> float:
    """A fit as the number of leading nines of its R^2 (0.999 gives 3); a missing or hopeless fit gives 0."""
    if r2 is None or not np.isfinite(float(r2)):
        return 0.0
    return float(-np.log10(max(1.0 - float(r2), FLOOR))) if float(r2) < 1.0 else float(-np.log10(FLOOR))


def features(pool: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One row of ``FEATURES`` for each candidate of a list, each read beside the others on the list."""
    if not pool:
        return np.zeros((len(pool), len(FEATURES)), dtype=float)
    fit = np.array([_digits(item.get("train_r2")) for item in pool], dtype=float)
    held = np.array([_digits(item.get("held_r2")) for item in pool], dtype=float)
    bits = np.array([float(item.get("law_bits") or 0.0) for item in pool], dtype=float)
    rows = [[index / len(pool), float(fit.max() - fit[index]), float(fit[index] - held[index]), float(held.max() - held[index]),
             bits[index] / 100.0, float(bits[index] - bits.min()) / 16.0, float(item.get("constants") or 0),
             float(item.get("small_constants") or 0), float(item.get("large_constants") or 0),
             float(item.get("operations") or 0), float(item.get("steps") or 0), float(held[index] - fit[index])]
            for index, item in enumerate(pool)]
    table: np.ndarray = np.asarray(rows, dtype=float)
    return table


def simplest_within(pool: Sequence[Mapping[str, Any]], tolerance: float = 0.3) -> list[int]:
    """The shortest first among candidates whose fit on unseen rows is within ``tolerance`` digits of the best; then the rest as they were."""
    held = [_digits(item.get("held_r2")) for item in pool]
    best = max(held, default=0.0)
    near = sorted((index for index in range(len(pool)) if best - held[index] <= tolerance),
                  key=lambda index: (float(pool[index].get("law_bits") or 0.0), index))
    chosen = set(near)
    return [*near, *(index for index in range(len(pool)) if index not in chosen)]


class LearnedOrder:
    """A logistic model of "counted solved", used to order a list. Its whole state is a few numbers."""

    def __init__(self, strength: float = 1.0) -> None:
        self._strength = float(strength)
        self.names: tuple[str, ...] = tuple(FEATURES)          #: what the numbers below are numbers of
        self.mean: np.ndarray = np.zeros(len(FEATURES))
        self.scale: np.ndarray = np.ones(len(FEATURES))
        self.weights: np.ndarray = np.zeros(len(FEATURES))
        self.offset = 0.0
        self.trained_on = {"problems": 0, "candidates": 0, "solved": 0}

    def fit(self, pools: Sequence[Sequence[Mapping[str, Any]]], label: str = LABEL) -> LearnedOrder:
        """Learn from lists whose candidates carry ``label``. With no solved candidate, or none unsolved, nothing is learned."""
        from sklearn.linear_model import LogisticRegression

        usable = [pool for pool in pools if pool]
        if not usable:
            return self
        X = np.concatenate([features(pool) for pool in usable])
        y = np.array([bool(item.get(label)) for pool in usable for item in pool], dtype=int)
        self.trained_on = {"problems": len(usable), "candidates": len(y), "solved": int(y.sum())}
        if y.min() == y.max():
            return self
        self.mean, self.scale = X.mean(axis=0), np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        model = LogisticRegression(C=self._strength, class_weight="balanced", max_iter=2000)
        model.fit((X - self.mean) / self.scale, y)
        self.weights, self.offset = np.asarray(model.coef_[0], dtype=float), float(model.intercept_[0])
        return self

    def scores(self, pool: Sequence[Mapping[str, Any]]) -> np.ndarray:
        scored: np.ndarray = ((features(pool) - self.mean) / self.scale) @ self.weights + self.offset
        return scored

    def order(self, pool: Sequence[Mapping[str, Any]]) -> list[int]:
        """Indices into the list, most likely solved first; equal scores keep the list's own order. Untrained: the list's order."""
        if not np.any(self.weights):
            return list(range(len(pool)))
        scored = self.scores(pool)
        return sorted(range(len(pool)), key=lambda index: (-float(scored[index]), index))

    def as_dict(self) -> dict[str, Any]:
        return {"features": list(self.names), "mean": [float(v) for v in self.mean], "scale": [float(v) for v in self.scale],
                "weights": [float(v) for v in self.weights], "offset": self.offset, "strength": self._strength,
                "trained_on": dict(self.trained_on)}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> LearnedOrder:
        if list(record.get("features", [])) != list(FEATURES):
            raise ValueError("this order was learned on other features than the module now computes")
        out = cls(float(record.get("strength", 1.0)))
        out.mean, out.scale = np.asarray(record["mean"], dtype=float), np.asarray(record["scale"], dtype=float)
        out.weights, out.offset = np.asarray(record["weights"], dtype=float), float(record["offset"])
        out.trained_on = dict(record.get("trained_on", {}))
        return out


def place_under(pool: Sequence[Mapping[str, Any]], order: Sequence[int], label: str = LABEL) -> int | None:
    """The place (from 1) of the first candidate carrying ``label`` when the list is read in ``order``."""
    return next((rank for rank, index in enumerate(order, 1) if pool[index].get(label)), None)


def reorder(rows: Sequence[Mapping[str, Any]], model: LearnedOrder | None = None, tolerance: float | None = None
            ) -> list[dict[str, Any]]:
    """Rows as ``order_score`` reads them, with the places each list would have under a learned order or the rule."""
    out = []
    for row in rows:
        pool = row["candidates"]
        order = model.order(pool) if model is not None else simplest_within(pool, tolerance if tolerance is not None else 0.3)
        out.append({"key": row["key"], "level": row["level"], "inputs": row.get("inputs"), "noise": row.get("noise"),
                    "solved_place": place_under(pool, order, "solved"), "form_place": place_under(pool, order, "same_form"),
                    "accurate_place": place_under(pool, order, "accurate")})
    return out


def out_of_fold(rows: Sequence[Mapping[str, Any]], folds: int = 5, seed: int = 0, strength: float = 1.0) -> list[dict[str, Any]]:
    """Every problem's places under an order learned from the other folds' problems only."""
    keys = sorted({row["key"] for row in rows})
    rng = np.random.default_rng(seed)
    assignment = {key: int(fold) for key, fold in zip(keys, rng.permutation(len(keys)) % folds, strict=True)}
    out: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [row["candidates"] for row in rows if assignment[row["key"]] != fold]
        model = LearnedOrder(strength).fit(train)
        out.extend(reorder([row for row in rows if assignment[row["key"]] == fold], model))
    return sorted(out, key=lambda row: row["key"])
