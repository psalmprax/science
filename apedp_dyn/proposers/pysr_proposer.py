"""PySR as a proposer: the main symbolic search, emitting candidates and deciding nothing.

Status: experimental

On the shared benchmark (PREREG-SRBENCH-EXTERNAL-2026-11) PySR recovered 73 and
68 of the 119 Feynman laws at 0 and 1 % noise, against 24 for the best in-house
engine, and recovered every law that engine did. It is therefore the main
search. This module wraps it as a :class:`ProposerBase`: it fits on the
development rows it is given and returns its hall of fame as
``CandidateProposal`` objects (AGENTS.md 1.1). It never sees a confirmation row
and never ranks its own output as accepted; ``apedp_dyn.discovery_l2.main_search``
sends the candidates to contrast confirmation.

PySR needs Julia. It is an optional dependency (``pip install pysr``), imported
only when a search is run; without it :class:`PySRUnavailableError` is raised
rather than a weaker engine being substituted silently. The settings are those
of the benchmark run (serial, deterministic, a wall-clock budget).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

BINARY = ("+", "-", "*", "/")
UNARY = ("sqrt", "exp", "log", "sin", "cos", "square")


class PySRUnavailableError(RuntimeError):
    """PySR (or the Julia runtime it needs) could not be loaded."""


def default_regressor(*, budget_seconds: float, binary: Sequence[str], unary: Sequence[str], seed: int,
                      guesses: Sequence[str] | None) -> Any:
    """A ``PySRRegressor`` with the benchmark's settings."""
    try:
        from pysr import PySRRegressor
    except Exception as exc:  # noqa: BLE001 - a missing package and a failed Julia start are the same answer
        raise PySRUnavailableError(f"PySR could not be loaded ({type(exc).__name__}); install it with `pip install pysr`") from exc
    options: dict[str, Any] = {}
    if guesses:
        options["guesses"] = list(guesses)
    return PySRRegressor(
        niterations=10**7, timeout_in_seconds=budget_seconds, binary_operators=list(binary), unary_operators=list(unary),
        model_selection="best", parallelism="serial", deterministic=True, random_state=seed,
        progress=False, verbosity=0, temp_equation_file=True, **options,
    )


def hall_of_fame(model: Any) -> list[dict[str, Any]]:
    """The fitted model's equations as (expression, complexity, loss, score), PySR's own pick first.

    PySR's ``best`` rule: among equations whose loss is within 1.5 times the
    lowest, the one with the highest score. The rest follow by increasing loss.
    """
    frame = model.equations_
    rows = [{"expr": sp.sympify(row["sympy_format"]), "complexity": int(row["complexity"]), "loss": float(row["loss"]),
             "score": float(row["score"]) if "score" in frame.columns else float("nan")}
            for _, row in frame.iterrows()]
    rows = [r for r in rows if np.isfinite(r["loss"])]
    if not rows:
        return []
    floor = min(r["loss"] for r in rows)
    near = [r for r in rows if r["loss"] <= 1.5 * floor]
    best = max(near, key=lambda r: (r["score"] if np.isfinite(r["score"]) else -np.inf, -r["loss"]))
    rest = sorted((r for r in rows if r is not best), key=lambda r: r["loss"])
    return [best, *rest]


class PySRProposer(ProposerBase):
    """PySR's hall of fame as candidate proposals."""

    def __init__(self, budget_seconds: float = 240.0, max_proposals: int = 8, binary: Sequence[str] = BINARY,
                 unary: Sequence[str] = UNARY, seed: int = 0, target_col: str = "target",
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL,
                 regressor_factory: Callable[..., Any] = default_regressor) -> None:
        self._budget = float(budget_seconds)
        self._max_proposals = int(max_proposals)
        self._binary, self._unary = tuple(binary), tuple(unary)
        self._seed = int(seed)
        self._target_col = target_col
        self._strategy_archetype = strategy_archetype
        self._factory = regressor_factory

    @property
    def proposer_id(self) -> str:
        return "pysr_proposer"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        """Search on the development rows. ``guesses`` (expressions in the feature names) seed the search."""
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        # PySR is given neutral names, so a column called ``sin`` or ``E`` cannot collide with a function or a constant
        internal = [f"x{i}" for i in range(len(features))]
        to_internal = {sp.Symbol(f): sp.Symbol(x) for f, x in zip(features, internal, strict=True)}
        to_feature = {sp.Symbol(x): sp.Symbol(f) for f, x in zip(features, internal, strict=True)}
        # PySR parses a seed as Julia, where a power is written ^
        seeds = [str(sp.sympify(g, locals={f: sp.Symbol(f) for f in features}).xreplace(to_internal)).replace("**", "^")
                 for g in guesses or ()]
        model = self._factory(budget_seconds=self._budget, binary=self._binary, unary=self._unary, seed=self._seed,
                              guesses=seeds or None)
        seed_rejected = ""
        try:
            model.fit(X, y, variable_names=internal)
        except Exception as exc:  # noqa: BLE001 - a seed PySR cannot use must not cost the search; anything else is re-raised below
            if not seeds:
                raise
            seed_rejected = f"{type(exc).__name__}: {exc}"[:200]
            model = self._factory(budget_seconds=self._budget, binary=self._binary, unary=self._unary, seed=self._seed,
                                  guesses=None)
            model.fit(X, y, variable_names=internal)
        hall = hall_of_fame(model)
        proposals, seen = [], set()
        for rank, row in enumerate(hall):
            expr = row["expr"].xreplace(to_feature)
            text = str(expr)
            if text in seen:
                continue
            seen.add(text)
            fn = sp.lambdify([sp.Symbol(f) for f in features], expr, "numpy")

            def compute_score(df: pd.DataFrame, fn: Any = fn, features: list[str] = features) -> np.ndarray:
                values = df[features].to_numpy(dtype=float)
                with np.errstate(all="ignore"):
                    out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
                return out

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id, candidate_id="pysr:" + hashlib.sha256(text.encode()).hexdigest()[:16],
                formula=text, strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
                compute_score=compute_score,
                parameters={"dev_loss": row["loss"], "complexity": row["complexity"], "score": row["score"]},
                metadata={"engine": "PySR", "hall_of_fame_rank": rank, "selection_pool_size": max(1, len(hall)),
                          "budget_seconds": self._budget, "seeded_with": 0 if seed_rejected else len(seeds),
                          "seed_rejected": seed_rejected, "seed": self._seed},
            ))
            if len(proposals) >= self._max_proposals:
                break
        return tuple(proposals)
