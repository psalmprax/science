"""A discovery loop that learns from the referee's refusals without fitting to them.

The open discovery engine searches one hypothesis space per call. The loop
climbs a fixed ladder of spaces, cheapest first -- dimensionally strict
expressions; expressions whose coefficients may carry units (a law with a
dimensional constant such as g); a superposition basis of up to two terms --
and stops at the first rung the referee promotes on. It does not try to read
from a refusal whether a bigger space would help: the referee reports a law
outside the space as ``intrinsic_observational_nonidentifiability`` when the
wrong candidates it does see cannot be told apart, so an attribution-based
stopping rule would stop exactly where it should climb.

Three rules keep the loop from turning into a way to search until something
passes:

1. **Proposer-validation split.** The development data are split once, by a
   seeded permutation, into a *proposer-validation* half and a *confirmation*
   half. Every round runs on the proposer-validation half only (the engine
   splits it again internally for its own holdout). The rung chosen there runs
   once more with the whole proposer-validation half as training data and the
   confirmation half as holdout; that verdict is the loop's result. The
   confirmation half is spent: a second confirmation for the same
   ``(domain, dataset_id)`` raises :class:`ConfirmationAlreadySpent`.
2. **Every round is a trial.** Each round, and the confirmation, is written to
   the :class:`~apedp_referee.trial_ledger.CandidateTrialLedger` under one
   ``search_id`` with the number of expressions generated so far as its
   ``selection_pool_size``, so the multiplicity charge covers every rung the
   loop tried, not only the one that worked.
3. **Memory is keyed by domain and reads verdicts only.** A rung is skipped
   for a new dataset only when, on at least ``MIN_DATASETS_TO_SKIP`` other
   datasets *of the same domain*, it was run, refused, and a higher rung was
   then promoted, every time it was run. The memory sees a record through
   :data:`READABLE_FIELDS` and nothing else: no fitted value, error, holdout
   statistic or selection statistic reaches the policy
   (``tests/test_refusal_memory.py`` pins this).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_referee.trial_ledger import CandidateTrialLedger

PROPOSER_ID = "refusal_memory_loop"
MIN_DATASETS_TO_SKIP = 2
#: The only parts of a ledger record the memory may read.
READABLE_FIELDS = frozenset({"domain", "dataset_id", "rung", "stage", "status", "attribution"})
ALGEBRAIC = ("+", "-", "*", "/", "pow")


class ConfirmationAlreadySpent(RuntimeError):
    """The confirmation half of this dataset has already been used once."""


@dataclass(frozen=True)
class Rung:
    """One hypothesis space, cheapest first."""

    name: str
    kind: str  # "single" | "multi"
    options: Mapping[str, Any] = field(default_factory=dict)


LADDER: tuple[Rung, ...] = (
    Rung("strict", "single", {"max_depth": 2, "max_complexity": 6, "coefficient_synthesis": False}),
    Rung("unit_coefficients", "single", {"max_depth": 2, "max_complexity": 6, "coefficient_synthesis": True}),
    Rung("superposition", "multi", {"max_depth": 2, "max_term_complexity": 4, "max_terms": 2}),
)


@dataclass(frozen=True)
class RoundOutcome:
    """What one engine call returned, reduced to what the loop may act on."""

    status: str
    attribution: str
    promoted: str | None
    generated: int


@dataclass
class LoopResult:
    domain: str
    dataset_id: str
    rounds: list[dict[str, Any]]
    chosen_rung: str | None
    confirmation: dict[str, Any] | None
    stopped_because: str

    @property
    def confirmed(self) -> bool:
        return self.confirmation is not None and self.confirmation["status"] == EpistemicStatus.UNIQUE_SUPPORT.value


def _value(x: Any) -> str:
    return str(getattr(x, "value", x))


def run_rung(
    rung: Rung,
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    dimensions: Mapping[str, Dimension],
    target_dimension: Dimension,
    *,
    noise_sigma: float | None,
    seed: int,
    X_holdout: Mapping[str, np.ndarray] | None = None,
    Y_holdout: np.ndarray | None = None,
) -> RoundOutcome:
    """Run one rung's engine and keep only its verdict."""
    if rung.kind == "single":
        from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine

        opts = dict(rung.options)
        synth = bool(opts.pop("coefficient_synthesis", False))
        result = L2DiscoveryEngine(operators=ALGEBRAIC, **opts).discover(
            X=X, Y=Y, dimensions=dimensions, target_dimension=target_dimension, target_name="y",
            noise_sigma=noise_sigma, seed=seed, X_holdout=X_holdout, Y_holdout=Y_holdout,
            coefficient_synthesis=synth,
        )
        adj = result.adjudication
        promoted = adj.promoted.candidate.candidate_id if adj.promoted is not None else None
        generated = len(result.generated_nodes)
    elif rung.kind == "multi":
        from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine

        mresult = MultiTermL2Engine(operators=ALGEBRAIC, **dict(rung.options)).discover_multiterm(
            X=X, Y=Y, dimensions=dimensions, target_dimension=target_dimension, target_name="y",
            noise_sigma=noise_sigma, seed=seed, X_holdout=X_holdout, Y_holdout=Y_holdout,
        )
        adj = mresult.adjudication
        promoted = mresult.promoted_equation if adj.promoted is not None else None
        generated = int(mresult.all_evaluated_count)
    else:  # pragma: no cover - LADDER is fixed
        raise ValueError(f"unknown rung kind {rung.kind!r}")
    return RoundOutcome(_value(adj.status), _value(adj.epistemic_attribution), promoted, generated)


def split_development(
    X: Mapping[str, np.ndarray], Y: np.ndarray, seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Seeded half/half split into (proposer-validation X, Y, confirmation X, Y)."""
    y = np.asarray(Y, dtype=float)
    order = np.random.default_rng(seed).permutation(len(y))
    a, b = np.sort(order[: len(y) // 2]), np.sort(order[len(y) // 2:])
    xs = {k: np.asarray(v, dtype=float) for k, v in X.items()}
    return {k: v[a] for k, v in xs.items()}, y[a], {k: v[b] for k, v in xs.items()}, y[b]


class RefusalMemory:
    """Verdicts of earlier loop rounds, for one domain, read through READABLE_FIELDS."""

    def __init__(self, ledger: CandidateTrialLedger, domain: str) -> None:
        self.ledger = ledger
        self.domain = domain

    def _views(self) -> list[dict[str, str]]:
        views = []
        for record in self.ledger.get_records():
            if record.proposer_id != PROPOSER_ID:
                continue
            view = {k: str(record.metadata[k]) for k in READABLE_FIELDS if k in record.metadata}
            if view.get("domain") == self.domain:
                views.append(view)
        return views

    def confirmation_spent(self, dataset_id: str) -> bool:
        return any(v.get("dataset_id") == dataset_id and v.get("stage") == "confirmation" for v in self._views())

    def rungs_to_try(self, dataset_id: str) -> list[int]:
        """Ladder indices for this dataset, cheapest first, minus rungs the domain has outgrown.

        The last rung is never skipped, so the loop always tries something.
        """
        names = [rung.name for rung in LADDER]
        refused: dict[str, set[str]] = {}  # dataset -> rungs refused there
        promoted: dict[str, set[str]] = {}  # dataset -> rungs promoted there
        for view in self._views():
            if view.get("stage") != "proposer_validation" or view.get("dataset_id") == dataset_id:
                continue
            bucket = promoted if view.get("status") == EpistemicStatus.UNIQUE_SUPPORT.value else refused
            bucket.setdefault(view.get("dataset_id", ""), set()).add(view.get("rung", ""))
        by_rung: dict[str, dict[str, bool]] = {}
        for dataset in {*refused, *promoted}:
            higher = {names.index(r) for r in promoted.get(dataset, set()) if r in names}
            for rung_name in {*refused.get(dataset, set()), *promoted.get(dataset, set())}:
                if rung_name not in names:
                    continue
                outgrown = rung_name in refused.get(dataset, set()) and any(h > names.index(rung_name) for h in higher)
                seen = by_rung.setdefault(rung_name, {})
                seen[dataset] = seen.get(dataset, True) and outgrown
        keep = []
        for index, rung in enumerate(LADDER):
            seen = by_rung.get(rung.name, {})
            skip = len(seen) >= MIN_DATASETS_TO_SKIP and all(seen.values())
            if not skip or index == len(LADDER) - 1:
                keep.append(index)
        return keep


def run_discovery_loop(
    X: Mapping[str, np.ndarray],
    Y: np.ndarray,
    dimensions: Mapping[str, Dimension],
    target_dimension: Dimension,
    *,
    domain: str,
    dataset_id: str,
    ledger: CandidateTrialLedger,
    noise_sigma: float | None = None,
    seed: int = 0,
    runner: Callable[..., RoundOutcome] = run_rung,
) -> LoopResult:
    """Climb the ladder on proposer-validation data, then confirm the chosen rung once."""
    memory = RefusalMemory(ledger, domain)
    if memory.confirmation_spent(dataset_id):
        raise ConfirmationAlreadySpent(
            f"the confirmation half of {domain}/{dataset_id} was used by an earlier loop; "
            "a new confirmation needs new data")
    xa, ya, xb, yb = split_development(X, Y, seed)
    search_id = f"{PROPOSER_ID}:{domain}:{dataset_id}:{seed}"
    pool = 0
    rounds: list[dict[str, Any]] = []

    def log(stage: str, rung: Rung, outcome: RoundOutcome) -> dict[str, Any]:
        nonlocal pool
        pool += max(1, outcome.generated)
        meta = {"domain": domain, "dataset_id": dataset_id, "rung": rung.name, "stage": stage,
                "status": outcome.status, "attribution": outcome.attribution,
                "search_id": search_id, "selection_pool_size": pool, "seed": seed}
        ledger.record_trial(
            candidate_id=f"{search_id}:{stage}:{rung.name}:{outcome.promoted or 'none'}",
            proposer_id=PROPOSER_ID, formula=outcome.promoted or "", strategy_archetype="equation_discovery",
            in_sample_sharpe=0.0, n_bars=len(ya), status=outcome.status, metadata=meta,
            selection_statistic=float("nan"),
        )
        return {**meta, "promoted": outcome.promoted, "generated": outcome.generated}

    chosen: Rung | None = None
    stopped = "every rung tried was refused on proposer-validation data"
    for index in memory.rungs_to_try(dataset_id):
        rung = LADDER[index]
        outcome = runner(rung, xa, ya, dimensions, target_dimension, noise_sigma=noise_sigma, seed=seed)
        rounds.append(log("proposer_validation", rung, outcome))
        if outcome.status == EpistemicStatus.UNIQUE_SUPPORT.value:
            chosen = rung
            stopped = f"{rung.name} promoted on proposer-validation data"
            break
    confirmation = None
    if chosen is not None:
        final = runner(chosen, xa, ya, dimensions, target_dimension, noise_sigma=noise_sigma, seed=seed,
                       X_holdout=xb, Y_holdout=yb)
        confirmation = log("confirmation", chosen, final)
    return LoopResult(domain, dataset_id, rounds, chosen.name if chosen else None, confirmation, stopped)
