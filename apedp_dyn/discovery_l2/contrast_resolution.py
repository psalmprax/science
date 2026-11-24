"""What to do after the referee abstains: measure where the candidates disagree.

Status: experimental

The referee's ``MULTI_HYPOTHESIS`` ends a passive analysis. Where more can be
measured (a pool of unlabelled cases, an experiment, a simulator), this module
takes the engine's own candidates to :mod:`contrast_confirmation` and turns the
outcome into the next action:

- ``confirmed``: one law survived and eliminated at least one rival.
- ``equivalent_class``: the survivors cannot be told apart on this domain.
- ``unchallenged``: one candidate, nothing to compare it with.
- ``widen_search``: no candidate survived (or there were none): the true law is
  not among them, so search a larger space.

Only ``widen_search`` says anything about the search space. PREREG-CONTRAST-2026-11
measured that a ``unique`` verdict is relative to the candidates given (with the
true law removed, a wrong law was named alone in 19 of 80 runs), so
:func:`climb` (over the three rungs of ``refusal_memory.LADDER``) moves up a rung on ``widen_search`` and never treats ``confirmed``
as proof that the space was large enough.

It promotes nothing. The referee's verdict is returned untouched beside the
contrast result, and every round is written to the trial ledger.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import refusal_memory
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.epistemic.adjudicate import EpistemicStatus
from apedp_referee.trial_ledger import CandidateTrialLedger

PROPOSER_ID = "contrast_resolution"
ACTIONS = {cc.UNIQUE: "confirmed", cc.EQUIVALENT_CLASS: "equivalent_class",
           cc.UNCHALLENGED: "unchallenged", cc.NO_SURVIVOR: "widen_search"}


@dataclass
class Resolution:
    referee_status: str
    action: str
    law: str | None
    contrast: cc.ContrastResult | None
    candidates: int


def fitted_candidate(expr: sp.Expr, names: Sequence[str], dev_X: np.ndarray, dev_y: np.ndarray,
                     noise: str = "relative") -> cc.FittedCandidate:
    """An expression with one coefficient per additive term, fitted on the development rows."""
    symbols = [sp.Symbol(n) for n in names]
    fns = [sp.lambdify(symbols, term.as_coeff_Mul()[1], "numpy") for term in sp.Add.make_args(sp.expand(expr))]

    def basis(X: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.column_stack([np.asarray(f(*X.T), dtype=float) * np.ones(len(X)) for f in fns])
        return out

    return cc.fit_candidate(str(expr), basis, dev_X, dev_y, noise)


def engine_candidates(result: Any, limit: int = 12) -> list[sp.Expr]:
    """The engine's best distinct structures by its internal holdout error."""
    pool = getattr(result, "candidates", None) or getattr(result, "pareto_candidates", [])
    scored = sorted((s for s in pool if s.holdout_rmse is not None and np.isfinite(s.holdout_rmse)),
                    key=lambda s: s.holdout_rmse)
    out: list[sp.Expr] = []
    seen: set[str] = set()
    for item in scored:
        node = item.candidate.metadata.get("ast")
        nodes = item.candidate.metadata.get("nodes")  # the multi-term engine stores one node per term
        if node is None and not nodes:
            continue
        expr = node.to_sympy() if node is not None else sp.Add(*[n.to_sympy() for n in nodes])
        key = str(sp.Add(*[t.as_coeff_Mul()[1] for t in sp.Add.make_args(sp.expand(expr))]))
        if key not in seen:
            seen.add(key)
            out.append(expr)
        if len(out) == limit:
            break
    return out


def resolve(result: Any, names: Sequence[str], dev_X: np.ndarray, dev_y: np.ndarray, pool_X: np.ndarray,
            measure: Callable[[np.ndarray], np.ndarray], *, noise: str = "relative", limit: int = 12,
            budget: int = 30) -> Resolution:
    """Contrast confirmation over the candidates of one engine result; the referee's verdict rides along."""
    status = str(getattr(result.adjudication.status, "value", result.adjudication.status))
    exprs = engine_candidates(result, limit)
    usable = []
    for expr in exprs:
        cand = fitted_candidate(expr, names, dev_X, dev_y, noise)
        if np.all(np.isfinite(cand.coef)) and np.all(np.isfinite(cand.dev_pred)):
            usable.append(cand)
    if not usable:
        return Resolution(status, "widen_search", None, None, 0)
    outcome = cc.contrast_confirm(usable, pool_X, measure, noise=noise, budget=budget)
    return Resolution(status, ACTIONS[outcome.verdict], outcome.survivor, outcome, len(usable))


def _run_rung(rung: refusal_memory.Rung, X: Mapping[str, np.ndarray], Y: np.ndarray,
              dimensions: Mapping[str, Dimension], target_dimension: Dimension, noise_sigma: float | None,
              seed: int, operators: Sequence[str]) -> tuple[Any, int]:
    """One rung of refusal_memory.LADDER; returns the engine result and how many expressions it tried."""
    options = dict(rung.options)
    if rung.kind == "single":
        from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine

        synth = bool(options.pop("coefficient_synthesis", False))
        result = L2DiscoveryEngine(operators=tuple(operators), **options).discover(
            X=X, Y=Y, dimensions=dimensions, target_dimension=target_dimension, target_name="y",
            noise_sigma=noise_sigma, seed=seed, coefficient_synthesis=synth)
        return result, len(result.generated_nodes)
    from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine

    result = MultiTermL2Engine(operators=tuple(operators), **options).discover_multiterm(
        X=X, Y=Y, dimensions=dimensions, target_dimension=target_dimension, target_name="y",
        noise_sigma=noise_sigma, seed=seed)
    return result, int(result.all_evaluated_count)


def climb(
    X: Mapping[str, np.ndarray], Y: np.ndarray, dimensions: Mapping[str, Dimension], target_dimension: Dimension,
    pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *, ledger: CandidateTrialLedger,
    search_id: str, noise_sigma: float | None = None, noise: str = "relative", seed: int = 0,
    operators: Sequence[str] = ("+", "-", "*", "/", "pow"), domain: str | None = None, dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run the rungs of ``refusal_memory.LADDER``, resolve each by contrast, move up only on ``widen_search``.

    ``pool_X`` columns follow the order of ``X``. Returns one record per rung
    tried; the last one holds the outcome. Each rung is one trial in ``ledger``.

    With ``domain`` and ``dataset_id`` the climb also uses and feeds the refusal
    memory: rungs the domain has outgrown are skipped, and each round is
    recorded in the form the memory reads (verdict fields only), a rung counting
    as having worked when it did not end in ``widen_search``.
    """
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    dev_y = np.asarray(Y, dtype=float)
    remembered = domain is not None and dataset_id is not None
    order = (refusal_memory.RefusalMemory(ledger, str(domain)).rungs_to_try(str(dataset_id)) if remembered
             else list(range(len(refusal_memory.LADDER))))
    rounds: list[dict[str, Any]] = []
    pool = 0
    for index in order:
        rung = refusal_memory.LADDER[index]
        result, generated = _run_rung(rung, X, dev_y, dimensions, target_dimension, noise_sigma, seed, operators)
        resolution = resolve(result, names, dev_X, dev_y, pool_X, measure, noise=noise)
        pool += max(1, generated)
        record = {"rung": rung.name, "referee_status": resolution.referee_status, "action": resolution.action,
                  "law": resolution.law, "candidates": resolution.candidates,
                  "equivalent": resolution.contrast.equivalent if resolution.contrast else [],
                  "rows_read": resolution.contrast.rows_read if resolution.contrast else 0,
                  "design_sha256": resolution.contrast.design_sha256 if resolution.contrast else None}
        metadata: dict[str, Any] = {"search_id": search_id, "selection_pool_size": pool, "rung": rung.name,
                                    "referee_status": resolution.referee_status, "rows_read": record["rows_read"],
                                    "design_sha256": record["design_sha256"], "resolved_by": "contrast"}
        if remembered:
            worked = resolution.action != "widen_search"
            metadata.update({"domain": domain, "dataset_id": dataset_id, "stage": "proposer_validation",
                             "status": EpistemicStatus.UNIQUE_SUPPORT.value if worked else "widen_search",
                             "attribution": resolution.action})
        ledger.record_trial(
            candidate_id=f"{search_id}:{rung.name}:{resolution.law or 'none'}",
            proposer_id=refusal_memory.PROPOSER_ID if remembered else PROPOSER_ID,
            formula=resolution.law or "", strategy_archetype="equation_discovery", in_sample_sharpe=0.0,
            n_bars=len(dev_y), status=resolution.action, selection_statistic=float("nan"), metadata=metadata)
        rounds.append(record)
        if resolution.action != "widen_search":
            break
    return rounds


def referee_abstained(result: Any) -> bool:
    """True when the referee promoted nothing, which is when contrast resolution has something to add."""
    return result.adjudication.status is not EpistemicStatus.UNIQUE_SUPPORT
