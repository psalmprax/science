"""The main search path: a cheap exhaustive first pass, then PySR, then contrast confirmation.

Status: experimental

Order of work, chosen from the benchmark results (PREREG-SRBENCH-BASELINE,
-EXTERNAL and PREREG-MMLM-L, 2026-11):

1. **First pass.** The exhaustive search in the order of a prior, with the
   confirmed early stop (``confirmed_stop.search_confirmed``). Seconds, and it
   returns laws the enumeration can express.
2. **PySR.** Run when the first pass did not stop early (or always, with
   ``always_search=True``), seeded with the first pass's best expression. Any
   ``ProposerBase`` can stand in; ``None`` means the first pass alone.
3. **Confirmation.** The candidates of both stages go to contrast confirmation
   (``contrast_confirmation.contrast_confirm``): rows are chosen from the input
   pool where candidates disagree, each target is read once through ``measure``,
   and the survivor has to match its own development error. The outcome is one
   of ``confirmed``, ``equivalent_class``, ``unchallenged`` or ``widen_search``.
4. **Adequacy.** Confirmation is relative to the candidates it was given: with
   the true law missing it named a wrong one alone in 19 of 80 runs of
   PREREG-CONTRAST-2026-11. So the survivor also has to leave no structure in
   its development residuals (the residual check of the guided search). If it
   does, the outcome is ``widen_search`` and the law is reported as rejected,
   not as found. This check only ever turns a confirmation into a refusal.

The searches see the development rows only. Targets of the pool are read only by
the confirmation step. This path does not call the template search or the
referee's sole-survivor rule, which promoted a wrong law in 62.7 % of its
promotions on SRBench (audit M-10).

What this path has not been measured on is recorded in the audit (M-11): the
three stages were each tested separately; the path as a whole has no
preregistered result yet.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed
from apedp_dyn.discovery_l2.guided_search import GuidedSearch, neighbours, residual_structure
from apedp_dyn.grammar.proposer_interface import ProposerBase

PROPOSER_ID = "main_search"
ADEQUACY_ROWS = 1500  # the residual check compares every pair of rows; larger samples are subsampled


def size_prior(node: es.Node) -> float:
    """Shorter expressions first: the order used when no learned prior is supplied."""
    return float(1 if node[0] == "x" else 1 + sum(size_prior(child) for child in node[1:]))


@dataclass
class MainSearchResult:
    action: str                      # confirmed | equivalent_class | unchallenged | widen_search
    law: str | None
    found_by: str | None             # first_pass | search
    candidates: list[dict[str, str]] = field(default_factory=list)
    first_pass: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    contrast: cc.ContrastResult | None = None
    adequacy: dict[str, Any] = field(default_factory=dict)
    rejected_law: str | None = None


def discover(
    X: Mapping[str, np.ndarray], y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    proposer: ProposerBase | None = None, structure_bits: Callable[[es.Node], float] | None = None,
    guided: GuidedSearch | None = None, always_search: bool = False, noise: str = "relative", budget: int = 30,
    limit: int = 12, seed: int = 0, ledger: Any = None, search_id: str | None = None, domain: str | None = None,
) -> MainSearchResult:
    """Run the path on development rows ``(X, y)``; ``pool_X`` columns follow the order of ``X``.

    ``guided`` reuses a prepared search order across datasets (pricing every
    candidate with a learned prior is the expensive part); otherwise one is built
    from ``structure_bits`` or the size prior. With ``ledger`` the outcome is
    recorded as one trial, in the form ``apedp_mmlm.law_ledger`` reads.
    """
    names = list(X)
    dev_X = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    dev_y = np.asarray(y, dtype=float)
    guided = guided or GuidedSearch(structure_bits or size_prior)
    first = search_confirmed(guided, dev_X, dev_y, names, 6 if len(names) <= 5 else 5)
    first_expr = sp.sympify(es.model_string(first.best), locals={n: sp.Symbol(n) for n in names})
    sources: list[tuple[str, sp.Expr]] = [("first_pass", first_expr)]
    report: dict[str, Any] = {"ran": False, "proposals": 0}
    if proposer is not None and (always_search or not first.stopped_early):
        frame = pd.DataFrame({**{n: dev_X[:, i] for i, n in enumerate(names)}, "__target__": dev_y})
        proposals = proposer.fit_and_propose(frame, target_col="__target__", guesses=[str(first_expr)])
        report = {"ran": True, "proposals": len(proposals), "proposer_id": proposer.proposer_id}
        searched = [("search", sp.sympify(p.formula, locals={n: sp.Symbol(n) for n in names})) for p in proposals]
        # an early stop keeps its place at the head; otherwise the search's own pick leads
        sources = [*sources, *searched] if first.stopped_early else [*searched, *sources]
    usable: list[cc.FittedCandidate] = []
    origin: dict[str, str] = {}
    for source, expr in sources:
        if len(usable) >= limit or not expr.free_symbols:
            continue
        cand = cr.fitted_candidate(expr, names, dev_X, dev_y, noise)
        if cand.candidate_id in origin or not (np.all(np.isfinite(cand.coef)) and np.all(np.isfinite(cand.dev_pred))):
            continue
        origin[cand.candidate_id] = source
        usable.append(cand)
    first_report = {"expression": str(first_expr), "evaluated": first.evaluated, "candidates": first.candidates,
                    "stopped_early": bool(first.stopped_early)}
    listed = [{"source": origin[c.candidate_id], "formula": c.candidate_id} for c in usable]
    if not usable:
        result = MainSearchResult("widen_search", None, None, listed, first_report, report, None)
    else:
        outcome = cc.contrast_confirm(usable, np.asarray(pool_X, dtype=float), measure, noise=noise, budget=budget, seed=seed)
        result = MainSearchResult(cr.ACTIONS[outcome.verdict], outcome.survivor,
                                  origin.get(outcome.survivor) if outcome.survivor else None,
                                  listed, first_report, report, outcome)
        survivor = next((c for c in usable if c.candidate_id == outcome.survivor), None)
        if survivor is not None:
            rows = (np.arange(len(dev_y)) if len(dev_y) <= ADEQUACY_ROWS
                    else np.sort(np.random.default_rng(seed).choice(len(dev_y), ADEQUACY_ROWS, replace=False)))
            structure = residual_structure(survivor.dev_resid[rows], neighbours(dev_X[rows], min(guided.k, len(rows) - 1)))
            result.adequacy = {"residual_structure": structure, "passed": structure <= 0.0}
            if structure > 0.0:
                result.action, result.rejected_law, result.law, result.found_by = "widen_search", result.law, None, None
    if ledger is not None:
        contrast = result.contrast
        metadata: dict[str, Any] = {
            "search_id": search_id, "resolved_by": "contrast", "found_by": result.found_by,
            "adequacy": result.adequacy, "rejected_law": result.rejected_law,
            "selection_pool_size": max(1, first.evaluated + report["proposals"]),
            "rows_read": contrast.rows_read if contrast else 0,
            "design_sha256": contrast.design_sha256 if contrast else None,
        }
        if domain is not None:
            metadata["domain"] = domain
        ledger.record_trial(
            candidate_id=f"{search_id}:{result.law or 'none'}", proposer_id=PROPOSER_ID, formula=result.law or "",
            strategy_archetype="equation_discovery", in_sample_sharpe=0.0, n_bars=len(dev_y), status=result.action,
            selection_statistic=float("nan"), metadata=metadata)
    return result
