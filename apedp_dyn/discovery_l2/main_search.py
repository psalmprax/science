"""The main search path: a cheap exhaustive first pass, then PySR, then contrast confirmation.

Status: experimental

Order of work, chosen from the benchmark results (PREREG-SRBENCH-BASELINE,
-EXTERNAL and PREREG-MMLM-L, 2026-11):

1. **First pass.** The exhaustive search in the order of a prior, with the
   confirmed early stop (``confirmed_stop.search_confirmed``). Seconds, and it
   returns laws the enumeration can express.
2. **PySR.** Run when the first pass did not settle the matter (or always, with
   ``always_search=True``), seeded with the first pass's best expression. Any
   ``ProposerBase`` can stand in; ``None`` means the first pass alone. The
   numeric constants inside each returned expression are refitted on the
   development rows (``polish``).
3. **Confirmation.** The candidates of both stages go to contrast confirmation
   (``contrast_confirmation.contrast_confirm``): rows are chosen from the input
   pool where candidates disagree, each target is read once through ``measure``,
   and the survivor has to match its own development error. The outcome is one
   of ``confirmed``, ``equivalent_class``, ``unchallenged`` or ``widen_search``.
   When several candidates cannot be told apart on the domain, the simplest of
   them is the one reported: a search run for minutes returns accurate
   approximations beside the law, and on the rows available they are equal.
4. **Adequacy.** Confirmation is relative to the candidates it was given: with
   the true law missing it named a wrong one alone in 19 of 80 runs of
   PREREG-CONTRAST-2026-11. So the survivor also has to leave no structure in
   its development residuals (the residual check of the guided search). If it
   does, the outcome is ``widen_search`` and the law is reported as rejected,
   not as found. The residual check alone has little power on clustered rows
   (a law with test R^2 0.95 passed it on a Strogatz trajectory set), so the
   survivor must also predict the development targets at least as well as a
   model-free smoother does (each target from its nearest neighbours' targets).
   A true law's error is the noise; the smoother's is the noise plus its own
   bias, so a true law is not refused by this, and a law that a smoother beats
   is missing something. These checks only ever turn a confirmation into a
   refusal, and the first pass settles the search only if its law passes them.

Data without noise is handled before step 3. A candidate whose development
residual is below ``NUMERIC_FLOOR`` (relative) fits to numerical precision;
contrast confirmation has no noise level to work with there. Such candidates
are checked directly on ``budget`` pool rows, and those that stay exact are
confirmed (the simplest is reported). An approximation that is merely close is
not exact, and on noise-free data its residuals have structure, so step 4
refuses it.

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
from scipy.optimize import least_squares

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2.confirmed_stop import search_confirmed
from apedp_dyn.discovery_l2.guided_search import GuidedSearch, neighbours, residual_structure
from apedp_dyn.grammar.proposer_interface import ProposerBase

PROPOSER_ID = "main_search"
ADEQUACY_ROWS = 1500  # the residual check compares every pair of rows; larger samples are subsampled
NUMERIC_FLOOR = 1e-6  # relative residual below which a fit is exact to numerical precision


def adequacy(resid: np.ndarray, X: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> dict[str, Any]:
    """Whether residuals show no structure and the fit is at least as good as a nearest-neighbour smoother."""
    rows = (np.arange(len(y)) if len(y) <= ADEQUACY_ROWS
            else np.sort(np.random.default_rng(seed).choice(len(y), ADEQUACY_ROWS, replace=False)))
    nn = neighbours(X[rows], min(k, len(rows) - 1))
    structure = residual_structure(resid[rows], nn)
    smoother = y[rows] - y[rows][nn].mean(axis=1)
    ratio = float(np.sqrt(np.mean(resid[rows] ** 2)) / max(float(np.sqrt(np.mean(smoother ** 2))), 1e-300))
    return {"residual_structure": structure, "error_vs_smoother": ratio, "passed": bool(structure <= 0.0 and ratio <= 1.0)}


def polish(expr: sp.Expr, names: list[str], X: np.ndarray, y: np.ndarray, max_constants: int = 12) -> sp.Expr:
    """Refit the numeric constants inside a searched expression on the development rows.

    A search stops refining constants at its own tolerance. On noise-free data a
    true law with a constant right to five digits then looks like an
    approximation, and would be refused. Returns the expression unchanged when
    there is nothing to refit, too much to refit, or the refit is not better.
    """
    floats = sorted(expr.atoms(sp.Float), key=str)
    if not floats or len(floats) > max_constants:
        return expr
    params = [sp.Symbol(f"_c{i}") for i in range(len(floats))]
    fn = sp.lambdify([*(sp.Symbol(n) for n in names), *params], expr.xreplace(dict(zip(floats, params, strict=True))), "numpy")
    scale = float(np.sqrt(np.mean(y ** 2))) or 1.0

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            r = np.asarray(fn(*X.T, *theta), dtype=float) * np.ones(len(y)) - y
        out: np.ndarray = np.where(np.isfinite(r), r, 1e6 * scale)
        return out

    start = np.array([float(v) for v in floats])
    try:
        fit = least_squares(resid, start, max_nfev=200)
    except (ValueError, TypeError, FloatingPointError):
        return expr
    if not np.all(np.isfinite(fit.x)) or float(np.sum(fit.fun ** 2)) >= float(np.sum(resid(start) ** 2)):
        return expr
    return expr.xreplace({old: sp.Float(float(new)) for old, new in zip(floats, fit.x, strict=True)})


def _relative(resid: np.ndarray, y: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.asarray(y, dtype=float) ** 2)))
    return float(np.sqrt(np.mean(np.asarray(resid, dtype=float) ** 2))) / max(scale, 1e-300)


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
    rows_read: int = 0


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
    # On noise-free data the residual check sees rounding error and may never stop early; a first-pass
    # law that fits to numerical precision settles the first pass all the same.
    first_fit = cr.fitted_candidate(first_expr, names, dev_X, dev_y, noise) if first_expr.free_symbols else None
    settled = False
    if first_fit is not None and bool(np.all(np.isfinite(first_fit.dev_pred))):
        settled = _relative(first_fit.dev_resid, dev_y) < NUMERIC_FLOOR or (
            bool(first.stopped_early) and adequacy(first_fit.dev_resid, dev_X, dev_y, guided.k, seed)["passed"])
    if proposer is not None and (always_search or not settled):
        frame = pd.DataFrame({**{n: dev_X[:, i] for i, n in enumerate(names)}, "__target__": dev_y})
        proposals = proposer.fit_and_propose(frame, target_col="__target__", guesses=[str(first_expr)])
        report = {"ran": True, "proposals": len(proposals), "proposer_id": proposer.proposer_id}
        searched = []
        for p in proposals:
            try:
                parsed = sp.sympify(p.formula, locals={n: sp.Symbol(n) for n in names})
                if parsed.free_symbols <= {sp.Symbol(n) for n in names}:
                    parsed = polish(parsed, names, dev_X, dev_y)
                searched.append(("search", parsed))
            except (sp.SympifyError, TypeError, SyntaxError):
                report["unparseable"] = report.get("unparseable", 0) + 1
        # an early stop keeps its place at the head; otherwise the search's own pick leads
        sources = [*sources, *searched] if settled else [*searched, *sources]
    usable: list[cc.FittedCandidate] = []
    known = {sp.Symbol(n) for n in names}
    origin: dict[str, str] = {}
    size: dict[str, int] = {}  # operations plus fitted constants: what "simplest" means below
    for source, expr in sources:
        if len(usable) >= limit or not expr.free_symbols or not expr.free_symbols <= known:
            continue  # a constant is not a law in the inputs; a formula in unknown names cannot be evaluated
        cand = cr.fitted_candidate(expr, names, dev_X, dev_y, noise)
        if cand.candidate_id in origin or not (np.all(np.isfinite(cand.coef)) and np.all(np.isfinite(cand.dev_pred))):
            continue
        origin[cand.candidate_id] = source
        size[cand.candidate_id] = int(sp.count_ops(expr)) + len(expr.atoms(sp.Float))
        usable.append(cand)
    first_report = {"expression": str(first_expr), "evaluated": first.evaluated, "candidates": first.candidates,
                    "stopped_early": bool(first.stopped_early), "settled": settled}
    listed = [{"source": origin[c.candidate_id], "formula": c.candidate_id} for c in usable]
    pool = np.asarray(pool_X, dtype=float)
    result = MainSearchResult("widen_search", None, None, listed, first_report, report, None)
    exact = [c for c in usable if _relative(c.dev_resid, dev_y) < NUMERIC_FLOOR]
    if exact:
        rows = np.sort(np.random.default_rng(seed).choice(len(pool), size=min(budget, len(pool)), replace=False))
        targets = np.asarray(measure(rows), dtype=float)
        result.rows_read = len(rows)
        with np.errstate(all="ignore"):
            errors = {c.candidate_id: _relative(c.predict(pool[rows]) - targets, targets) for c in exact}
        held = [c.candidate_id for c in exact if np.isfinite(errors[c.candidate_id]) and errors[c.candidate_id] < 10 * NUMERIC_FLOOR]
        if held:
            law = min(held, key=lambda m: (size[m], held.index(m)))
            result.action = "confirmed" if len(held) == 1 else "equivalent_class"
            result.law, result.found_by = law, origin[law]
            result.adequacy = {"exact_fit": True, "pool_relative_error": errors[law], "passed": True}
        else:  # exact on the development rows and not on fresh ones: not a law; the others may still be
            usable = [c for c in usable if c not in exact]
    if result.law is None and usable:
        try:
            outcome = cc.contrast_confirm(usable, pool, measure, noise=noise, budget=budget, seed=seed)
        except ValueError:  # no candidate with finite pool predictions and a positive development error
            outcome = None
        if outcome is not None:
            law = outcome.survivor
            if law is not None and outcome.verdict == cc.EQUIVALENT_CLASS:
                members = [m for m in dict.fromkeys([law, *outcome.equivalent]) if m in size]
                law = min(members, key=lambda m: (size[m], members.index(m)))
            result.action, result.law, result.found_by = cr.ACTIONS[outcome.verdict], law, origin.get(law) if law else None
            result.contrast, result.rows_read = outcome, result.rows_read + outcome.rows_read
            survivor = next((c for c in usable if c.candidate_id == law), None)
            if survivor is not None and _relative(survivor.dev_resid, dev_y) >= NUMERIC_FLOOR:
                result.adequacy = adequacy(survivor.dev_resid, dev_X, dev_y, guided.k, seed)
                if not result.adequacy["passed"]:
                    result.action, result.rejected_law, result.law, result.found_by = "widen_search", result.law, None, None
    if ledger is not None:
        contrast = result.contrast
        metadata: dict[str, Any] = {
            "search_id": search_id, "resolved_by": "contrast", "found_by": result.found_by,
            "adequacy": result.adequacy, "rejected_law": result.rejected_law,
            "selection_pool_size": max(1, first.evaluated + report["proposals"]),
            "rows_read": result.rows_read,
            "design_sha256": contrast.design_sha256 if contrast else None,
        }
        if domain is not None:
            metadata["domain"] = domain
        ledger.record_trial(
            candidate_id=f"{search_id}:{result.law or 'none'}", proposer_id=PROPOSER_ID, formula=result.law or "",
            strategy_archetype="equation_discovery", in_sample_sharpe=0.0, n_bars=len(dev_y), status=result.action,
            selection_statistic=float("nan"), metadata=metadata)
    return result
