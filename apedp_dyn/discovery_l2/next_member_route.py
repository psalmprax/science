"""The route in force, and the next law of a class when the one it took fails the residual check.

Status: experimental

**Not in force. Development only: nothing here has been tested blind.**

**What it adds.** When several laws cannot be told apart on a table, the main search takes the shortest of them and
then checks its residuals. If that check fails, the reading ends with "widen the search", and the other laws of the
class are not looked at. A development study (``scripts/study_handed_truth.py``, 28 November 2026) handed the route
the true form last of ten laws on 122 tables: 13 readings ended this way, and in all 13 the true form was in the
class. This module takes the route's own record of such a reading and tries the class's other laws in the route's
own order (shortest first). Each is treated as the route treats the member it takes: fitted with one coefficient
for each additive term (``contrast_resolution.fitted_candidate``), its residuals checked
(``main_search.adequacy``), written in the fitted form with unnoticed terms dropped
(``refined_search.confirmed_and_pruned``), and handed to the last checks in force (``vouching_v2.vouch``) with every
other member of the class as its rivals. Those checks may report another member in its place; that member has to
pass the residual check as well, or nothing is reported. Where the route was asked for the first version of the
checks, or for none, the step does nothing.

The first build of this step read each member's text as it stood. A member's text is what a search wrote, and the
route never checks a law in that form (an outside review of 28 November 2026 pointed at the route's handing on of
unfitted texts, which is how it was noticed here). The step now fits before it checks. In the study beside it that
changed 7 of 366 readings: a member that failed the check as text passed it once fitted, so fewer were tried, and
in one reading a law was found, and not vouched for, where none had been.

**What it does not do.** It does not search, and it does not change what the route decides anywhere else: a reading
that ends in any other way, or with a law reported, is returned as it is, so the step can add a report and cannot
take one away. The frozen files are called and not edited. Its rivals are every other member of the class, the one
the contrast called the survivor included; the route in force hands its last checks the survivor's equivalents
only, which leaves the survivor out whenever a shorter member was taken in its place.

**Why it could be wrong to do this.** Every further law tried is one more chance for a wrong law to pass the two
checks. The study beside it counts what it reports on lists that hold no true law.

The owner's word for building it: "try all of these" (28 November 2026).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import contrast_confirmation as cc
from apedp_dyn.discovery_l2 import contrast_resolution as cr
from apedp_dyn.discovery_l2 import (
    default_route,
    main_search,
    refined_search,
    vouched_route,
    vouched_route_v2,
    vouching_v2,
)
from apedp_dyn.discovery_l2.guided_search import GuidedSearch
from apedp_dyn.grammar.proposer_interface import ProposerBase

MAX_MEMBERS = 6          #: laws of a class tried after the first; a class has rarely more
NEIGHBOURS = int(inspect.signature(GuidedSearch.__init__).parameters["k"].default)   #: as the main search's residual check


def _fitted(law: str, names: list[str], X: np.ndarray, y: np.ndarray, noise: str) -> cc.FittedCandidate | None:
    """The law as the main search fits a candidate: one coefficient for each additive term. None if it cannot be."""
    try:
        expr = sp.sympify(law, locals={name: sp.Symbol(name) for name in names})
        with np.errstate(all="ignore"):
            candidate = cr.fitted_candidate(expr, names, X, y, noise)
    except Exception:  # noqa: BLE001 - a member that cannot be fitted is passed over, as the main search passes over a candidate
        return None
    return candidate if np.all(np.isfinite(candidate.coef)) and np.all(np.isfinite(candidate.dev_pred)) else None


def _size(law: str, names: list[str]) -> int:
    """The main search's measure of a law's length: operations and constants."""
    expr = sp.sympify(law, locals={name: sp.Symbol(name) for name in names})
    return int(sp.count_ops(expr)) + len(expr.atoms(sp.Float))


def next_member(record: Mapping[str, Any], X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, *,
                searched: bool, noise: str = "relative", seed: int = 0) -> dict[str, Any]:
    """``record`` as the route gave it, or with the class's next law in place of the one that failed.

    ``searched``: whether the reading had a search stage. The route reports a law from a class only then, and so
    does this. ``noise`` and ``seed`` as the route was given them.
    """
    out = dict(record)
    state: dict[str, Any] = {"applied": False, "tried": []}
    out["next_member"] = state
    path = record.get("main_search")
    contrast = getattr(path, "contrast", None)
    rejected = getattr(path, "rejected_law", None)
    if record.get("reported") and record.get("law"):
        state["why_not"] = "the route reported a law"
    elif record.get("route") != default_route.REFINED:
        state["why_not"] = "measured behind the refined path only"
    elif (record.get("vouching") or {}).get("version") != vouched_route_v2.SECOND:
        state["why_not"] = "measured with the second version of the last checks only"
    elif not searched:
        state["why_not"] = "no search stage: the route reports nothing from a class then"
    elif getattr(path, "action", None) != "widen_search" or not rejected:
        state["why_not"] = "the reading did not end at the residual check"
    elif contrast is None or contrast.verdict != cc.EQUIVALENT_CLASS:
        state["why_not"] = "the law that failed was not taken from a class"
    if "why_not" in state or contrast is None:
        return out
    names = [str(name) for name in X]
    dev_X = np.column_stack([np.asarray(X[name], dtype=float) for name in X])
    dev_y = np.asarray(Y, dtype=float)
    listed = [str(c["formula"]) for c in (getattr(path, "candidates", None) or [])]
    members = [m for m in dict.fromkeys([contrast.survivor, *contrast.equivalent]) if m]
    others = [m for m in members if m != rejected]
    try:
        others.sort(key=lambda m: (_size(m, names), listed.index(m) if m in listed else len(listed)))
    except Exception as exc:  # noqa: BLE001 - a class that cannot be ordered is left as the route left it
        state["why_not"] = f"the class could not be ordered: {type(exc).__name__}"
        return out
    state.update({"applied": True, "class": len(members), "rejected_first": str(rejected)})
    keep = vouched_route.rows_given(len(dev_y), len(names))            # the rows the last checks read, as in the route
    check_X, check_y = dev_X, dev_y
    if keep < len(dev_y):
        picked = np.sort(np.random.default_rng(0).choice(len(dev_y), keep, replace=False))
        check_X, check_y = dev_X[picked], dev_y[picked]
    for member in others[:MAX_MEMBERS]:
        fitted = _fitted(member, names, dev_X, dev_y, noise)
        if fitted is None:
            state["tried"].append({"residual_check": None})
            continue
        residuals = main_search.adequacy(fitted.dev_resid, dev_X, dev_y, NEIGHBOURS, seed)
        tried: dict[str, Any] = {"residual_check": bool(residuals["passed"])}
        state["tried"].append(tried)
        if not residuals["passed"]:
            continue
        # written as the refined path writes the law it reports: fitted term by term, unnoticed terms dropped
        written, dropped = refined_search.confirmed_and_pruned(member, names, dev_X, dev_y, noise)
        rivals = [m for m in members if m != member]
        try:
            done = vouching_v2.vouch(str(written), rivals, names, check_X, check_y, np.asarray(pool_X, dtype=float))
        except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing, as in the route
            done = vouching_v2.Vouched(None, f"the checks failed: {type(exc).__name__}: {exc}"[:200])
        law, reason = done.law, done.reason
        if law is not None and bool(done.switched):            # the checks chose another member: it has to pass the residual check too
            chosen = _fitted(law, names, dev_X, dev_y, noise)
            if chosen is None or not main_search.adequacy(chosen.dev_resid, dev_X, dev_y, NEIGHBOURS, seed)["passed"]:
                law, reason = None, "the member the checks chose in its place fails the residual check"
        tried.update({"vouched": law is not None, "reason": reason, "switched": bool(done.switched), "dropped_terms": list(dropped)})
        if law is None:
            out.update({"law": None, "reported": False, "tentative_law": str(written), "law_as_confirmed": str(written),
                        "tentative_reason": f"{vouched_route_v2.NOT_VOUCHED}: {reason}", "action": "equivalent_class"})
        else:
            out.update({"law": law, "reported": True, "law_as_confirmed": law, "action": "equivalent_class",
                        "tentative_law": None, "dropped_terms": list(dropped)})
        state["took"] = len(state["tried"])
        return out                                             # the first member with clean residuals decides, either way
    return out


def discover_law(X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
                 proposer: ProposerBase | None = None, noise: str = "relative", seed: int = 0, **route: Any) -> dict[str, Any]:
    """``vouched_route_v2.discover_law`` and then :func:`next_member`. Same arguments; the record gains ``next_member``."""
    record = vouched_route_v2.discover_law(X, Y, pool_X, measure, proposer=proposer, noise=noise, seed=seed, **route)
    return next_member(record, X, Y, pool_X, searched=proposer is not None, noise=noise, seed=seed)
