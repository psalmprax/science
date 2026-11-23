"""The measured discovery routes, for the UI: what they conclude on a table of data, and what the status page says.

Status: experimental

Until 2026-11-20 every discovery endpoint of the UI called the template engines
(``L2DiscoveryEngine``, ``MultiTermL2Engine``) and showed the referee's verdict
on their candidates. The benchmarks since then changed what should be shown
first: the default route is the refined path (``discovery_l2/default_route.py``,
PREREG-REFINED-SEARCH; the main search before it, PREREG-MAIN-SEARCH), the
template engine's promotions were wrong in 60 % of cases on SRBench (audit
M-10), and return-like series need their own route (PREREG-MARKET-PLANTED).

This module is what the endpoints call for that. It decides nothing itself:

* ``route_reading`` runs ``discovery_l2.vouched_route.discover_law`` on tabular
  rows (the default route unless one is named, with the checks of
  ``discovery_l2/vouching.py`` as its last step since 2026-11-22) and returns a
  JSON-safe record. A law the checks do not let through is shown as tentative,
  with the check it failed. The UI has no physical units for a table, so the
  refined path runs without its units reduction here.
* ``series_reading`` runs ``discovery_l2.series_route`` on rows in time order;
  ``market_series_reading`` does that on a market window's train slice.
* ``status_block`` is the list the status page shows: the default route, the
  routes that exist, the engines that were retired and why, and the recorded
  result behind each statement. ``status_fields`` is what ``/api/status``
  merges in.
* ``mount`` registers ``POST /api/discover/route`` on the app: the routes on a
  table the caller sends. (Registered directly, not through an ``APIRouter``,
  so the endpoint is listed in ``/api/status`` on every FastAPI version.)

The endpoint and the status fields live here and not in ``ui/server.py``
because that file is at its size ceiling (``scripts/audit_exception_hygiene.py``)
and should not grow; it calls ``mount`` and the two functions.

**The search stage.** PySR needs a Julia runtime and its first call in a
process can take minutes, which an interactive request should not spend by
surprise. So PySR runs only when a budget is given (``search_seconds`` in the
request, or the environment variable ``APEDP_UI_SEARCH_SECONDS``); without one
the route runs its exhaustive first pass and, on the refined path, the
polynomial library, and the record says the reading is partial. With a budget and no PySR installed, the
record says that too.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI

from apedp_dyn.discovery_l2 import default_route, series_route, vouched_route
from apedp_dyn.market.assets import compute_strictly_causal_features, load_series_with_dates
from apedp_dyn.orchestration.competition import RETIRED_ENGINES
from apedp_dyn.ui.helpers import _audit, _json_obj, _require_api_key

SEARCH_SECONDS_VARIABLE = "APEDP_UI_SEARCH_SECONDS"
MAX_SEARCH_SECONDS = 300.0
POOL_SHARE = 0.3
MIN_ROWS = 24


def _search_budget(requested: float | None) -> float:
    if requested is None:
        try:
            requested = float(os.environ.get(SEARCH_SECONDS_VARIABLE, "0") or 0.0)
        except ValueError:
            requested = 0.0
    return float(min(max(requested, 0.0), MAX_SEARCH_SECONDS))


def _proposer(budget: float) -> tuple[Any, str]:
    """The search stage in force with the budget (four searches where four cores are free, else PySR, and
    the memory of reported laws beside them), or ``None`` with the reason there is no search stage."""
    if budget <= 0.0:
        return None, f"off: first pass alone (give search_seconds, or set {SEARCH_SECONDS_VARIABLE}, to run PySR)"
    try:
        import pysr  # noqa: F401 - only to learn whether the search can run at all

    except Exception as exc:  # noqa: BLE001 - a missing package and a failed Julia start are the same answer here
        return None, f"unavailable: PySR could not be loaded ({type(exc).__name__}); first pass alone"
    from apedp_dyn.discovery_l2 import stage_in_force

    stage: tuple[Any, str] = stage_in_force.search_stage(budget)
    return stage


def _clean(X: Mapping[str, Any], y: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = [str(n) for n in X]
    columns = np.column_stack([np.asarray(X[n], dtype=float).ravel() for n in X]) if names else np.empty((0, 0))
    target = np.asarray(y, dtype=float).ravel()
    if not names or len(target) != len(columns):
        raise ValueError("inputs and target must have the same number of rows, and there must be at least one input")
    keep = np.all(np.isfinite(columns), axis=1) & np.isfinite(target)
    return names, columns[keep], target[keep]


def route_reading(X: Mapping[str, Any], y: Any, *, route_name: str | None = None, noise: str = "absolute",
                  search_seconds: float | None = None, seed: int = 0) -> dict[str, Any]:
    """What the discovery route concludes on tabular rows. Never raises on bad data: the record carries the reason."""
    started = time.perf_counter()
    record: dict[str, Any] = {"route": route_name, "default_route": default_route.DEFAULT_ROUTE, "action": "not_run",
                              "law": None, "law_as_confirmed": None, "reported": False}
    try:
        names, columns, target = _clean(X, y)
        if len(target) < MIN_ROWS:
            raise ValueError(f"{len(target)} usable rows; the route needs at least {MIN_ROWS}")
        chosen = record["route"] = route_name or default_route.default_route()
        if chosen == "l2":
            raise ValueError("route 'l2' needs physical dimensions; this reading runs the refined path or the main search")
        order = np.random.default_rng(seed).permutation(len(target))
        cut = round((1.0 - POOL_SHARE) * len(target))
        dev, pool = order[:cut], order[cut:]
        pool_y = target[pool]
        proposer, stage = _proposer(_search_budget(search_seconds))
        out = vouched_route.discover_law({n: columns[dev, i] for i, n in enumerate(names)}, target[dev], columns[pool],
                                         lambda rows: pool_y[rows], ledger=None, search_id="ui", route_name=chosen,
                                         proposer=proposer, noise=noise, seed=seed)
        result = out["main_search"]
        record.update({
            "route": out["route"], "action": out["action"], "law": out["law"], "law_as_confirmed": out["law_as_confirmed"],
            # a law that is not reported is shown apart, with the reason: nothing challenged it (audit M-14),
            # the search stage did not run, or the last checks did not vouch for it (audit M-31, M-32)
            "reported": bool(out["reported"]),
            "vouching": dict(out.get("vouching") or {}),
            "tentative": out.get("tentative_law") is not None,
            "tentative_reason": out.get("tentative_reason"),
            "dropped_terms": [str(t) for t in out.get("dropped_terms", [])],
            "found_by": result.found_by, "search_stage": stage, "candidates": len(result.candidates),
            # the measured route is first pass AND PySR; a reading without PySR is partial
            "complete": proposer is not None,
            "rows_development": len(dev), "rows_pool": len(pool), "rows_read_from_pool": int(result.rows_read),
            "rejected_law": result.rejected_law,
            "adequacy": {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else bool(v) if isinstance(v, (bool, np.bool_)) else v)
                         for k, v in (result.adequacy or {}).items()},
        })
    except Exception as exc:  # noqa: BLE001 - a reading that cannot be made is reported in the record, not raised into the page
        record.update({"action": "not_run", "error": f"{type(exc).__name__}: {exc}"[:300]})
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["evidence"] = (default_route.EVIDENCE if record["route"] == default_route.REFINED
                          else "preregistration/RESULT-MAIN-SEARCH-2026-11.md")
    return record


def series_reading(X: Mapping[str, Any], y: Any, *, search_seconds: float | None = None, seed: int = 0,
                   development_share: float = 0.6) -> dict[str, Any]:
    """What the series route concludes on rows in time order: a law only if it forecasts the later rows."""
    started = time.perf_counter()
    record: dict[str, Any] = {"route": "series_route", "action": "not_run", "law": None, "reported": False}
    try:
        names, columns, target = _clean(X, y)
        cut = round(development_share * len(target))
        if cut < MIN_ROWS or len(target) - cut < 20:
            raise ValueError(f"{len(target)} usable rows; the series route needs {MIN_ROWS} to develop on and 20 after them")
        proposer, stage = _proposer(_search_budget(search_seconds))
        result = series_route.discover_predictive_law({n: columns[:cut, i] for i, n in enumerate(names)}, target[:cut],
                                                      columns[cut:], target[cut:], proposer=proposer, seed=seed)
        record.update({
            "action": result.action, "law": result.law, "reported": result.law is not None,
            "chosen_on_development": result.selected.get("chosen"),
            "forecast_test": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in result.forecast_test.items()},
            "path_alone": {"action": result.ungated_action, "law": result.ungated_law,
                           "note": "the main search path by itself reports laws in pure noise on such data (audit M-16); do not act on this line"},
            "search_stage": stage, "complete": proposer is not None, "rows_development": cut, "rows_later": len(target) - cut,
        })
    except Exception as exc:  # noqa: BLE001 - reported in the record, not raised into the page
        record.update({"action": "not_run", "error": f"{type(exc).__name__}: {exc}"[:300]})
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["evidence"] = "preregistration/RESULT-MARKET-PLANTED-2026-11.md"
    return record


MAX_ROWS, MAX_COLUMNS = 20_000, 12


def route_request(req: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a request body and run the route it asks for. Bad bodies give a record with ``error``, not an exception."""
    columns, target = req.get("columns"), req.get("target")
    if not isinstance(columns, Mapping) or not isinstance(target, (list, tuple)) or not columns:
        return {"route": None, "action": "not_run", "law": None, "reported": False,
                "error": "body needs 'columns' (name -> list of numbers) and 'target' (list of numbers)"}
    if len(columns) > MAX_COLUMNS or len(target) > MAX_ROWS:
        return {"route": None, "action": "not_run", "law": None, "reported": False,
                "error": f"at most {MAX_COLUMNS} columns and {MAX_ROWS} rows"}
    seconds = req.get("search_seconds")
    budget = float(seconds) if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None
    if bool(req.get("time_ordered")):
        return series_reading(columns, target, search_seconds=budget)
    return route_reading(columns, target, search_seconds=budget)


def discover_route(req: dict[str, Any]) -> dict[str, Any]:
    """The measured discovery routes on a table of the caller's own.

    Body: ``columns`` (name -> list of numbers), ``target`` (list of numbers),
    optional ``time_ordered`` (true: the series route, a law only if it forecasts
    later rows), optional ``search_seconds`` (run PySR for that long, if installed).
    """
    out = route_request(req)
    _audit("discover/route", {"verdict": str(out.get("action")), "data_origin": "caller", "route": str(out.get("route"))})
    safe: dict[str, Any] = _json_obj(out)
    return safe


def mount(app: FastAPI) -> None:
    """Register this module's endpoint on the UI app, behind the same key check as the others."""
    app.post("/api/discover/route", dependencies=[Depends(_require_api_key)])(discover_route)


def market_series_reading(asset: str, window: Any) -> dict[str, Any]:
    """The series route on a market window's TRAIN slice (the holdout never enters): a law only if it forecasts later rows.

    Return-like data is the case the main search must not be used on alone (it
    reported a law in 28 of 40 simulated series that contained none; audit
    M-16). The first 60 % of the train slice is the development period and the
    rest the later period the forecast test reads. The same features as the
    engine competition on that window.
    """
    dates, prices = load_series_with_dates(asset)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    causal = compute_strictly_causal_features(prices)
    sl = slice(date_to_idx[window.train_start], date_to_idx[window.train_end] + 1)
    X = {"r_lag": causal["r_lag"][sl], "vol": causal["vol"][sl], "ma_dev": causal["ma_dev"][sl]}
    return series_reading(X, causal["returns"][sl])


def status_fields() -> dict[str, Any]:
    """What ``/api/status`` merges in: the engine list it always carried, and the measured routes."""
    return {"engines": ["L2DiscoveryEngine", "MultiTermL2Engine", "L2ODEDiscoveryEngine"], "discovery": status_block()}


def status_block() -> dict[str, Any]:
    """What the status page says about routes and engines, each line with the recorded result behind it."""
    return {
        "default_route": default_route.DEFAULT_ROUTE,
        "routes": [
            {"name": "refined", "for": "tabular data where a law explains nearly all of the target; the default since 2026-11-20",
             "what": ("the main search with the units' dimensionless products where units are given, a polynomial library, "
                      "PySR on every problem, and the law written in its confirmed form with unnoticed terms dropped"),
             "measured": ("SRBench: 0 of 212 reported laws not accurate; 78 and 77 of 119 Feynman laws recovered; "
                          "one of three expectations missed by one law"),
             "evidence": default_route.EVIDENCE},
            {"name": "main_search", "for": "the default until 2026-11-20; set APEDP_LAW_DISCOVERY_ROUTE=main_search to use it",
             "what": "exhaustive first pass, PySR, contrast confirmation, adequacy check",
             "measured": "SRBench: 4 of 186 reported laws not accurate; 69 and 70 of 119 Feynman laws recovered",
             "evidence": "preregistration/RESULT-MAIN-SEARCH-2026-11.md"},
            {"name": "ode", "for": "trajectories of a dynamical system (apedp_dyn/ode/route_in_force.py; no endpoint here runs it)",
             "what": ("one state and one rate per disjoint window; four searches and sparse regression side by side; terms "
                      "below the rows' precision dropped; where the system can be started again, a follow-up experiment"),
             "measured": ("46 systems it was not built on: 117 of 276 components recovered, 3 of 165 reported laws wrong; "
                          "with the experiment 120, read off that run's rows. At 1 % noise two of three reported laws are "
                          "accurate and are not the law"),
             "evidence": "preregistration/RESULT-ODE-ADDITIONS-2026-11.md"},
            {"name": "series_route", "for": "noisy series in time order (return-like data)",
             "what": "candidates from the main search, one chosen on the development period, one forecast test on later rows",
             "measured": "simulated series: a law in 0 of 40 null series; 31 of 40 where it explains 5 or 20 % of the variance",
             "evidence": "preregistration/RESULT-MARKET-PLANTED-2026-11.md"},
            {"name": "l2", "for": "kept for replication",
             "what": "the template engine and the referee's verdict on its candidates",
             "measured": "SRBench: 160 of 268 promotions not accurate (audit M-10)",
             "evidence": "preregistration/RESULT-SRBENCH-BASELINE-2026-11.md"},
        ],
        "retired_engines": {
            **dict(RETIRED_ENGINES),
            "gagp": ("retired 2026-11-19 as a law-discovery engine: 53 and 7 of 119 Feynman laws against 73 and 68 for PySR "
                     "(PREREG-MAIN-SEARCH-2026-11). Kept importable."),
        },
        "search_stage": _proposer(_search_budget(None))[1],
        "search_stage_measured": ("four searches side by side: SRBench, 73 and 63 of 103 problems recovered against 61 and "
                                  "61 for one search (preregistration/RESULT-PORTFOLIO-SEARCH-2026-11.md). A memory of "
                                  "reported laws beside them: 23 of 80 new laws recovered against 17, none lost, no wrong "
                                  "law; nearly all of the gain without noise, and on laws of one grammar "
                                  "(preregistration/RESULT-MEMORY-2026-11.md)"),
        "endpoints": {
            "/api/discover/physics": "template engine and referee, with the main search reading beside it (route_reading)",
            "/api/discover/market": "template engine and referee, with the series route reading beside it (series_reading)",
            "/api/discover/route": "the measured routes alone, on a table the caller sends",
            "/api/discover/registry": ("template engine and referee only: its data is one trajectory of a driven system, "
                                       "which the ODE route was not measured on (ODEBench systems are autonomous, several trajectories)"),
        },
        "last_step": {
            "in_force": vouched_route.vouching_in_force(),
            "what": ("checks on a law the refined path is about to report: structure left in its residuals, members of "
                     "an unseparated class that disagree, and whether an error large enough to make it wrong could "
                     "have been seen. A law that fails is shown as tentative with the reason"),
            "measured": ("Cost first: on SRBench behind the four searches they kept 150 of the 193 laws the route "
                         "recovered (47 of 90 at 1 % noise) and stopped nothing, since no report there was inaccurate; "
                         "one expectation missed, in force until the owner decides "
                         "(preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md). On 60 new laws handed over with their "
                         "counterfeits, noise 0, 1 % and 5 %: wrong laws 16 of 229 reports before the checks and 2 of "
                         "191 after, all at 5 % noise; 139 of 146 right laws kept"),
            "evidence": vouched_route.EVIDENCE,
            "cost": "preregistration/RESULT-VOUCHED-SRBENCH-2026-11.md",
            "switch": f"{vouched_route.ENVIRONMENT_VARIABLE}={vouched_route.OFF}"},
        "caution": ("A law marked tentative was not challenged by any rival, or the last checks did not vouch for it: "
                    "the reason is shown with it. On return-like data use the series route; "
                    "the search alone reports laws in pure noise there. The UI passes no physical units, so the "
                    "refined path runs here without its units reduction."),
    }
