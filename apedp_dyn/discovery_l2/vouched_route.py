"""The route in force for tables since 2026-11-22: the default route, and a law is reported only if the checks let it through.

Status: experimental

**Why this module exists.** PREREG-COUNTERFEIT-2026-11 fixed, before its run,
that if its third, fourth and fifth expectations held "the checks become the
last step of the route in force for tables, in a new module around the frozen
ones, with a switch that turns them off. A law that fails them is shown as not
vouched for, with the reason, and is not reported." They held.
``default_route``, ``refined_search`` and ``vouching`` are not edited. Code
that wants "the route for a table" calls ``discover_law`` here.

**The evidence** (``preregistration/RESULT-COUNTERFEIT-2026-11.md``, run once,
2026-11-22): 60 laws drawn after the checks were written, each handed to the
route with its counterfeits as candidates, with the law among them or with
only wrong ones, at noise 0, 1 % and 5 %. The route reported 16 wrong laws in
229 reports, all at 5 % noise; after the checks 2 in 191. Of the 146 right
laws the checks kept 139 and they turned 7 other reports into the law. Scored
with a wrong law costing nineteen right ones: -0.88 before, 0.60 after.

**What that does and does not say.**

* At noise up to 1 % the route reported no wrong law without the checks (0 of
  141) and the checks changed little there: four approximations became the
  law, five were withheld. What they are for is noise beyond what the earlier
  runs measured.
* At 5 % noise they let 2 of 16 wrong laws through, 2 of the 55 laws they
  passed (3.6 %; the 95 % interval reaches 12.5 %), and withheld 7 of 35 right
  laws. Both wrong laws were the route's own forms, wrong by a small margin.
  For an error of another shape than those the power rule plants, its
  estimate is too high (audit M-33).
* The candidates were handed over. No search engine ran, and what the checks
  cost in recovered laws on the physics benchmark has not been measured.
  Rows were uniform and the noise Gaussian.
* Nothing has searched for data that defeats the checks.

**Cost.** About a third of a second for a law on 700 rows. The checks compare
every pair of rows, so they read at most 1,500 rows, and fewer where a table
has more than seven inputs (``rows_given``): on such tables they have less to
go on than where they were measured.

``APEDP_TABLE_VOUCHING=off`` gives the route without the checks, as it was
before.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import default_route, vouching
from apedp_dyn.grammar.proposer_interface import ProposerBase

ENVIRONMENT_VARIABLE = "APEDP_TABLE_VOUCHING"
OFF = "off"
EVIDENCE = "preregistration/RESULT-COUNTERFEIT-2026-11.md"
NOT_VOUCHED = "not vouched for"
#: the checks compare every pair of rows in every input; wide tables are given fewer rows so that this stays in bounds
PAIR_BYTES = 256e6
FEWEST_ROWS = 64


def vouching_in_force() -> bool:
    """Whether the checks are the route's last step: yes, unless the environment switches them off."""
    return os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower() != OFF


def rows_given(count: int, inputs: int) -> int:
    """How many rows the checks read: all, up to their own limit; fewer where a table has more than seven inputs."""
    fits = int(np.sqrt(PAIR_BYTES / (16.0 * max(inputs, 1))))
    return int(min(count, vouching.MAX_ROWS, max(FEWEST_ROWS, fits)))


def checked(record: Mapping[str, Any], X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray) -> dict[str, Any]:
    """The record of ``default_route.discover_law`` after the checks.

    A reported law stays, gives way to another member of its class, or is
    withheld: then ``law`` is ``None``, ``reported`` is false, and the route's
    law is kept as ``tentative_law`` with ``tentative_reason`` saying which
    check it failed. ``vouching`` in the record says what was done. The checks
    read the rows the law was fitted on and the pool's inputs, nothing else.
    """
    out = dict(record)
    state: dict[str, Any] = {"in_force": vouching_in_force(), "applied": False, "evidence": EVIDENCE}
    out["vouching"] = state
    law = out.get("law")
    if not state["in_force"]:
        state["why_not"] = f"switched off ({ENVIRONMENT_VARIABLE}={OFF})"
    elif out.get("route") != default_route.REFINED:
        state["why_not"] = "the checks were measured behind the refined path only"
    elif not out.get("reported") or not law:
        state["why_not"] = "the route reported no law"
    if "why_not" in state:
        return out
    names = [str(name) for name in X]
    dev_X = np.column_stack([np.asarray(X[name], dtype=float) for name in X])
    dev_y = np.asarray(Y, dtype=float)
    keep = rows_given(len(dev_y), len(names))
    if keep < len(dev_y):                                   # the checks' own draw where their limit decides
        picked = np.sort(np.random.default_rng(0).choice(len(dev_y), keep, replace=False))
        dev_X, dev_y = dev_X[picked], dev_y[picked]
    contrast = getattr(getattr(out.get("refined"), "path", None), "contrast", None)
    others = [str(text) for text in (contrast.equivalent if contrast is not None else [])]
    try:
        done = vouching.vouch(str(law), others, names, dev_X, dev_y, np.asarray(pool_X, dtype=float))
    except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing: the rule of the run that measured them
        done = vouching.Vouched(None, f"the checks failed: {type(exc).__name__}: {exc}"[:200])
    state.update({"applied": True, "route_law": str(law), "reason": done.reason, "switched": bool(done.switched),
                  "power": done.power, "members": len(done.members), "refused": sum(m.refused for m in done.members),
                  "rows": len(dev_y)})
    if done.law is None:
        out.update({"law": None, "reported": False, "tentative_law": str(law),
                    "tentative_reason": f"{NOT_VOUCHED}: {done.reason}", "law_as_confirmed": str(law)})
    elif done.switched:
        out.update({"law": done.law, "law_as_confirmed": done.law})
    return out


def discover_law(
    X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray, measure: Callable[[np.ndarray], np.ndarray], *,
    ledger: Any = None, search_id: str | None = None, route_name: str | None = None,
    proposer: ProposerBase | None = None, units: Mapping[str, Mapping[str, Any]] | None = None,
    target_units: Mapping[str, Any] | None = None, dimensions: Mapping[str, Any] | None = None,
    target_dimension: Any = None, noise: str = "relative", seed: int = 0, domain: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """``default_route.discover_law``, with the checks as the last step of the refined path.

    Same arguments and the same record, with ``vouching`` added. The other
    routes are handed on as they were measured.
    """
    record = default_route.discover_law(X, Y, pool_X, measure, ledger=ledger, search_id=search_id, route_name=route_name,
                                        proposer=proposer, units=units, target_units=target_units, dimensions=dimensions,
                                        target_dimension=target_dimension, noise=noise, seed=seed, domain=domain,
                                        dataset_id=dataset_id)
    return checked(record, X, Y, pool_X)
