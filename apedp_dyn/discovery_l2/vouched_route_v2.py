"""The route in force for tables since 2026-11-23: the default route, with the second version of the last checks as its last step.

Status: experimental

**Why this module exists.** PREREG-VOUCHING-V2-2026-11 fixed before its run
that if all five of its expectations held, "the second version becomes the
last step of the route in force for tables, in a new module around the frozen
ones, with its cost and benefit here stated wherever the route is described,
and M-34 and M-36 stated as open: it was tested up to 5 % noise only, and
above that it is known to be weaker than the first version." All five held.
``default_route``, ``vouching``, ``vouching_v2``, ``vouched_route`` and
``table_route`` are not edited. Code that wants "the route for a table" calls
``discover_law`` here.

**The evidence** (``preregistration/RESULT-VOUCHING-V2-2026-11.md``, run once,
2026-11-23, on 100 laws drawn after the module's last change):

* Handed counterfeits of 60 laws, noise 0, 1 % and 5 %, no search engine: the
  route reported 19 wrong laws in 231 reports (18 of 92 at 5 % noise); after
  the second version none in 189. It kept 109 of the route's 110 right laws at
  noise up to 1 % and 29 of 30 at 5 %. The first version, beside it: 1 wrong
  law in 188 reports, and 109 and 27 kept.
* Behind the search stage in force on 40 laws: at 5 % noise the route reported
  31 laws, 11 of them inaccurate; the second version reported 17, none
  inaccurate. It kept 26 of the 27 laws the route recovered at noise up to
  1 %, and 3 of 4 at 5 %.

**Cost, stated first wherever the route is described.** It withholds or
replaces some right laws: one of 27 recovered laws behind the search at noise
up to 1 % and one of 4 at 5 %; one of 110 and one of 30 handed right laws.
Earlier, in development on the physics benchmark's recorded run, it kept 192
of the 193 laws the route recovered. It takes about a third of a second for a
law on 700 rows.

**Open.** Above 5 % noise it was not tested here, and on the red-team
campaign's worlds it let more wrong laws through than the first version
(audit M-36). A law with a pole in the domain can pass (M-34). The tables were
up to three inputs, the noise Gaussian, the rows uniform.

``APEDP_TABLE_VOUCHING=off`` gives the route without last checks.
``APEDP_TABLE_VOUCHING=first`` (or ``on``, which asked for the first version
until this module) gives the first version, as ``vouched_route`` applies it.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from apedp_dyn.discovery_l2 import default_route, vouched_route, vouching_v2
from apedp_dyn.grammar.proposer_interface import ProposerBase

ENVIRONMENT_VARIABLE = vouched_route.ENVIRONMENT_VARIABLE
SECOND = "second"
FIRST = "first"
OFF = "off"
#: what asked for the first version while it was on request only (``table_route``); it still does
FIRST_ALIASES = (FIRST, "on")
EVIDENCE = "preregistration/RESULT-VOUCHING-V2-2026-11.md"
DECISION = ("the second version is the last step of the route in force since 2026-11-23, by the consequence "
            "PREREG-VOUCHING-V2-2026-11 fixed before its run; all five expectations held")
NOT_VOUCHED = vouched_route.NOT_VOUCHED


def version() -> str:
    """Which last checks the route applies: the second version unless the environment asks otherwise."""
    value = os.environ.get(ENVIRONMENT_VARIABLE, "").strip().lower()
    if value == OFF:
        return OFF
    if value in FIRST_ALIASES:
        return FIRST
    return SECOND


def in_force() -> bool:
    """Whether any last checks are applied."""
    return version() != OFF


def checked(record: Mapping[str, Any], X: Mapping[str, np.ndarray], Y: np.ndarray, pool_X: np.ndarray) -> dict[str, Any]:
    """The record of ``default_route.discover_law`` after the last checks in force.

    As in ``vouched_route.checked``: a reported law stays, gives way to another
    member of its class, or is withheld, and then ``law`` is ``None``,
    ``reported`` is false and the route's law is kept as ``tentative_law``
    with the reason. ``vouching`` in the record says which version ran and
    what it did. The checks read the rows the law was fitted on and the
    pool's inputs, nothing else.
    """
    chosen = version()
    if chosen == FIRST:
        first: dict[str, Any] = vouched_route.checked(record, X, Y, pool_X)
        first["vouching"] = {**first["vouching"], "version": FIRST,
                             "note": f"the first version, asked for with {ENVIRONMENT_VARIABLE}"}
        return first
    out = dict(record)
    state: dict[str, Any] = {"in_force": chosen != OFF, "version": chosen, "applied": False, "evidence": EVIDENCE}
    out["vouching"] = state
    law = out.get("law")
    if chosen == OFF:
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
    keep = vouched_route.rows_given(len(dev_y), len(names))   # the structure test compares pairs of rows, as in the first
    if keep < len(dev_y):
        picked = np.sort(np.random.default_rng(0).choice(len(dev_y), keep, replace=False))
        dev_X, dev_y = dev_X[picked], dev_y[picked]
    contrast = getattr(getattr(out.get("refined"), "path", None), "contrast", None)
    others = [str(text) for text in (contrast.equivalent if contrast is not None else [])]
    try:
        done = vouching_v2.vouch(str(law), others, names, dev_X, dev_y, np.asarray(pool_X, dtype=float))
    except Exception as exc:  # noqa: BLE001 - checks that crash vouch for nothing: the rule of the run that measured them
        done = vouching_v2.Vouched(None, f"the checks failed: {type(exc).__name__}: {exc}"[:200])
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
    """``default_route.discover_law``, with the last checks in force as the last step of the refined path.

    Same arguments and the same record, with ``vouching`` added. The other
    routes are handed on as they were measured.
    """
    record = default_route.discover_law(X, Y, pool_X, measure, ledger=ledger, search_id=search_id, route_name=route_name,
                                        proposer=proposer, units=units, target_units=target_units, dimensions=dimensions,
                                        target_dimension=target_dimension, noise=noise, seed=seed, domain=domain,
                                        dataset_id=dataset_id)
    return checked(record, X, Y, pool_X)
