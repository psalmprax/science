"""A family of tables read together through the route in force, with the fallbacks behind it.

Status: experimental

**Not in force.** Nothing calls it but its study script and its tests. The route in force for a table
is ``vouched_route_v2.discover_law`` and stays so until a blind test and the owner's word (AGENTS.md 1.5).
Built on the owner's word of 27 November 2026 ("build the best of the above and make the others
fallback"; the designs are ``docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md``).

**What it does**, every step through the route in force and its referee (``read``):

1. *Each member by itself*, with the search stage it is given: what the route does today.
2. *Siblings.* Every law the route listed for any member is refitted on every member and the shapes are
   ordered by all rows of the family (``across_tables.score_shapes``). A member without a reported law is
   handed the best shapes that came from its siblings, refitted on its own rows, beside the laws its own
   search had proposed, and read again.
3. *The wider law.* If the family carries a condition, the constants of its best shapes are set against
   it, and the laws that result over the inputs and the condition go to the route on the pooled table:
   all rows of all members, the condition one more input (``across_tables.wider_laws``). If one is
   reported, it is written out for each member that still has no law (the condition at that member's
   value) and that member is read once more.
4. *Fallbacks* (``fallbacks.py``), for a member that still has no reported law: solved cases, the
   tensor-train reading, a language model, in the order given and only as far as needed.

**What decides.** The referee, on a member's rows for a member's law and on the family's rows for the
wider law. Nothing here reports, withholds or reorders a law the route confirmed (AGENTS.md 1.1): a member
whose first reading reported a law keeps it and is not read again.

**What it costs.** A second or third reading of a member replays laws; it runs no search. It does ask the
referee again, and the referee reads pool rows: with the same seed the same rows, so a replay measures
nothing the first reading had not measured unless a new rival makes it ask for more. ``readings`` in the
record counts them.

**A single table** (``discover_table``) is read as a family when one of its inputs takes few values
(``across_tables.by_levels``); its wider law is then a law of the table itself. Otherwise it is the
route, and then the fallbacks.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import sympy as sp

from apedp_dyn.discovery_l2 import across_tables, vouched_route_v2
from apedp_dyn.discovery_l2.across_tables import Family, Scored, Table
from apedp_dyn.discovery_l2.fallbacks import Fallback, try_in_order
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.ode.extended_route import Recording, replayed

Stage = Callable[[Table], "ProposerBase | None"]
SHAPES_RECORDED = 5


def read(table: Table, proposer: ProposerBase | None, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
    """One table by the route in force. Beside its record: the laws its stage proposed and the laws it listed."""
    stage = Recording(proposer) if proposer is not None else None
    started = time.perf_counter()
    record = vouched_route_v2.discover_law(table.columns(), table.y, table.pool_X, table.measure, proposer=stage,
                                           noise=noise, seed=seed)
    path = record.get("main_search")
    listed = [str(c["formula"]) for c in (getattr(path, "candidates", None) or [])]
    reported = bool(record.get("reported") and record.get("law"))
    return {"record": record, "reported": reported, "law": str(record["law"]) if reported else None,
            "tentative_law": str(record["tentative_law"]) if record.get("tentative_law") else None,
            "action": record.get("action"), "formulas": list(stage.formulas) if stage is not None else [], "listed": listed,
            "rows_read": getattr(path, "rows_read", None), "seconds": round(time.perf_counter() - started, 2)}


def summary(reading: dict[str, Any] | None) -> dict[str, Any] | None:
    """What of a reading can be written down."""
    if reading is None:
        return None
    record = reading["record"]
    vouching = record.get("vouching") or {}
    return {"reported": reading["reported"], "law": reading["law"], "tentative_law": reading["tentative_law"],
            "action": reading["action"], "listed": len(reading["listed"]), "rows_read": reading["rows_read"],
            "last_checks": {key: vouching.get(key) for key in ("in_force", "applied", "version") if key in vouching},
            "measured_range": (record.get("measured_range") or {}).get("state"), "seconds": reading["seconds"]}


def _laws_of(reading: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for law in (reading["law"], reading["tentative_law"], *reading["listed"]):
        if law and law not in out:
            out.append(law)
    return out


def _one_of(law: str | None, laws: Sequence[str], names: Sequence[str]) -> bool:
    shape = across_tables.shape_of(str(law), names) if law else None
    return shape is not None and shape.key in {s.key for s in (across_tables.shape_of(item, names) for item in laws) if s is not None}


def _shapes_record(scored: Sequence[Scored]) -> list[dict[str, Any]]:
    return [{"shape": item.shape.key, "sum_of_bic": None if item.score == float("inf") else round(item.score, 2),
             "from_members": sorted(item.from_members), "reported_on": sorted(item.reported_on)} for item in scored[:SHAPES_RECORDED]]


def discover_family(family: Family, stage: Stage, *, noise: str = "absolute", seed: int = 0, condition: str | None = None,
                    fallbacks: Sequence[Fallback] = (), siblings: bool = True, wider: bool = True) -> dict[str, Any]:
    """The family read together. The record: each member's first and final reading and where its law came from,
    the shapes, the wider law, and how many readings it took.

    ``stage(table)`` gives the search stage for a table (or ``None``). ``condition`` names the condition the wider law
    is read over; left out, the family's own if it carries exactly one.
    """
    started = time.perf_counter()
    names, tables = family.names, family.tables
    if condition is None and len(family.conditions) == 1:
        condition = family.conditions[0]
    readings = 0

    def again(table: Table, laws: Sequence[str]) -> dict[str, Any]:
        nonlocal readings
        readings += 1
        return read(table, replayed(list(dict.fromkeys(laws))), noise=noise, seed=seed)

    firsts = [read(table, stage(table), noise=noise, seed=seed) for table in tables]
    readings += len(tables)
    finals = list(firsts)
    sources: list[str | None] = ["route" if r["reported"] else None for r in firsts]
    handed: list[list[str]] = [[] for _ in tables]
    trails: list[list[dict[str, Any]]] = [[] for _ in tables]

    scored = across_tables.score_shapes(family, [_laws_of(r) for r in firsts], [r["law"] for r in firsts], seed=seed)
    if siblings:
        for i, table in enumerate(tables):
            if finals[i]["reported"]:
                continue
            handed[i] = across_tables.sibling_laws(family, scored, i)
            if not handed[i]:
                continue
            second = again(table, [*handed[i], *firsts[i]["formulas"]])
            if second["reported"]:
                finals[i] = second
                sources[i] = "siblings" if _one_of(second["law"], handed[i], names) else "second reading"
        newly = [(across_tables.shape_of(str(finals[i]["law"]), names), i) for i in range(len(tables))
                 if finals[i]["reported"] and not firsts[i]["reported"]]
        for item in scored:
            item.reported_on |= {i for shape, i in newly if shape is not None and shape.key == item.shape.key}
        scored = sorted(scored, key=lambda s: (0 if s.reported_on else 1, -len(s.reported_on), s.score, s.shape.key))

    wide: dict[str, Any] | None = None
    if wider and condition is not None:
        laws, how = across_tables.wider_laws(family, scored, condition, seed=seed)
        wide = {"condition": condition, "handed": laws, "how": how, "reading": None, "law": None, "reported": False}
        if laws:
            whole = across_tables.pooled(family, condition)
            reading = again(whole, laws)
            wide.update({"reading": summary(reading), "law": reading["law"], "reported": reading["reported"]})
            if reading["reported"]:
                symbol = sp.Symbol(condition)
                expr = sp.sympify(reading["law"], locals={n: sp.Symbol(n) for n in whole.names})
                for i, table in enumerate(tables):
                    if finals[i]["reported"]:
                        continue
                    own = str(expr.xreplace({symbol: sp.Float(float(table.conditions[condition]))}))
                    third = again(table, [own, *handed[i], *firsts[i]["formulas"]])
                    if third["reported"]:
                        finals[i] = third
                        sources[i] = "wider law" if _one_of(third["law"], [own], names) else "third reading"

    for i, table in enumerate(tables):
        if finals[i]["reported"] or not fallbacks:
            continue
        after, trails[i] = try_in_order(table, [*handed[i], *firsts[i]["formulas"]], fallbacks, again)
        if after is not None:
            finals[i] = after
            sources[i] = trails[i][-1]["fallback"]

    members = [{"key": table.key, "conditions": dict(table.conditions), "first": summary(firsts[i]), "final": summary(finals[i]),
                "source": sources[i], "handed_from_siblings": handed[i], "fallbacks": trails[i],
                "first_proposed": list(firsts[i]["formulas"]), "first_listed": _laws_of(firsts[i])} for i, table in enumerate(tables)]
    return {"family": family.key, "inputs": names, "members": members, "shapes": _shapes_record(scored), "wider": wide,
            "reported_first": sum(r["reported"] for r in firsts), "reported_final": sum(r["reported"] for r in finals),
            "readings": readings, "seconds": round(time.perf_counter() - started, 2)}


def discover_table(table: Table, stage: Stage, *, noise: str = "absolute", seed: int = 0, fallbacks: Sequence[Fallback] = (),
                   levels: bool = True) -> dict[str, Any]:
    """One table: the route in force; then, if one of its inputs takes few values, the table as a family; then the fallbacks.

    The record's ``law`` is the law reported for the table itself, by the route, by the family's wider law (which is a
    law of this table: the condition is one of its inputs), or after a fallback; ``source`` says which.
    """
    started = time.perf_counter()
    first = read(table, stage(table), noise=noise, seed=seed)
    out: dict[str, Any] = {"table": table.key, "first": summary(first), "law": first["law"], "reported": first["reported"],
                           "source": "route" if first["reported"] else None, "family": None, "fallbacks": []}
    known = list(first["formulas"])
    if not out["reported"] and levels:
        family = across_tables.by_levels(table)
        if family is not None:
            record = discover_family(family, stage, noise=noise, seed=seed)
            out["family"] = record
            wide = record.get("wider") or {}
            if wide.get("reported"):
                out.update({"law": wide["law"], "reported": True, "source": "family"})
            known = [*(wide.get("handed") or []), *known]
    if not out["reported"] and fallbacks:
        reading, out["fallbacks"] = try_in_order(
            table, known, fallbacks, lambda t, laws: read(t, replayed(list(dict.fromkeys(laws))), noise=noise, seed=seed))
        if reading is not None:
            out.update({"law": reading["law"], "reported": True, "source": out["fallbacks"][-1]["fallback"]})
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out
