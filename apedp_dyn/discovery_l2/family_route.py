"""A family of tables read together through the route in force, with the fallbacks behind it.

Status: experimental

**Not in force.** Nothing calls it but its study script and its tests. The route in force for a table
is ``vouched_route_v2.discover_law`` and stays so until a blind test and the owner's word (AGENTS.md 1.5).
Built on the owner's word of 27 November 2026 ("build the best of the above and make the others
fallback"; the designs are ``docs/SOLUTIONS_FOR_NO_GAIN_ITEMS_2026-11-27.md``).

**What it does**, every step through the route in force and its referee (``read``):

1. *Each member by itself*, with the search stage it is given: what the route does today.
2. *Siblings.* Every law any member's search proposed is refitted on every member and the shapes are
   ordered by all rows of the family (``across_tables.score_shapes``). A member without a reported law is
   handed the best shapes its referee has not had before it, refitted on its own rows, and read again.
3. *The wider law.* If the family carries a condition, the constants of its best shapes are set against
   it, and the laws that result over the inputs and the condition go to the route on the pooled table:
   all rows of all members, the condition one more input (``across_tables.wider_laws``). With
   ``pooled_stage`` the pooled table is searched as well and the family's laws stand beside what that
   search proposed. A wider law that is reported is written out for each member that still has no law
   (the condition at that member's value), and that member is read once more.
4. *Fallbacks* (``fallbacks.py``), for a member that still has no reported law: solved cases, the
   tensor-train reading, a language model, in the order given and only as far as needed.

**Two rules for every reading after the first**, both learned from the first development run
(``benchmarks/family_stream/dev_run1.json``), in which a member put into a family it did not belong to
gained a law almost as often as one in its own:

* *A re-reading brings in only what was handed in for it.* What the referee reports on a table depends on
  the list of laws before it. With six unrelated laws added to a fixed list, some tables it had refused
  get a law and some it had reported lose theirs (``benchmarks/family_stream/dev_list_sensitivity.json``:
  3 gained and 5 lost of 160). The route as first built read again only the tables without a law and
  counted whatever was then reported, so it could collect only changes of the first kind. Now a re-reading
  counts only if the law it reports is one that was handed in for it (``brought_in``). If the referee
  reports one of the table's own laws instead, that is the same candidates before other rivals, not new
  knowledge: the member stays as its first reading left it, and the record says a law was passed over
  (``passed_over``).
* *Beside, never in place.* The route gives a search stage ten of the twelve candidates its referee
  contrasts, in the order given, the first its starting champion. The route as first built put the laws it
  handed in at the head of the list, where six of them led it and left room for the table's first four.
  Now the table's own laws and the laws handed in take turns, the table's own first
  (``across_tables.beside``), so the table's own first five stay before the referee as rivals.

**What decides.** The referee, on a member's rows for a member's law and on the family's rows for the
wider law. Nothing here reports a law the referee did not confirm, and a member whose first reading
reported a law keeps it and is not read again (AGENTS.md 1.1).

**What it costs.** A re-reading replays laws; it runs no search (the pooled search of ``pooled_stage`` is
the one exception, and it is a first reading of that table). It does ask the referee again, and the
referee reads pool rows. ``readings`` in the record counts them.

**A single table** (``discover_table``) is read as a family when one of its inputs takes few values
(``across_tables.by_levels``); its wider law is then a law of the table itself, and stands beside what
the table's own search proposed. Otherwise it is the route, and then the fallbacks.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import sympy as sp

from apedp_dyn.discovery_l2 import across_tables, vouched_route_v2
from apedp_dyn.discovery_l2.across_tables import Family, Scored, Table, beside
from apedp_dyn.discovery_l2.fallbacks import ANY_CONSTANTS, Fallback, try_in_order
from apedp_dyn.grammar.proposer_interface import ProposerBase
from apedp_dyn.ode.extended_route import Recording, replayed

Stage = Callable[[Table], "ProposerBase | None"]
SHAPES_RECORDED = 5


def read(table: Table, proposer: ProposerBase | None, *, noise: str = "absolute", seed: int = 0) -> dict[str, Any]:
    """One table by the route in force. Beside its record: the laws its stage proposed, the laws it listed, and the
    reported law in the form the referee confirmed it (``confirmed_as``), before the route pruned and rewrote it.

    A reading the route cannot make is written down, not raised, as the product does
    (``ui/route_endpoint.route_reading``): no law, action ``not_run``, and the error. The route in force can end in
    a fitting error on a law a search proposed (the second development run lost a unit of ten tables to one), and
    one member's failed reading must not cost its family the others.
    """
    stage = Recording(proposer) if proposer is not None else None
    started = time.perf_counter()
    try:
        record = vouched_route_v2.discover_law(table.columns(), table.y, table.pool_X, table.measure, proposer=stage,
                                               noise=noise, seed=seed)
    except Exception as exc:  # noqa: BLE001 - whatever stops a reading is that table's answer: no law, and why
        error = f"{type(exc).__name__}: {exc}"[:300]
        return {"record": {"error": error}, "reported": False, "law": None, "confirmed_as": None, "tentative_law": None,
                "action": "not_run", "error": error, "formulas": list(stage.formulas) if stage is not None else [],
                "listed": [], "rows_read": None, "seconds": round(time.perf_counter() - started, 2)}
    path = record.get("main_search")
    listed = [str(c["formula"]) for c in (getattr(path, "candidates", None) or [])]
    reported = bool(record.get("reported") and record.get("law"))
    confirmed = getattr(path, "law", None)
    return {"record": record, "reported": reported, "law": str(record["law"]) if reported else None,
            "confirmed_as": str(confirmed) if reported and confirmed else None,
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
            "action": reading["action"], "error": reading.get("error"), "listed": len(reading["listed"]),
            "rows_read": reading["rows_read"],
            "last_checks": {key: vouching.get(key) for key in ("in_force", "applied", "version") if key in vouching},
            "measured_range": (record.get("measured_range") or {}).get("state"), "seconds": reading["seconds"]}


def _laws_of(reading: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for law in (reading["law"], reading["tentative_law"], *reading["listed"]):
        if law and law not in out:
            out.append(law)
    return out


def _one_of(law: str | None, laws: Sequence[str], names: Sequence[str]) -> bool:
    shape = across_tables.shape_of(str(law), names, ANY_CONSTANTS) if law else None
    return shape is not None and shape.key in {
        s.key for s in (across_tables.shape_of(item, names, ANY_CONSTANTS) for item in laws) if s is not None}


def brought_in(reading: dict[str, Any], handed: Sequence[str], names: Sequence[str]) -> bool:
    """Whether a re-reading reported a law that was handed in for it: the second rule of this module.

    Judged on the law as the referee confirmed it, since the route may prune and rewrite what it reports.
    """
    if not reading.get("reported"):
        return False
    return _one_of(reading.get("confirmed_as") or reading.get("law"), handed, names)


def _shapes_record(scored: Sequence[Scored]) -> list[dict[str, Any]]:
    return [{"shape": item.shape.key, "sum_of_bic": None if item.score == float("inf") else round(item.score, 2),
             "from_members": sorted(item.from_members), "seen_on": sorted(item.seen_on),
             "reported_on": sorted(item.reported_on)} for item in scored[:SHAPES_RECORDED]]


def discover_family(family: Family, stage: Stage, *, noise: str = "absolute", seed: int = 0, condition: str | None = None,
                    fallbacks: Sequence[Fallback] = (), siblings: bool = True, wider: bool = True,
                    pooled_stage: Stage | None = None, pooled_known: Sequence[str] | None = None) -> dict[str, Any]:
    """The family read together. The record: each member's first and final reading and where its law came from,
    the shapes, the wider law, and how many readings it took.

    ``stage(table)`` gives the search stage for a table (or ``None``). ``condition`` names the condition the wider law
    is read over; left out, the family's own if it carries exactly one. ``pooled_stage(table)`` gives a search stage for
    the pooled table: with it the pooled table is searched too, and the family's wider laws stand beside what that
    search proposed. ``pooled_known`` are laws a search has already proposed for the pooled table, on which the route
    has reported nothing: the wider laws stand beside those, and no new search is run.
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
    proposed = [list(r["formulas"]) for r in firsts]            # what each member's own search proposed, in its order
    sources: list[str | None] = ["route" if r["reported"] else None for r in firsts]
    handed: list[list[str]] = [[] for _ in tables]
    trails: list[list[dict[str, Any]]] = [[] for _ in tables]
    passed_over: list[list[dict[str, Any]]] = [[] for _ in tables]

    def take(i: int, reading: dict[str, Any], new: Sequence[str], source: str) -> None:
        """A re-reading of member ``i``: taken if it reported a law that was handed in, written down if it reported another."""
        if brought_in(reading, new, names):
            finals[i], sources[i] = reading, source
        elif reading["reported"]:
            passed_over[i].append({"reading": source, "law": reading["law"]})

    scored = across_tables.score_shapes(family, [_laws_of(r) for r in firsts], [r["law"] for r in firsts],
                                        unseen=proposed, seed=seed)
    if siblings:
        for i, table in enumerate(tables):
            if finals[i]["reported"]:
                continue
            handed[i] = across_tables.sibling_laws(family, scored, i)
            if handed[i]:
                take(i, again(table, beside(proposed[i], handed[i])), handed[i], "siblings")
        newly = [(across_tables.shape_of(str(finals[i].get("confirmed_as") or finals[i]["law"]), names), i)
                 for i in range(len(tables)) if finals[i]["reported"] and not firsts[i]["reported"]]
        for item in scored:
            item.reported_on |= {i for shape, i in newly if shape is not None and shape.key == item.shape.key}
        scored = sorted(scored, key=lambda s: (0 if s.reported_on else 1, -len(s.reported_on), s.score, s.shape.key))

    wide: dict[str, Any] | None = None
    if wider and condition is not None:
        laws, how = across_tables.wider_laws(family, scored, condition, seed=seed)
        wide = {"condition": condition, "handed": laws, "how": how, "alone": None, "searched": None, "beside": None,
                "reading": None, "law": None, "reported": False, "source": None, "passed_over": []}
        whole = across_tables.pooled(family, condition)
        final: dict[str, Any] | None = None
        if laws:
            # the family's laws with no search of the pooled table beside them: a first reading of that table
            alone = again(whole, laws)
            wide["alone"] = summary(alone)
            if pooled_stage is None and pooled_known is None and alone["reported"]:
                final, wide["source"] = alone, "family"
        if pooled_stage is not None or pooled_known is not None:
            known = list(pooled_known or [])
            if pooled_stage is not None:
                searched = read(whole, pooled_stage(whole), noise=noise, seed=seed)
                readings += 1
                wide["searched"] = summary(searched)
                known = list(searched["formulas"])
                if searched["reported"]:
                    final, wide["source"] = searched, "pooled search"
            if final is None and laws:
                both = again(whole, beside(known, laws))
                wide["beside"] = summary(both)
                if brought_in(both, laws, whole.names):
                    final, wide["source"] = both, "family"
                elif both["reported"]:
                    wide["passed_over"].append({"reading": "family", "law": both["law"]})
        if final is not None:
            wide.update({"reading": summary(final), "law": final["law"], "reported": True})
            symbol = sp.Symbol(condition)
            expr = sp.sympify(final["law"], locals={n: sp.Symbol(n) for n in whole.names})
            for i, table in enumerate(tables):
                if finals[i]["reported"]:
                    continue
                own = str(expr.xreplace({symbol: sp.Float(float(table.conditions[condition]))}))
                take(i, again(table, beside(proposed[i], [own])), [own], "wider law")

    for i, table in enumerate(tables):
        if finals[i]["reported"] or not fallbacks:
            continue
        after, trails[i] = try_in_order(table, proposed[i], fallbacks, again)
        if after is not None:
            finals[i] = after
            sources[i] = trails[i][-1]["fallback"]

    members = [{"key": table.key, "conditions": dict(table.conditions), "first": summary(firsts[i]), "final": summary(finals[i]),
                "source": sources[i], "handed_from_siblings": handed[i], "fallbacks": trails[i], "passed_over": passed_over[i],
                "first_proposed": proposed[i], "first_listed": _laws_of(firsts[i])} for i, table in enumerate(tables)]
    return {"family": family.key, "inputs": names, "members": members, "shapes": _shapes_record(scored), "wider": wide,
            "reported_first": sum(r["reported"] for r in firsts), "reported_final": sum(r["reported"] for r in finals),
            "readings": readings, "seconds": round(time.perf_counter() - started, 2)}


def discover_table(table: Table, stage: Stage, *, noise: str = "absolute", seed: int = 0, fallbacks: Sequence[Fallback] = (),
                   levels: bool = True) -> dict[str, Any]:
    """One table: the route in force; then, if one of its inputs takes few values, the table as a family; then the fallbacks.

    The record's ``law`` is the law reported for the table itself, by the route, by the family's wider law (which is a
    law of this table: the condition is one of its inputs, and it stood beside what the table's own search proposed),
    or by a fallback's own law; ``source`` says which.
    """
    started = time.perf_counter()
    first = read(table, stage(table), noise=noise, seed=seed)
    out: dict[str, Any] = {"table": table.key, "first": summary(first), "law": first["law"], "reported": first["reported"],
                           "source": "route" if first["reported"] else None, "family": None, "fallbacks": []}
    proposed = list(first["formulas"])
    if not out["reported"] and levels:
        family = across_tables.by_levels(table)
        if family is not None:
            record = discover_family(family, stage, noise=noise, seed=seed, pooled_known=proposed)
            out["family"] = record
            wide = record.get("wider") or {}
            if wide.get("reported"):
                out.update({"law": wide["law"], "reported": True, "source": "family"})
    if not out["reported"] and fallbacks:
        reading, out["fallbacks"] = try_in_order(
            table, proposed, fallbacks, lambda t, laws: read(t, replayed(list(dict.fromkeys(laws))), noise=noise, seed=seed))
        if reading is not None:
            out.update({"law": reading["law"], "reported": True, "source": out["fallbacks"][-1]["fallback"]})
    out["seconds"] = round(time.perf_counter() - started, 2)
    return out
