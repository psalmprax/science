#!/usr/bin/env python3
"""Development: do the tables of a family help each other? The family route beside the route table by table.

Status: audit-only

**Development, not a result.** Nothing was fixed beforehand, the families are drawn from the invented-law
grammar with a seed of this study, and what is counted here decides nothing: it says whether the family
route (``apedp_dyn/discovery_l2/family_route.py``, built on the owner's word of 27 November 2026) is worth
a preregistered test, and with which marks.

**The families.** Hidden shapes of one and two inputs (``curriculum.draw``, levels 1 to 3; none with a
structure of the held-back benchmark laws). Each has ``MEMBERS`` tables, a member a value of one condition
``c``. One or two of the shape's constants follow a hidden law of ``c``; the others are one number for the
whole family. Half the families use laws of ``c`` that ``across_tables`` can write (a line, a reciprocal, a
power, an exponential, a root, a logarithm); the other half use laws it cannot (a sine, a saturation, a
bump, a parabola), so that half measures what an approximation is worth. Noise is 1 %, 5 % and 10 % of the
family's size. A member has 700 rows to fit on, 300 in its pool and 400 without noise to score on, as a
curriculum problem has. ``--families`` and ``--skip`` choose which of the seed's families are read: the
first run read families 0 to 15, the second 16 to 31.

**Two runs, two versions of the route.** The first run (``benchmarks/family_stream/dev_run1.json``, made by
this script as it stood at commit e38a897) read the route as first built, which put the laws it handed in
at the head of a member's list and counted whatever law a re-reading then reported. Its control showed
that most of what it gained was not the family's. What the referee reports on a table depends on the list
before it (``scripts/study_list_sensitivity.py``), and a route that reads again only the tables without a
law collects the changes in its favour. The route was changed (``family_route.py``: a re-reading brings in
only what was handed in for it; beside, never in place) and this script now reads the changed route, with
arms that show the first build's effect beside it.

**What is read, each through the route in force with the product's own search stage:**

* **A** each member by itself: the route as it is today.
* **B** the family: siblings' shapes, then the wider law over the inputs and ``c``.
* **F** B and then the fallbacks for a member still without a law: solved cases from a store of ``STORE``
  other invented laws, then the tensor-train reading. No language model: that needs a key. The store is drawn
  from the same grammar, so now and then a family's hidden shape is also the shape of a solved case (2 of the
  second run's 16 families, none of the first's); those are counted apart.
* **R** the control for B: each member without a law is put into the *other* family of its pair, a family
  it does not belong to, and handed what that family gives, under the same two rules.
* **O** a member's own laws replayed with nothing beside them: whether reading again changes anything.
* **B1** and **R1** the first build's way, kept only to show its effect: the handed laws at the head of
  the list, and whatever law the referee then reports counted, with its origin written down (a handed
  law, or one of the member's own).

**The wider law, three ways on the pooled table** (all rows of a family, ``c`` one more input):
*alone*, the family's laws with no search beside them; *searched*, the route in force searching the
pooled table with the five members' search time together; *final*, what the family route reports when it
does both: the search's law if the search reports one, else a family law that stood beside the search's
proposals and was the one reported.

**What is counted.** For a member: a law reported, the reported law accurate (R squared 0.999 on rows
without noise), and the law itself by the benchmark's test of form. For the wider law: the same on the
members' rows without noise, and accuracy at values of ``c`` between the members', which no row was ever
measured at. Wrong reports are counted in every arm. For each member without a law it is also written down
where the law itself was: among the candidates its referee had, lower on its own search's list, on a
sibling's lists, among the laws it was handed.

    python scripts/study_family_stream.py run --out DIR --shard 0 --shards 24 --searches 30 --families 32 --skip 16
    python scripts/study_family_stream.py collect --out DIR --families 32 --skip 16
    python scripts/study_family_stream.py merge --out DIR --families 32 --skip 16 --records A.json B.json
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import itertools
import json
import math
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_srbench_baseline as base  # noqa: E402 - after the path is set

from apedp_dyn.discovery_l2 import across_tables, fallbacks, family_route  # noqa: E402
from apedp_dyn.discovery_l2.across_tables import Family, Table, beside  # noqa: E402
from apedp_dyn.ode.extended_route import replayed  # noqa: E402
from apedp_dyn.proposers import experience  # noqa: E402
from apedp_dyn.recursion import curriculum  # noqa: E402

TAG = "DEV-FAMILY-STREAM-2026-11"
SEED, STORE_SEED = 2026112811, 2026112812        #: drawn by this study only
FAMILIES, MEMBERS, STORE = 16, 5, 60
POOLED_TIMES = 5                                   #: the pooled table's search gets the members' search time together
FROM_THE_FAMILY = ("route", "siblings", "wider law")   #: where a member's law may come from before the fallbacks
NOISES = (0.01, 0.05, 0.10)
LEVELS = (1, 2, 3)                                 #: one input and two blocks; two inputs and two or three blocks
CONDITION = "c"
VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
ROWS, POOL, TEST_ROWS = curriculum.ROWS, curriculum.POOL, curriculum.TEST_ROWS
MOVES_ENOUGH = 0.10                                #: the first and last member's laws differ by at least this share
STALL_REPORT_S = 1800
_c = sp.Symbol(CONDITION)
#: What a constant is multiplied by along the condition. The first six are forms ``across_tables`` can write.
WRITABLE = {"line": 0.4 + 0.3 * _c, "reciprocal": 0.5 + 1.5 / _c, "power": 2.2 * _c ** sp.Float(-0.8),
            "exponential": 2.4 * sp.exp(-0.35 * _c), "root": 0.2 + 0.7 * sp.sqrt(_c), "logarithm": 0.6 + 0.8 * sp.log(_c)}
NOT_WRITABLE = {"sine": 1.2 + 0.6 * sp.sin(1.1 * _c), "saturation": 2.6 * _c / (1.5 + _c),
                "bump": 0.6 + 1.6 * sp.exp(-(_c - 3) ** 2 / 2), "parabola": 0.5 + 1.0 * _c - 0.14 * _c ** 2}


def _rng(*parts: Any) -> np.random.Generator:
    key = hashlib.sha256("|".join(str(p) for p in ("family-stream", *parts)).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(key, "big"))


def _with_moving(expr: sp.Expr, moving: dict[int, sp.Expr]) -> sp.Expr:
    """``expr`` with its ``k``-th fitted constant (in the order ``shape_of`` meets them) multiplied by ``moving[k]``."""
    met = [0]

    def walk(node: sp.Expr) -> sp.Expr:
        if node.is_Float:
            met[0] += 1
            return node * moving[met[0] - 1] if met[0] - 1 in moving else node
        if not node.args:
            return node
        return node.func(*[walk(arg) for arg in node.args])

    return walk(expr)


def draw_families(count: int = FAMILIES, seed: int = SEED) -> list[dict[str, Any]]:
    """The study's families, a pair at a time: both of a pair have the same inputs, so each is the other's wrong family."""
    rng = np.random.default_rng(seed)
    held_back = curriculum.benchmark_fingerprints()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(out) < count:
        level = LEVELS[(len(out) // 2) % len(LEVELS)]         # a pair shares its level, so its number of inputs
        expr, inputs = curriculum.draw(rng, level)
        constants = len(expr.atoms(sp.Float))
        structure = curriculum.structure(expr)
        if constants < 2 or structure in seen or curriculum.fingerprint(expr) in held_back or not curriculum.acceptable(expr, inputs, rng):
            continue
        writable = len(out) % 4 < 2                              # pairs alternate: both writable, both not
        forms = WRITABLE if writable else NOT_WRITABLE
        places = sorted(int(v) for v in rng.choice(sum(1 for _ in _floats(expr)), size=1 if rng.random() < 0.7 else 2, replace=False))
        chosen = {place: str(rng.choice(sorted(forms))) for place in places}
        wide = _with_moving(expr, {place: forms[name] for place, name in chosen.items()})
        values = sorted(float(v) for v in rng.choice(VALUES, size=MEMBERS, replace=False))
        names = [f"x{i}" for i in range(inputs)]
        laws = [wide.xreplace({_c: sp.Float(v)}) for v in values]
        if not all(curriculum.acceptable(law, inputs, rng) for law in laws):
            continue
        X = rng.uniform(curriculum.LOW, curriculum.HIGH, (400, inputs))
        ends = [np.asarray(sp.lambdify([sp.Symbol(n) for n in names], law, "numpy")(*X.T), dtype=float) * np.ones(len(X))
                for law in (laws[0], laws[-1])]
        if float(np.sqrt(np.mean((ends[0] - ends[1]) ** 2))) < MOVES_ENOUGH * float(np.sqrt(np.mean(ends[0] ** 2))):
            continue
        seen.add(structure)
        out.append({"family": len(out), "level": level, "inputs": inputs, "shape": str(expr), "wide": str(wide),
                    "moving": {str(place): name for place, name in chosen.items()}, "writable": writable, "values": values,
                    "laws": [str(law) for law in laws]})
    return out


def _floats(expr: sp.Expr) -> list[sp.Expr]:
    """The fitted constants of a law in the order ``shape_of`` meets them, one entry an occurrence."""
    if expr.is_Float:
        return [expr]
    return [f for arg in expr.args for f in _floats(arg)]


def tables(spec: dict[str, Any], noise: float) -> tuple[Family, list[dict[str, Any]]]:
    """A family's tables at one noise level, and for each member what it is scored against."""
    inputs = spec["inputs"]
    names = [f"x{i}" for i in range(inputs)]
    symbols = [sp.Symbol(n) for n in names]
    drawn = []
    for index, (value, law) in enumerate(zip(spec["values"], spec["laws"], strict=True)):
        rng = _rng(spec["family"], index, noise)
        X = rng.uniform(curriculum.LOW, curriculum.HIGH, (ROWS + POOL + TEST_ROWS, inputs))
        truth = sp.sympify(law, locals=dict(zip(names, symbols, strict=True)))
        clean = np.asarray(sp.lambdify(symbols, truth, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        drawn.append((value, truth, X, clean, rng))
    size = float(np.sqrt(np.mean(np.concatenate([clean[: ROWS + POOL] for _, _, _, clean, _ in drawn]) ** 2)))
    members, scoring = [], []
    for index, (value, truth, X, clean, rng) in enumerate(drawn):
        y = clean + noise * size * rng.standard_normal(len(X))
        pool_y = y[ROWS: ROWS + POOL]
        members.append(Table(f"family {spec['family']}, {CONDITION}={value:g}", list(names), X[:ROWS], y[:ROWS], X[ROWS: ROWS + POOL],
                             lambda asked, pool_y=pool_y: pool_y[np.asarray(asked, dtype=int)], {CONDITION: float(value)}))
        scoring.append({"truth": truth, "test_X": X[ROWS + POOL:], "test_clean": clean[ROWS + POOL:], "index": index})
    return Family(f"family {spec['family']}", members), scoring


def score(law: str | None, truth: sp.Expr, names: Sequence[str], X: np.ndarray, clean: np.ndarray) -> dict[str, bool]:
    """Whether a law is accurate on rows without noise, and whether it is the law itself by the benchmark's test of form."""
    out = {"accurate": False, "recovered": False}
    if not law:
        return out
    try:
        symbols = [sp.Symbol(n) for n in names]
        model = sp.sympify(str(law), locals=dict(zip(names, symbols, strict=True)))
        with np.errstate(all="ignore"):
            predicted = np.asarray(sp.lambdify(symbols, model, "numpy")(*X.T), dtype=float) * np.ones(len(X))
        r2 = base._r2(clean, predicted)
        out["accurate"] = bool(r2 is not None and r2 >= base.R2_ACCURACY)
        out["recovered"] = bool(base.symbolic_recovery(model, truth, list(names), X)["recovered"])
    except Exception:  # noqa: BLE001 - a law that cannot be scored is neither accurate nor recovered
        return out
    return out


def store(count: int = STORE, seed: int = STORE_SEED) -> experience.Experience:
    """Solved cases for the fallback: other invented laws of one and two inputs with a table of each."""
    rng = np.random.default_rng(seed)
    held_back = curriculum.benchmark_fingerprints()
    out = experience.Experience()
    seen: set[str] = set()
    while len(out.cases) < count:
        expr, inputs = curriculum.draw(rng, int(rng.integers(0, 4)))
        if curriculum.structure(expr) in seen or curriculum.fingerprint(expr) in held_back or not curriculum.acceptable(expr, inputs, rng):
            continue
        seen.add(curriculum.structure(expr))
        X = rng.uniform(curriculum.LOW, curriculum.HIGH, (ROWS, inputs))
        y = np.asarray(sp.lambdify([sp.Symbol(f"x{i}") for i in range(inputs)], expr, "numpy")(*X.T), dtype=float) * np.ones(ROWS)
        out.add(f"store {len(out.cases)}", str(expr), X, y)
    return out


def product_stage(seconds: float) -> family_route.Stage:
    """The search stage the product gives a reading (``ui/route_endpoint.py``), built afresh for each table."""
    from apedp_dyn.ui import route_endpoint

    def stage(_table: Table) -> Any:
        return route_endpoint._proposer(route_endpoint._search_budget(seconds))[0]

    return stage


def _again(member: Table, laws: Sequence[str], noise: str, seed: int) -> dict[str, Any]:
    return family_route.read(member, replayed(list(dict.fromkeys(laws))), noise=noise, seed=seed)


def _scored(reported: bool, law: str | None, how: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    law = law if reported else None
    return {"reported": bool(law), **score(law, how["truth"], names, how["test_X"], how["test_clean"])}


def _other_arms(member: Table, own: dict[str, Any], how: dict[str, Any], other: Family, other_record: dict[str, Any],
                noise: str, seed: int) -> dict[str, dict[str, Any]]:
    """For a member the route reported no law on: its own laws replayed (O), the first build's way with its own
    family (B1), and the wrong family under the two rules (R) and the first build's way (R1)."""
    names = member.names
    proposed, handed = list(own["first_proposed"]), list(own["handed_from_siblings"])
    out: dict[str, dict[str, Any]] = {}
    again = _again(member, proposed, noise, seed) if proposed else None
    out["O"] = {**_scored(bool(again and again["reported"]), again["law"] if again else None, how, names), "readings": int(bool(again))}

    def first_build(laws: Sequence[str]) -> dict[str, Any]:
        if not laws:
            return {**_scored(False, None, how, names), "handed": 0, "origin": None}
        reading = _again(member, [*laws, *proposed], noise, seed)
        origin = None if not reading["reported"] else "handed" if family_route.brought_in(reading, laws, names) else "own"
        return {**_scored(reading["reported"], reading["law"], how, names), "handed": len(laws), "origin": origin}

    out["B1"] = first_build(handed)
    pseudo = Family("wrong family", [member, *other.tables])
    others = other_record["members"]
    scored = across_tables.score_shapes(pseudo, [own["first_listed"], *[m["first_listed"] for m in others]],
                                        [None, *[m["first"]["law"] if m["first"]["reported"] else None for m in others]],
                                        unseen=[proposed, *[m["first_proposed"] for m in others]], seed=seed)
    wrong = across_tables.sibling_laws(pseudo, scored, 0)
    out["R1"] = first_build(wrong)
    control: dict[str, Any] = {"handed": len(wrong), "readings": 0, "passed_over": 0, "law": None}
    if wrong:
        reading = _again(member, beside(proposed, wrong), noise, seed)
        control["readings"] += 1
        if family_route.brought_in(reading, wrong, names):
            control["law"] = reading["law"]
        elif reading["reported"]:
            control["passed_over"] += 1
    wide = other_record["wider"] or {}
    if control["law"] is None and wide.get("reported"):
        # as the family route does after its wider law: that law at this member's value of the condition
        expr = sp.sympify(wide["law"], locals={n: sp.Symbol(n) for n in (*names, CONDITION)})
        at_value = str(expr.xreplace({_c: sp.Float(float(member.conditions[CONDITION]))}))
        reading = _again(member, beside(proposed, [at_value]), noise, seed)
        control["readings"] += 1
        if family_route.brought_in(reading, [at_value], names):
            control["law"] = reading["law"]
        elif reading["reported"]:
            control["passed_over"] += 1
    law = control.pop("law")
    out["R"] = {**_scored(law is not None, law, how, names), **control}
    return out


def _which(laws: Sequence[str], how: dict[str, Any], names: Sequence[str]) -> dict[str, bool]:
    """Whether the law itself, and whether an accurate law, is among ``laws``."""
    marks = [score(law, how["truth"], names, how["test_X"], how["test_clean"]) for law in laws]
    return {"law_itself": any(m["recovered"] for m in marks), "accurate": any(m["accurate"] for m in marks)}


def run_unit(pair: Sequence[dict[str, Any]], noise: float, searches: float, *, seed: int,
             stage: family_route.Stage | None = None, pooled_stage: family_route.Stage | None = None,
             cases: experience.Experience | None = None) -> dict[str, Any]:
    """One pair of families at one noise level through every arm."""
    stage = stage or product_stage(searches)
    pooled_stage = pooled_stage or product_stage(POOLED_TIMES * searches)
    cases = cases if cases is not None else store()
    chain = [fallbacks.SolvedCases(cases), fallbacks.TensorTrain()]
    # a family whose hidden shape is also the shape of a solved case: there the cases can simply know the answer
    solved = {curriculum.structure(sp.sympify(case.law)) for case in cases.cases}
    built = [tables(spec, noise) for spec in pair]
    records = []
    for family, _ in built:
        faulthandler.dump_traceback_later(STALL_REPORT_S, repeat=True, file=sys.stderr)
        try:
            records.append(family_route.discover_family(family, stage, noise="absolute", seed=seed, fallbacks=chain,
                                                        pooled_stage=pooled_stage))
        finally:
            faulthandler.cancel_dump_traceback_later()
    member_rows, family_rows = [], []
    for which, (spec, (family, scoring), record) in enumerate(zip(pair, built, records, strict=True)):
        other, other_record = built[1 - which][0], records[1 - which]
        names = family.names
        where = [{"listed": _which(m["first_listed"], how, names), "proposed": _which(m["first_proposed"], how, names)}
                 for m, how in zip(record["members"], scoring, strict=True)]
        for index, (member, table, how) in enumerate(zip(record["members"], family.tables, scoring, strict=True)):
            first, final = member["first"], member["final"]
            row: dict[str, Any] = {
                "family": spec["family"], "member": how["index"], "noise": noise, "inputs": spec["inputs"], "level": spec["level"],
                "writable": spec["writable"], "shape_among_the_solved_cases": curriculum.structure(sp.sympify(spec["shape"])) in solved,
                "source": member["source"], "first_action": first["action"],
                "first_tentative": bool(first["tentative_law"]), "passed_over": [p["reading"] for p in member["passed_over"]],
                "fallbacks": [{k: step.get(k) for k in ("fallback", "tried", "proposed", "reported", "why_not", "error")}
                              | {"passed_over": bool(step.get("passed_over"))} for step in member["fallbacks"]]}
            after_family = final if member["source"] in FROM_THE_FAMILY else first
            row["A"] = _scored(first["reported"], first["law"], how, names)
            row["B"] = _scored(after_family["reported"], after_family["law"], how, names)
            row["F"] = _scored(final["reported"], final["law"], how, names)
            handed = member["handed_from_siblings"]
            row["where"] = {
                "among_its_referee_s_candidates": where[index]["listed"],
                "on_its_own_search_s_list": where[index]["proposed"],
                "on_a_sibling_s_lists": {"law_itself": any(where[j][k]["law_itself"] for j in range(len(where)) if j != index
                                                           for k in ("listed", "proposed"))},
                "among_the_laws_handed": {**_which(handed, how, names), "count": len(handed)},
                "among_the_first_three_handed": _which(handed[:3], how, names)}
            if first["reported"]:
                for arm in ("O", "B1", "R", "R1"):
                    row[arm] = dict(row["A"])
            else:
                faulthandler.dump_traceback_later(STALL_REPORT_S, repeat=True, file=sys.stderr)
                try:
                    row.update(_other_arms(table, member, how, other, other_record, "absolute", seed))
                finally:
                    faulthandler.cancel_dump_traceback_later()
            member_rows.append(row)
        family_rows.append(_family_row(spec, family, scoring, record, noise))
    return {"members": member_rows, "families": family_rows}


def _family_row(spec: dict[str, Any], family: Family, scoring: list[dict[str, Any]], record: dict[str, Any],
                noise: float) -> dict[str, Any]:
    names = [*family.names, CONDITION]
    truth = sp.sympify(spec["wide"], locals={n: sp.Symbol(n) for n in names})
    at_members = np.vstack([np.column_stack([s["test_X"], np.full(len(s["test_X"]), t.conditions[CONDITION])])
                            for s, t in zip(scoring, family.tables, strict=True)])
    clean = np.concatenate([s["test_clean"] for s in scoring])
    rng = _rng("between", spec["family"], noise)
    values = sorted(t.conditions[CONDITION] for t in family.tables)
    between = np.column_stack([rng.uniform(curriculum.LOW, curriculum.HIGH, (TEST_ROWS, len(family.names))),
                               rng.choice([(a + b) / 2 for a, b in itertools.pairwise(values)], TEST_ROWS)])
    between_clean = np.asarray(sp.lambdify([sp.Symbol(n) for n in names], truth, "numpy")(*between.T), dtype=float) * np.ones(TEST_ROWS)

    def scored(part: dict[str, Any] | None) -> dict[str, Any]:
        law = part["law"] if part and part.get("reported") else None
        return {"read": part is not None, "reported": bool(law), **score(law, truth, names, at_members, clean),
                "accurate_between": score(law, truth, names, between, between_clean)["accurate"]}

    wide = record["wider"] or {}
    handed = list(wide.get("handed") or [])
    marks = [score(law, truth, names, at_members, clean) for law in handed]
    return {"family": spec["family"], "noise": noise, "inputs": spec["inputs"], "level": spec["level"],
            "writable": spec["writable"], "moving": spec["moving"], "readings": record["readings"],
            "seconds": record["seconds"], "shapes": record["shapes"][:2],
            "wider": {"handed": len(handed), "law_itself_handed": any(m["recovered"] for m in marks),
                      "accurate_handed": any(m["accurate"] for m in marks), "source": wide.get("source"),
                      "passed_over": len(wide.get("passed_over") or []),
                      "alone": scored(wide.get("alone")), "searched": scored(wide.get("searched")),
                      "final": scored({"law": wide.get("law"), "reported": wide.get("reported")} if wide else None),
                      "searched_seconds": (wide.get("searched") or {}).get("seconds")}}


def units(families: Sequence[dict[str, Any]]) -> list[tuple[int, float]]:
    """(pair, noise), every pair at every noise."""
    return [(pair, noise) for pair in range(len(families) // 2) for noise in NOISES]


def sign_test(gained: int, lost: int) -> float:
    """Two-sided: the chance of a split at least this uneven if a gain and a loss were equally likely."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def _paired(rows: Sequence[dict[str, Any]], arm: str, what: str, against: str = "A") -> dict[str, Any]:
    """Members that gained and lost, and families: a family's members share one hidden shape, so they are not
    five separate trials. The sign test is over families (more of its members with the mark, or fewer)."""
    gained = sum(bool(r[arm][what]) and not r[against][what] for r in rows)
    lost = sum(bool(r[against][what]) and not r[arm][what] for r in rows)
    net: dict[int, int] = {}
    for r in rows:
        net[r["family"]] = net.get(r["family"], 0) + int(bool(r[arm][what])) - int(bool(r[against][what]))
    more, fewer = sum(v > 0 for v in net.values()), sum(v < 0 for v in net.values())
    return {"gained": gained, "lost": lost, "families_with_more": more, "families_with_fewer": fewer,
            "sign_test_p_over_families": round(sign_test(more, fewer), 5)}


ARMS = ("A", "B", "F", "R", "O", "B1", "R1")
WAYS = ("alone", "searched", "final")


def _cell(rows: Sequence[dict[str, Any]], fams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cell: dict[str, Any] = {"members": len(rows), "families": len(fams)}
    for arm in ARMS:
        reported = [r for r in rows if r[arm]["reported"]]
        cell[arm] = {"reported": len(reported), "reported_and_not_accurate": sum(not r[arm]["accurate"] for r in reported),
                     "reported_and_the_law_itself": sum(r[arm]["recovered"] for r in reported)}
        if arm != "A":
            cell[arm]["against_A"] = {what: _paired(rows, arm, what) for what in ("reported", "recovered")}
    cell["B_against_R"] = {what: _paired(rows, "B", what, "R") for what in ("reported", "recovered")}
    cell["B1_against_R1"] = {what: _paired(rows, "B1", what, "R1") for what in ("reported", "recovered")}
    cell["sources"] = {source: sum(str(r["source"]) == source for r in rows) for source in sorted({str(r["source"]) for r in rows})}
    cell["passed_over"] = {"siblings": sum("siblings" in r["passed_over"] for r in rows),
                           "wider law": sum("wider law" in r["passed_over"] for r in rows),
                           "control": sum(r["R"].get("passed_over", 0) > 0 for r in rows)}
    cell["first_build_origin"] = {arm: {origin: sum(r[arm].get("origin") == origin for r in rows) for origin in ("handed", "own")}
                                  for arm in ("B1", "R1")}
    cell["fallbacks"] = {name: {"tried": sum(any(s["fallback"] == name and s["tried"] for s in r["fallbacks"]) for r in rows),
                                "its_own_law_reported": sum(r["source"] == name for r in rows),
                                "of_those_where_the_shape_was_a_solved_case":
                                    sum(r["source"] == name and r["shape_among_the_solved_cases"] for r in rows),
                                "of_those_the_law_itself": sum(r["source"] == name and r["F"]["recovered"] for r in rows),
                                "of_those_accurate": sum(r["source"] == name and r["F"]["accurate"] for r in rows),
                                "another_law_reported_and_passed_over":
                                    sum(any(s["fallback"] == name and s["passed_over"] for s in r["fallbacks"]) for r in rows)}
                         for name in ("solved cases", "tensor train")}
    missed = [r for r in rows if not r["A"]["reported"]]
    cell["members_without_a_first_report"] = {
        "count": len(missed),
        "first_action": {action: sum(str(r["first_action"]) == action for r in missed)
                         for action in sorted({str(r["first_action"]) for r in missed})},
        "the_law_itself_was": {key: sum(r["where"][key]["law_itself"] for r in missed)
                               for key in ("among_its_referee_s_candidates", "on_its_own_search_s_list", "on_a_sibling_s_lists",
                                           "among_the_laws_handed", "among_the_first_three_handed")},
        "an_accurate_law_was": {key: sum(r["where"][key]["accurate"] for r in missed)
                                for key in ("among_its_referee_s_candidates", "on_its_own_search_s_list",
                                            "among_the_laws_handed", "among_the_first_three_handed")},
        "handed_at_least_one_law": sum(r["where"]["among_the_laws_handed"]["count"] > 0 for r in missed)}
    wider = {"families_handed_a_wider_law": sum(f["wider"]["handed"] > 0 for f in fams),
             "the_law_itself_among_those_handed": sum(f["wider"]["law_itself_handed"] for f in fams),
             "an_accurate_law_among_those_handed": sum(f["wider"]["accurate_handed"] for f in fams),
             "passed_over": sum(f["wider"]["passed_over"] > 0 for f in fams),
             "final_source": {source: sum(str(f["wider"]["source"]) == source for f in fams)
                              for source in sorted({str(f["wider"]["source"]) for f in fams})}}
    for way in WAYS:
        wider[way] = {key: sum(bool(f["wider"][way][key]) for f in fams) for key in ("reported", "accurate", "accurate_between", "recovered")}
        wider[way]["reported_and_not_accurate"] = sum(f["wider"][way]["reported"] and not f["wider"][way]["accurate"] for f in fams)
    for one, two in (("alone", "searched"), ("final", "searched")):
        wider[f"{one}_and_{two}"] = {
            "both": sum(f["wider"][one]["reported"] and f["wider"][two]["reported"] for f in fams),
            f"only_{one}": sum(f["wider"][one]["reported"] and not f["wider"][two]["reported"] for f in fams),
            f"only_{two}": sum(f["wider"][two]["reported"] and not f["wider"][one]["reported"] for f in fams)}
    for label, chosen in (("writable", True), ("not_writable", False)):
        sel = [f for f in fams if f["writable"] is chosen]
        wider[label] = {"families": len(sel), **{way: sum(f["wider"][way]["reported"] for f in sel) for way in WAYS}}
    cell["wider"] = wider
    return cell


def summarise(members: Sequence[dict[str, Any]], families: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"all noise levels": _cell(members, families)}
    for noise in sorted({r["noise"] for r in members}):
        out[f"noise {noise:g}"] = _cell([r for r in members if r["noise"] == noise], [r for r in families if r["noise"] == noise])
    return out


def merged(records: Sequence[dict[str, Any]], names: Sequence[str], expected: int) -> dict[str, Any]:
    """Records of runs that read different units of the same families, as one: units that failed in one run and were
    read again in another. A unit (a family at a noise level) found in two records is refused: nothing is chosen
    between two readings of the same unit."""
    members = [row for record in records for row in record["members"]]
    fams = [row for record in records for row in record["families"]]
    keys = [(row["family"], row["noise"]) for row in fams]
    twice = sorted({key for key in keys if keys.count(key) > 1})
    if twice:
        raise ValueError(f"read in more than one record, family and noise: {twice}")
    first = records[0]
    return {"tag": first["tag"], "development_not_a_result": True, "what": first["what"],
            "merged_from": [{"record": name, "units_found": record["units_found"],
                             "units_failed": [{k: f[k] for k in ("pair", "noise")} for f in record.get("units_failed", [])]}
                            for name, record in zip(names, records, strict=True)],
            "settings": first["settings"], "units_expected": expected, "units_found": len(fams) // 2,
            "complete": len(fams) // 2 == expected,
            "summary": summarise(members, fams) if members else {}, "families": fams, "members": members}


def main(argv: list[str] | None = None, stage: Callable[[float], family_route.Stage] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=("run", "collect", "families", "merge"))
    parser.add_argument("--records", type=Path, nargs="*", default=[], help="for merge: records of runs of the same families")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--searches", type=float, default=30.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--families", type=int, default=FAMILIES, help="families drawn from the seed")
    parser.add_argument("--skip", type=int, default=0, help="of those, how many to leave out from the start (an even number)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.skip % 2:
        parser.error("--skip must be even: the families come in pairs")
    args.out.mkdir(parents=True, exist_ok=True)
    families = draw_families(args.families)[args.skip:]
    if args.command == "families":
        (args.out / "families.json").write_text(json.dumps(families, indent=1) + "\n")
        return 0
    if args.command == "merge":
        record = merged([json.loads(path.read_text()) for path in args.records], [path.name for path in args.records],
                        len(units(families)))
        (args.out / "family_stream_merged.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"units": f"{record['units_found']} of {record['units_expected']}",
                          "summary": record["summary"].get("all noise levels")}, indent=1))
        return 0
    if args.command == "run":
        mine = [unit for index, unit in enumerate(units(families)) if index % args.shards == args.shard]
        if args.limit:
            mine = mine[: args.limit]
        cases = store()
        failed = 0
        for pair, noise in mine:
            started = time.perf_counter()
            first = families[2 * pair]["family"]
            try:
                got = run_unit(families[2 * pair: 2 * pair + 2], noise, args.searches, seed=first // 2,
                               stage=None if stage is None else stage(args.searches),
                               pooled_stage=None if stage is None else stage(POOLED_TIMES * args.searches), cases=cases)
            except Exception:  # noqa: BLE001 - written down, and the shard's other units still run; the shard ends red
                failed += 1
                (args.out / f"failed_{first:02d}_{noise:g}.json").write_text(
                    json.dumps({"pair": [first, first + 1], "noise": noise, "error": traceback.format_exc()[-4000:]}) + "\n")
                print(f"families {first} and {first + 1}, noise {noise:g}: FAILED\n{traceback.format_exc()}", flush=True)
                continue
            got.update({"pair": [first, first + 1], "noise": noise, "searches": args.searches,
                        "seconds": round(time.perf_counter() - started, 1)})
            (args.out / f"unit_{first:02d}_{noise:g}.json").write_text(json.dumps(got, sort_keys=True) + "\n")
            print(f"families {first} and {first + 1}, noise {noise:g}: {got['seconds']} s", flush=True)
        return 1 if failed else 0
    got_units = [json.loads(path.read_text()) for path in sorted(args.out.rglob("unit_*.json"))]
    failures = [json.loads(path.read_text()) for path in sorted(args.out.rglob("failed_*.json"))]
    members = [row for unit in got_units for row in unit["members"]]
    fams = [row for unit in got_units for row in unit["families"]]
    expected = len(units(families))
    record = {"tag": TAG, "development_not_a_result": True,
              "what": "development: families of invented laws read by the route table by table (A), as families under the two "
                      "rules (B), with the fallbacks behind (F), with the wrong family as control (R), their own laws replayed "
                      "(O), the first build's way (B1, R1), and the wider law alone, by a search of the pooled table, and by both",
              "settings": {"families_drawn": args.families, "families_skipped": args.skip,
                           "families_read": [f["family"] for f in families], "members": MEMBERS, "noises": list(NOISES),
                           "levels": list(LEVELS), "store": STORE, "seed": SEED, "store_seed": STORE_SEED,
                           "searches": sorted({u["searches"] for u in got_units}), "pooled_search_times": POOLED_TIMES},
              "units_expected": expected, "units_found": len(got_units), "complete": len(got_units) == expected,
              "units_failed": [{"pair": f["pair"], "noise": f["noise"], "error": f["error"][-600:]} for f in failures],
              "summary": summarise(members, fams) if members else {}, "families": fams, "members": members}
    (args.out / "family_stream.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"units": f"{len(got_units)} of {expected}", "summary": record["summary"].get("all noise levels")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
