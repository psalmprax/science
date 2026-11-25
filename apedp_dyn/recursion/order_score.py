"""Where the law lands in a proposer's order: a measurement that can see a small gain.

Status: experimental

**Why.** The recursion decided between two searchers on ten problems with nought to three solved
(``levels.better``), and measured a generation on 48. A real gain of a few points cannot be seen at
that size, so its choices were chance: in the run with the searches on, the "improved" searcher
solved 7 of 144 fresh problems where the starting one solved 10
(``benchmarks/recursion/dev_search_run2.json``). A loop cannot improve what it cannot measure.

**What is measured here.** A proposer hands the referee an ordered list of candidate laws. On a
problem whose true law is known (the curriculum makes its own), each candidate is judged with the
judge's own rule, and three places are read off the list (1 is the first candidate):

* ``solved_place``: the first candidate that is accurate on clean rows and no longer than the true
  law. This is ``judge.verdict``'s "solved", asked of every candidate and not only of the one the
  route reported;
* ``form_place``: the first candidate that rewrites show equal to the true law;
* ``accurate_place``: the first candidate at the accuracy line, whatever its length.

A place carries more than "solved or not": a law at place 2 and a law at place 40 are both
"proposed", and only the first is likely to be the one the route confirms. The score of a problem
is the reciprocal of its place (0 if the law is not on the list), and a set of problems is
summarised by the share found, the share at place 1 and the mean reciprocal place.

**Comparing two proposers** (``compare``) is done problem by problem on the same problems: how
often the law came earlier, how often later, an exact sign test on those, and a bootstrap interval
for the difference of the mean reciprocal place.

**What can be seen** (``power``). For a number of problems, how often a planted gain of a given
size would be told from no gain, by simulation from the places actually measured. This is the
number to look at before trusting a comparison: with ten problems almost nothing can be seen.

**Limits.** The measurement needs the true law, so it exists only on a curriculum the system makes
itself; on a table from the world there is no place to read. It scores a proposer's list, not the
route's report: a law high on the list can still be refused by the referee. It decides nothing.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp
from scipy.stats import binomtest

from apedp_dyn.conf.discovery_checks import DISCOVERY_ACCURACY_R2
from apedp_dyn.recursion import judge, mdl
from apedp_dyn.recursion.curriculum import Problem

PLACES = ("solved_place", "form_place", "accurate_place")
BOOTSTRAP = 2000


@dataclass(frozen=True)
class Judged:
    """One candidate law beside a problem's true law."""

    text: str
    r2: float | None
    bits: float
    accurate: bool          #: at the accuracy line on clean rows
    solved: bool            #: accurate and no longer than the true law: the judge's rule
    same_form: bool         #: rewrites show it equal to the true law, constants rounded
    readable: bool = True   #: False if the text could not be read as a law in the problem's inputs


def judge_candidate(problem: Problem, text: str, rows: Mapping[str, np.ndarray] | None = None,
                    truth_bits: float | None = None, *, with_form: bool = True) -> Judged:
    """A candidate law judged as ``judge.verdict`` judges a reported one. Never raises on a bad candidate."""
    rows = problem.rows() if rows is None else rows
    names = problem.names
    symbols = [sp.Symbol(name) for name in names]
    limit = mdl.law_bits(problem.truth, len(names)) if truth_bits is None else truth_bits
    try:
        expr = sp.sympify(str(text), locals=dict(zip(names, symbols, strict=True)))
        if not expr.free_symbols <= set(symbols):
            raise ValueError("a symbol that is not an input")
        with np.errstate(all="ignore"):
            pred = np.asarray(sp.lambdify(symbols, expr, "numpy")(*rows["test_X"].T), dtype=float) * np.ones(len(rows["test_X"]))
        score = judge.r2(rows["test_clean"], pred)
        bits = mdl.law_bits(expr, len(names))
    except Exception:  # noqa: BLE001 - whatever a proposer wrote that cannot be read or evaluated is not the law
        return Judged(str(text), None, math.inf, False, False, False, readable=False)
    accurate = bool(score is not None and score >= DISCOVERY_ACCURACY_R2)
    form = False
    if with_form:
        try:
            form = bool(judge.same_form(expr, problem.truth))
        except (ValueError, TypeError, RecursionError):
            form = False
    return Judged(str(text), None if score is None else round(float(score), 6), round(float(bits), 1), accurate,
                  bool(accurate and bits <= limit), form)


def places(judged: Sequence[Judged]) -> dict[str, int | None]:
    """The place (from 1) of the first solved, same-form and accurate candidate; ``None`` where there is none."""
    def first(flag: str) -> int | None:
        return next((rank for rank, item in enumerate(judged, 1) if getattr(item, flag)), None)
    return {"solved_place": first("solved"), "form_place": first("same_form"), "accurate_place": first("accurate")}


def reciprocal(place: int | None) -> float:
    return 0.0 if place is None else 1.0 / float(place)


def measure(problem: Problem, proposals: Sequence[str], *, with_form: bool = True) -> dict[str, Any]:
    """One problem's row: the three places of a proposer's ordered list, and what each candidate was."""
    rows = problem.rows()
    limit = mdl.law_bits(problem.truth, len(problem.names))
    judged = [judge_candidate(problem, text, rows, limit, with_form=with_form) for text in proposals]
    return {"key": problem.key, "level": problem.level, "inputs": problem.inputs, "noise": problem.noise,
            "proposals": len(judged), "truth_bits": round(float(limit), 1), **places(judged),
            "candidates": [asdict(item) for item in judged]}


def summarise(rows: Sequence[Mapping[str, Any]], what: str = "solved_place") -> dict[str, Any]:
    """Share found, share at place 1 and within 3, and the mean reciprocal place, overall and by level."""
    def block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        found = [row[what] for row in selected if row.get(what) is not None]
        n = len(selected)
        return {"problems": n, "found": len(found), "first": sum(place == 1 for place in found),
                "within_3": sum(place <= 3 for place in found),
                "share_found": round(len(found) / n, 4) if n else None,
                "mean_reciprocal_place": round(float(np.mean([reciprocal(row.get(what)) for row in selected])), 4) if n else None}
    out = block(rows)
    out["what"] = what
    out["by_level"] = {str(level): block([row for row in rows if row["level"] == level])
                       for level in sorted({row["level"] for row in rows})}
    return out


def compare(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], what: str = "solved_place",
            seed: int = 0) -> dict[str, Any]:
    """The second proposer against the first on the problems both were measured on, problem by problem."""
    a = {row["key"]: row.get(what) for row in first}
    b = {row["key"]: row.get(what) for row in second}
    keys = sorted(set(a) & set(b))
    earlier = sum(reciprocal(b[k]) > reciprocal(a[k]) for k in keys)
    later = sum(reciprocal(b[k]) < reciprocal(a[k]) for k in keys)
    gained = sum(a[k] is None and b[k] is not None for k in keys)
    lost = sum(a[k] is not None and b[k] is None for k in keys)
    difference = np.array([reciprocal(b[k]) - reciprocal(a[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    if len(keys):
        draws = rng.integers(0, len(keys), size=(BOOTSTRAP, len(keys)))
        means = difference[draws].mean(axis=1)
        interval: list[float | None] = [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]
    else:
        interval = [None, None]
    return {"what": what, "problems": len(keys), "earlier": int(earlier), "later": int(later),
            "same": len(keys) - int(earlier) - int(later), "gained": int(gained), "lost": int(lost),
            "sign_test_p": round(float(binomtest(int(earlier), int(earlier + later), 0.5).pvalue), 6) if earlier + later else 1.0,
            "found_p": round(float(binomtest(int(gained), int(gained + lost), 0.5).pvalue), 6) if gained + lost else 1.0,
            "mean_reciprocal_difference": round(float(difference.mean()), 4) if len(keys) else None,
            "interval_95": interval}


def power(rows: Sequence[Mapping[str, Any]], problems: int, gain: float, *, churn: float = 0.0, what: str = "solved_place",
          level: float = 0.05, trials: int = 2000, seed: int = 0) -> float:
    """How often a planted gain would be told from no gain with ``problems`` problems, by simulation.

    Problems are drawn with replacement from the measured ``rows``. The improved proposer finds the
    law on ``(1 + gain)`` times as many of them: it loses a share ``churn`` of those the measured one
    found and gains enough of the others to end ``gain`` ahead; on a problem both find, the place is
    the same. The test is the exact sign test of ``compare`` on found against not found, two-sided at
    ``level``. Returns the share of trials in which it rejects.
    """
    found = np.array([row.get(what) is not None for row in rows], dtype=bool)
    if not len(found) or not 0.0 < found.mean() < 1.0:
        return 0.0
    rng = np.random.default_rng(seed)
    rejected = 0
    for _ in range(int(trials)):
        sample = found[rng.integers(0, len(found), size=int(problems))]
        have = int(sample.sum())
        lose = int(rng.binomial(have, churn)) if have else 0
        want = round(have * gain) + lose
        gained = min(want, len(sample) - have)
        if gained + lose and _sign_test(gained, lose) <= level:
            rejected += 1
    return rejected / float(trials)


@lru_cache(maxsize=65536)
def _sign_test(gained: int, lost: int) -> float:
    """The exact two-sided sign test of ``gained`` against ``lost``; the same few pairs recur in a simulation."""
    return float(binomtest(int(gained), int(gained + lost), 0.5).pvalue)


def smallest_visible_gain(rows: Sequence[Mapping[str, Any]], problems: int, *, churn: float = 0.0, wanted: float = 0.8,
                          gains: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
                          what: str = "solved_place", trials: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The smallest of ``gains`` seen with probability ``wanted`` on ``problems`` problems, and the whole curve."""
    curve = {f"{gain:g}": round(power(rows, problems, gain, churn=churn, what=what, trials=trials, seed=seed), 3) for gain in gains}
    visible = next((gain for gain in gains if curve[f"{gain:g}"] >= wanted), None)
    return {"problems": int(problems), "churn": churn, "wanted_power": wanted, "smallest_visible_gain": visible, "power_by_gain": curve}


def run(problems: Sequence[Problem], propose: Callable[[Problem], Sequence[str]], *, with_form: bool = True
        ) -> list[dict[str, Any]]:
    """Every problem's row for one proposer; ``propose`` returns its ordered candidate laws for a problem."""
    return [measure(problem, list(propose(problem)), with_form=with_form) for problem in problems]
