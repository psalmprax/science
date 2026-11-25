"""A table searched by its parts: the shape read from the rows, a search for each part, the parts' laws put together.

Status: experimental

**What it is.** ``proposers/table_shape.py`` reads from a table how its law is put together from sums
and products of smaller parts, and hands back a table for each part. This proposer gives each part's
table to a search of its own (any proposer; a part of one input is the usual case), puts the laws
found back together in the shape read, refits the constants of the whole on the rows it was given,
and proposes what comes out. A table of five inputs is then never searched as one.

**Why the constants are refitted here.** A part is fixed only up to the changes that leave the whole
unchanged, and its table is the smooth model's part, not the rows. The laws of eight parts carry
more constants than the path refits (``main_search.polish`` stops at twelve), so the whole is
refitted here, on the real rows, before it is proposed.

**Why the rows choose each part's law, and not the part's own table.** A part's table is smooth and
looks free of noise, so a search fits it ever more closely with ever longer laws: the first run of
this proposer with PySR (25 November, ``table-split-search-run-1``) proposed accurate wholes for 38
of 80 tables of four and five inputs and the law itself for none, each part's law being the right
term with the smooth model's own wiggles fitted on top. What a part's table cannot say, the real
rows can: a fifth of the rows given is set aside, and for each part in turn the **shortest** of its
laws is taken for which the whole, refitted on the other four fifths, misses the rows set aside by
no more than ``TOLERANCE`` above what the best-fitting whole misses them by. That shortest whole is
proposed first, the best-fitting whole second.

**What it is not.** It decides nothing (AGENTS.md 1.1): what it proposes is confirmed, refused and
checked by the path like any other candidate. It reads only the rows it is given. Where the table
does not split it proposes nothing, and a wrong split yields a candidate of the wrong form.

**Not in force.** Nothing builds it but its study script and its tests
(``scripts/study_split_search.py``). Whether it joins the stage for tables is the owner's decision
after a measurement through the route (AGENTS.md 1.5).
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.discovery_l2.main_search import polish
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import table_shape
from apedp_dyn.proposers.composed import formula_proposal

SECONDS = 90.0            #: for the searches of all parts together
MIN_SECONDS = 5.0         #: the least a part's search is given
PER_PART = 30             #: laws kept for each part: a search's whole list, one law for each length it tried
MAX_PROPOSALS = 3
MAX_CONSTANTS = 64        #: constants of a whole that are refitted together
FALLBACK_DEGREE = 3       #: where a part's search returns nothing, a polynomial of this degree stands in for it
HELD_SHARE = 0.2          #: share of the rows given that is set aside to choose each part's law
TOLERANCE = 0.02          #: how much further a shorter whole may miss the rows set aside than the best-fitting whole does
MAX_WHOLES = 80           #: wholes refitted for one table while its parts' laws are chosen
Reader = Callable[..., table_shape.Reading]


def size(law: str | sp.Expr) -> int:
    """The length of a law: the nodes of its expression. A law that cannot be read is as long as can be."""
    try:
        return sum(1 for _ in sp.preorder_traversal(sp.sympify(law)))
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError):
        return 10**6


def set_aside(X: np.ndarray, y: np.ndarray, share: float = HELD_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """Rows to refit on and rows set aside, chosen by the table itself: the same table, the same rows."""
    digest = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(len(y))
    held = max(1, round(share * len(y)))
    return np.sort(order[held:]), np.sort(order[:held])


def stand_in(names: Sequence[str], X: np.ndarray, y: np.ndarray, degree: int = FALLBACK_DEGREE) -> str:
    """A polynomial in each input of a part, by least squares: what stands in for a part no search gave a law for."""
    columns, terms = [np.ones(len(y))], ["1"]
    for i, name in enumerate(names):
        for power in range(1, degree + 1):
            columns.append(X[:, i] ** power)
            terms.append(f"{name}**{power}" if power > 1 else name)
    weights, *_ = np.linalg.lstsq(np.column_stack(columns), y, rcond=None)
    return " + ".join(f"({float(w)!r})*{term}" if term != "1" else f"({float(w)!r})" for w, term in zip(weights, terms, strict=True))


class SplitSearchProposer(ProposerBase):
    """The parts of a table searched one by one and put together."""

    def __init__(self, inner: Callable[[float], ProposerBase], seconds: float = SECONDS, max_proposals: int = MAX_PROPOSALS,
                 per_part: int = PER_PART, min_seconds: float = MIN_SECONDS, target_col: str = "target",
                 reader: Reader = table_shape.read) -> None:
        self._inner, self._reader = inner, reader
        self._seconds, self._min_seconds = float(seconds), float(min_seconds)
        self._max, self._per_part = int(max_proposals), int(per_part)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "table_split"

    def _part_laws(self, names: list[str], X: np.ndarray, y: np.ndarray, seconds: float, note: dict[str, Any]) -> list[str]:
        frame = pd.DataFrame(X, columns=names)
        frame[self._target_col] = y
        laws: list[str] = []
        try:
            for proposal in self._inner(seconds).fit_and_propose(frame, target_col=self._target_col):
                text = str(proposal.formula)
                if text not in laws:
                    laws.append(text)
                if len(laws) >= self._per_part:
                    break
        except Exception as exc:  # noqa: BLE001 - one part's search failing must not cost the table; the stand-in is used and it is said
            note["error"] = f"{type(exc).__name__}: {exc}"[:160]
        if not laws:
            note["stand_in"] = True
            laws = [stand_in(names, X, y)]
        return laws

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        started = time.perf_counter()
        reading = self._reader(X, y)
        leaves = reading.shape.leaves()
        self.last_run = {"read": bool(reading.report.get("read")), "shape": reading.shape.text(features), "parts": len(leaves),
                         "widest": reading.shape.widest(), "reading_seconds": round(time.perf_counter() - started, 2),
                         "part_laws": [], "proposed": 0}
        if reading.shape.form == table_shape.LEAF:
            return ()
        each = max(self._seconds / len(leaves), self._min_seconds)
        rows = reading.rows
        found: dict[table_shape.Path, list[str]] = {}
        for path, group in leaves:
            names = [features[i] for i in group]
            note: dict[str, Any] = {"at": list(path), "inputs": names}
            found[path] = self._part_laws(names, X[np.ix_(rows, list(group))], reading.tables[path], each, note)
            note["laws"] = list(found[path])
            self.last_run["part_laws"].append(note)
        paths = [path for path, _ in leaves]
        symbols = {name: sp.Symbol(name) for name in features}
        ordered = list(symbols.values())
        fit, held = set_aside(X, y)
        refitted = 0

        def whole(choice: dict[table_shape.Path, int], rows: np.ndarray | None = fit) -> tuple[float, sp.Expr] | None:
            """The parts' chosen laws put together and refitted; how far the whole misses the rows set aside."""
            nonlocal refitted
            refitted += 1
            text = reading.compose({path: found[path][choice[path]] for path in paths})
            on = np.arange(len(y)) if rows is None else rows
            try:
                expr = polish(sp.sympify(text, locals=symbols), features, X[on], y[on], max_constants=MAX_CONSTANTS)
                with np.errstate(all="ignore"):
                    value = np.asarray(sp.lambdify(ordered, expr, "numpy")(*X[held].T), dtype=float) * np.ones(len(held))
            except Exception as exc:  # noqa: BLE001 - a whole that cannot be written or evaluated is not proposed, and it is said
                self.last_run.setdefault("wholes_not_written", []).append(f"{type(exc).__name__}: {exc}"[:160])
                return None
            if not np.all(np.isfinite(value)):
                return None
            return float(np.sqrt(np.mean((value - y[held]) ** 2))), expr

        # each part's laws shortest first; to begin with, each part has the law that fits its own table best (the search's first)
        first = {path: found[path][0] for path in paths}
        for path in paths:
            found[path] = sorted(found[path], key=size)
        best = {path: found[path].index(first[path]) for path in paths}
        start = whole(best)
        chosen = dict(best)
        if start is not None:
            limit = start[0] * (1.0 + TOLERANCE)
            for path in sorted(paths, key=lambda at: -size(found[at][best[at]])):      # the longest law first
                for shorter in range(chosen[path]):
                    if refitted >= MAX_WHOLES:
                        break
                    got = whole({**chosen, path: shorter})
                    if got is not None and got[0] <= limit:
                        chosen[path] = shorter
                        break
        proposals: list[CandidateProposal] = []
        seen: set[str] = set()
        notes = []
        for name, choice in (("shortest the rows allow", chosen), ("best fit of each part", best)):
            final = whole(choice, rows=None)                  # the constants of what is proposed are refitted on every row given
            if final is None or str(final[1]) in seen:
                continue
            seen.add(str(final[1]))
            notes.append({"whole": name, "size": size(final[1]), "laws": {"/".join(map(str, path)) or "-": found[path][choice[path]]
                                                                         for path in paths}})
            proposals.append(formula_proposal(self.proposer_id, str(final[1]), features,
                                              {"rank": len(proposals), "whole": name, "shape": self.last_run["shape"], "parts": len(leaves)}))
        self.last_run.update({"proposed": len(proposals[: self._max]), "wholes": notes, "wholes_refitted": refitted,
                              "missed_on_the_rows_set_aside": None if start is None else round(start[0], 6),
                              "searching_seconds": round(time.perf_counter() - started, 2)})
        return tuple(proposals[: self._max])
