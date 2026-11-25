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

**What it is not.** It decides nothing (AGENTS.md 1.1): what it proposes is confirmed, refused and
checked by the path like any other candidate. It reads only the rows it is given. Where the table
does not split it proposes nothing, and a wrong split yields a candidate of the wrong form.

**Not in force.** Nothing builds it but its study script and its tests
(``scripts/study_split_search.py``). Whether it joins the stage for tables is the owner's decision
after a measurement through the route (AGENTS.md 1.5).
"""
from __future__ import annotations

import itertools
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
PER_PART = 2              #: laws kept for each part
MAX_PROPOSALS = 3
MAX_CONSTANTS = 64        #: constants of a whole that are refitted together
FALLBACK_DEGREE = 3       #: where a part's search returns nothing, a polynomial of this degree stands in for it
Reader = Callable[..., table_shape.Reading]


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
        # the first law of every part; then every way to take one part's second law instead
        paths = [path for path, _ in leaves]
        choices = [dict.fromkeys(paths, 0)]
        choices += [{**choices[0], path: 1} for path in paths if len(found[path]) > 1]
        symbols = {name: sp.Symbol(name) for name in features}
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for choice in itertools.islice(choices, 1 + 2 * self._max):
            text = reading.compose({path: found[path][choice[path]] for path in paths})
            try:
                expr = polish(sp.sympify(text, locals=symbols), features, X, y, max_constants=MAX_CONSTANTS)
                with np.errstate(all="ignore"):
                    value = np.asarray(sp.lambdify(list(symbols.values()), expr, "numpy")(*X.T), dtype=float) * np.ones(len(y))
            except Exception as exc:  # noqa: BLE001 - a whole that cannot be written or evaluated is not proposed, and it is said
                self.last_run.setdefault("wholes_not_written", []).append(f"{type(exc).__name__}: {exc}"[:160])
                continue
            if not np.all(np.isfinite(value)) or str(expr) in seen:
                continue
            seen.add(str(expr))
            scored.append((float(np.sqrt(np.mean((value - y) ** 2))), str(expr)))
        scored.sort()
        self.last_run.update({"proposed": min(len(scored), self._max), "searching_seconds": round(time.perf_counter() - started, 2),
                              "errors_on_the_rows_given": [round(error, 6) for error, _ in scored[: self._max]]})
        return tuple(formula_proposal(self.proposer_id, text, features, {"rank": rank, "shape": self.last_run["shape"],
                                                                          "parts": len(leaves)})
                     for rank, (_, text) in enumerate(scored[: self._max]))
