"""A real language model as one of MMLM's experts: it reads a table's numbers and says which shapes of law to try first.

Status: experimental

**Development; not in force.** MMLM's equation model (``equation_language``) is a prior that does not look at
the data: it says how plausible a shape is before any row is seen, and the exhaustive search tries shapes in
that order. Its experts are counting models and one small network. This module adds an expert of another kind,
asked for by the owner on 2026-11-24: a large language model that **does** look at the data. It is shown rows of
the table, with the inputs named ``x0, x1, ...`` and nothing else, and replies with shapes. Two things are done
with them:

* the shapes it names are tried **first**, in its order;
* the equation model is taught them (``SelfAdjustingEquationModel.observe``) on a copy, so that the rest of the
  search is ordered by a prior that has moved towards what the language model said. The copy is thrown away
  after the table: nothing the language model says is learned for good.

The language model proposes an order of search and nothing else. Every shape is still fitted and judged by the
search and the referee as before; a wrong hint costs time, not correctness.

The model is any ``client(prompt) -> text`` callable; this module has no network code and never sees a key.
Only text made of the allowed names, numbers and operators is parsed (``llm_proposer.parse_skeleton``).
Whether the expert helps is a measurement: ``scripts/study_mmlm_llm_expert.py``.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sympy as sp

from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.proposers import llm_proposer
from apedp_mmlm import equation_language as el

ROWS_SHOWN = 24              # rows of the table the language model is shown
SHAPES_WANTED = 8            # shapes it is asked for
REPEATS = 3                  # times each shape is taught to the copy of the equation model
REPLY_KEPT = 1500            # characters of a reply kept in a record


def table_prompt(X: np.ndarray, y: np.ndarray, *, shown: int = ROWS_SHOWN, wanted: int = SHAPES_WANTED) -> str:
    """The prompt: rows of the table spread over its range, numbers only, and the form the reply must take."""
    n_vars = X.shape[1]
    names = [f"x{i}" for i in range(n_vars)]
    picked = np.argsort(y, kind="stable")[np.linspace(0, len(y) - 1, min(shown, len(y))).round().astype(int)]
    rows = [", ".join(f"{v:.4g}" for v in (*X[i], y[i])) for i in picked]
    return "\n".join([
        f"Here are {len(rows)} measurements of y for the inputs {', '.join(names)}, taken from a table of {len(y)} rows.",
        f"Find the shape of the law y = f({', '.join(names)}).",
        f"Write up to {wanted} candidate shapes, the most likely first, one per line.",
        f"Use only {', '.join(names)}, the operators + - * / **, the functions sqrt exp log sin cos, and small whole numbers.",
        "Do not write fitted constants: an overall scale and an offset are fitted for you (write x0/x1**2, not 1.73*x0/x1**2 + 0.2).",
        "Reply with the shapes only, no explanation.",
        "",
        f"Rows ({', '.join([*names, 'y'])}):",
        *rows,
    ])


def shapes_in(reply: str, n_vars: int, wanted: int = SHAPES_WANTED) -> tuple[list[sp.Expr], int]:
    """The shapes a reply names, in its order, each once, and how many of its lines could not be read as one."""
    shapes: list[sp.Expr] = []
    unread = 0
    for line in str(reply).splitlines():
        match = llm_proposer._LINE.match(line)
        text = (match.group(1) if match else line).strip().strip("`")
        if not text:
            continue
        try:
            expr = llm_proposer.parse_skeleton(text, n_vars)
        except Exception:  # noqa: BLE001 - a line that is not an allowed expression is counted and left out
            unread += 1
            continue
        expr = expr.xreplace({s: sp.Integer(1) for s in expr.free_symbols if str(s).startswith("c")})
        if not expr.free_symbols or any(expr.equals(seen) for seen in shapes):
            continue
        shapes.append(expr)
        if len(shapes) == wanted:
            break
    return shapes, unread


def place_in_search(expr: sp.Expr, n_vars: int, index: dict[bytes, int]) -> int | None:
    """Where the exhaustive search has this shape (scale and offset aside), or ``None`` if it does not enumerate it."""
    symbols = [sp.Symbol(f"x{i}") for i in range(n_vars)]
    try:
        fn = sp.lambdify(symbols, expr, "numpy")
        with np.errstate(all="ignore"):
            key = es._fingerprint(np.asarray(fn(*es._probes(n_vars).T), dtype=float) * np.ones(es.N_PROBE))
    except Exception:  # noqa: BLE001 - a shape that cannot be evaluated has no place in the search
        return None
    return index.get(key) if key is not None else None


def _tree_of(expr: sp.Expr) -> el.Tree | None:
    """The shape in the equation language's dialect, or ``None`` if it is outside it (such a shape teaches nothing)."""
    try:
        return el.to_tree(expr)
    except Exception:  # noqa: BLE001 - the dialect refuses what it cannot write; the reason does not matter here
        return None


def taught(model: el.SelfAdjustingEquationModel, shapes: Sequence[sp.Expr], repeats: int = REPEATS) -> el.SelfAdjustingEquationModel:
    """A copy of the equation model that has been shown the shapes. The model itself is not changed."""
    informed = copy.deepcopy(model)
    for expr in shapes:
        tree = _tree_of(expr)
        for _ in range(repeats if tree is not None else 0):
            informed.observe(tree)
    return informed


def rank_by(scores: np.ndarray, where: int) -> int:
    """The place of candidate ``where`` when candidates are tried in order of ``scores``, ties by enumeration order."""
    mine = scores[where]
    return int(np.sum(scores < mine) + np.sum((scores == mine)[:where]) + 1)


def rank_with_expert(where: int, named: Sequence[int], informed_scores: np.ndarray) -> int:
    """The place of candidate ``where`` when the shapes the expert named are tried first, then the rest by the taught prior."""
    first = list(dict.fromkeys(named))
    if where in first:
        return first.index(where) + 1
    scores: np.ndarray = informed_scores.astype(float).copy()
    scores[first] = -np.inf                                    # already tried
    return rank_by(scores, where)


class LlmExpert:
    """A language model that reads a table and names shapes of law to try first."""

    def __init__(self, client: Callable[[str], str], *, shown: int = ROWS_SHOWN, wanted: int = SHAPES_WANTED) -> None:
        self.client = client
        self.shown, self.wanted = shown, wanted

    def shapes(self, X: np.ndarray, y: np.ndarray) -> tuple[list[sp.Expr], dict[str, Any]]:
        """The shapes named for this table, and the record of the call (reply, token counts, lines not read)."""
        prompt = table_prompt(X, y, shown=self.shown, wanted=self.wanted)
        record: dict[str, Any] = {"prompt_characters": len(prompt)}
        try:
            reply = self.client(prompt)
        except Exception as exc:  # noqa: BLE001 - an expert that cannot be reached names nothing; the search goes on without it
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return [], record
        shapes, unread = shapes_in(reply, X.shape[1], self.wanted)
        record.update({"reply": str(reply)[:REPLY_KEPT], "usage": dict((getattr(self.client, "last", None) or {}).get("usage") or {}),
                       "shapes": [str(s) for s in shapes], "lines_not_read": unread})
        return shapes, record
