"""A search of one input, built step by step, every step scored on the table before the next (look-ahead).

Status: experimental

**Why it exists.** The split search (``split_search.py``) cuts a wide table into parts, most of them
of one input, and gives each part to PySR for a few seconds. That step is the weak one: on the second
development set the wholes were accurate for about half of the tables and the law itself for none,
because PySR, given a smooth part table and little time, returns long laws that fit the table's
wiggles, or misses a two-block part altogether (``benchmarks/table_split/dev_search_second_run1.json``).

**The idea, and where it comes from.** The owner's non-neural program-search studies
(``psalmprax/non-neural-program-search``, study 3) found that the largest gain per candidate checked
came from *look-ahead through the interpreter*: building a program one step at a time and running
every possible next step on the examples before choosing which to keep. Here a step adds one block of
the input to the expression built so far, by a sum, a product or a quotient, and every possible step
(every block, every value of its constant on a grid) is scored on the part's table at once, in closed
form, before the best are kept for the next step (a beam). The constants of the kept candidates are
then refitted together. What is borrowed is the look-ahead and the step-by-step building; the
studies' learned guide (boosted trees choosing the next step) is not, because with sixteen blocks and
three joins every next step can simply be tried.

**The vocabulary is fixed here, before any set it is tested on was drawn**: sixteen blocks of one
input that are common in physical laws (powers, logarithms, exponentials, sines and cosines, a
hyperbolic tangent, a saturation, a shifted square root, a Gaussian), with one inner constant where a
block has a scale. It contains every block of the development grammar
(``apedp_dyn/recursion/curriculum.py``), so on tables drawn from that grammar it measures
in-distribution performance and must be read as such; a set drawn from another generator, with blocks
outside this vocabulary, is the test of it (AGENTS.md 1.6).

**What it misses, seen in development** (25-26 November, on tables of one input written by hand and
on the first development set): a quotient of two blocks that both have a constant is found on some
samples of rows and not on others (``1.7x/(x + 0.7)/(sin(2.3x) + 1.3)``), and
``x/(x + 1.3)/(0.7 + exp(-0.45x))`` was not found on three samples, a law of three blocks fitting the
table as closely; blocks outside the vocabulary (``atan``, ``cosh``) are only approximated. On noisy
tables a law of one block often fits the rows set aside as well as the right law of two, so the split
search, which takes the shortest law within its tolerance, may take it.

**What it is not.** It decides nothing (AGENTS.md 1.1). It returns a list of laws of different lengths,
best fit to its table first; the split search's rows choose among them, and the path confirms or
refuses what is proposed. Tables of more than one input are not searched here (nothing is proposed).
Not in force: nothing builds it but its tests and the split-search study.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares, minimize_scalar

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal

GRID = np.geomspace(0.03, 16.0, 28)           #: values tried for a block's inner constant
OFFSETS = np.geomspace(0.05, 20.0, 12)        #: values tried for the offset ``b`` of ``E/(b + g)`` and ``E*(b + g)``
BEAM = 10                                      #: structures kept at each step
DEPTH = 3                                      #: blocks at most
KEEP = (12, 8, 6)                              #: laws returned of one, two and three blocks, each with and without its constant
REFINED = 30                                   #: candidates of each length whose constants are refitted before ranking
LARGEST = 1.0e8                                #: a column with a larger value is not used (overflow, conditioning)
FLAT = 0.05                                    #: a scaled block whose argument stays below this is in its polynomial regime
CANCELS = 1.0e3                                #: a term this many times larger than the table is cancelling another
SECONDS = 30.0


@dataclasses.dataclass(frozen=True)
class Block:
    name: str
    template: str                              #: ``{x}`` the input, ``{a}`` the inner constant
    fn: Callable[[np.ndarray, float], np.ndarray]
    scaled: bool


BLOCKS: tuple[Block, ...] = (
    Block("x", "{x}", lambda x, a: x, False),
    Block("x2", "{x}**2", lambda x, a: x**2, False),
    Block("x3", "{x}**3", lambda x, a: x**3, False),
    Block("inv", "1/{x}", lambda x, a: 1.0 / x, False),
    Block("inv2", "{x}**(-2)", lambda x, a: x**-2.0, False),
    Block("sqrt", "sqrt({x})", lambda x, a: np.sqrt(x), False),
    Block("log", "log({x})", lambda x, a: np.log(x), False),
    Block("exp_neg", "exp(-{a}*{x})", lambda x, a: np.exp(-a * x), True),
    Block("exp_pos", "exp({a}*{x})", lambda x, a: np.exp(a * x), True),
    Block("log1p", "log(1 + {a}*{x})", lambda x, a: np.log1p(a * x), True),
    Block("sin", "sin({a}*{x})", lambda x, a: np.sin(a * x), True),
    Block("cos", "cos({a}*{x})", lambda x, a: np.cos(a * x), True),
    Block("tanh", "tanh({a}*{x})", lambda x, a: np.tanh(a * x), True),
    Block("sat", "{x}/({a} + {x})", lambda x, a: x / (a + x), True),
    Block("sqrt_shift", "sqrt({a} + {x})", lambda x, a: np.sqrt(a + x), True),
    Block("gauss", "exp(-{a}*{x}**2)", lambda x, a: np.exp(-a * x**2), True),
)

# A candidate is a tuple tree. Leaves: ("leaf", block, a). Steps:
#   ("inverse", b, leaf)        1/(b + leaf)
#   ("sum", E, w, leaf)         E + w*leaf
#   ("product", E, leaf)        E*leaf
#   ("quotient", E, b, leaf)    E/(b + leaf)
#   ("scaled", E, b, leaf)      E*(b + leaf)
# The whole law is c0 + c1*E, with c0 and c1 fitted linearly.
Node = tuple[Any, ...]


def _leaf_value(node: Node, x: np.ndarray) -> np.ndarray:
    _, b, a = node
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(BLOCKS[b].fn(x, a), dtype=float)
    return out


def value(node: Node, x: np.ndarray) -> np.ndarray:
    kind = node[0]
    out: np.ndarray
    with np.errstate(all="ignore"):
        if kind == "leaf":
            out = _leaf_value(node, x)
        elif kind == "inverse":
            out = 1.0 / (node[1] + value(node[2], x))
        elif kind == "sum":
            out = value(node[1], x) + node[2] * value(node[3], x)
        elif kind == "product":
            out = value(node[1], x) * value(node[2], x)
        elif kind == "scaled":
            out = value(node[1], x) * (node[2] + value(node[3], x))
        else:
            out = value(node[1], x) / (node[2] + value(node[3], x))
    return out


def text(node: Node, name: str) -> str:
    kind = node[0]
    if kind == "leaf":
        block = BLOCKS[node[1]]
        return block.template.format(x=name, a=repr(float(node[2])))
    if kind == "inverse":
        return f"1/({float(node[1])!r} + {text(node[2], name)})"
    if kind == "sum":
        return f"({text(node[1], name)} + ({float(node[2])!r})*{text(node[3], name)})"
    if kind == "product":
        return f"({text(node[1], name)})*({text(node[2], name)})"
    if kind == "scaled":
        return f"({text(node[1], name)})*({float(node[2])!r} + {text(node[3], name)})"
    return f"({text(node[1], name)})/({float(node[2])!r} + {text(node[3], name)})"


def structure(node: Node) -> str:
    """The candidate with its constants left out: what the beam keeps one of."""
    kind = node[0]
    if kind == "leaf":
        return BLOCKS[node[1]].name
    if kind == "inverse":
        return f"inv({structure(node[2])})"
    if kind == "sum":
        return f"sum({structure(node[1])},{structure(node[3])})"
    if kind == "product":
        return f"prod({structure(node[1])},{structure(node[2])})"
    if kind == "scaled":
        return f"shprod({structure(node[1])},{structure(node[3])})"
    return f"quot({structure(node[1])},{structure(node[3])})"


def blocks_in(node: Node) -> int:
    kind = node[0]
    if kind == "leaf":
        return 1
    if kind == "inverse":
        return blocks_in(node[2])
    if kind in ("sum", "quotient", "scaled"):
        return blocks_in(node[1]) + blocks_in(node[3])
    return blocks_in(node[1]) + blocks_in(node[2])


def _constants(node: Node) -> list[float]:
    kind = node[0]
    if kind == "leaf":
        return [float(node[2])] if BLOCKS[node[1]].scaled else []
    if kind == "inverse":
        return [float(node[1]), *_constants(node[2])]
    if kind == "sum":
        return [*_constants(node[1]), float(node[2]), *_constants(node[3])]
    if kind == "product":
        return [*_constants(node[1]), *_constants(node[2])]
    return [*_constants(node[1]), float(node[2]), *_constants(node[3])]


def _with_constants(node: Node, values: list[float]) -> Node:
    """The same structure with its constants taken, in order, from ``values`` (consumed)."""
    kind = node[0]
    if kind == "leaf":
        return ("leaf", node[1], values.pop(0)) if BLOCKS[node[1]].scaled else node
    if kind == "inverse":
        b = values.pop(0)
        return ("inverse", b, _with_constants(node[2], values))
    if kind == "sum":
        left = _with_constants(node[1], values)
        w = values.pop(0)
        return ("sum", left, w, _with_constants(node[3], values))
    if kind == "product":
        left = _with_constants(node[1], values)
        return ("product", left, _with_constants(node[2], values))
    left = _with_constants(node[1], values)
    b = values.pop(0)
    return (kind, left, b, _with_constants(node[3], values))


def _usable(columns: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        ok: np.ndarray = np.all(np.abs(columns) < LARGEST, axis=0)   # a NaN compares False, an infinity is not below
    return ok


def _two_column_rss(y: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Residual sum of squares of ``y ≈ c0 + c1*v`` for every column ``v``, in closed form."""
    yc = y - y.mean()
    ok = _usable(columns)
    vc = np.where(ok, columns, 0.0)
    vc = vc - vc.mean(axis=0)
    norm = np.einsum("ij,ij->j", vc, vc)
    along = yc @ vc
    rss = float(yc @ yc) - np.where(norm > 1e-12, along**2 / np.where(norm > 1e-12, norm, 1.0), 0.0)
    out: np.ndarray = np.where(ok & (norm > 1e-12), rss, np.inf)
    return out


def _three_column(y: np.ndarray, e: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``y ≈ c0 + c1*E + c2*g`` for every column ``g``: the residual sums of squares and ``w = c2/c1``."""
    yc, ec = y - y.mean(), e - e.mean()
    ee = float(ec @ ec)
    n = len(columns.T)
    if ee <= 1e-12:
        return np.full(n, np.inf), np.zeros(n)
    ok = _usable(columns)
    gc = np.where(ok, columns, 0.0)
    gc = gc - gc.mean(axis=0)
    r = yc - (yc @ ec / ee) * ec
    g_res = gc - np.outer(ec, (ec @ gc) / ee)
    gg = np.einsum("ij,ij->j", g_res, g_res)
    safe = np.where(gg > 1e-12, gg, 1.0)
    c2 = (r @ g_res) / safe
    rss = float(r @ r) - (r @ g_res) ** 2 / safe
    c1 = ((yc @ ec) - c2 * (gc.T @ ec)) / ee
    good = ok & (gg > 1e-12) & (np.abs(c1) > 1e-12)
    return np.where(good, rss, np.inf), np.where(good, c2 / np.where(np.abs(c1) > 1e-12, c1, 1.0), 0.0)


def _block_columns(x: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """A block's values for every value of its constant on the grid (one column if it has none)."""
    block = BLOCKS[b]
    grid = GRID if block.scaled else np.array([1.0])
    with np.errstate(all="ignore"):
        return np.column_stack([np.asarray(block.fn(x, float(a)), dtype=float) * np.ones(len(x)) for a in grid]), grid


def _first_step(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, Node]]:
    """Every block alone, and every block as ``1/(b + g)``, with its constants chosen on the grids and then between."""
    best: dict[str, tuple[float, Node]] = {}
    for b in range(len(BLOCKS)):
        columns, grid = _block_columns(x, b)
        rss = _two_column_rss(y, columns)
        k = int(np.argmin(rss))
        if np.isfinite(rss[k]):
            a, score = float(grid[k]), float(rss[k])
            if BLOCKS[b].scaled:
                def alone(v: float, b: int = b) -> float:
                    return float(_two_column_rss(y, _block_columns_at(x, b, v))[0])
                better = _between(alone, grid, k)
                if better is not None and alone(better) < score:
                    a, score = better, alone(better)
            node: Node = ("leaf", b, a)
            best[structure(node)] = (score, node)
        with np.errstate(all="ignore"):
            table = np.stack([_two_column_rss(y, 1.0 / (o + columns)) for o in OFFSETS])
        i, k = (int(v) for v in np.unravel_index(int(np.argmin(table)), table.shape))
        if not np.isfinite(table[i, k]):
            continue
        offset, a, score = float(OFFSETS[i]), float(grid[k]), float(table[i, k])

        def inverse(o: float, v: float, b: int = b) -> float:
            with np.errstate(all="ignore"):
                return float(_two_column_rss(y, 1.0 / (o + _block_columns_at(x, b, v)))[0])

        for _ in range(2):                                    # the offset, then the block's constant, twice
            def by_offset(o: float, a: float = a) -> float:
                return inverse(o, a)

            better = _between(by_offset, OFFSETS, int(np.clip(np.searchsorted(OFFSETS, offset), 0, len(OFFSETS) - 1)))
            if better is not None and inverse(better, a) < score:
                offset, score = better, inverse(better, a)

            def by_constant(v: float, o: float = offset) -> float:
                return inverse(o, v)

            if BLOCKS[b].scaled:
                better = _between(by_constant, np.array([a / 1.3, a, a * 1.3]), 1)
                if better is not None and inverse(offset, better) < score:
                    a, score = better, inverse(offset, better)
        node = ("inverse", offset, ("leaf", b, a))
        best[structure(node)] = (score, node)
    return best


def _between(score: Callable[[float], float], grid: np.ndarray, k: int) -> float | None:
    """The constant that scores best between the grid's neighbours of its best point (bounded Brent, on its logarithm)."""
    if len(grid) < 3:
        return None
    low, high = np.log(grid[max(k - 1, 0)]), np.log(grid[min(k + 1, len(grid) - 1)])
    found = minimize_scalar(lambda u: score(float(np.exp(u))), bounds=(low, high), method="bounded",
                            options={"xatol": 1e-6, "maxiter": 30})
    return float(np.exp(found.x)) if found.success else None


def _joined(y: np.ndarray, kind: str, e: np.ndarray, g: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Scores of joining ``E`` with every column ``g`` by ``kind``: residual sums of squares, and ``w`` for a sum."""
    with np.errstate(all="ignore"):
        if kind == "sum":
            return _three_column(y, e, g)
        if kind == "product":
            v = e[:, None] * g
        elif kind == "quotient":
            v = e[:, None] / (offset + g)
        else:
            v = e[:, None] * (offset + g)
    return _two_column_rss(y, v), np.zeros(g.shape[1])


def _node(kind: str, e_node: Node, offset: float, w: float, leaf: Node) -> Node:
    if kind == "sum":
        return ("sum", e_node, w, leaf)
    if kind == "product":
        return ("product", e_node, leaf)
    return (kind, e_node, offset, leaf)


def _next_step(x: np.ndarray, y: np.ndarray, beam: Sequence[Node]) -> dict[str, tuple[float, Node]]:
    """Every block joined to every candidate of the beam by every join, each scored on the table.

    The new block's constant is chosen on the grid and then between its neighbours. Where the candidate
    is a single block with a constant of its own, that constant is chosen again with the new block in
    place, and the new block's after it: a constant fitted to the table alone is rarely the one it has
    in a law of two blocks.
    """
    best: dict[str, tuple[float, Node]] = {}

    def keep(score: float, node: Node) -> None:
        key = structure(node)
        if np.isfinite(score) and (key not in best or score < best[key][0]):
            best[key] = (float(score), node)

    columns_of = [_block_columns(x, b) for b in range(len(BLOCKS))]
    for e_node in beam:
        e = value(e_node, x)
        if not np.all(np.isfinite(e)):
            continue
        own = e_node[0] == "leaf" and BLOCKS[e_node[1]].scaled
        for b, (columns, grid) in enumerate(columns_of):
            scaled = BLOCKS[b].scaled
            for kind in ("sum", "product", "quotient", "scaled"):
                offsets = OFFSETS if kind in ("quotient", "scaled") else np.array([0.0])
                scores = [_joined(y, kind, e, columns, float(o)) for o in offsets]
                table = np.stack([rss for rss, _ in scores])
                i, k = (int(v) for v in np.unravel_index(int(np.argmin(table)), table.shape))
                offset, a, score = float(offsets[i]), float(grid[k]), float(table[i, k])
                e_now, e_values = e_node, e
                if own and kind in ("sum", "product"):        # both constants on the grid together
                    e_columns, e_grid = _block_columns(x, e_node[1])
                    both = np.stack([_joined(y, kind, e_columns[:, j], columns, 0.0)[0] for j in range(len(e_grid))])
                    j, k = (int(v) for v in np.unravel_index(int(np.argmin(both)), both.shape))
                    if both[j, k] < score:
                        e_now = ("leaf", e_node[1], float(e_grid[j]))
                        e_values = e_columns[:, j]
                        a, score = float(grid[k]), float(both[j, k])
                elif own:                                     # the first block's constant on every third grid point, with the offset
                    e_columns, e_grid = _block_columns(x, e_node[1])
                    for j in range(1, len(e_grid), 3):
                        cube = np.stack([_joined(y, kind, e_columns[:, j], columns, float(o))[0] for o in offsets])
                        i2, k2 = (int(v) for v in np.unravel_index(int(np.argmin(cube)), cube.shape))
                        if cube[i2, k2] < score:
                            e_now = ("leaf", e_node[1], float(e_grid[j]))
                            e_values = e_columns[:, j]
                            offset, a, score = float(offsets[i2]), float(grid[k2]), float(cube[i2, k2])
                            i, k = i2, k2
                if not np.isfinite(score):
                    continue

                def new_score(v: float, e_values: np.ndarray = e_values, offset: float = offset, kind: str = kind, b: int = b) -> float:
                    return float(_joined(y, kind, e_values, _block_columns_at(x, b, v), offset)[0][0])

                if scaled:
                    better = _between(new_score, grid, int(k))
                    if better is not None and new_score(better) < score:
                        a, score = better, new_score(better)
                if kind in ("quotient", "scaled"):                # the offset between its grid neighbours
                    def by_offset(o: float, a: float = a, e_values: np.ndarray = e_values, kind: str = kind, b: int = b) -> float:
                        return float(_joined(y, kind, e_values, _block_columns_at(x, b, a), o)[0][0])
                    better = _between(by_offset, OFFSETS, int(i))
                    if better is not None and by_offset(better) < score:
                        offset, score = better, by_offset(better)
                if own:
                    g = _block_columns_at(x, b, a)
                    e_columns, e_grid = _block_columns(x, e_node[1])
                    along = np.array([float(_joined(y, kind, e_columns[:, j], g, offset)[0][0]) for j in range(len(e_grid))])
                    j = int(np.argmin(along))

                    def e_score(v: float, g: np.ndarray = g, offset: float = offset, kind: str = kind, e_block: int = e_node[1]) -> float:
                        return float(_joined(y, kind, _block_columns_at(x, e_block, v)[:, 0], g, offset)[0][0])

                    a_e = float(e_grid[j])
                    better = _between(e_score, e_grid, j)
                    if better is not None and e_score(better) < e_score(a_e):
                        a_e = better
                    if e_score(a_e) < score:
                        e_now = ("leaf", e_node[1], a_e)
                        e_values = value(e_now, x)
                        score = e_score(a_e)
                        if scaled:
                            def again_score(v: float, ev: np.ndarray = e_values) -> float:
                                return new_score(v, ev)

                            again = _between(again_score, np.array([a / 1.3, a, a * 1.3]), 1)
                            if again is not None and new_score(again, e_values) < score:
                                a, score = again, new_score(again, e_values)
                w = float(_joined(y, kind, e_values, _block_columns_at(x, b, a), offset)[1][0])
                keep(score, _node(kind, e_now, offset, w, ("leaf", b, a)))
    return best


def _block_columns_at(x: np.ndarray, b: int, a: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        out: np.ndarray = (np.asarray(BLOCKS[b].fn(x, float(a)), dtype=float) * np.ones(len(x)))[:, None]
    return out


def _outer(y: np.ndarray, e: np.ndarray, intercept: bool = True) -> tuple[float, float, np.ndarray]:
    """``y ≈ c0 + c1*E`` by least squares, or ``y ≈ c1*E`` without the intercept."""
    if intercept:
        design = np.column_stack([np.ones(len(e)), e])
        (c0, c1), *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        ee = float(e @ e)
        c0, c1 = 0.0, (float(e @ y) / ee if ee > 0 else 0.0)
    return float(c0), float(c1), y - (c0 + c1 * e)


def refine(node: Node, x: np.ndarray, y: np.ndarray, intercept: bool = True) -> tuple[float, Node, float, float]:
    """All constants of a candidate refitted together; the outer two by least squares inside (variable projection)."""
    start = _constants(node)

    def residual(values: np.ndarray) -> np.ndarray:
        e = value(_with_constants(node, list(values)), x)
        if not np.all(np.isfinite(e)):
            return np.full(len(y), 1e6)
        return _outer(y, e, intercept)[2]

    if start:
        try:
            fitted = least_squares(residual, np.asarray(start, dtype=float), method="lm", max_nfev=40 * (len(start) + 1))
            node = _with_constants(node, list(fitted.x))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    e = value(node, x)
    if not np.all(np.isfinite(e)):
        return np.inf, node, 0.0, 0.0
    c0, c1, left = _outer(y, e, intercept)
    return float(left @ left), node, c0, c1


def _flat(node: Node, x: np.ndarray) -> bool:
    """A block that only imitates a simpler one: ``cos(0.01 x)`` is a parabola, ``x/(500 + x)`` a line."""
    kind = node[0]
    if kind == "leaf":
        block, a = BLOCKS[node[1]], float(node[2])
        if not block.scaled:
            return False
        span = float(np.max(np.abs(x)))
        if block.name in ("sat", "sqrt_shift"):
            return a > span / FLAT
        reach = a * span**2 if block.name == "gauss" else a * span
        return reach < FLAT
    return any(_flat(part, x) for part in node[1:] if isinstance(part, tuple))


def degenerate(node: Node, c0: float, c1: float, x: np.ndarray, y: np.ndarray) -> bool:
    """A law whose fit rests on a block in its polynomial regime, or on terms far larger than the table cancelling."""
    if _flat(node, x):
        return True
    size = float(np.max(np.abs(y))) + float(np.std(y)) + 1e-12
    with np.errstate(all="ignore"):
        e = value(node, x)
    return bool(abs(c0) > CANCELS * size or float(np.max(np.abs(c1 * e))) > CANCELS * size)


def search(x: np.ndarray, y: np.ndarray, name: str = "x", depth: int = DEPTH, beam: int = BEAM,
           keep: Sequence[int] = KEEP, deadline: float | None = None) -> list[tuple[float, str, int]]:
    """Laws of one input for the table ``(x, y)``: ``(mean squared error, law, blocks)``, best fit first."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    found: list[tuple[float, str, int]] = []
    level = _first_step(x, y)
    for step in range(1, depth + 1):
        ranked = sorted(level.values(), key=lambda item: item[0])
        refined = []
        for _, node in ranked[:REFINED]:
            rss, fitted, c0, c1 = refine(node, x, y)
            if np.isfinite(rss):
                refined.append((rss, fitted, c0, c1))
        refined.sort(key=lambda item: item[0])
        # a degenerate law is not returned, but its blocks stay in the beam: the next step refits their constants
        kept = [item for item in refined if not degenerate(item[1], item[2], item[3], x, y)]
        for rss, fitted, c0, c1 in kept[: keep[step - 1]]:
            found.append((rss / len(y), f"{c0!r} + ({c1!r})*({text(fitted, name)})", blocks_in(fitted)))
            # the same law without its additive constant, refitted: shorter, and where the part has none the
            # right blocks fit without it and the wrong ones do not, which is what lets the rows tell them apart
            bare = refine(fitted, x, y, intercept=False)
            if np.isfinite(bare[0]) and not degenerate(bare[1], 0.0, bare[3], x, y):
                found.append((bare[0] / len(y), f"({bare[3]!r})*({text(bare[1], name)})", blocks_in(bare[1])))
        if step < depth:                                      # the next step starts from the refitted candidates
            ranked = [(rss, fitted) for rss, fitted, _, _ in refined] + list(ranked[REFINED:])
        if step == depth or (deadline is not None and time.perf_counter() > deadline):
            break
        chosen = [node for _, node in ranked[:beam]]
        if step == 1:                                         # every block, alone, may start a law of two or three
            chosen += [node for _, node in ranked[beam:] if node[0] == "leaf"]
        level = _next_step(x, y, chosen)
    seen: set[str] = set()
    out = []
    for item in sorted(found, key=lambda item: item[0]):
        if item[1] not in seen:
            seen.add(item[1])
            out.append(item)
    return out


class BlockSearchProposer(ProposerBase):
    """The block search as a proposer for a table of one input; a wider table gets nothing."""

    def __init__(self, seconds: float = SECONDS, max_proposals: int = 2 * sum(KEEP), target_col: str = "target") -> None:
        self._seconds, self._max, self._target_col = float(seconds), int(max_proposals), target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "block_search"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        if len(features) != 1:
            self.last_run = {"searched": False, "inputs": len(features)}
            return ()
        began = time.perf_counter()
        laws = search(frame[features[0]].to_numpy(dtype=float), frame[target].to_numpy(dtype=float), features[0],
                      deadline=began + self._seconds)
        self.last_run = {"searched": True, "laws": len(laws), "seconds": round(time.perf_counter() - began, 2)}
        return tuple(formula_proposal(self.proposer_id, law, features, {"rank": i, "blocks": blocks, "mse": mse})
                     for i, (mse, law, blocks) in enumerate(laws[: self._max]))


def _size(law: str) -> int:
    try:
        return sum(1 for _ in sp.preorder_traversal(sp.sympify(law)))
    except (sp.SympifyError, TypeError, SyntaxError, AttributeError):
        return 10**6


def _mse(proposal: CandidateProposal, frame: pd.DataFrame, target: str) -> float:
    try:
        with np.errstate(all="ignore"):
            pred = np.asarray(proposal.compute_score(frame), dtype=float)
        y = frame[target].to_numpy(dtype=float)
        out = float(np.mean((pred - y) ** 2))
        return out if np.isfinite(out) else np.inf
    except Exception:  # noqa: BLE001 - a law that cannot be evaluated on the table goes to the end of the list
        return np.inf


class BlocksFirst(ProposerBase):
    """The block search, then another search with the seconds left; both lists together, best fit first.

    Neither search is given more time than the part was given: the block search's seconds are taken
    from the other's. Where the block search proposes nothing (a part of more inputs), the other
    search has the whole time.
    """

    def __init__(self, seconds: float, inner: Callable[[float], ProposerBase], min_seconds: float = 5.0,
                 target_col: str = "target") -> None:
        self._seconds, self._inner, self._min = float(seconds), inner, float(min_seconds)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "blocks_first"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        began = time.perf_counter()
        blocks = BlockSearchProposer(seconds=self._seconds).fit_and_propose(frame, target_col=target)
        spent = time.perf_counter() - began
        other = list(self._inner(max(self._seconds - spent, self._min)).fit_and_propose(frame, target_col=target))
        merged = sorted([*blocks, *other], key=lambda p: _mse(p, frame, target))
        self.last_run = {"block_laws": len(blocks), "other_laws": len(other), "block_seconds": round(spent, 2)}
        if not merged:
            return ()
        # the best fit first (the split search starts from it); then the rest shortest first, so that a list
        # cut short loses the longest laws, the ones least likely to be the law
        return (merged[0], *sorted(merged[1:], key=lambda p: _size(p.formula)))
