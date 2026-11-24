"""Building blocks learned from solved laws by compression, and a proposer that builds candidates from them.

Status: experimental

**Learning the library** (``Library.learn``). Every part of every solved law
that depends on an input is a *fragment*: its inputs become numbered holes
(``h0``, ``h1``; up to ``MAX_HOLES``), its fitted constants become ``C``, its
scale and offset are dropped. A fragment met in several laws can be named once
and reused. Naming it is worth, in bits (``mdl``),

    uses * (cost of writing it out - cost of one use of the name) - cost of defining it

and the library is the fragments with the largest positive worth, up to
``max_blocks``. A fragment that only ever appears inside a chosen one, as often,
is not counted again. This is DreamCoder's compression step in a small form:
the worths are computed once on the laws as written, not re-computed after each
choice, so two overlapping blocks can both be chosen.

It differs from the memory of confirmed laws (``proposers.structure_memory``),
which keeps every one-variable part it meets and each whole law: the library
keeps parts in **more than one input** (``h0/(C + h1)``), and keeps a part only
if naming it shortens the laws.

**Building candidates** (``LibraryProposer``). Each block is placed on the
problem's inputs in every way, and fitted alone (scale, offset and its own
constants). The best ``top`` placements are then joined two at a time (sum,
product, quotient), and three at a time when the problem has three inputs or
more, and fitted again. Candidates that use every input are ranked by their
total description length. A candidate that rewrites show equal to a better one
(``egraph``) is dropped, and so is one whose denominator a solver proves reaches
zero inside the rows' range (``smt``). What is left goes to the referee as
proposals, like any other proposer's; the proposer decides nothing.
"""
from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal
from apedp_dyn.recursion import egraph, mdl, smt

C = sp.Symbol("C")
MAX_HOLES = 2
MIN_OPS, MAX_OPS = 1, 9
MAX_BLOCKS = 24
STARTS = (1.0, 0.3, 3.0)
BAD = 1e3
JOINS = ("add", "mul", "div")


def _core(part: sp.Expr) -> sp.Expr:
    """A part without its scale and offset."""
    return part.as_coeff_Mul()[1].as_coeff_Add()[1].as_coeff_Mul()[1]


def fragment_of(part: sp.Expr) -> tuple[str, int, int] | None:
    """A part of a law as a block: ``(text in h0, h1, C; holes; constants)``, or ``None`` if it is not one."""
    core = _core(part)
    symbols = sorted(core.free_symbols, key=lambda s: (str(core).find(str(s)), str(s)))
    if not 1 <= len(symbols) <= MAX_HOLES or core.is_Symbol:
        return None
    if not MIN_OPS <= int(sp.count_ops(core)) <= MAX_OPS:
        return None
    renamed = core.xreplace({s: sp.Symbol(f"h{i}") for i, s in enumerate(symbols)})
    floats = renamed.atoms(sp.Float)
    text = str(renamed.xreplace(dict.fromkeys(floats, C)))
    return text, len(symbols), text.count("C")


def fragments_of(law: str | sp.Expr) -> dict[str, tuple[int, int, list[str]]]:
    """Every block in a law: text -> (holes, constants, the smaller blocks inside it)."""
    expr = sp.sympify(law) if isinstance(law, str) else law
    out: dict[str, tuple[int, int, list[str]]] = {}
    for node in sp.preorder_traversal(expr):
        if not node.free_symbols:
            continue
        found = fragment_of(node)
        if found is None:
            continue
        inside = []
        for inner in sp.preorder_traversal(_core(node)):
            sub = fragment_of(inner) if inner.free_symbols and inner != _core(node) else None
            if sub is not None and sub[0] != found[0]:
                inside.append(sub[0])
        out.setdefault(found[0], (found[1], found[2], sorted(set(inside))))
    return out


@dataclass
class Library:
    """The chosen blocks: text -> holes, constants, how many laws use it, what naming it saves in bits."""

    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    laws: int = 0

    @classmethod
    def learn(cls, laws: Sequence[str], *, max_blocks: int = MAX_BLOCKS, variables: int = 3) -> Library:
        uses: dict[str, int] = {}
        shape: dict[str, tuple[int, int, list[str]]] = {}
        read = 0
        for law in laws:
            try:
                found = fragments_of(law)
            except (sp.SympifyError, TypeError, ValueError, RecursionError):
                continue
            read += 1
            for text, info in found.items():
                uses[text] = uses.get(text, 0) + 1
                shape.setdefault(text, info)
        candidates = []
        for text, count in uses.items():
            holes, constants, _inside = shape[text]
            written = mdl.law_bits(sp.sympify(text.replace("C", "1.5")), variables, max_blocks)
            named = mdl.block_use_bits(holes, constants, variables, max_blocks)
            worth = count * (written - named) - written
            if count >= 2 and worth > 0:
                candidates.append((worth, text))
        chosen: dict[str, dict[str, Any]] = {}
        for worth, text in sorted(candidates, key=lambda pair: (-pair[0], pair[1])):
            if len(chosen) == max_blocks:
                break
            if any(text in shape[other][2] and uses[other] >= uses[text] for other in chosen):
                continue                      # only ever met inside a block already chosen
            holes, constants, _inside = shape[text]
            chosen[text] = {"holes": holes, "constants": constants, "uses": uses[text], "worth_bits": round(worth, 2)}
        return cls(chosen, read)

    def as_dict(self) -> dict[str, Any]:
        return {"laws": self.laws, "blocks": dict(sorted(self.blocks.items()))}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Library:
        return cls({k: dict(v) for k, v in record.get("blocks", {}).items()}, int(record.get("laws", 0)))

    def compression(self, laws: Sequence[str], variables: int = 3) -> dict[str, float]:
        """Bits to write the laws without the library and with it (each law's largest blocks named; an estimate)."""
        plain = named = 0.0
        size = len(self.blocks)
        for law in laws:
            try:
                expr = sp.sympify(law)
                found = fragments_of(expr)
            except (sp.SympifyError, TypeError, ValueError, RecursionError):
                continue
            cost = mdl.law_bits(expr, variables)
            plain += cost
            saved = 0.0
            taken: set[str] = set()
            for text in sorted((t for t in found if t in self.blocks), key=lambda t: -len(t)):
                if any(text in found[other][2] for other in taken):
                    continue
                holes, constants, _inside = found[text]
                saved += mdl.law_bits(sp.sympify(text.replace("C", "1.5")), variables) - mdl.block_use_bits(
                    holes, constants, variables, size)
                taken.add(text)
            named += max(cost - saved, 0.0)
        return {"bits_without": round(plain, 1), "bits_with": round(named, 1)}


# --------------------------------------------------------------------------- candidates
@dataclass
class _Placed:
    """A block on some of the problem's inputs, with its fitted constants."""

    expr: sp.Expr                 #: in the inputs and its parameters
    params: list[sp.Symbol]
    theta: np.ndarray
    used: frozenset[int]
    mse: float = float("inf")


def _parametrised(text: str, variables: dict[str, sp.Symbol], prefix: str) -> tuple[sp.Expr, list[sp.Symbol]]:
    params: list[sp.Symbol] = []

    def fresh(expr: sp.Expr) -> sp.Expr:
        if expr == C:
            params.append(sp.Symbol(f"{prefix}{len(params)}"))
            return params[-1]
        if not expr.args:
            return expr
        return expr.func(*[fresh(arg) for arg in expr.args])

    return fresh(sp.sympify(text, locals={"C": C, **variables})), params


def _fit(expr: sp.Expr, params: Sequence[sp.Symbol], symbols: Sequence[sp.Symbol], X: np.ndarray, y: np.ndarray,
         start: np.ndarray, evaluations: int = 40) -> tuple[float, np.ndarray]:
    """Least squares over ``params``; the relative mean squared error and where it was reached."""
    fn = sp.lambdify([*symbols, *params], expr, "numpy")
    spread = float(np.std(y)) or 1.0

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*X.T, *theta), dtype=complex) * np.ones(len(X))
        ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
        out: np.ndarray = np.where(ok, (value.real - y) / spread, BAD)
        return out

    try:
        fit = least_squares(resid, start, max_nfev=evaluations * max(len(start), 1))
    except (ValueError, TypeError, FloatingPointError):
        return float("inf"), start
    final = resid(fit.x)
    if np.any(final >= BAD) or not np.all(np.isfinite(fit.x)):
        return float("inf"), start
    return float(np.mean(final ** 2)), fit.x


def _join(kind: str, a: sp.Expr, b: sp.Expr) -> sp.Expr:
    if kind == "add":
        return a + sp.Symbol("_w") * b
    if kind == "mul":
        return a * b
    return a / (sp.Symbol("_w") + b)


def _written(expr: sp.Expr, params: Sequence[sp.Symbol], theta: np.ndarray) -> sp.Expr:
    return expr.xreplace({p: sp.Float(float(f"{v:.8g}")) for p, v in zip(params, theta, strict=True)})


def candidates(library: Library, names: Sequence[str], X: np.ndarray, y: np.ndarray, *, seconds: float = 20.0,
               top: int = 6, limit: int = 3, prove_poles: bool = True) -> tuple[list[sp.Expr], dict[str, Any]]:
    """Laws in every input built from the library's blocks, the shortest total description first."""
    started = time.monotonic()
    deadline = started + float(seconds)
    symbols = [sp.Symbol(n) for n in names]
    n = len(names)
    scale, offset = sp.Symbol("_a"), sp.Symbol("_b")
    report: dict[str, Any] = {"blocks": len(library.blocks), "placed": 0, "joined": 0, "out_of_time": False}

    placed: list[_Placed] = [_Placed(symbols[i], [], np.zeros(0), frozenset([i])) for i in range(n)]
    for text, info in library.blocks.items():
        for chosen in itertools.permutations(range(n), int(info["holes"])):
            try:
                expr, params = _parametrised(text, {f"h{k}": symbols[j] for k, j in enumerate(chosen)}, f"_p{len(placed)}_")
            except (sp.SympifyError, TypeError, SyntaxError, ValueError):
                continue
            placed.append(_Placed(expr, params, np.ones(len(params)), frozenset(chosen)))
    for item in placed:                                      # each block alone: scale * block + offset
        if time.monotonic() > deadline:
            report["out_of_time"] = True
            break
        best = (float("inf"), np.ones(len(item.params) + 2))
        for first in STARTS[: 1 if not item.params else len(STARTS)]:
            start = np.concatenate([np.full(len(item.params), first), [1.0, 0.0]])
            got = _fit(scale * item.expr + offset, [*item.params, scale, offset], symbols, X, y, start)
            if got[0] < best[0]:
                best = got
        item.mse, item.theta = best[0], best[1][: len(item.params)]
        report["placed"] += 1
    ranked = sorted((p for p in placed if np.isfinite(p.mse)), key=lambda p: p.mse)[: max(int(top), 1)]

    found: list[tuple[float, sp.Expr]] = []

    def keep(expr: sp.Expr, params: Sequence[sp.Symbol], start: np.ndarray, used: frozenset[int]) -> tuple[float, np.ndarray]:
        mse, theta = _fit(expr, params, symbols, X, y, start)
        if np.isfinite(mse) and len(used) == n:
            law = _written(expr, params, theta)
            if law.free_symbols == set(symbols):
                fn = sp.lambdify(symbols, law, "numpy")
                with np.errstate(all="ignore"):
                    values = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
                if np.all(np.isfinite(values)):
                    found.append((mdl.total_bits(law, values - y, y, n, len(library.blocks)), law))
        return mse, theta

    for item in ranked:
        if len(item.used) == n:
            keep(scale * item.expr + offset, [*item.params, scale, offset], np.concatenate([item.theta, [1.0, 0.0]]), item.used)
    weight = sp.Symbol("_w")
    pairs: list[tuple[float, sp.Expr, list[sp.Symbol], np.ndarray, frozenset[int]]] = []
    for a, b in itertools.permutations(ranked, 2):
        for kind in JOINS:
            if kind != "div" and str(a.expr) > str(b.expr):
                continue                                     # a sum and a product once each
            if time.monotonic() > deadline:
                report["out_of_time"] = True
                break
            joined = _join(kind, a.expr, b.expr)
            params = [*a.params, *b.params, *([weight] if kind != "mul" else []), scale, offset]
            start = np.concatenate([a.theta, b.theta, *([[1.0]] if kind != "mul" else []), [1.0, 0.0]])
            mse, theta = keep(scale * joined + offset, params, start, a.used | b.used)
            report["joined"] += 1
            if np.isfinite(mse):
                pairs.append((mse, joined, params[:-2], theta[:-2], a.used | b.used))
    if n >= 3:
        second = sp.Symbol("_v")
        for _mse, joined, params, theta, used in sorted(pairs, key=lambda p: p[0])[: max(int(top), 1)]:
            for c in ranked:
                if c.used <= used:
                    continue
                for kind in ("add", "mul"):
                    if time.monotonic() > deadline:
                        report["out_of_time"] = True
                        break
                    triple = joined + second * c.expr if kind == "add" else joined * c.expr
                    more = [*params, *c.params, *([second] if kind == "add" else []), scale, offset]
                    start = np.concatenate([theta, c.theta, *([[1.0]] if kind == "add" else []), [1.0, 0.0]])
                    keep(scale * triple + offset, more, start, used | c.used)
                    report["joined"] += 1

    box = {name: (float(np.min(X[:, i])), float(np.max(X[:, i]))) for i, name in enumerate(names)}
    out: list[sp.Expr] = []
    report["dropped_pole"] = report["dropped_equal"] = 0
    shapes: set[str] = set()
    for _bits, law in sorted(found, key=lambda pair: pair[0]):
        if len(out) == int(limit):
            break
        shape = str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), C)))
        if shape in shapes:                                  # the same law with other constants: the shorter came first
            report["dropped_equal"] += 1
            continue
        shapes.add(shape)
        if prove_poles and smt.pole_in_box(law, box).get("pole") is True:
            report["dropped_pole"] += 1
            continue
        if any(egraph.equal(law, other, iterations=3, node_limit=600)["equal"] for other in out):
            report["dropped_equal"] += 1
            continue
        out.append(law)
    report["seconds"] = round(time.monotonic() - started, 2)
    report["candidates"] = len(found)
    return out, report


class LibraryProposer(ProposerBase):
    """Candidates built from a learned library's blocks, as proposals."""

    def __init__(self, library: Library, *, max_proposals: int = 3, seconds: float = 20.0, top: int = 6,
                 target_col: str = "target") -> None:
        self._library, self._max = library, int(max_proposals)
        self._seconds, self._top = float(seconds), int(top)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "library_blocks"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        if not self._library.blocks or self._seconds <= 0:
            self.last_run = {"blocks": len(self._library.blocks), "proposed": 0}
            return []
        laws, report = candidates(self._library, features, frame[features].to_numpy(dtype=float),
                                  frame[target].to_numpy(dtype=float), seconds=self._seconds, top=self._top, limit=self._max)
        self.last_run = {**report, "proposed": len(laws)}
        return [formula_proposal(self.proposer_id, str(law), features, {"rank": rank}) for rank, law in enumerate(laws)]
