"""The memory's proposals with the work counted in fits, not in seconds.

``structure_memory.grow`` and ``remembered_whole`` stop at a clock. On a slower or busier machine
they try fewer laws, so the same table can be read differently from one run to the next (measured
on the development tables of the small-table reading: 4 of 120 gave another outcome on a repeat).
Here both stop at a number of fits instead. Every fit starts from fixed or seeded points, so the
same table and the same memory give the same proposals on any machine.

The steps are the same as in ``structure_memory`` and use its own fitting code; only the stopping
rule differs. **Not in force anywhere**: the readings and routes keep the timed growth their frozen
plans name. Replacing it is a change of behaviour, to be compared first and decided by the owner.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import structure_memory as timed
from apedp_dyn.proposers.composed import formula_proposal
from apedp_dyn.proposers.structure_memory import Grown, Memory

# Fits allowed to the growth and to the remembered whole laws. Chosen on the development tables of the
# small-table reading as about the work the timed growth did there in its 5 seconds (see
# benchmarks/small_tables/dev_counted_memory.json); a number of fits, so it means the same on any machine.
GROW_FITS = 1500
WHOLE_FITS = 400


def grow_counted(memory: Memory, X: np.ndarray, y: np.ndarray, *, beam: int = 6, max_motifs: int = 4,
                 max_fits: int = GROW_FITS, keep: int = 12, refit: int = 10
                 ) -> tuple[list[Grown], dict[str, Any], dict[str, Any]]:
    """``structure_memory.grow`` with a number of fits where it has a clock; the same three results.

    The count is checked where the clock was: before each law of the beam is extended. A law being
    extended is finished, so the work can pass ``max_fits`` by one law's extensions, the same on
    every machine.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    motifs = timed._compiled(sorted(memory.motifs, key=lambda shape: (-memory.motifs[shape], shape)))
    report: dict[str, Any] = {"motifs": len(motifs) - 1, "fits": 0, "steps": 0, "stopped": "grown to the limit",
                              "max_fits": int(max_fits)}
    found: dict[tuple[tuple[str, str, int], ...], Grown] = {}

    def extended(steps: tuple[tuple[str, str, int], ...], before: Sequence[float]) -> Grown | None:
        motif = motifs[steps[-1][1]]
        own = [] if steps[-1][0] == "multiply" else [1.0]
        best: Grown | None = None
        firsts = (1.0,) if not motif.params else motif.scan(X[:, steps[-1][2]]) if motif.periodic else timed.STARTS
        for first in firsts:
            start = np.array([*before, *own, *([first] + [1.0] * (motif.params - 1) if motif.params else [])], dtype=float)
            report["fits"] += 1
            fitted = timed._fit(steps, start, X, y, motifs, free=len(own) + motif.params)
            if fitted is not None and (best is None or fitted.bic < best.bic):
                best = fitted
        return best

    def polished(rough: Grown) -> Grown:
        report["fits"] += 1
        fitted = timed._fit(rough.steps, np.array(rough.theta), X, y, motifs)
        return fitted if fitted is not None and fitted.bic < rough.bic else rough

    held: list[Grown] = []
    for shape, variable in itertools.product(motifs, range(X.shape[1])):
        if shape == timed.CONSTANT:
            continue
        first = extended((("start", shape, variable),), [])
        if first is not None:
            found[first.steps] = first
            held.append(first)
    report["steps"] = 1
    for _ in range(max_motifs - 1):
        held = sorted(held, key=lambda g: g.bic)[:beam]
        if not held or held[0].relative_error < timed.EXACT:
            report["stopped"] = "an exact fit" if held else "nothing fitted"
            break
        grown: list[Grown] = []
        for parent in held:
            if report["fits"] >= max_fits:
                report["stopped"] = "out of fits"
                break
            rough: list[Grown] = []
            for join, shape, variable in itertools.product(timed.JOINS, motifs, range(X.shape[1])):
                if shape == timed.CONSTANT and (join != "add" or variable != 0
                                                or any(step[1] == timed.CONSTANT for step in parent.steps)):
                    continue                    # one intercept at most, and a constant is only ever added
                steps = (*parent.steps, (join, shape, variable))
                if steps in found:
                    continue
                candidate = extended(steps, parent.theta)
                if candidate is not None:
                    rough.append(candidate)
            for candidate in sorted(rough, key=lambda g: g.bic)[:refit]:
                fitted = polished(candidate)
                found[fitted.steps] = fitted
                grown.append(fitted)
        report["steps"] += 1
        held = grown
        if report["stopped"] == "out of fits":
            break
    return sorted(found.values(), key=lambda g: g.bic)[:keep], motifs, report


def remembered_whole_counted(memory: Memory, names: Sequence[str], X: np.ndarray, y: np.ndarray, limit: int = 4,
                             max_fits: int = WHOLE_FITS, tries: int = 8) -> tuple[list[tuple[float, sp.Expr]], dict[str, Any]]:
    """``structure_memory.remembered_whole`` with a number of fits where it has a clock, and what was done."""
    out: list[tuple[float, sp.Expr]] = []
    spread = float(np.std(y)) or 1.0
    rng = np.random.default_rng(0)
    report: dict[str, Any] = {"fits": 0, "stopped": "every remembered law tried", "max_fits": int(max_fits)}
    for shape, variables in memory.skeletons.items():
        if variables != len(names):                     # a law of the problem uses every one of its variables
            continue
        for chosen in itertools.permutations(range(len(names))):
            if report["fits"] >= max_fits:
                report["stopped"] = "out of fits"
                return sorted(out, key=lambda pair: pair[0])[:limit], report
            fitted = _whole(shape, [names[j] for j in chosen], names, X, y, spread, rng, tries, report)
            if fitted is not None:
                out.append(fitted)
    return sorted(out, key=lambda pair: pair[0])[:limit], report


def _whole(shape: str, assigned: Sequence[str], names: Sequence[str], X: np.ndarray, y: np.ndarray, spread: float,
           rng: np.random.Generator, tries: int, report: dict[str, Any]) -> tuple[float, sp.Expr] | None:
    """One remembered shape with its variables assigned, constants fitted from ``tries`` starts; its BIC and the law."""
    try:
        expr, params = timed.with_parameters(shape, {f"u{i}": sp.Symbol(name) for i, name in enumerate(assigned)}, "_s")
        fn = sp.lambdify([*(sp.Symbol(n) for n in names), *params], expr, "numpy")
    except (sp.SympifyError, TypeError, SyntaxError, ValueError):
        return None
    if not params or len(params) > len(y):
        return None

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*X.T, *theta), dtype=complex) * np.ones(len(X))
        ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
        res: np.ndarray = np.where(ok, (value.real - y) / spread, timed.BAD)
        return res

    best: tuple[float, np.ndarray] | None = None
    for attempt in range(tries):
        start = (np.ones(len(params)) if attempt == 0
                 else rng.choice([-1.0, 1.0], len(params)) * rng.choice(timed.STARTS, len(params)))
        report["fits"] += 1
        try:
            fit = least_squares(resid, start, max_nfev=60 * len(params))
        except (ValueError, TypeError, FloatingPointError):
            continue
        final = resid(fit.x)
        if np.any(final >= timed.BAD) or not np.all(np.isfinite(fit.x)):
            continue
        mse = float(np.mean(final ** 2))
        if best is None or mse < best[0]:
            best = (mse, fit.x)
        if mse < timed.EXACT ** 2:
            break
    if best is None:
        return None
    bic = len(y) * float(np.log(max(best[0], 1e-30))) + len(params) * float(np.log(len(y)))
    law = expr.xreplace({p: sp.Float(float(f"{v:.8g}")) for p, v in zip(params, best[1], strict=True)})
    return bic, law


class CountedMemoryProposer(ProposerBase):
    """``StructureMemoryProposer`` whose work is a number of fits: the same proposals on any machine."""

    def __init__(self, memory: Memory, max_proposals: int = 3, beam: int = 6, max_motifs: int = 4,
                 grow_fits: int = GROW_FITS, whole_fits: int = WHOLE_FITS, target_col: str = "target") -> None:
        self._memory, self._max = memory, int(max_proposals)
        self._beam, self._max_motifs = int(beam), int(max_motifs)
        self._grow_fits, self._whole_fits = int(grow_fits), int(whole_fits)
        self._target_col = target_col
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "structure_memory_counted"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        best, motifs, report = grow_counted(self._memory, X, y, beam=self._beam, max_motifs=self._max_motifs,
                                            max_fits=self._grow_fits)
        whole, whole_report = remembered_whole_counted(self._memory, features, X, y, max_fits=self._whole_fits)
        texts: list[str] = []
        seen: set[str] = set()
        candidates = [*[(g.bic, timed.written(g, features, motifs)) for g in best], *whole]
        for _bic, law in sorted(candidates, key=lambda pair: pair[0]):
            if law.free_symbols != {sp.Symbol(f) for f in features}:
                continue                        # a law of the problem uses every one of its variables
            key = str(law.xreplace(dict.fromkeys(law.atoms(sp.Float), timed.C)))
            if key not in seen:
                seen.add(key)
                texts.append(str(law))
            if len(texts) == self._max:
                break
        self.last_run = {**report, "whole": whole_report, "laws_remembered": self._memory.laws, "proposed": len(texts)}
        return [formula_proposal(self.proposer_id, text, features, {"rank": rank}) for rank, text in enumerate(texts)]
