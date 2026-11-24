"""An LLM as an equation proposer, with a guard against reciting known formulas.

Status: experimental

The language model is a callable ``client(prompt) -> text`` supplied by the
caller; this module contains no network code and no model. It asks for equation
skeletons with free constants ``c0, c1, ...``, fits the constants, feeds the
scores of earlier attempts back, and emits ``CandidateProposal`` objects. It
decides nothing: the referee adjudicates what it proposes.

Three guards, because a model that has read the literature can pass a benchmark
by remembering it (the concern LLM-SRBench was built around):

1. **Anonymous variables.** The model sees ``x0, x1, ...`` and, optionally, a
   description the caller wrote; never the dataset's own column names.
2. **Recitation probe.** Before any data is shown, the model is asked what the
   law is from the description alone. Every proposal whose structure matches one
   of those answers is marked ``recited_before_data``: the model knew it without
   looking, so it is recall, not discovery, however well it fits.
3. **Only whitelisted text is parsed.** A reply is evaluated only if it consists
   of the allowed variables, constants, functions, numbers and operators; a
   model's output is never passed to an evaluator otherwise.

The scores fed back come from the rows this proposer was given. Giving it only
development rows is the caller's job, as for every proposer (AGENTS.md 1.2).
Tested with a scripted stand-in for the model; no real model has been run
through it in this repository.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype

FUNCTIONS = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh}
_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z_]\w*)|(\*\*|[-+*/(),]))")
# A list marker ("1. ", "2) ", "- ") and "y =" may precede the expression; a leading number such as 0.5 may not be eaten.
_LINE = re.compile(r"^\s*(?:(?:\d+[.)]|[-*])\s+)?(?:y\s*=\s*)?(.+?)\s*$")


class UnsafeExpressionError(ValueError):
    """The text contains something other than the allowed names, numbers and operators."""


def parse_skeleton(text: str, n_vars: int, max_constants: int = 6) -> sp.Expr:
    """``text`` as an expression over x0..x{n-1} and c0..c{k-1}; refuses anything else."""
    allowed = {f"x{i}" for i in range(n_vars)} | {f"c{i}" for i in range(max_constants)} | set(FUNCTIONS)
    text = text.replace("^", "**")
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match or match.end() == position:
            if text[position:].strip() == "":
                break
            raise UnsafeExpressionError(f"unexpected text at {position}: {text[position:position + 20]!r}")
        if match.group(2) and match.group(2) not in allowed:
            raise UnsafeExpressionError(f"name {match.group(2)!r} is not allowed")
        position = match.end()
    symbols = {name: sp.Symbol(name) for name in allowed if name not in FUNCTIONS}
    expr = sp.sympify(text, locals={**symbols, **FUNCTIONS})
    if not isinstance(expr, sp.Expr):
        raise UnsafeExpressionError("not an expression")
    return expr


def structure_key(expr: sp.Expr) -> str:
    """The expression with every constant and number replaced by one placeholder, for comparing structure."""
    c = sp.Symbol("C")
    constants = {s: c for s in expr.free_symbols if re.fullmatch(r"c\d+", str(s))}
    generic = expr.xreplace(constants)
    generic = generic.xreplace(dict.fromkeys(generic.atoms(sp.Float), c))
    return str(sp.simplify(generic))


def fit_constants(expr: sp.Expr, X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[dict[str, float], float]:
    """Least-squares constants (a few seeded starts) and the relative RMS error they leave."""
    consts = sorted((s for s in expr.free_symbols if re.fullmatch(r"c\d+", str(s))), key=str)
    xs = [sp.Symbol(f"x{i}") for i in range(X.shape[1])]
    fn = sp.lambdify([*consts, *xs], expr, "numpy")
    rms = float(np.sqrt(np.mean(y ** 2))) or 1.0

    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(fn(*theta, *X.T), dtype=float) * np.ones(len(y))
        out: np.ndarray = np.where(np.isfinite(value), value - y, 1e6 * rms) / rms
        return out

    if not consts:
        return {}, float(np.sqrt(np.mean(resid(np.array([])) ** 2)))
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray] | None = None
    for start in [np.ones(len(consts)), *rng.normal(0.0, 2.0, (4, len(consts)))]:
        fit = least_squares(resid, start, max_nfev=400)
        err = float(np.sqrt(np.mean(fit.fun ** 2)))
        if best is None or err < best[0]:
            best = (err, fit.x)
    assert best is not None
    return {str(s): float(v) for s, v in zip(consts, best[1], strict=True)}, best[0]


_SUBSCRIPT = re.compile(r"\b([xc])_\{?(\d+)\}?")
_LABEL = re.compile(r"^\s*(?:[A-Za-z]\w*\s*(?:\([^()=]*\))?\s*=|[A-Za-z ]{0,24}:)\s*")
_WORD = re.compile(r"[A-Za-z]{3,}")


def _replies(text: str) -> list[str]:
    """Candidate expression lines from a model's reply.

    Models decorate their answers: code fences, backticks, dollar signs, list
    markers, ``y =`` or ``f(x0, x1) =`` in front, ``x_0`` for ``x0``, a full stop
    at the end. Those are removed. A sentence (three or more words that are not
    function names) is dropped as prose. Anything else that mentions a variable
    or calls something goes on to :func:`parse_skeleton`, which decides whether
    it is an allowed expression and counts it as rejected if it is not.
    """
    out = []
    for line in text.splitlines():
        match = _LINE.match(line.replace("`", "").replace("$", "").replace("\\(", "").replace("\\)", ""))
        if not match or not match.group(1):
            continue
        cleaned = match.group(1).replace("\u00d7", "*").replace("\u00b7", "*").replace("\u2212", "-")
        cleaned = _SUBSCRIPT.sub(r"\1\2", _LABEL.sub("", cleaned)).strip().rstrip(".;,").strip()
        prose = sum(word not in FUNCTIONS for word in _WORD.findall(cleaned)) >= 3
        if cleaned and not prose and (re.search(r"x\d", cleaned) or "(" in cleaned):
            out.append(cleaned)
    return out


class LlmEquationProposer(ProposerBase):
    """Iterative skeleton proposals from a caller-supplied language model."""

    def __init__(self, client: Callable[[str], str], target_col: str = "target", rounds: int = 3,
                 per_round: int = 5, description: str = "", max_proposals: int = 5,
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> None:
        self._client = client
        self._target_col = target_col
        self._rounds, self._per_round = int(rounds), int(per_round)
        self._description = description
        self._max_proposals = int(max_proposals)
        self._strategy_archetype = strategy_archetype
        #: what the last call saw, kept even when nothing could be proposed
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "llm_equation_proposer"

    def _rules(self, n_vars: int) -> str:
        return (f"Variables: {', '.join(f'x{i}' for i in range(n_vars))}. Free constants: c0, c1, ... "
                f"Functions: {', '.join(sorted(FUNCTIONS))}. Operators: + - * / **. "
                f"Reply with up to {self._per_round} expressions for y, one per line, nothing else.")

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        n_vars = X.shape[1]
        context = f"Context: {self._description}\n" if self._description else ""

        recited: set[str] = set()
        probe = self._client(f"{context}No data is given. If you already know the law relating y to the variables, "
                             f"state it.\n{self._rules(n_vars)}")
        for reply in _replies(probe):
            try:
                recited.add(structure_key(parse_skeleton(reply, n_vars)))
            except (UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError):
                continue

        sample = np.random.default_rng(0).choice(len(y), size=min(12, len(y)), replace=False)
        rows = "\n".join(", ".join(f"{v:.4g}" for v in (*X[i], y[i])) for i in sample)
        tried: dict[str, dict[str, Any]] = {}
        rejected = 0
        for _ in range(self._rounds):
            history = "\n".join(f"{t['skeleton']}  -> relative error {t['error']:.3g}"
                                for t in sorted(tried.values(), key=lambda t: t["error"])[:8])
            reply = self._client(f"{context}Rows ({', '.join(f'x{i}' for i in range(n_vars))}, y):\n{rows}\n"
                                 f"Tried so far:\n{history or 'nothing'}\n{self._rules(n_vars)}")
            for text in _replies(reply)[: self._per_round]:
                try:
                    expr = parse_skeleton(text, n_vars)
                except (UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError):
                    rejected += 1
                    continue
                key = structure_key(expr)
                if key in tried:
                    continue
                constants, error = fit_constants(expr, X, y)
                tried[key] = {"skeleton": str(expr), "expr": expr, "constants": constants, "error": error, "key": key}

        self.last_run = {"skeletons_tried": len(tried), "replies_rejected": rejected, "recited_before_data": len(recited),
                         "last_reply": reply[-300:] if self._rounds > 0 else "",
                         "last_answer": dict(getattr(self._client, "last", {}) or {})}
        proposals = []
        for item in sorted(tried.values(), key=lambda t: t["error"])[: self._max_proposals]:
            fitted = item["expr"].xreplace({sp.Symbol(k): sp.Float(v) for k, v in item["constants"].items()})
            fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(n_vars)], fitted, "numpy")

            def compute_score(df: pd.DataFrame, fn: Any = fn, features: list[str] = features) -> np.ndarray:
                values = df[features].to_numpy(dtype=float)
                out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
                return out

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id, candidate_id=f"llm:{item['key']}", formula=str(fitted),
                strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
                compute_score=compute_score, parameters={"dev_relative_rmse": item["error"], **item["constants"]},
                metadata={"skeleton": item["skeleton"], "recited_before_data": item["key"] in recited,
                          "selection_pool_size": max(1, len(tried)), "replies_rejected_as_unsafe_or_unparseable": rejected,
                          "variables_anonymised": True},
            ))
        return tuple(proposals)
