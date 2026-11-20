"""A language model that configures a search instead of writing the equations.

Status: experimental

PREREG-LLM-PROPOSER-2026-11 put a language model in the search's place: it
wrote the candidate equations itself, from numbers. It recovered about half of
what PySR recovers in the same place. The published systems that do well with
language models use them the other way round: the model shapes the search
space and a conventional search does the enumerating and fitting. This module
is that design.

``ControlledSearchProposer`` asks the model one question per problem: given the
ranges of the inputs, a few rows and the first pass's best expression,

* which unary operators the search should allow (a subset of the search's own),
* up to four derived inputs worth adding (``x0*x1``, ``x0/x2``, ``log(x1)``),
* up to three complete guesses with free constants.

Everything the model returns is checked before it is used. Operators outside
the search's set are dropped. Derived inputs and guesses are parsed by the
whitelist parser of ``llm_proposer`` (arithmetic, six functions, ``x``-variables
and ``c``-constants only; nothing is evaluated as code), must be finite on the
data, and derived inputs may not contain free constants. A reply that cannot be
read, or a model that cannot be reached, leaves the search with its default
configuration, and the fact is recorded: the model can widen or narrow a
search, it cannot stop one.

The search then runs on the inputs plus the derived inputs, seeded with the
guesses, and its formulas are rewritten in the original variables. The model
decides nothing about acceptance; that is the path's.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers import llm_proposer
from apedp_dyn.proposers.composed import formula_proposal

DEFAULT_UNARY = ("sqrt", "exp", "log", "sin", "cos", "square")
MAX_FEATURES = 4
MAX_GUESSES = 3
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def read_configuration(reply: str, n_vars: int, allowed_unary: Sequence[str] = DEFAULT_UNARY) -> dict[str, Any]:
    """The model's reply as a checked configuration; whatever cannot be used is listed under ``rejected``."""
    out: dict[str, Any] = {"unary": list(allowed_unary), "features": [], "guesses": [], "rejected": [], "readable": False}
    match = _OBJECT.search(reply or "")
    if not match:
        out["rejected"].append("no JSON object in the reply")
        return out
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        out["rejected"].append(f"JSON could not be read: {exc.msg}")
        return out
    if not isinstance(raw, dict):
        out["rejected"].append("the reply is not a JSON object")
        return out
    out["readable"] = True
    asked = [str(u) for u in raw.get("unary", []) if isinstance(u, str)] if isinstance(raw.get("unary"), list) else None
    if asked is not None:
        chosen = [u for u in allowed_unary if u in asked]
        out["rejected"] += [f"operator {u!r} is not one the search has" for u in asked if u not in allowed_unary]
        out["unary"] = chosen  # an empty list is a valid choice: arithmetic only
    for key, limit, constants in (("features", MAX_FEATURES, 0), ("guesses", MAX_GUESSES, 6)):
        items = raw.get(key, [])
        for text in items[: 3 * limit] if isinstance(items, list) else []:
            if len(out[key]) >= limit:
                break
            try:
                expr = llm_proposer.parse_skeleton(str(text), n_vars, max_constants=constants)
            except (llm_proposer.UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError) as exc:
                out["rejected"].append(f"{key[:-1]} {str(text)[:40]!r}: {type(exc).__name__}")
                continue
            if not any(re.fullmatch(r"x\d+", str(s)) for s in expr.free_symbols):
                out["rejected"].append(f"{key[:-1]} {str(text)[:40]!r}: uses no input")
                continue
            if key == "features" and (expr.is_Symbol or str(expr) in {str(e) for e in out["features"]}):
                continue  # an input itself, or a repeat, adds nothing
            out[key].append(expr)
    return out


class ControlledSearchProposer(ProposerBase):
    """A search whose operators, extra inputs and seeds are chosen by a language model, within checked limits."""

    def __init__(self, client: Callable[[str], str] | None, search_factory: Callable[..., ProposerBase], *,
                 allowed_unary: Sequence[str] = DEFAULT_UNARY, rows_shown: int = 12, description: str = "",
                 target_col: str = "target") -> None:
        self._client = client
        self._factory = search_factory          # called as search_factory(unary=...) -> ProposerBase
        self._allowed = tuple(allowed_unary)
        self._rows_shown = rows_shown
        self._description = description
        self._target_col = target_col
        #: what the last call asked, got and used
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "controlled_search"

    def _question(self, X: np.ndarray, y: np.ndarray, first_pass: str) -> str:
        n_vars = X.shape[1]
        sample = np.random.default_rng(0).choice(len(y), size=min(self._rows_shown, len(y)), replace=False)
        rows = "\n".join(", ".join(f"{v:.4g}" for v in (*X[i], y[i])) for i in sample)
        ranges = "; ".join(f"x{i} in [{X[:, i].min():.3g}, {X[:, i].max():.3g}]" for i in range(n_vars))
        context = f"Context: {self._description}\n" if self._description else ""
        return (f"{context}A symbolic regression search will look for y as a function of {', '.join(f'x{i}' for i in range(n_vars))}.\n"
                f"Ranges: {ranges}; y in [{y.min():.3g}, {y.max():.3g}].\n"
                f"Rows ({', '.join(f'x{i}' for i in range(n_vars))}, y):\n{rows}\n"
                f"Best simple expression found so far: {first_pass or 'none'}\n"
                "Configure the search. Reply with one JSON object and nothing else:\n"
                f'{{"unary": [operators to allow, chosen from {list(self._allowed)}], '
                f'"features": [up to {MAX_FEATURES} expressions in the variables to add as extra inputs, such as "x0*x1" or "x0/x2"], '
                f'"guesses": [up to {MAX_GUESSES} complete expressions for y; free constants are written c0, c1, ...]}}\n'
                "Functions allowed in expressions: sqrt, exp, log, sin, cos, tanh. Operators: + - * / **.")

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        n_vars = X.shape[1]
        to_x = {sp.Symbol(f): sp.Symbol(f"x{i}") for i, f in enumerate(features)}
        to_feature = {sp.Symbol(f"x{i}"): sp.Symbol(f) for i, f in enumerate(features)}
        first_pass = ""
        if guesses:
            try:
                first_pass = str(sp.sympify(guesses[0], locals={f: sp.Symbol(f) for f in features}).xreplace(to_x))
            except (sp.SympifyError, TypeError, SyntaxError):
                first_pass = ""
        config = read_configuration("", n_vars, self._allowed)
        asked = False
        failure = ""
        if self._client is not None:
            asked = True
            try:
                config = read_configuration(self._client(self._question(X, y, first_pass)), n_vars, self._allowed)
            except Exception as exc:  # noqa: BLE001 - a model that cannot be reached leaves the default search, recorded below
                failure = f"{type(exc).__name__}: {exc}"[:200]
        xs = [sp.Symbol(f"x{i}") for i in range(n_vars)]
        derived: list[tuple[str, sp.Expr, np.ndarray]] = []
        for expr in config["features"]:
            with np.errstate(all="ignore"):
                column = np.asarray(sp.lambdify(xs, expr, "numpy")(*X.T), dtype=float) * np.ones(len(y))
            if np.all(np.isfinite(column)) and float(np.std(column)) > 0.0:
                derived.append((f"f{len(derived)}", expr, column))
            else:
                config["rejected"].append(f"feature {str(expr)[:40]!r}: not finite, or constant, on the data")
        seeds: list[str] = []
        fitted: list[tuple[str, float]] = []
        for expr in config["guesses"]:
            try:
                constants, error = llm_proposer.fit_constants(expr, X, y)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                config["rejected"].append(f"guess {str(expr)[:40]!r}: could not be fitted")
                continue
            if not np.isfinite(error):
                config["rejected"].append(f"guess {str(expr)[:40]!r}: not finite on the data")
                continue
            concrete = expr.xreplace({sp.Symbol(k): sp.Float(v) for k, v in constants.items()}).xreplace(to_feature)
            fitted.append((str(concrete), float(error)))
            seeds.append(str(concrete))
        search = self._factory(unary=tuple(config["unary"]))
        augmented = frame.copy()
        for name, _, column in derived:
            augmented[name] = column
        back = {sp.Symbol(name): expr.xreplace(to_feature) for name, expr, _ in derived}
        local = {n: sp.Symbol(n) for n in [*features, *(name for name, _, _ in derived)]}
        proposals: list[CandidateProposal] = []
        seen: set[str] = set()
        for formula, error in sorted(fitted, key=lambda item: item[1]):
            if formula not in seen:
                seen.add(formula)
                proposals.append(formula_proposal(self.proposer_id, formula, features, {"found_by": "model_guess", "relative_error": error}))
        searched = search.fit_and_propose(augmented, target_col=target, guesses=[*(guesses or ()), *seeds][:4], **kwargs)
        for proposal in searched:
            try:
                formula = str(sp.sympify(proposal.formula, locals=local).xreplace(back))
            except (sp.SympifyError, TypeError, SyntaxError):
                continue
            if formula not in seen:
                seen.add(formula)
                proposals.append(formula_proposal(self.proposer_id, formula, features, {"found_by": proposal.proposer_id}))
        self.last_run = {"asked": asked, "model_failure": failure, "readable": config["readable"], "unary": list(config["unary"]),
                         "features": [str(e) for _, e, _ in derived], "guesses": [f for f, _ in fitted],
                         "rejected": list(config["rejected"]), "search_proposals": len(searched)}
        return proposals
