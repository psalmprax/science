"""A candidate generator written by a model, as a task for the loop and as a proposer for the path.

Status: experimental

The piece of the search that the loop may rewrite first is the cheapest one to
judge: a function ``propose(X, y, names)`` that looks at the development rows
and returns a few candidate laws as text. The polynomial library that the
refined path already adds is such a function, written by hand; the misses of
the recorded runs are largely forms no such library covers (a ratio with an
offset, a saturating term, a decay with a rate inside).

* ``read_candidates`` is the only door from a candidate to the rest of the
  system: each returned string goes through the whitelist parser of
  ``llm_proposer``, and what it refuses is dropped and counted.
* ``CandidateGeneratorTask`` is the judge. On problems whose law is known it
  asks: is the law among the few candidates returned, by the criterion the
  caller supplies; and is any of them accurate on clean rows. It needs no
  search engine and takes a fraction of a second a problem, so that many
  versions can be scored.
* ``EvolvedProposer`` runs one checked version in the jail for the path, like
  any other proposer: development rows in, ``CandidateProposal`` objects out,
  nothing decided.

**What the judge's score is not.** It says whether the law is *proposed*. Which
candidate the path then reports, and how often a wrong one, is the path's
confirmation at work, and has to be measured through the path on problems no
version was scored on.
"""
from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.conf import discovery_checks
from apedp_dyn.evolution import jail, restricted
from apedp_dyn.evolution.operator import Contract, Evaluation
from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.proposers import llm_proposer

FUNCTION = "propose"
MAX_CANDIDATES = 4          # candidates of one call that are read; the path confirms few, and each takes a search's place
MAX_TEXT = 400              # characters of one candidate that are read
MAX_NUMBERS = 12            # constants one candidate may carry
MAX_EXPONENT = 12.0         # the largest constant power
ACCURATE_R2 = discovery_checks.DISCOVERY_ACCURACY_R2   # the benchmarks' own threshold for an accurate law
WORTH_CHECKING_R2 = 0.9     # a candidate below this on clean rows is not checked for the law's form

CONTRACT = Contract(
    function=FUNCTION, parameters=3, signature="propose(X, y, names)",
    description=(
        "X is a numpy array of shape (rows, variables), y an array of length rows (the target, with a little "
        "measurement noise), names a list of the variables' names, one per column of X. Return a list of at most "
        f"{MAX_CANDIDATES} strings. Each string is one candidate law for y: an expression in the names with its "
        "constants already fitted and written as numbers, using + - * / ** and the functions "
        + ", ".join(sorted(llm_proposer.FUNCTIONS)) + ". Put the candidates you trust most first. "
        "A version is scored on many problems whose law is known: first by the number of problems where the true "
        "law (the same form, whatever the fitted numbers) is among the candidates, then by the number of problems "
        "with an accurate candidate, then by returning fewer candidates. A call that takes longer than a few "
        "seconds counts as returning nothing."),
    notes=("Fit the constants by least squares (np.linalg.lstsq) for forms that are linear in them, and by a "
           "small grid over a constant inside a function for forms that are not, with least squares for the rest. "
           "Prefer the simplest form that fits: many forms fit noisy rows almost equally well."),
)


def _has_variable(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id.startswith("x") for n in ast.walk(node))


def _number(node: ast.AST) -> float | None:
    """The value of a plain number, a negated one, or a quotient of two; else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        return None if inner is None else -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        top, bottom = _number(node.left), _number(node.right)
        return None if top is None or bottom is None or bottom == 0 else top / bottom
    return None


def safe_to_parse(text: str) -> bool:
    """Whether the whitelist parser can be given this text without a number in it exploding.

    The parser evaluates arithmetic on whole numbers exactly, so ``9**9**9`` would keep it busy
    until memory ran out. A power is therefore accepted only if its base contains a variable and
    its exponent either contains one too or is a plain number of size at most ``MAX_EXPONENT``.
    Only the syntax is read here; nothing is evaluated.
    """
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    numbers = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            numbers += 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not _has_variable(node.left):
                return False
            if not _has_variable(node.right):
                exponent = _number(node.right)
                if exponent is None or abs(exponent) > MAX_EXPONENT:
                    return False
    return numbers <= MAX_NUMBERS


def read_candidates(returned: Any, n_vars: int, limit: int = MAX_CANDIDATES) -> tuple[list[sp.Expr], int]:
    """What a candidate returned, as expressions in ``x0, x1, ...``; and how many strings were refused.

    Only the first ``limit`` strings are read. A string is refused if it is not text, is too long,
    could stall the parser (``safe_to_parse``), does not pass the whitelist parser, has a free
    constant left in it, or uses no variable. Whatever the parser raises on a string is a refusal.
    """
    if not isinstance(returned, (list, tuple)):
        return [], 1
    expressions: list[sp.Expr] = []
    refused = 0
    symbols = {sp.Symbol(f"x{i}") for i in range(n_vars)}
    for text in list(returned)[:limit]:
        if not isinstance(text, str) or len(text) > MAX_TEXT or not safe_to_parse(text):
            refused += 1
            continue
        try:
            expr = llm_proposer.parse_skeleton(text, n_vars, max_constants=0)
            usable = bool(expr.free_symbols) and expr.free_symbols <= symbols \
                and not expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I)
        except Exception:  # noqa: BLE001 - whatever the parser raises on a candidate's text is a refusal of that text
            usable = False
        if not usable:
            refused += 1
            continue
        expressions.append(expr)
    return expressions, refused


def _values(expr: sp.Expr, n_vars: int, X: np.ndarray) -> np.ndarray:
    fn = sp.lambdify([sp.Symbol(f"x{i}") for i in range(n_vars)], expr, "numpy")
    with np.errstate(all="ignore"):
        out: np.ndarray = np.asarray(fn(*X.T), dtype=float) * np.ones(len(X))
    return out


@dataclass(frozen=True)
class Problem:
    """One problem whose law is known. ``X``, ``y`` are what a candidate sees; the test rows are clean."""

    name: str
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    truth: sp.Expr               #: in ``x0, x1, ...``

    @property
    def names(self) -> list[str]:
        return [f"x{i}" for i in range(self.X.shape[1])]


class CandidateGeneratorTask:
    """The judge of ``propose`` versions on problems with known laws.

    **The laws that score a version are never shown to the model** (since
    2026-11-23). Before then the feedback printed the true law of each problem
    a version missed. Those same problems made the score, so a version could
    fingerprint a problem by its rows and return the law it had been shown. A
    scripted adversary did exactly that (``tests/test_judge_adversary.py``).
    Now the feedback names the true law only for ``teaching`` problems, which
    are run and reported but count for nothing. For a scored problem it says
    what the version returned and how accurate it was, never the law.
    """

    contract = CONTRACT

    def __init__(self, problems: Sequence[Problem], recovered: Callable[[sp.Expr, Problem], bool], *,
                 seconds_per_call: float = 3.0, shown_failures: int = 6, teaching: Sequence[Problem] = ()) -> None:
        self.problems = list(problems)
        self.teaching = list(teaching)
        self._recovered = recovered
        self._seconds = float(seconds_per_call)
        self._shown = int(shown_failures)

    def evaluate(self, code: str) -> Evaluation:
        everything = self.problems + self.teaching
        try:
            run = jail.run(code, FUNCTION, [[p.X, p.y, p.names] for p in everything], parameters=3,
                           seconds_per_call=self._seconds)
        except restricted.RejectedCodeError as exc:
            return Evaluation(False, error=f"refused before running: {exc}")
        if not run.defined:
            return Evaluation(False, error=f"could not be defined: {run.error}")
        answered = [c for c in run.calls if c.ok]
        if not answered:
            first = next((c.error for c in run.calls if c.error), run.error)
            return Evaluation(False, error=f"failed on every problem; the first: {first}")
        solved = accurate = returned = refused_total = 0
        missed: list[str] = []
        taught: list[str] = []
        per_problem: dict[str, dict[str, Any]] = {}
        errors = [c.error for c in run.calls[:len(self.problems)] if not c.ok]
        for index, (problem, call) in enumerate(zip(everything, run.calls, strict=True)):
            teaching = index >= len(self.problems)
            n_vars = problem.X.shape[1]
            expressions, refused = read_candidates(call.result, n_vars) if call.ok else ([], 0)
            if not teaching:                             # the score's counts are the scored problems' alone
                refused_total += refused
                returned += len(expressions)
            truth = _values(problem.truth, n_vars, problem.X_test)
            spread = float(np.sum((truth - truth.mean()) ** 2))
            best_r2, best_text, found = -np.inf, "", False
            for expr in expressions:
                values = _values(expr, n_vars, problem.X_test)
                r2 = 1.0 - float(np.sum((values - truth) ** 2)) / spread if np.all(np.isfinite(values)) and spread > 0 else -np.inf
                if r2 > best_r2:
                    best_r2, best_text = r2, str(expr)
                if not found and r2 >= WORTH_CHECKING_R2 and self._recovered(expr, problem):
                    found = True
            per_problem[f"teaching: {problem.name}" if teaching else problem.name] = {"solved": found, "best_r2": None if not np.isfinite(best_r2) else round(best_r2, 6),
                                         "candidates": len(expressions), "refused": refused, "teaching": teaching,
                                         "error": "" if call.ok else call.error}
            shown = f"best returned: {best_text[:120]} (R2 {best_r2:.4f})" if best_text else \
                ("returned nothing usable" if call.ok else f"failed: {call.error[:80]}")
            if teaching:
                if not found:
                    taught.append(f"  law {problem.truth} | {shown}")
                continue
            solved += found
            accurate += bool(best_r2 >= ACCURATE_R2)
            if not found:
                missed.append(f"  a problem with {n_vars} input{'s' if n_vars != 1 else ''} | {shown}")
        n = len(self.problems)
        average = returned / n
        summary = (f"law proposed in {solved} of {n} problems, an accurate candidate in {accurate}, "
                   f"{average:.1f} candidates a problem, {run.seconds:.1f} s")
        notes = []
        if errors:
            notes.append(f"  {len(errors)} calls failed; the first: {errors[0][:120]}")
        if refused_total:
            notes.append(f"  {refused_total} returned strings were not readable as expressions and were dropped")
        feedback = "\n".join([f"  {n - solved} scored problems where the law was not among the candidates (their laws "
                              "are not shown); some of them:", *missed[: self._shown]]) if missed else \
            "  the law was proposed in every scored problem"
        if taught:
            feedback += "\n" + "\n".join(["  laws to learn from, which count for nothing in the score; missed here:",
                                          *taught[: self._shown]])
        if notes:
            feedback += "\n" + "\n".join(notes)
        return Evaluation(True, (float(solved), float(accurate), -round(average, 2)), summary, feedback,
                          details={"problems": per_problem, "seconds": round(run.seconds, 2), "failed_calls": len(errors),
                                   "refused_strings": refused_total, "processes_killed": run.restarts})


class EvolvedProposer(ProposerBase):
    """One checked version of ``propose``, run in the jail on the rows it is given."""

    def __init__(self, code: str, *, target_col: str = "target", seconds: float = 10.0,
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> None:
        restricted.check(code, FUNCTION, 3)             # refuse at construction what would be refused at every call
        self._code = code
        self._target_col = target_col
        self._seconds = float(seconds)
        self._strategy_archetype = strategy_archetype
        #: what the last call returned and what was dropped
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "evolved_proposer"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        names = [f"x{i}" for i in range(len(features))]
        run = jail.run(self._code, FUNCTION, [[X, y, names]], parameters=3, seconds_per_call=self._seconds)
        call = run.calls[0] if run.calls else None
        expressions, refused = read_candidates(call.result, len(features)) if call is not None and call.ok else ([], 0)
        self.last_run = {"defined": run.defined, "error": run.error or (call.error if call is not None else ""),
                         "candidates": len(expressions), "refused_strings": refused, "seconds": round(run.seconds, 2)}
        to_feature = {sp.Symbol(f"x{i}"): sp.Symbol(f) for i, f in enumerate(features)}
        proposals = []
        for rank, expr in enumerate(expressions):
            fitted = expr.xreplace(to_feature)
            fn = sp.lambdify([sp.Symbol(f) for f in features], fitted, "numpy")

            def compute_score(df: pd.DataFrame, fn: Any = fn, features: list[str] = features) -> np.ndarray:
                values = df[features].to_numpy(dtype=float)
                out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
                return out

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id,
                candidate_id=f"evolved:{hashlib.sha256(str(expr).encode()).hexdigest()[:16]}",
                formula=str(fitted), strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
                compute_score=compute_score, parameters={},
                metadata={"rank_given_by_the_candidate": rank, "selection_pool_size": max(1, len(expressions)),
                          "replies_rejected_as_unsafe_or_unparseable": refused, "variables_anonymised": True}))
        return tuple(proposals)
