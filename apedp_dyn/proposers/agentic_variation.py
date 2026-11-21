"""A language-model agent as the variation operator of a search.

Status: experimental

**Where this comes from.** Evolutionary program search with a language model
(FunSearch, AlphaEvolve) asks the model for one candidate per call inside a
loop that the framework fixes. Agentic Variation Operators (Chen et al., 2026,
arXiv:2603.24517) hand the variation step to an agent instead: it reads the
lineage of earlier candidates, a knowledge base and the results of running
what it wrote, and proposes, repairs and criticises before it commits. Their
evidence is on GPU kernels, where the score of a candidate is exact and cheap.

**What this repository had.** Two measured uses of a language model, both a
fixed pipeline: ``llm_proposer`` (the model writes equations and is told the fit
error of its earlier ones, three rounds; PREREG-LLM-PROPOSER: 78 laws recovered
against 152 for PySR) and ``llm_controller`` (one question per problem about
how to configure the search; PREREG-LLM-CONTROLLER: 173 against 174 without the
model). In neither can the model look at a residual, choose what to test, or
see what the other searches found.

**What this module is.** The same idea in this repository's terms:

* **A workbench** (``Workbench``) holds the development rows and every
  expression tried on them, whoever tried it: the lineage. Fitting an
  expression returns more than an error: which simple functions of the inputs
  the residual still follows, where the largest errors sit, and how many
  operations and constants the expression costs.
* **An agent session** (``run_session``) gives the model the lineage and lets
  it act for a bounded number of steps: ``FIT`` an expression, ask the
  knowledge base (``KNOW``), ``SUBMIT`` its answers. What it does next is its
  choice; what each action returns is computed here.
* **The proposer** (``AgenticVariationProposer``) alternates a conventional
  search with agent sessions: the search's expressions enter the lineage, the
  agent varies them with the residuals in view, and the best structures seed
  the next round of the search. With no search it is the agent alone.

**What the agent cannot do.** It sees development rows only, as every proposer
does (AGENTS.md 1.2), and decides nothing about acceptance: the path's
confirmation and adequacy checks judge what comes out. Its replies are never
executed: an expression is used only if the whitelist parser of
``llm_proposer`` accepts it, and everything else in a reply is dropped and
counted. Every structure fitted, by the agent or the search, is counted in
``selection_pool_size``. A model that cannot be reached, or that replies with
nothing usable, costs the search nothing: its expressions are still returned.
Before any row is shown the model is asked what the law is from the
description alone, and a proposal with the structure of one of those answers
is marked ``recited_before_data``, as in ``llm_proposer``.

**What is returned.** Not the best fits: on noisy rows the best fits are the
most elaborate expressions, and an agent told to lower the error will keep
adding terms below the noise (the first run with a real model did exactly
that: it wrote the true law in its first step and then decorated it with
sines for eleven more). So an expression has to earn its cost: going from the
cheapest expression upwards, one is kept only if its error is at least ``GAIN``
lower than that of the last one kept. Those come first, best fit first; then
the rest by fit. The model is told the same rule, and that a residual with no
structure in it is a reason to simplify and submit, not to add.

**What is not known.** Whether this recovers more laws than the search alone,
at equal time and without more wrong laws, has not been measured. The earlier
runs suggest the freely callable model is the limit; and a language model may
know textbook laws, so a fair test uses laws it cannot have read.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import rankdata

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase, StrategyArchetype
from apedp_dyn.proposers import llm_proposer

EXACT = 1e-6            # relative error below which a fit is exact to numerical precision (the path's own floor)
HINT_FLOOR = 0.25       # a rank correlation of the residual below this is not reported
GAIN = 0.05             # an expression earns a higher cost only by an error lower by this share
MAX_CONSTANTS = 6
FITS_PER_REPLY = 3
SUBMITS = 3
_ACTION = re.compile(r"^\s*(?:[-*>]\s*)?(THINK|FIT|KNOW|SUBMIT)\s*[:=]\s*(.*?)\s*$", re.IGNORECASE)

#: Notes written by hand: what to try when a residual looks a certain way. General modelling
#: practice, not answers to any benchmark. ``knowledge`` returns the notes whose key words match.
NOTES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("residual", "term", "missing"),
     "If the residual rises or falls with one input, a term in that input is missing: add c*x or multiply by (1 + c*x)."),
    (("interaction", "product", "sign"),
     "If the residual follows a product of two inputs, add c*xi*xj; many rates are sums of products of two states."),
    (("u-shaped", "even", "square"),
     "A residual that is U-shaped in an input points to an even term: c*x**2, or a square of a difference of inputs."),
    (("saturation", "plateau", "limit"),
     "A quantity that levels off: x/(c + x), 1 - exp(-c*x), tanh(c*x). A logistic rate is x*(1 - x/c)."),
    (("oscillation", "periodic", "wave"),
     "Periodic behaviour: sin(c*x + d) or cos(c*x); the difference of two angles often enters as sin(xi - xj)."),
    (("power", "scale", "log"),
     "If log y is a straight line in log x the law is a power: c*x**d. Try square roots and inverse squares first."),
    (("ratio", "units", "dimensionless"),
     "Laws often depend on ratios or products of inputs with the same units; try xi/xj inside a function."),
    (("rational", "pole", "fraction"),
     "A sharp rise near a value of an input suggests a pole: 1/(c + x), x/(c + x**2), (a + b*x)/(c + d*x)."),
    (("exponential", "decay", "growth"),
     "Decay or growth: exp(-c*x), x*exp(-c*x); a rate proportional to the state itself gives c*x."),
    (("simplify", "constants", "overfit"),
     "At equal error prefer fewer constants. Remove one term at a time and fit again; a term the data do not need hides the law."),
)


def knowledge(topic: str, limit: int = 2) -> str:
    """The notes whose key words appear in ``topic``; the list of key words if none does."""
    words = set(re.findall(r"[a-z-]+", topic.lower()))
    found = [text for keys, text in NOTES if words & set(keys)]
    if found:
        return " ".join(found[:limit])
    return "No note on that. Topics: " + ", ".join(keys[0] for keys, _ in NOTES) + "."


def complexity(expr: sp.Expr) -> int:
    """Operations plus constants: what "simpler" means when two expressions fit equally well."""
    constants = len(expr.atoms(sp.Float)) + sum(bool(re.fullmatch(r"c\d+", str(s))) for s in expr.free_symbols)
    return int(sp.count_ops(expr)) + constants


def _ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.asarray(rankdata(values), dtype=float)
    centred = ranks - ranks.mean()
    norm = float(np.sqrt((centred ** 2).sum()))
    out: np.ndarray = centred / norm if norm > 0 else np.zeros_like(centred)
    return out


def probe_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """Simple functions of the inputs a residual is checked against, as unit-length centred ranks.

    Each input (a monotone trend), its squared distance from its median (a U shape), and each
    product of two inputs. A feature whose ranks repeat an earlier one's is left out.
    """
    features: dict[str, np.ndarray] = {}
    seen: list[np.ndarray] = []

    def add(name: str, values: np.ndarray) -> None:
        if not np.all(np.isfinite(values)):
            return
        ranks = _ranks(values)
        if not ranks.any() or any(abs(float(ranks @ other)) > 0.999 for other in seen):
            return
        seen.append(ranks)
        features[name] = ranks

    count = X.shape[1]
    for i in range(count):
        add(f"x{i}", X[:, i])
    for i in range(count):
        add(f"U-shape in x{i}", (X[:, i] - np.median(X[:, i])) ** 2)
    for i in range(count):
        for j in range(i + 1, count):
            add(f"x{i}*x{j}", X[:, i] * X[:, j])
    return features


def residual_hints(features: dict[str, np.ndarray], resid: np.ndarray, limit: int = 3) -> list[tuple[str, float]]:
    """Which probe features the residual still follows: (name, rank correlation), strongest first."""
    if not np.all(np.isfinite(resid)) or not len(resid):
        return []
    ranks = _ranks(resid)
    scored = [(name, float(ranks @ feature)) for name, feature in features.items()]
    strong = [item for item in scored if abs(item[1]) >= HINT_FLOOR]
    return sorted(strong, key=lambda item: -abs(item[1]))[:limit]


@dataclass
class Trial:
    """One expression tried on the development rows."""

    key: str                    #: its structure, constants replaced by one placeholder
    skeleton: str               #: the expression with free constants c0, c1, ...
    fitted: sp.Expr             #: the same with the fitted numbers, in x0, x1, ...
    error: float                #: relative RMS error on the rows
    cost: int
    source: str                 #: "agent", "search" or "seed"
    hints: list[tuple[str, float]] = field(default_factory=list)
    worst: str = ""
    submitted: bool = False

    def line(self) -> str:
        hints = "; ".join(f"residual follows {name} ({rho:+.2f})" for name, rho in self.hints)
        state = "exact fit" if self.error < EXACT else (hints or "no structure seen in the residual")
        return f"{self.skeleton} | error {self.error:.3g} | cost {self.cost} | {state} | from {self.source}"


class Workbench:
    """The development rows and everything tried on them."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n_vars = self.X.shape[1]
        self.scale = float(np.sqrt(np.mean(self.y ** 2))) or 1.0
        self.features = probe_features(self.X)
        self.trials: dict[str, Trial] = {}
        self.rejected = 0
        self._seed = seed
        self._symbols = [sp.Symbol(f"x{i}") for i in range(self.n_vars)]

    def _residual(self, fitted: sp.Expr) -> np.ndarray:
        """Data minus fit: positive where the expression is too low."""
        try:
            fn = sp.lambdify(self._symbols, fitted, "numpy")
            with np.errstate(all="ignore"):
                out: np.ndarray = self.y - np.asarray(fn(*self.X.T), dtype=float) * np.ones(len(self.y))
        except Exception:  # noqa: BLE001 - an expression that cannot be evaluated (a function numpy lacks) is a failed trial
            out = np.full(len(self.y), np.nan)
        return out

    def _record(self, key: str, skeleton: str, fitted: sp.Expr, source: str) -> Trial:
        resid = self._residual(fitted)
        finite = bool(np.all(np.isfinite(resid)))
        error = float(np.sqrt(np.mean(resid ** 2))) / self.scale if finite else float("inf")
        worst = ""
        if finite and error >= EXACT:
            row = int(np.argmax(np.abs(resid)))
            worst = "largest error at " + ", ".join(f"x{i}={self.X[row, i]:.3g}" for i in range(self.n_vars))
        trial = Trial(key, skeleton, fitted, error, complexity(fitted), source,
                      residual_hints(self.features, resid) if finite and error >= EXACT else [], worst)
        known = self.trials.get(key)
        if known is None or trial.error < known.error:
            trial.submitted = bool(known and known.submitted)
            self.trials[key] = trial
        return self.trials[key]

    def fit(self, text: str, source: str = "agent") -> Trial | str:
        """Parse ``text`` with the whitelist parser, fit its free constants, record it. A string says why not."""
        try:
            expr = llm_proposer.parse_skeleton(text, self.n_vars, max_constants=MAX_CONSTANTS)
        except (llm_proposer.UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
            self.rejected += 1
            return f"not usable ({type(exc).__name__}): only x0..x{self.n_vars - 1}, c0..c{MAX_CONSTANTS - 1}, " \
                   f"{', '.join(sorted(llm_proposer.FUNCTIONS))} and + - * / ** are read"
        if not any(s in expr.free_symbols for s in self._symbols):
            self.rejected += 1
            return "not usable: the expression uses no input"
        key = llm_proposer.structure_key(expr)
        if key in self.trials and self.trials[key].source == source:
            return self.trials[key]
        constants, _ = llm_proposer.fit_constants(expr, self.X, self.y, seed=self._seed)
        fitted = expr.xreplace({sp.Symbol(name): sp.Float(value) for name, value in constants.items()})
        return self._record(key, str(expr), fitted, source)

    def admit(self, fitted: sp.Expr, source: str) -> Trial | None:
        """Record an expression that already carries its numbers (a search's, a seed), as it is."""
        if not fitted.free_symbols or not fitted.free_symbols <= set(self._symbols):
            return None
        numbers = sorted(fitted.atoms(sp.Float), key=str)
        skeleton = fitted.xreplace({number: sp.Symbol(f"c{i}") for i, number in enumerate(numbers)}) \
            if len(numbers) <= MAX_CONSTANTS else fitted
        return self._record(llm_proposer.structure_key(fitted), str(skeleton), fitted, source)

    def ranked(self) -> list[Trial]:
        """Best first: by error, exact fits together, the cheaper expression first among equals."""
        usable = [t for t in self.trials.values() if np.isfinite(t.error)]
        return sorted(usable, key=lambda t: (0.0 if t.error < EXACT else t.error, t.cost, t.skeleton))

    def earned(self) -> list[Trial]:
        """The expressions that earn their cost, best fit first.

        Of the trials no other beats on both error and cost, taken from the cheapest upwards, one
        is kept only if its error is at least ``GAIN`` lower than that of the last one kept (an
        exact fit after one that is not exact always is). What a dearer expression gains below
        that is within what noise gives any expression with one more term.
        """
        front, cheapest = [], float("inf")
        for trial in self.ranked():               # by error: on the front if cheaper than all that fit better
            if trial.cost < cheapest:
                front.append(trial)
                cheapest = trial.cost
        kept: list[Trial] = []
        for trial in reversed(front):             # from the cheapest (and worst) upwards
            error = 0.0 if trial.error < EXACT else trial.error
            last = None if not kept else (0.0 if kept[-1].error < EXACT else kept[-1].error)
            if last is None or error < (1.0 - GAIN) * last:
                kept.append(trial)
        return kept[::-1]

    def shortlist(self) -> list[Trial]:
        """Every usable trial: those that earn their cost first (best fit first), then the others by fit."""
        earned = self.earned()
        chosen = {id(trial) for trial in earned}
        return earned + [trial for trial in self.ranked() if id(trial) not in chosen]

    def lineage(self, limit: int = 8) -> str:
        earned = {id(trial) for trial in self.earned()}
        rows = self.shortlist()[:limit]
        return "\n".join(f"  {i + 1}. {t.line()}" + ("" if id(t) in earned else " | fits no better than a cheaper one")
                         for i, t in enumerate(rows)) or "  nothing tried yet"


def read_actions(reply: str) -> list[tuple[str, str]]:
    """The lines of a reply that are actions, as (kind, text); everything else is dropped."""
    out = []
    for line in reply.replace("`", "").splitlines():
        match = _ACTION.match(line)
        if match:
            out.append((match.group(1).upper(), match.group(2)))
    return out


def _expressions(text: str) -> list[str]:
    """One or more expressions after ``FIT:`` or ``SUBMIT:``, separated by semicolons; ``y =`` in front is dropped."""
    parts = [re.sub(r"^\s*(?:y|f\([^()]*\))\s*=\s*", "", part).strip().rstrip(".") for part in text.split(";")]
    return [part for part in parts if part]


@dataclass
class Session:
    """What an agent session did."""

    steps: int = 0
    fits: int = 0
    notes_read: int = 0
    submitted: list[str] = field(default_factory=list)
    unusable_replies: int = 0
    stopped: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)


def _prompt(bench: Workbench, session: Session, observations: list[str], steps_left: int, fits_left: int,
            description: str, rows_shown: int) -> str:
    X, y = bench.X, bench.y
    names = ", ".join(f"x{i}" for i in range(bench.n_vars))
    sample = np.random.default_rng(0).choice(len(y), size=min(rows_shown, len(y)), replace=False)
    rows = "\n".join(", ".join(f"{v:.4g}" for v in (*X[i], y[i])) for i in sample)
    ranges = "; ".join(f"x{i} in [{X[:, i].min():.3g}, {X[:, i].max():.3g}]" for i in range(bench.n_vars))
    context = f"Context: {description}\n" if description else ""
    last = "\n".join(f"  {line}" for line in observations) or "  none yet"
    return (
        f"{context}You are searching for the law that gives y from {names}. You work on a bench that fits "
        "expressions for you and tells you what their residuals (data minus fit) still follow.\n"
        f"{len(y)} rows. Ranges: {ranges}; y in [{y.min():.3g}, {y.max():.3g}].\n"
        f"A few rows ({names}, y):\n{rows}\n"
        "Lineage (error is relative RMS; cost counts operations and constants). First the expressions that "
        f"earn their cost, best fit first; then the others:\n{bench.lineage()}\n"
        f"Results of your last actions:\n{last}\n"
        f"Steps left: {steps_left}. Fits left: {fits_left}.\n"
        "Rules for expressions: variables " + names + "; free constants c0, c1, ... (fitted for you); functions "
        + ", ".join(sorted(llm_proposer.FUNCTIONS)) + "; operators + - * / **.\n"
        "Reply with lines of these kinds and nothing else:\n"
        "THINK: one sentence on what the residuals say\n"
        f"FIT: an expression (at most {FITS_PER_REPLY} such lines)\n"
        "KNOW: a topic, to read a note (" + ", ".join(keys[0] for keys, _ in NOTES) + ")\n"
        f"SUBMIT: your final expression (at most {SUBMITS} such lines; ends the session)\n"
        f"An expression earns a higher cost only if its error is at least {100 * GAIN:.0f} % lower than a cheaper one's; "
        "a smaller difference is noise. Where the leading expression's residual shows structure, repair it. "
        "Where it shows none, do not add terms: try cheaper forms of it, then SUBMIT the cheapest that fits as well."
    )


def run_session(client: Callable[[str], str], bench: Workbench, *, steps: int = 6, max_fits: int = 18,
                description: str = "", rows_shown: int = 12, deadline: float | None = None) -> Session:
    """Let the model act on the bench for at most ``steps`` replies and ``max_fits`` fits.

    Ends when the model submits, when a budget is spent, when ``deadline`` (a ``time.monotonic``
    value) has passed, or when two replies in a row contain no action. An exception from
    ``client`` ends it too, and is recorded.
    """
    session = Session()
    observations: list[str] = []
    idle = 0
    while session.steps < steps and session.fits < max_fits:
        if deadline is not None and time.monotonic() >= deadline:
            session.stopped = "out of time"
            break
        prompt = _prompt(bench, session, observations, steps - session.steps, max_fits - session.fits,
                         description, rows_shown)
        try:
            reply = client(prompt)
        except Exception as exc:  # noqa: BLE001 - a model that cannot be reached ends the session, never the search
            session.stopped = f"model unavailable: {type(exc).__name__}: {exc}"[:200]
            break
        session.steps += 1
        actions = read_actions(reply)
        session.transcript.append({"reply": reply[-600:], "actions": "; ".join(f"{k} {t}"[:80] for k, t in actions)})
        if not any(kind != "THINK" for kind, _ in actions):
            session.unusable_replies += 1
            idle += 1
            if idle >= 2:
                session.stopped = "two replies without an action"
                break
            observations = ["Your reply contained no FIT, KNOW or SUBMIT line."]
            continue
        idle = 0
        observations = []
        fits_this_reply = 0
        done = False
        for kind, text in actions:
            if kind == "FIT":
                for expression in _expressions(text):
                    if fits_this_reply >= FITS_PER_REPLY or session.fits >= max_fits:
                        break
                    fits_this_reply += 1
                    session.fits += 1
                    result = bench.fit(expression)
                    shown = result if isinstance(result, str) else result.line() + (f" | {result.worst}" if result.worst else "")
                    observations.append(f"FIT {expression[:60]} -> {shown}")
            elif kind == "KNOW":
                session.notes_read += 1
                observations.append(f"KNOW {text[:40]} -> {knowledge(text)}")
            elif kind == "SUBMIT":
                for expression in _expressions(text)[: SUBMITS - len(session.submitted)]:
                    result = bench.fit(expression)
                    if not isinstance(result, str):
                        result.submitted = True
                        session.submitted.append(result.key)
                done = True
        if done:
            session.stopped = "submitted"
            break
    session.stopped = session.stopped or "budget spent"
    return session


class AgenticVariationProposer(ProposerBase):
    """A search and an agent taking turns on one lineage; the agent alone if there is no search.

    ``search`` is called as ``search(budget_seconds)`` and must return a proposer whose
    ``fit_and_propose(frame, target_col=..., guesses=[...])`` returns formulas in the frame's
    column names (``PySRProposer``, a portfolio, a guarded portfolio). ``budget_seconds`` is the
    time for all its rounds together; the agent's sessions are bounded by ``session_seconds`` each.
    """

    def __init__(self, client: Callable[[str], str] | None, search: Callable[[float], ProposerBase | None] | None = None, *,
                 budget_seconds: float = 240.0, rounds: int = 2, steps: int = 6, max_fits: int = 18,
                 session_seconds: float = 90.0, max_proposals: int = 8, seeds_per_round: int = 3,
                 description: str = "", target_col: str = "target", seed: int = 0, probe: bool = True,
                 strategy_archetype: StrategyArchetype = StrategyArchetype.TACTICAL) -> None:
        self._client = client
        self._probe = bool(probe)
        self._search = search
        self._budget, self._rounds = float(budget_seconds), max(1, int(rounds))
        self._steps, self._max_fits, self._session_seconds = int(steps), int(max_fits), float(session_seconds)
        self._max_proposals, self._seeds_per_round = int(max_proposals), int(seeds_per_round)
        self._description, self._target_col, self._seed = description, target_col, seed
        self._strategy_archetype = strategy_archetype
        #: what the last call did, kept even when nothing could be proposed
        self.last_run: dict[str, Any] = {}

    @property
    def proposer_id(self) -> str:
        return "agentic_variation"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or self._target_col
        features = [c for c in frame.columns if c != target]
        to_x = {sp.Symbol(f): sp.Symbol(f"x{i}") for i, f in enumerate(features)}
        to_feature = {sp.Symbol(f"x{i}"): sp.Symbol(f) for i, f in enumerate(features)}
        bench = Workbench(frame[features].to_numpy(dtype=float), frame[target].to_numpy(dtype=float), seed=self._seed)

        def admit(text: str, source: str) -> None:
            try:
                expr = sp.sympify(text, locals={f: sp.Symbol(f) for f in features})
            except (sp.SympifyError, SyntaxError, TypeError):
                return
            if isinstance(expr, sp.Expr):
                bench.admit(expr.xreplace(to_x), source)

        recited: set[str] = set()
        if self._client is not None and self._probe:
            names = ", ".join(f"x{i}" for i in range(bench.n_vars))
            context = f"Context: {self._description}\n" if self._description else ""
            try:
                answer = self._client(f"{context}No data is given. If you already know the law relating y to {names}, "
                                      "state it as expressions for y, one per line, with free constants c0, c1, ...")
            except Exception:  # noqa: BLE001 - a model that cannot be reached recites nothing
                answer = ""
            for text in llm_proposer._replies(answer):
                try:
                    recited.add(llm_proposer.structure_key(llm_proposer.parse_skeleton(text, bench.n_vars)))
                except (llm_proposer.UnsafeExpressionError, sp.SympifyError, SyntaxError, TypeError, ValueError):
                    continue

        for guess in guesses or ():
            admit(str(guess), "seed")
        sessions: list[Session] = []
        search_runs: list[dict[str, Any]] = []
        search_proposals: list[CandidateProposal] = []
        for round_index in range(self._rounds):
            if self._search is not None:
                engine = self._search(self._budget / self._rounds)
                if engine is not None:
                    seeds = [str(t.fitted.xreplace(to_feature)) for t in bench.shortlist()[: self._seeds_per_round]]
                    found = list(engine.fit_and_propose(frame, target_col=target, guesses=seeds or None))
                    search_proposals += found
                    for proposal in found:
                        admit(proposal.formula, "search")
                    search_runs.append({"round": round_index, "proposals": len(found), "seeded_with": len(seeds),
                                        "proposer_id": engine.proposer_id})
            if self._client is not None:
                sessions.append(run_session(self._client, bench, steps=self._steps, max_fits=self._max_fits,
                                            description=self._description,
                                            deadline=time.monotonic() + self._session_seconds))

        ranked = bench.shortlist()
        by_agent = sum(t.source == "agent" for t in ranked)
        transcript = hashlib.sha256(repr([s.transcript for s in sessions]).encode()).hexdigest()
        self.last_run = {
            "structures_tried": len(bench.trials), "by_agent": by_agent, "replies_rejected": bench.rejected,
            "sessions": [{"steps": s.steps, "fits": s.fits, "notes_read": s.notes_read, "submitted": len(s.submitted),
                          "unusable_replies": s.unusable_replies, "stopped": s.stopped} for s in sessions],
            "search_runs": search_runs, "transcript_sha256": transcript, "recited_before_data": len(recited),
            "last_reply": sessions[-1].transcript[-1]["reply"][-300:] if sessions and sessions[-1].transcript else "",
            "last_answer": dict(getattr(self._client, "last", {}) or {}),
        }
        proposals = []
        for trial in ranked[: self._max_proposals]:
            fitted = trial.fitted.xreplace(to_feature)
            fn = sp.lambdify([sp.Symbol(f) for f in features], fitted, "numpy")

            def compute_score(df: pd.DataFrame, fn: Any = fn, features: list[str] = features) -> np.ndarray:
                values = df[features].to_numpy(dtype=float)
                out: np.ndarray = np.asarray(fn(*values.T), dtype=float) * np.ones(len(df))
                return out

            proposals.append(CandidateProposal(
                proposer_id=self.proposer_id, candidate_id=f"agentic:{trial.key}", formula=str(fitted),
                strategy_archetype=self._strategy_archetype, declared_scale_invariant=False,
                compute_score=compute_score, parameters={"dev_relative_rmse": trial.error},
                metadata={"skeleton": trial.skeleton, "found_by": trial.source, "submitted_by_agent": trial.submitted,
                          "recited_before_data": trial.key in recited,
                          "selection_pool_size": max(1, len(bench.trials)),
                          "replies_rejected_as_unsafe_or_unparseable": bench.rejected, "variables_anonymised": True},
            ))
        if not proposals:      # nothing usable on the bench: what the search returned stands as it was
            return tuple(search_proposals)
        return tuple(proposals)
