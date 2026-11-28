"""A proposer that looks for a polynomial relation between the inputs and the target, and proposes its branches.

Status: experimental

**Not in force. Development only: nothing here has been tested blind.**

**Why.** Every other search stage here looks for the target as a formula in the inputs, ``y = f(x)``. Some laws
are short only when written as a relation: a saturation curve ``y = a*x/(1 + b*x)`` is three terms as
``y + b*x*y - a*x = 0``, and a law with a square root, ``y = sqrt(p**2 + m**2)``, is three terms as
``y**2 - p**2 - m**2 = 0``. A probe on tables drawn for the purpose (``scripts/probe_outside_proposals.py``, 28
November 2026) ranked the law's own relation first in every draw without noise; at 1 % noise in 20 to 29 draws of
30 for five of six algebraic laws and in none for a straight line; and on the three real development tables for
one of three. An outside note proposed the idea; SINDy-PI (Kaheman, Kutz and Brunton, 2020) is the published form
of it for differential equations.

**What.** ``ImplicitFormProposer`` fits every small set of monomials in the inputs and the target (two to four
terms, total degree at most three, the target to the second power at most) as ``P(x, y) = 0``, ranks the sets by
the error of the solved formula on the rows with a price for each term, and proposes the best ten solved for the
target. A relation of first degree in the target gives a ratio of two polynomials; one of
second degree gives the quadratic formula, and of its two branches the one nearer the rows is proposed. It decides
nothing: what it proposes goes to the contrast confirmation and the last checks like any other stage's proposals,
and the route refits the constants itself.

**What it cannot propose.** A law that is not algebraic in this small sense: an exponential, a logarithm, a sine,
or an algebraic law whose relation needs a higher degree or more terms. On such a table it still proposes ten
relations, each an approximation, so the referee's refusals are what stands between them and a report. With more
than two inputs the number of sets grows quickly, and the degree or the number of terms is lowered to keep to
``MAX_SUPPORTS``.

**Fitting.** The columns are divided by their medians first, a change of units that is undone in the formulas.
The coefficients of a set are the direction of least spread (the last singular vector), reweighted four times by
the size of ``dP/dy`` so that what is made small is each row's distance in the target, to first order, and not
the size of ``P``. On rows without noise every set that contains the right one also fits; the error is
therefore not counted below one part in ten million of the target's spread, and the shorter set leads.

**Ranking, and why it is not the probe's.** The probe ranked a set by that first-order distance. Building this
showed what that does on a straight line with noise: the relation ``(y - a*x)**2 = 0`` is a perfect square, its
``dP/dy`` vanishes on the curve with ``P``, the first-order distance comes out at half the true one, and the
square leads every honest relation. It is why the probe lost the plain line at 1 % noise. Here a set is ranked by
the error of its solved formula on the rows, and a relation of second degree whose two branches coincide (a
perfect square) is not proposed.

**Which are proposed.** Not the ten best of one list. On a straight line with 1 % noise, seventeen relations of
three and four terms ranked above the line itself, each fitting one chance bend in the noise a little better
than its price in terms: the line would not have been among ten. The best relation of two terms, of three and of
four are therefore proposed first, and the rest follow in the order of the one list. A formula that divides by something
that changes sign among the rows has a pole between them and is not proposed: the common case is the law itself
multiplied through by a factor, whose pole and zero nearly cancel. A law with a real pole among its rows is
therefore out of this stage's reach.

The owner's word for building it: "try all of these" (28 November 2026).
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

from apedp_dyn.grammar.proposer_interface import CandidateProposal, ProposerBase
from apedp_dyn.proposers.composed import formula_proposal

DEGREE_IN_Y = 2                    #: the target to the second power at most: a higher power has no formula worth proposing
LADDER = ((3, 4), (3, 3), (2, 4), (2, 3), (2, 2))   #: (total degree, most terms); the first within the budget is used
MAX_SUPPORTS = 20_000              #: sets of monomials fitted for one table, at most
SEARCH_ROWS = 400                  #: rows the sets are fitted on; a larger table is sampled, the same rows every time
REWEIGHTS = 4                      #: rounds of reweighting by the size of dP/dy
EXACT = 1e-7                       #: a distance below this share of the target's spread is not counted (see above)
PROPOSALS = 10                     #: formulas proposed, at most: what the route takes from a stage
ATTEMPTS = 400                     #: relations written out as formulas for one table, at most (most are repeats on exact rows)
DIGITS = 12                        #: significant digits written for a coefficient; the route refits them
SAME = 1e-9                        #: two formulas whose values differ by less than this share of the spread are one
NEGLIGIBLE = 1e-10                 #: a set with a coefficient this small beside its largest (columns divided by their
                                   #: medians) is the set without that term
DOUBLE_ROOT = 1e-3                 #: a relation whose two branches lie this close together is a perfect square

Exponents = tuple[int, ...]


@dataclass(frozen=True)
class Relation:
    """One fitted set of monomials: ``sum(coef * x**e * y**e_y) = 0`` in the columns' own units."""

    support: tuple[Exponents, ...]
    coef: tuple[float, ...]
    bic: float
    distance: float          #: root mean square error of the solved formula on the rows fitted, in units of the target's median


def monomials(n_inputs: int, degree: int) -> list[Exponents]:
    """Exponents over (inputs..., target): total degree at most ``degree``, the target to ``DEGREE_IN_Y`` at most."""
    found: list[Exponents] = []
    for total in range(1, degree + 1):                 # by which variables are multiplied, not by every table of powers
        for chosen in itertools.combinations_with_replacement(range(n_inputs + 1), total):
            exps = tuple(chosen.count(i) for i in range(n_inputs + 1))
            if exps[-1] <= DEGREE_IN_Y:
                found.append(exps)
    return [*sorted(found), (0,) * (n_inputs + 1)]


def setting(n_inputs: int, rows: int, budget: int = MAX_SUPPORTS) -> tuple[int, int] | None:
    """The total degree and the most terms used for a table of this many inputs and rows, or None if none is affordable."""
    for degree, terms in LADDER:
        terms = min(terms, rows - 2)                   # a set of k terms is fitted on more than k + 1 rows
        if terms < 2:
            return None
        if sum(math.comb(len(monomials(n_inputs, degree)), k) for k in range(2, terms + 1)) <= budget:
            return degree, terms
    return None


def _usable(support: Sequence[Exponents]) -> bool:
    """Whether a set says something about the target in terms of an input, and is not another set times a monomial."""
    holds_target = any(e[-1] > 0 for e in support)
    holds_input = any(sum(e[:-1]) > 0 for e in support)
    has_factor = any(min(column) > 0 for column in zip(*support, strict=True))
    # y**2 = c * x**2 says what y = c * x says, and that set is fitted on its own; y**2 = c * x**3 is a new form
    a_square = len(support) == 2 and all(p % 2 == 0 for e in support for p in e)
    return holds_target and holds_input and not has_factor and not a_square


def _solved(support: Sequence[Exponents], coef: np.ndarray, x_parts: np.ndarray) -> np.ndarray | None:
    """The relation solved for the target on the rows, one column for each branch; None if it gives no formula there."""
    by_power = [np.zeros(len(x_parts)), np.zeros(len(x_parts)), np.zeros(len(x_parts))]
    for exps, c, column in zip(support, coef, x_parts.T, strict=True):
        by_power[exps[-1]] += c * column
    c0, b, a = by_power
    quadratic = any(e[-1] == 2 for e in support)
    below = a if quadratic else b
    if not np.all(np.isfinite(below)) or float(np.min(below)) * float(np.max(below)) <= 0.0:
        return None                             # what the formula divides by changes sign among the rows: a pole between them
    with np.errstate(all="ignore"):
        if not quadratic:
            out = (-c0 / b)[:, None]
        else:
            inside = b ** 2 - 4 * a * c0
            size = b ** 2 + 4 * np.abs(a * c0)
            if np.any(inside < 0) or float(np.median(inside / np.where(size > 0, size, 1.0))) < DOUBLE_ROOT:
                return None                     # no real formula on some row, or a perfect square (see the module's text)
            root = np.sqrt(inside)
            out = np.column_stack([(-b + root) / (2 * a), (-b - root) / (2 * a)])
    usable = np.all(np.isfinite(out), axis=0)
    return out[:, usable] if np.any(usable) else None


def relations(X: np.ndarray, y: np.ndarray, degree: int, max_terms: int) -> list[Relation]:
    """Every usable set of 2..``max_terms`` monomials fitted on the rows, best first. Coefficients are for the rows as given."""
    n = len(y)
    exps = monomials(X.shape[1], degree)
    with np.errstate(all="ignore"):
        x_all = np.column_stack([np.prod(X ** np.asarray(e[:-1]), axis=1) for e in exps])
        A_all = np.column_stack([x_all[:, j] * y ** e[-1] for j, e in enumerate(exps)])
        D_all = np.column_stack([e[-1] * x_all[:, j] * y ** max(e[-1] - 1, 0) for j, e in enumerate(exps)])
    norms = np.linalg.norm(A_all, axis=0)
    fine = np.all(np.isfinite(A_all), axis=0) & (norms > 0)
    spread = float(np.std(y)) or 1.0
    floor_rss = n * (EXACT * spread) ** 2
    found: list[Relation] = []
    for k in range(2, max_terms + 1):
        if k >= n - 1:
            break
        for picked in itertools.combinations(range(len(exps)), k):
            support = tuple(exps[i] for i in picked)
            index = list(picked)
            if not _usable(support) or not np.all(fine[index]):
                continue
            A, D, scale = A_all[:, index], D_all[:, index], norms[index]
            weights = np.ones(n)
            coef = np.zeros(k)
            usable = True
            for _ in range(REWEIGHTS):
                try:
                    coef = np.linalg.svd((A / scale) * weights[:, None], full_matrices=False)[2][-1] / scale
                except np.linalg.LinAlgError:
                    usable = False
                    break
                slope = np.abs(D @ coef)
                middle = float(np.median(slope))
                if not np.all(np.isfinite(slope)) or middle == 0.0:
                    usable = False
                    break
                weights = 1.0 / np.maximum(slope, 1e-3 * middle)
            if not usable or float(np.min(np.abs(coef))) <= NEGLIGIBLE * float(np.max(np.abs(coef))):
                continue                         # a term fitted as nothing: the set without it is fitted on its own
            solved = _solved(support, coef, x_all[:, index])
            if solved is None:
                continue
            rss = float(np.min(np.sum((solved - y[:, None]) ** 2, axis=0)))
            if not np.isfinite(rss):
                continue
            rss = max(rss, floor_rss)
            found.append(Relation(support, tuple(float(c) for c in coef),
                                  n * math.log(rss / n) + (k - 1) * math.log(n), math.sqrt(rss / n)))
    found.sort(key=lambda r: (r.bic, len(r.support), r.support))
    return found


def _number(value: float) -> sp.Float:
    return sp.Float(float(f"{value:.{DIGITS}g}"))


def _polynomial(terms: Sequence[tuple[Exponents, float]], symbols: Sequence[sp.Symbol], over: float = 1.0,
                lowered: Exponents | None = None) -> sp.Expr:
    """``sum(c/over * x**(e - lowered))``; a coefficient that comes to one exactly is written as one."""
    total: sp.Expr = sp.Integer(0)
    for exps, c in terms:
        powers = exps if lowered is None else tuple(a - b for a, b in zip(exps, lowered, strict=True))
        monomial = sp.Mul(*[s ** p for s, p in zip(symbols, powers, strict=True) if p != 0])
        total += monomial if c == over else _number(c / over) * monomial
    return total


def branches(relation: Relation, symbols: Sequence[sp.Symbol]) -> list[sp.Expr]:
    """The relation solved for the target: one formula if it is of first degree in the target, two if of second."""
    by_power: dict[int, list[tuple[Exponents, float]]] = {0: [], 1: [], 2: []}
    for exps, c in zip(relation.support, relation.coef, strict=True):
        if c != 0.0:
            by_power[exps[-1]].append((exps[:-1], c))
    lead = by_power[2] or by_power[1]
    if not lead or (not by_power[0] and not (by_power[2] and by_power[1])):
        return []                                      # nothing that gives the target in terms of the inputs
    exps_lead, c_lead = max(lead, key=lambda term: abs(term[1]))          # this coefficient becomes one
    lowered = exps_lead if len(lead) == 1 else None                       # a single leading monomial is divided through
    a = _polynomial(by_power[2], symbols, c_lead, lowered)
    b = _polynomial(by_power[1], symbols, c_lead, lowered)
    c = _polynomial(by_power[0], symbols, c_lead, lowered)
    if not by_power[2]:
        return [-c / b]
    if not by_power[1]:
        root = sp.sqrt(-c / a)
        return [root, -root]
    root = sp.sqrt(b ** 2 - 4 * a * c)
    return [(-b + root) / (2 * a), (-b - root) / (2 * a)]


def _values(expr: sp.Expr, symbols: Sequence[sp.Symbol], X: np.ndarray) -> np.ndarray | None:
    try:
        with np.errstate(all="ignore"):
            out = np.asarray(sp.lambdify(list(symbols), expr, "numpy")(*X.T), dtype=complex) * np.ones(len(X))
    except Exception:  # noqa: BLE001 - a branch that cannot be evaluated is not proposed
        return None
    if not np.all(np.isfinite(out)) or np.any(np.abs(out.imag) > 0):
        return None
    real: np.ndarray = out.real
    return real


def in_units(relation: Relation, x_scale: np.ndarray, y_scale: float) -> Relation:
    """The relation fitted on columns divided by their scales, written for the columns as they are."""
    scales = np.array([*x_scale, y_scale], dtype=float)
    coef = np.array([c / float(np.prod(scales ** np.asarray(e))) for e, c in zip(relation.support, relation.coef, strict=True)])
    largest = float(np.max(np.abs(coef)))
    return Relation(relation.support, tuple(float(c / largest) for c in coef), relation.bic, relation.distance)


class ImplicitFormProposer(ProposerBase):
    """Proposes formulas solved from the best polynomial relations between the inputs and the target. Decides nothing."""

    def __init__(self, proposals: int = PROPOSALS, budget: int = MAX_SUPPORTS, rows: int = SEARCH_ROWS) -> None:
        self._proposals, self._budget, self._rows = int(proposals), int(budget), int(rows)
        #: what the last call did: the setting used, sets fitted, and the relation behind each proposal
        self.last_run: dict[str, Any] = {}
        self.failures: list[str] = []

    @property
    def proposer_id(self) -> str:
        return "implicit_form"

    def fit_and_propose(self, train_data: Any, target_col: str | None = None, guesses: Sequence[str] | None = None,
                        **kwargs: Any) -> Sequence[CandidateProposal]:
        frame = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
        target = target_col or "target"
        names = [str(c) for c in frame.columns if c != target]
        self.last_run, self.failures = {"ran": False}, []
        if not names or target not in frame.columns:
            return []
        X = frame[names].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        keep = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]
        chosen = setting(len(names), len(y), self._budget)
        if chosen is None or len(y) < 4:
            self.last_run["why_not"] = "too few rows, or too many inputs for the budget"
            return []
        degree, max_terms = chosen
        if len(y) > self._rows:
            picked = np.sort(np.random.default_rng(0).choice(len(y), self._rows, replace=False))
            fit_X, fit_y = X[picked], y[picked]
        else:
            fit_X, fit_y = X, y
        x_scale = np.median(np.abs(fit_X), axis=0)
        x_scale = np.where((x_scale > 0) & np.isfinite(x_scale), x_scale, 1.0)
        y_scale = float(np.median(np.abs(fit_y)))
        y_scale = y_scale if y_scale > 0 and math.isfinite(y_scale) else 1.0
        ranked = relations(fit_X / x_scale, fit_y / y_scale, degree, max_terms)
        symbols = [sp.Symbol(name) for name in names]
        spread = float(np.std(y)) or 1.0
        out: list[CandidateProposal] = []
        kept: list[np.ndarray] = []
        behind: list[dict[str, Any]] = []

        def propose(place: int, relation: Relation) -> bool:
            """Adds the relation's better branch to the proposals; False if it gives no formula or one already proposed."""
            best: tuple[float, sp.Expr, np.ndarray] | None = None
            for branch in branches(in_units(relation, x_scale, y_scale), symbols):
                values = _values(branch, symbols, X)
                if values is None:
                    continue
                miss = float(np.mean((values - y) ** 2))
                if best is None or miss < best[0]:
                    best = (miss, branch, values)
            if best is None or any(float(np.max(np.abs(best[2] - seen))) <= SAME * spread for seen in kept):
                return False
            kept.append(best[2])
            behind.append({"rank": place + 1, "terms": len(relation.support), "support": [list(e) for e in relation.support],
                           "error": relation.distance})
            out.append(formula_proposal(self.proposer_id, str(best[1]), names, metadata=dict(behind[-1])))
            return True

        # the best relation of each number of terms first, then the rest in order (see "Which are proposed" above)
        heads: dict[int, list[int]] = {}
        for place, relation in enumerate(ranked):
            heads.setdefault(len(relation.support), []).append(place)
        written: set[int] = set()
        for places in sorted(heads.values(), key=lambda found: found[0]):
            for place in places:
                if len(written) >= ATTEMPTS or len(out) >= self._proposals:
                    break
                written.add(place)
                if propose(place, ranked[place]):
                    break
        for place, relation in enumerate(ranked):
            if len(written) >= ATTEMPTS or len(out) >= self._proposals:
                break
            if place not in written:
                written.add(place)
                propose(place, relation)
        self.last_run = {"ran": True, "degree": degree, "max_terms": max_terms, "rows_fitted": len(fit_y),
                         "relations_fitted": len(ranked), "written_out": len(written), "proposed": len(out), "behind": behind}
        return out
