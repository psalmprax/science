"""Split a table before it is searched: the sums and products a law is made of, read from its rows.

Status: experimental

**Why.** The table route has recovered no law of four or five inputs by form (wider-conditions runs of
24 and 25 November), and fifteen minutes of search in place of ninety seconds did not change that
(``benchmarks/budget_ladder/``). A law of five inputs is found by search only if it need not be
searched as one. If, for two groups of inputs A and B, ``y = g(A) * h(B) + k(B)`` (or one of the
plainer forms below), the table splits into smaller tables, and each of those may split again. This
module reads that shape from the rows. It is the first step of the first of the ideas the owner
ordered on 25 November (``docs/IDEAS_FROM_OUTSIDE_2026-11-25.md``, C1; the textbook form is the
separability test of AI Feynman, made there on a fitted network, one anchor point, sums and products).

**What the system already had.** The older package has a separability test of its own
(``apedp/separability.py``): every pair of inputs judged by the mixed curvature of local quadratic
fits, groups that add or groups that multiply, used to seed that package's search. The table route
does not use it. On the 300 tables this module was measured on, called as the older search calls
it, it split 2 (``benchmarks/table_split/``): its fits follow polynomials, not the sines, logarithms
and exponentials these laws are made of, and it has no form inside a group. This module is another
method for the same idea, written for the table route.

**The forms.** For two groups of inputs, simplest first:

=============  ==========================  =======================================
``g*h``        ``g(A) * h(B)``             a product
``g+h``        ``g(A) + h(B)``             a sum
``c+g*h``      ``c + g(A) * h(B)``         a product and a constant
``g*h+k(B)``   ``g(A) * h(B) + k(B)``      a product and a term of the second group
``g*h+k(A)``   ``g(A) * h(B) + k(A)``      a product and a term of the first group
=============  ==========================  =======================================

The last two say that one group enters the law through a single quantity, added to and multiplied
by terms of the other. Every law of the development grammar is such a shape down to tables of one
input (``scripts/study_table_split.py``); a law in which two inputs meet inside a sine or a root is not.

**How.** One smooth model of the whole table is fitted: a Gaussian process with a squared-exponential
kernel and one length scale an input. That kernel is a product of one factor an input, so the model's
value on any recombination of rows (the inputs of group A from one row, those of group B from
another) is a matrix product, and nothing is refitted. On that matrix each form is the nearest
matrix of rank one after a constant, the column means or the row means are taken off. Every part of
a form is again a function of the same kind, so a part can be split again by the same step.

**What decides.** Predictions on rows left out of the fit, pooled over ``FOLDS`` folds, squared errors
compared row by row. A form is borne out when it is worse by no more than ``Z_REFUSED`` standard
errors than each of: the unsplit model, the shape read so far, every other form for the same two
groups, and the matrix cut to rank two and to rank three. The last comparisons matter: a split model
is smoother than the unsplit one and so predicts a little better even when its form is only nearly
right, and only a model equally smooth and less constrained shows that. Of the forms borne out the
simplest is taken; of the splits on offer, the one with the fewest parts, then the one that predicts
best. Every shape, however deep, is judged on the same real rows; none is judged on values an
earlier split produced.

**What it is not.** Not a proposer and not a judge. It returns a shape and, for each part left
unsplit, a table of that part's inputs against its values. A part is fixed only up to the changes
that leave the whole unchanged (a factor moved between ``g`` and ``h``, a multiple of ``g`` moved
into ``k``), so a part's table need not be the simplest function that would do. A wrong split yields
candidates of the wrong form, which the path refuses or reports like any other candidate. It reads
no confirmation or pool row. **Not in force**; nothing builds it but its study script and its tests.

**Limits known at writing.** A departure from a form smaller than the left-out rows can show reads
as the form. A table the smooth model does not fit (``MAX_UNSPLIT_ERROR`` of the target's spread) is
not read at all. Recombined rows are only meaningful where inputs vary independently of each other;
every table it was tried on has inputs drawn independently.
"""
from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

LEAF = "leaf"
PRODUCT, SUM, PRODUCT_AND_CONSTANT, WITH_SECOND, WITH_FIRST = "g*h", "g+h", "c+g*h", "g*h+k(B)", "g*h+k(A)"
FORMS = (PRODUCT, SUM, PRODUCT_AND_CONSTANT, WITH_SECOND, WITH_FIRST)   #: simplest first
_TERMS = {PRODUCT: ((0, 1),), SUM: ((0,), (1,)), PRODUCT_AND_CONSTANT: ((0, 1),), WITH_SECOND: ((0, 1), (2,)),
          WITH_FIRST: ((0, 1), (2,))}
_COST = {PRODUCT: 2.0, SUM: 2.0, PRODUCT_AND_CONSTANT: 2.5, WITH_SECOND: 3.0, WITH_FIRST: 3.0}   #: parts, a constant as half
FOLDS = 5
MAX_ROWS = 1000            #: rows a reading is made from; more are thinned (a fit grows with the cube of the rows)
GRID = 200                 #: rows whose inputs are recombined to project the model on a form
Z_REFUSED = 2.0            #: a form worse than a model it is held against by more standard errors than this is refused
MAX_UNSPLIT_ERROR = 0.5    #: share of the target's spread the unsplit model may miss on left-out rows and still be read
MINORS = 20_000            #: two-by-two minors the constant of ``c+g*h`` is read from
MAX_CONSTANT = 1.0e3       #: a constant larger than this many times the part's largest value is a sum in disguise
_JITTER = 1.0e-10
Path = tuple[int, ...]


def bipartitions(group: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every way to put a group of inputs into two non-empty groups; the group holding the first input comes first."""
    members = tuple(group)
    out = []
    for size in range(0, len(members) - 1):
        for rest in itertools.combinations(members[1:], size):
            first = (members[0], *rest)
            second = tuple(m for m in members if m not in first)
            out.append((first, second))
    return out


@dataclass(frozen=True)
class Shape:
    """How a law is put together: a group of inputs left as one table, or one of ``FORMS`` over smaller shapes."""

    form: str
    group: tuple[int, ...]
    parts: tuple[Shape, ...] = ()

    def leaves(self, at: Path = ()) -> list[tuple[Path, tuple[int, ...]]]:
        """Each part left unsplit: where it stands in the shape, and its inputs."""
        if self.form == LEAF:
            return [(at, self.group)]
        return [leaf for i, part in enumerate(self.parts) for leaf in part.leaves((*at, i))]

    def widest(self) -> int:
        """The number of inputs of the largest table left to search."""
        return max(len(group) for _, group in self.leaves())

    def text(self, names: Sequence[str] | None = None) -> str:
        if self.form == LEAF:
            return "f(" + ", ".join(names[i] if names else f"x{i}" for i in self.group) + ")"
        inside = [part.text(names) if part.form == LEAF else f"({part.text(names)})" for part in self.parts]
        if self.form == SUM:
            return f"{inside[0]} + {inside[1]}"
        product = f"{inside[0]}*{inside[1]}"
        if self.form == PRODUCT_AND_CONSTANT:
            return f"c + {product}"
        return product if self.form == PRODUCT else f"{product} + {inside[2]}"

    def as_dict(self) -> dict[str, Any]:
        return {"form": self.form, "group": list(self.group), "parts": [part.as_dict() for part in self.parts]}


@dataclass
class _Node:
    """One fit's numbers for a shape: a leaf is ``c0 + K(group) @ beta``; a form is a constant and products of parts."""

    group: tuple[int, ...]
    form: str = LEAF
    c0: float = 0.0
    beta: np.ndarray | None = None
    constant: float = 0.0
    parts: list[_Node] = field(default_factory=list)
    terms: tuple[tuple[int, ...], ...] = ()

    def at(self, path: Path) -> _Node:
        node = self
        for i in path:
            node = node.parts[i]
        return node

    def replaced(self, path: Path, node: _Node) -> _Node:
        if not path:
            return node
        parts = list(self.parts)
        parts[path[0]] = parts[path[0]].replaced(path[1:], node)
        return _Node(self.group, self.form, constant=self.constant, parts=parts, terms=self.terms)

    def shape(self) -> Shape:
        return Shape(self.form, self.group, tuple(part.shape() for part in self.parts))


class Surface:
    """The smooth model's settings: how far the target swings, one length scale an input, and the noise."""

    def __init__(self, amplitude: float, scales: np.ndarray, noise: float) -> None:
        self.amplitude = float(amplitude)
        self.scales = np.asarray(scales, dtype=float)
        self.noise = float(noise)

    @classmethod
    def fitted(cls, unit: np.ndarray, y: np.ndarray) -> Surface:
        """Settings of highest marginal likelihood for a table whose inputs are scaled to the unit interval."""
        spread = float(np.std(y)) or 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(unit.shape[1]), (5e-2, 1e2)) + WhiteKernel(1e-2, (1e-8, 1e1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(unit, (y - float(np.mean(y))) / spread)
        fitted = model.kernel_
        return cls(fitted.k1.k1.constant_value * spread**2, np.atleast_1d(fitted.k1.k2.length_scale),
                   fitted.k2.noise_level * spread**2)

    def factor(self, d: int, here: np.ndarray, there: np.ndarray) -> np.ndarray:
        """The kernel's factor for input ``d`` between two sets of rows."""
        gap = (here[:, None] - there[None, :]) / self.scales[d]
        factor: np.ndarray = np.exp(-0.5 * gap * gap)
        return factor


def _leading(matrix: np.ndarray, rank: int = 1) -> list[tuple[float, np.ndarray, np.ndarray]] | None:
    """The leading singular triples of a matrix, or nothing where it has none to speak of."""
    try:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(s[0]) and s[0] > 0):
        return None
    return [(float(s[k]), u[:, k], vt[k]) for k in range(min(rank, len(s))) if s[k] > 0]


class _Fit:
    """The model fitted on some rows: what any shape of it says on a grid of those rows and on rows left out."""

    def __init__(self, surface: Surface, unit: np.ndarray, y: np.ndarray, fitted: np.ndarray, left_out: np.ndarray,
                 grid: np.ndarray, seed: int) -> None:
        inputs = unit.shape[1]
        self.on_grid = [surface.factor(d, unit[grid, d], unit[fitted, d]) for d in range(inputs)]
        self.on_left_out = [surface.factor(d, unit[left_out, d], unit[fitted, d]) for d in range(inputs)]
        self.y_left_out = y[left_out]
        self._products: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        mean = float(np.mean(y[fitted]))
        gram = np.full((len(fitted), len(fitted)), surface.amplitude)
        for d in range(inputs):
            gram *= surface.factor(d, unit[fitted, d], unit[fitted, d])
        gram[np.diag_indices_from(gram)] += surface.noise + _JITTER * surface.amplitude
        alpha = cho_solve(cho_factor(gram, lower=True), y[fitted] - mean)
        self.root = _Node(tuple(range(inputs)), c0=mean, beta=surface.amplitude * alpha)
        rng = np.random.default_rng(seed)
        self.minors = tuple(rng.integers(0, len(grid), MINORS) for _ in range(4))

    def _kernel(self, where: int, group: tuple[int, ...]) -> np.ndarray:
        key = (where, group)
        if key not in self._products:
            factors = self.on_grid if where == 0 else self.on_left_out
            product = factors[group[0]].copy()
            for d in group[1:]:
                product *= factors[d]
            self._products[key] = product
        return self._products[key]

    def value(self, node: _Node, where: int = 1) -> np.ndarray:
        """What a shape says on the grid rows (``where`` 0) or on the rows left out (1)."""
        if node.form == LEAF:
            leaf: np.ndarray = node.c0 + self._kernel(where, node.group) @ node.beta
            return leaf
        values = [self.value(part, where) for part in node.parts]
        total = np.full(len(values[0]), node.constant)
        for term in node.terms:
            product = values[term[0]]
            for i in term[1:]:
                product = product * values[i]
            total = total + product
        return total

    def _constant(self, matrix: np.ndarray) -> float | None:
        """The ``c`` for which ``matrix - c`` is nearest a product: in every two-by-two minor, det = c * mixed difference."""
        i, k, j, m = self.minors
        mixed = matrix[i, j] + matrix[k, m] - matrix[i, m] - matrix[k, j]
        det = matrix[i, j] * matrix[k, m] - matrix[i, m] * matrix[k, j]
        weight = float(np.sum(mixed * mixed))
        if not weight > 0:
            return None
        constant = float(np.sum(det * mixed) / weight)
        if not np.isfinite(constant) or abs(constant) > MAX_CONSTANT * float(np.max(np.abs(matrix))):
            return None
        return constant

    def forms(self, leaf: _Node, first: tuple[int, ...], second: tuple[int, ...]) -> dict[str, _Node]:
        """The leaf's nearest shape of each form over two groups of its inputs, and its cuts to rank two and three."""
        k_a, k_b = self._kernel(0, first), self._kernel(0, second)
        beta, c0, group = leaf.beta, leaf.c0, leaf.group
        mean_a, mean_b = k_a.mean(axis=0), k_b.mean(axis=0)
        matrix = c0 + (k_a * beta) @ k_b.T

        def part(inputs: tuple[int, ...], offset: float, weights: np.ndarray) -> _Node:
            return _Node(inputs, c0=float(offset), beta=weights)

        out: dict[str, _Node] = {}
        grand = c0 + float(np.sum(beta * mean_a * mean_b))
        out[SUM] = _Node(group, SUM, terms=_TERMS[SUM],
                         parts=[part(first, c0, beta * mean_b), part(second, c0 - grand, beta * mean_a)])

        def product(form: str, constant: float) -> None:
            triples = _leading(matrix - constant)
            if triples:
                sigma, u, v = triples[0]
                shift = c0 - constant
                out[form] = _Node(group, form, constant=constant, terms=_TERMS[form],
                                  parts=[part(first, shift * v.sum(), beta * (k_b.T @ v)),
                                         part(second, shift * u.sum() / sigma, beta * (k_a.T @ u) / sigma)])

        product(PRODUCT, 0.0)
        constant = self._constant(matrix)
        if constant is not None:
            product(PRODUCT_AND_CONSTANT, constant)

        # g(A) * h(B) + k(B): take each column's mean off; what is left has rank one.
        triples = _leading(matrix - matrix.mean(axis=0)[None, :])
        if triples:
            sigma, u, v = triples[0]
            w = k_b.T @ v
            out[WITH_SECOND] = _Node(group, WITH_SECOND, terms=_TERMS[WITH_SECOND], parts=[
                part(first, -np.sum(beta * mean_a * w), beta * w),
                part(second, 0.0, beta * (k_a.T @ u - mean_a * u.sum()) / sigma),
                part(second, c0, beta * mean_a)])
        # g(A) * h(B) + k(A): the same with each row's mean.
        triples = _leading(matrix - matrix.mean(axis=1)[:, None])
        if triples:
            sigma, u, v = triples[0]
            w = k_a.T @ u
            out[WITH_FIRST] = _Node(group, WITH_FIRST, terms=_TERMS[WITH_FIRST], parts=[
                part(first, 0.0, beta * (k_b.T @ v - mean_b * v.sum())),
                part(second, -np.sum(beta * w * mean_b) / sigma, beta * w / sigma),
                part(first, c0, beta * mean_b)])
        # Held against, never taken: the matrix cut to rank two and to rank three.
        triples = _leading(matrix, rank=3) or []
        for rank in (2, 3):
            if len(triples) >= rank:
                parts: list[_Node] = []
                terms: list[tuple[int, ...]] = []
                for sigma, u, v in triples[:rank]:
                    terms.append((len(parts), len(parts) + 1))
                    parts += [part(first, c0 * v.sum(), beta * (k_b.T @ v)), part(second, c0 * u.sum() / sigma, beta * (k_a.T @ u) / sigma)]
                out[f"rank {rank}"] = _Node(group, f"rank {rank}", parts=parts, terms=tuple(terms))
        return out


def _z(worse: np.ndarray, better: np.ndarray) -> float:
    """By how many standard errors the first set of errors exceeds the second, squared and compared row by row."""
    gap = worse * worse - better * better
    spread = float(np.std(gap, ddof=1)) if len(gap) > 1 else 0.0
    return float(np.mean(gap)) / (spread / np.sqrt(len(gap))) if spread > 0 else 0.0


def _rms(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors * errors)))


@dataclass
class Reading:
    """A table's shape, the table left to search for each part, and what the reading rests on."""

    shape: Shape
    rows: np.ndarray                 #: which rows of the table were read
    tables: dict[Path, np.ndarray]   #: for each part left unsplit, its values on those rows
    constants: dict[Path, float]     #: for each ``c+g*h`` in the shape, its constant
    report: dict[str, Any]

    def recomposed(self, values: dict[Path, np.ndarray] | None = None, shape: Shape | None = None, at: Path = ()) -> Any:
        """The whole from its parts: the shape's arithmetic on a value for each part left unsplit (by default its table)."""
        values = self.tables if values is None else values
        shape = shape or self.shape
        if shape.form == LEAF:
            return values[at]
        inside = [self.recomposed(values, part, (*at, i)) for i, part in enumerate(shape.parts)]
        total: Any = self.constants.get(at, 0.0)
        for term in _TERMS[shape.form]:
            product = inside[term[0]]
            for i in term[1:]:
                product = product * inside[i]
            total = total + product
        return total

    def compose(self, laws: dict[Path, str], shape: Shape | None = None, at: Path = ()) -> str:
        """One formula from a formula for each part left unsplit."""
        shape = shape or self.shape
        if shape.form == LEAF:
            return f"({laws[at]})"
        inside = [self.compose(laws, part, (*at, i)) for i, part in enumerate(shape.parts)]
        if shape.form == SUM:
            return f"({inside[0]} + {inside[1]})"
        product = f"{inside[0]}*{inside[1]}"
        if shape.form == PRODUCT:
            return f"({product})"
        if shape.form == PRODUCT_AND_CONSTANT:
            return f"({self.constants[at]!r} + {product})"
        return f"({product} + {inside[2]})"


def read(X: np.ndarray, y: np.ndarray, *, folds: int = FOLDS, seed: int = 0, max_rows: int = MAX_ROWS,
         grid: int = GRID, z_refused: float = Z_REFUSED, max_unsplit_error: float = MAX_UNSPLIT_ERROR,
         forms: Sequence[str] = FORMS, every_candidate: bool = False) -> Reading:
    """The shape of the law behind a table, as far as rows left out of the fit bear it out."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] < 1 or len(y) != len(X):
        raise ValueError("a reading needs a table: rows of inputs and one target a row")
    rng = np.random.default_rng(seed)
    rows = np.arange(len(y))
    if len(y) > max_rows:
        rows = np.sort(rng.choice(len(y), size=max_rows, replace=False))
    X, y = X[rows], y[rows]
    inputs = X.shape[1]
    whole = tuple(range(inputs))
    report: dict[str, Any] = {"rows": len(y), "inputs": inputs, "z_refused": z_refused, "forms": list(forms),
                              "steps": [], "read": False}
    alone = Reading(Shape(LEAF, whole), rows, {(): y.copy()}, {}, report)
    if inputs < 2 or len(y) < 4 * folds:
        return alone
    low, high = X.min(axis=0), X.max(axis=0)
    unit = (X - low) / np.where(high > low, high - low, 1.0)

    surface = Surface.fitted(unit, y)
    report["length_scales"] = [float(s) for s in surface.scales]
    order = rng.permutation(len(y))
    fits = []
    for fold in range(folds):
        left_out = np.sort(order[fold::folds])
        fitted = np.setdiff1d(order, left_out)
        on = fitted if len(fitted) <= grid else np.sort(rng.choice(fitted, size=grid, replace=False))
        fits.append(_Fit(surface, unit, y, fitted, left_out, on, seed + fold))
    trees = [fit.root for fit in fits]

    def errors(candidates: Sequence[_Node]) -> np.ndarray:
        gaps: np.ndarray = np.concatenate([fit.y_left_out - fit.value(tree) for fit, tree in zip(fits, candidates, strict=True)])
        return gaps

    unsplit = errors(trees)
    spread = float(np.std(y)) or 1.0
    report["unsplit_error"] = _rms(unsplit)
    report["unsplit_error_of_spread"] = report["unsplit_error"] / spread
    if not report["unsplit_error_of_spread"] <= max_unsplit_error:
        return alone
    report["read"] = True

    taken: list[tuple[Path, tuple[int, ...], tuple[int, ...], str]] = []
    so_far = unsplit
    while True:
        best: tuple[tuple[float, float], dict[str, Any], list[_Node], np.ndarray] | None = None
        seen = []
        for path, group in trees[0].shape().leaves():
            if len(group) < 2:
                continue
            for first, second in bipartitions(group):
                by_fit = [fit.forms(tree.at(path), first, second) for fit, tree in zip(fits, trees, strict=True)]
                on_offer = [name for name in by_fit[0] if all(name in one for one in by_fit)]
                candidates = {name: [tree.replaced(path, one[name]) for tree, one in zip(trees, by_fit, strict=True)] for name in on_offer}
                missed = {name: errors(nodes) for name, nodes in candidates.items()}
                for form in forms:
                    if form not in missed:
                        continue
                    against = {"unsplit": unsplit, "so far": so_far, **{name: e for name, e in missed.items() if name != form}}
                    held = {name: _z(missed[form], other) for name, other in against.items()}
                    entry: dict[str, Any] = {
                        "at": list(path), "first": list(first), "second": list(second), "form": form,
                        "ratio": _rms(missed[form]) / (_rms(so_far) or 1.0), "worst_z": max(held.values()),
                        "worst_against": max(held, key=lambda name: held[name])}
                    seen.append(entry)
                    rank = (_COST[form], entry["ratio"])
                    if entry["worst_z"] <= z_refused and (best is None or rank < best[0]):
                        best = (rank, entry, candidates[form], missed[form])
        if every_candidate:
            report.setdefault("candidates", []).append(seen)
        if best is None:
            break
        _, entry, trees, so_far = best
        taken.append((tuple(entry["at"]), tuple(entry["first"]), tuple(entry["second"]), entry["form"]))
        report["steps"].append(entry)

    everything = np.arange(len(y))
    on = everything if len(y) <= grid else np.sort(rng.choice(everything, size=grid, replace=False))
    final = _Fit(surface, unit, y, everything, everything, on, seed + folds)
    tree = final.root
    for path, first, second, form in taken:
        node = final.forms(tree.at(path), first, second).get(form)
        if node is None:           # the form held on every fold but cannot be built on all rows together: stop here
            report["stopped_at"] = len(report["steps"])
            break
        tree = tree.replaced(path, node)
    shape = tree.shape()
    # a table left whole is handed on as it came; a part of a split table is the smooth model's part, there being no other
    tables: dict[Path, np.ndarray] = {(): y.copy()}
    if shape.form != LEAF:
        tables = {path: final.value(tree.at(path)) for path, _ in shape.leaves()}
    constants: dict[Path, float] = {}

    def gather(node: _Node, at: Path) -> None:
        if node.form == PRODUCT_AND_CONSTANT:
            constants[at] = node.constant
        for i, inner in enumerate(node.parts):
            gather(inner, (*at, i))

    gather(tree, ())
    report["shape"] = shape.text()
    report["widest"] = shape.widest()
    report["parts"] = len(shape.leaves())
    report["shaped_error_of_spread"] = _rms(so_far) / spread
    return Reading(shape, rows, tables, constants, report)
