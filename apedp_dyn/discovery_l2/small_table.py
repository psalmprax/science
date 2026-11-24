"""A reading for tables of tens of rows: the plainest law the table cannot tell from the best, and whether it holds up.

Status: experimental

**Not in force; development only.** Nothing calls it but its tests and
``scripts/run_small_tables.py``. Putting it in force needs a blind test and the
owner's word (AGENTS.md 1.5).

**Why.** The route in force was measured on 700 rows a table, and the product
refuses fewer than 24. The tables laws were found in are smaller: Kepler had
six planets, Langmuir ten pressures. On those the route's way of confirming
(set 30 % of the rows aside) leaves two or three rows to confirm on, and its
choice among forms follows the best fit, which among thousands of forms tried
on ten rows is often an odd one (``benchmarks/real_data/dev_first_pass_alone.json``).

**What this does instead.**

1. *Candidates* from several sources, each a form whose constants are refitted
   whenever rows change: the exhaustive first pass; power laws fitted in
   logarithms, with the exponents free and with each rounded to the nearest
   simple fraction; lines and parabolas; plain shapes with one constant inside
   (a saturating ratio, an exponential, a logarithm) alone and, across inputs,
   multiplied or added (with three inputs only one of them carries a shape); the memory of confirmed laws in force; and a search
   stage if the caller hands one over. Two kinds of candidate are left out
   before any scoring: a form with more than one constant per
   ``ROWS_PER_CONSTANT`` rows, and a form with one transcendental function
   inside another (a sine of an exponential), which no table of ten rows can
   support and no reader would take for a law.
2. *What is known beforehand* (``known_beforehand``): a form shown to contradict
   a statement made before the table was read is set aside.
3. *Leaving rows out.* Errors are absolute or relative to each measurement
   (``noise``; by default relative when the target keeps one sign and spans
   more than a factor of ten, as a period from 88 days to 165 years does).
   Every form is scored by its error on rows it was not fitted to (each row left out in turn; ten folds above ``LEAVE_ONE_OUT_UP_TO``
   rows). A form that follows the scatter of a few rows predicts the others worse.
4. *The plainest within one standard error.* The forms whose left-out error is
   within one standard error of the best are a class the table cannot tell
   apart; the reading's law is the plainest of them, by description length
   (``recursion.mdl``). This is the one-standard-error rule of Breiman et al.
5. *Does the choice hold up?* The whole choice is repeated with each row
   removed. The reading is **stable** if the same form is chosen every time.

**What it reports.** A law is *reported* only if the choice is stable and its
error on left-out rows reaches ``DISCOVERY_ACCURACY_R2``, the line that defines
a wrong law (``apedp_dyn.conf.discovery_checks``). Otherwise the law is
*tentative* and the record says which of the two failed. Scatter keeps a noisy
table below that line however right its law is, so on most real tables the
reading is tentative: it names the plainest form the table supports and the
others it cannot tell from it, and claims nothing more.

**Not measured:** this module has development numbers only
(``scripts/run_small_tables.py``). The constants below are development constants
of this module, not thresholds of the referee.
"""
from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from apedp_dyn.conf.discovery_checks import DISCOVERY_ACCURACY_R2
from apedp_dyn.discovery_l2 import exhaustive_search as es
from apedp_dyn.discovery_l2 import known_beforehand
from apedp_dyn.recursion import mdl

TARGET = "__target__"
MIN_ROWS = 6                 # fewer rows than this and nothing is read
FIRST_PASS_KEEP = 30         # forms taken from the first pass, best description length first
LEAVE_ONE_OUT_UP_TO = 40     # rows; above this, ten folds
FOLDS = 10
STABILITY_FORMS = 8          # forms that take part when the choice is repeated with a row removed
EXPONENT_DENOMINATORS = (1, 2, 3)   # a fitted exponent is rounded to the nearest fraction with one of these below the line
MEMORY_SECONDS = 5.0
SHOWN = 5                    # members of the class listed in the record
ROWS_PER_CONSTANT = 3        # a form may have one fitted constant for every this many rows
MAX_NESTING = 1              # transcendental functions inside one another: exp(-c*x) is 1, sin(exp(x)) is 2
RELATIVE_SPAN = 10.0         # noise="auto": relative errors if the target keeps its sign and spans more than this factor
SHAPE_STARTS = (1.0, 0.3, 3.0, -1.0, -0.3)   # an inner constant's starting values, in units of 1 / the input's median size
TRANSCENDENTAL = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.tanh, sp.sinh, sp.cosh, sp.asin, sp.acos, sp.atan)
FITTED_EXPONENTS = "power law, exponents fitted"
BAD = 1e6
ROUNDING = 1e-10             # of the target's size: an error below this is rounding, not misfit (as in the first pass)


@dataclass
class Form:
    """A law's shape: its constants are parameters, refitted whenever the rows change."""

    key: str                       #: the shape with every parameter written ``C``
    expr: sp.Expr                  #: in the inputs and ``params``
    params: list[sp.Symbol]
    start: np.ndarray              #: the constants as proposed
    source: str
    bits: float                    #: description length of the shape, constants included
    fn: Any = field(repr=False, default=None)
    columns: Any = field(repr=False, default=None)   #: where the form is linear in its constants: its columns, for exact fits

    def written(self, theta: np.ndarray) -> sp.Expr:
        return self.expr.xreplace({p: sp.Float(float(f"{v:.8g}")) for p, v in zip(self.params, theta, strict=True)})


def form_of(law: str | sp.Expr, names: Sequence[str], source: str, *, free_exponents: bool = False) -> Form | None:
    """A candidate as a form. Its floats become parameters; exponents stay as written unless ``free_exponents``."""
    symbols = {n: sp.Symbol(n) for n in names}
    try:
        expr = sp.sympify(law, locals=symbols) if isinstance(law, str) else law
    except (sp.SympifyError, TypeError, SyntaxError, ValueError, RecursionError):
        return None
    if not expr.free_symbols or not expr.free_symbols <= set(symbols.values()):
        return None
    params: list[sp.Symbol] = []
    values: list[float] = []

    def shape(node: sp.Expr, in_exponent: bool) -> sp.Expr:
        """Each float met becomes a constant of its own (two that happen to be equal are still two)."""
        if isinstance(node, sp.Float):
            if in_exponent and not free_exponents:
                return node
            params.append(sp.Symbol(f"_c{len(params)}"))
            values.append(float(node))
            return params[-1]
        if not node.args:
            return node
        if node.is_Pow:
            return sp.Pow(shape(node.base, in_exponent), shape(node.exp, True))
        return node.func(*[shape(arg, in_exponent) for arg in node.args])

    shaped = shape(expr, False)
    if not params:                                   # a bare shape: it still needs a scale to be a law
        params, values = [sp.Symbol("_c0")], [1.0]
        shaped = params[0] * shaped
    floats = values
    key = str(shaped.xreplace(dict.fromkeys(params, sp.Symbol("C"))))
    try:
        fn = sp.lambdify([*params, *symbols.values()], shaped, "numpy")
        bits = mdl.law_bits(shaped.xreplace(dict.fromkeys(params, sp.Float(1.5))), len(names))
        slopes = [sp.diff(shaped, p) for p in params]
        linear = (all(not (slope.free_symbols & set(params)) for slope in slopes)
                  and sp.expand(shaped - sum(p * slope for p, slope in zip(params, slopes, strict=True))) == 0)
        columns = sp.lambdify(list(symbols.values()), slopes, "numpy") if linear else None
    except (TypeError, ValueError, SyntaxError, RecursionError):
        return None
    return Form(key, shaped, params, np.array([float(f) for f in floats]), source, bits, fn, columns)


def _design(form: Form, X: np.ndarray) -> np.ndarray | None:
    with np.errstate(all="ignore"):
        design = np.column_stack([np.asarray(column, dtype=complex) * np.ones(len(X)) for column in form.columns(*X.T)])
    if not np.all(np.isfinite(design.real)) or np.any(np.abs(design.imag) > 1e-12):
        return None
    real: np.ndarray = design.real
    return real


def nesting(expr: sp.Expr) -> int:
    """How deep transcendental functions sit inside one another."""
    inner = max((nesting(arg) for arg in expr.args), default=0)
    return inner + int(isinstance(expr, TRANSCENDENTAL))


def weights_of(y: np.ndarray, noise: str) -> np.ndarray:
    """One over each row's error size: ones for absolute errors, 1/|y| for relative ones."""
    if noise not in ("absolute", "relative", "auto"):
        raise ValueError("noise is 'absolute', 'relative' or 'auto'")
    size = np.abs(y)
    one_sign = bool(np.all(y > 0) or np.all(y < 0))
    relative = noise == "relative" or (noise == "auto" and one_sign and float(np.max(size)) > RELATIVE_SPAN * float(np.min(size)))
    if relative and not np.all(size > 0):
        raise ValueError("relative errors need a target that is never zero")
    out: np.ndarray = 1.0 / size if relative else np.ones(len(y))
    return out


def _residual(form: Form, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
    def resid(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            value = np.asarray(form.fn(*theta, *X.T), dtype=complex) * np.ones(len(y))
        ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
        out: np.ndarray = np.where(ok, (value.real - y) * w, BAD)
        return out
    return resid


def fit(form: Form, X: np.ndarray, y: np.ndarray, start: np.ndarray | None = None, w: np.ndarray | None = None
        ) -> np.ndarray | None:
    """The form's constants by (weighted) least squares on these rows, from ``start``; ``None`` if it cannot be fitted."""
    if len(y) < len(form.params):
        return None
    w = np.ones(len(y)) if w is None else w
    if form.columns is not None:                     # linear in its constants: the exact least-squares answer
        design = _design(form, X)
        if design is None:
            return None
        exact, _res, rank, _sv = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        return exact if rank == design.shape[1] and np.all(np.isfinite(exact)) else None
    spread = float(np.std(y * w)) or float(np.max(np.abs(y * w))) or 1.0
    resid = _residual(form, X, y, w / spread)
    begin = form.start if start is None else start
    try:
        got = least_squares(resid, begin, max_nfev=60 * len(begin))
    except (ValueError, TypeError, FloatingPointError):
        return None
    if not np.all(np.isfinite(got.x)) or np.any(resid(got.x) >= BAD):
        return None
    theta: np.ndarray = got.x
    return theta


def predict(form: Form, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        value = np.asarray(form.fn(*theta, *X.T), dtype=complex) * np.ones(len(X))
    out: np.ndarray = np.where(np.abs(value.imag) < 1e-12, value.real, np.nan)
    return out


def folds_of(n: int, seed: int) -> list[np.ndarray]:
    """Each row alone up to ``LEAVE_ONE_OUT_UP_TO`` rows; ten folds above."""
    if n <= LEAVE_ONE_OUT_UP_TO:
        return [np.array([i]) for i in range(n)]
    order = np.random.default_rng(seed).permutation(n)
    return [order[k::FOLDS] for k in range(FOLDS)]


def left_out_errors(form: Form, X: np.ndarray, y: np.ndarray, theta: np.ndarray, seed: int, w: np.ndarray | None = None
                    ) -> np.ndarray | None:
    """Each row's squared (weighted) error when the form is fitted without it. ``None`` if a refit fails."""
    w = np.ones(len(y)) if w is None else w
    errors = np.empty(len(y))
    keep: np.ndarray = np.ones(len(y), dtype=bool)
    if form.columns is not None and len(y) <= LEAVE_ONE_OUT_UP_TO:
        design = _design(form, X)                    # each row left out in one step: residual / (1 - leverage)
        if design is not None:
            weighted = design * w[:, None]
            leverage = np.einsum("ij,ji->i", weighted, np.linalg.pinv(weighted))
            if np.all(leverage < 1 - 1e-9):
                exact: np.ndarray = ((y - design @ theta) * w / (1 - leverage)) ** 2
                return exact
            return None                              # a row the others cannot predict at all
    for held in folds_of(len(y), seed):
        keep[:] = True
        keep[held] = False
        refit = fit(form, X[keep], y[keep], theta, w[keep])
        if refit is None:
            return None
        guess = predict(form, refit, X[held])
        if not np.all(np.isfinite(guess)):
            return None
        errors[held] = ((guess - y[held]) * w[held]) ** 2
    return errors


@dataclass
class Scored:
    form: Form
    theta: np.ndarray
    errors: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors))

    @property
    def standard_error(self) -> float:
        return float(np.std(self.errors, ddof=1) / np.sqrt(len(self.errors))) if len(self.errors) > 1 else 0.0


def choose(scored: Sequence[Scored]) -> tuple[Scored, list[Scored]]:
    """The plainest form within one standard error of the best, and the class it was chosen from (plainest first)."""
    best = min(scored, key=lambda s: s.mean)
    limit = best.mean + best.standard_error
    members = sorted((s for s in scored if s.mean <= limit), key=lambda s: (s.form.bits, s.mean))
    return members[0], members


def score_all(forms: Sequence[Form], X: np.ndarray, y: np.ndarray, seed: int, deadline: float | None = None,
              w: np.ndarray | None = None) -> list[Scored]:
    out = []
    weighted = y if w is None else y * w
    # two forms that both fit to rounding are equally exact; without this floor the plainer loses on the sixteenth digit
    floor = (ROUNDING * float(np.sqrt(np.mean(weighted ** 2)))) ** 2
    for form in forms:
        if deadline is not None and time.monotonic() > deadline:
            break
        theta = fit(form, X, y, None, w)
        if theta is None:
            continue
        errors = left_out_errors(form, X, y, theta, seed, w)
        if errors is not None:
            out.append(Scored(form, theta, np.maximum(errors, floor)))
    return out


# ------------------------------------------------------------------ candidates
def _power_laws(names: Sequence[str], X: np.ndarray, y: np.ndarray, units: Mapping[str, str] | None) -> list[tuple[str, str]]:
    """Power laws fitted in logarithms: exponents free, and each rounded to the nearest simple fraction.

    Two inputs that carry the same unit may be added, so their sum is tried in place of the pair.
    """
    if np.any(X <= 0) or not (np.all(y > 0) or np.all(y < 0)):
        return []
    out: list[tuple[str, str]] = []
    settings: list[list[str]] = [list(names)]
    if units:
        for a, b in itertools.combinations(names, 2):
            if units.get(a) and units.get(a) == units.get(b):
                settings.append([f"({a} + {b})", *(n for n in names if n not in (a, b))])
    columns = {n: X[:, i] for i, n in enumerate(names)}
    for factors in settings:
        values = np.column_stack([np.asarray(sp.lambdify([sp.Symbol(n) for n in names], sp.sympify(f), "numpy")(*X.T), dtype=float)
                                  * np.ones(len(y)) if "(" in f else columns[f] for f in factors])
        if np.any(values <= 0) or len(y) <= len(factors) + 1:
            continue
        design = np.column_stack([np.log(values), np.ones(len(y))])
        coef, _res, rank, _sv = np.linalg.lstsq(design, np.log(np.abs(y)), rcond=None)
        if rank < design.shape[1]:
            continue
        sign = "" if y[0] > 0 else "-"
        scale = float(np.exp(coef[-1]))
        free = " * ".join(f"{f}**{float(e)!r}" for f, e in zip(factors, coef[:-1], strict=True))
        out.append((f"{sign}{scale!r} * {free}", FITTED_EXPONENTS))
        rounded = []
        for exponent in coef[:-1]:
            options = [Fraction(round(float(exponent) * d), d) for d in EXPONENT_DENOMINATORS]
            rounded.append(min(options, key=lambda q: (abs(float(q) - float(exponent)), q.denominator)))
        if all(q != 0 for q in rounded):              # whether the rounding costs anything is for the left-out rows to say
            plain = " * ".join(f"{f}**Rational({q.numerator}, {q.denominator})" for f, q in zip(factors, rounded, strict=True))
            out.append((f"{sign}{scale!r} * {plain}", "power law, exponents rounded to simple fractions"))
    return out


SHAPES = ("{x}", "{x}/({c} + {x})", "1/({c} + {x})", "exp({c}*{x})", "log(1 + {c}*{x})")


def _shapes(names: Sequence[str], X: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[tuple[str, str]]:
    """Plain shapes with one constant inside, one per input, multiplied together or (two inputs) added; fitted from several starts."""
    out: list[tuple[str, str]] = []
    sizes = [float(np.median(np.abs(X[:, i]))) or 1.0 for i in range(len(names))]
    joins = ("*", "+") if len(names) == 2 else ("*",)
    starts = SHAPE_STARTS if len(names) == 1 else SHAPE_STARTS[:3]
    shaped_inputs = 2 if len(names) <= 2 else 1       # with three inputs, one of them carries a shape: the count stays small
    for choice in itertools.product(range(len(SHAPES)), repeat=len(names)):
        if all(k == 0 for k in choice) and len(names) > 1:
            continue                                  # a bare product or sum of the inputs: the power laws and lines cover it
        if sum(k != 0 for k in choice) > shaped_inputs:
            continue
        for join in joins:
            for offset in (False, True):
                parts, inner = [], []
                for i, (name, k) in enumerate(zip(names, choice, strict=True)):
                    constant = f"_k{i}"
                    parts.append(SHAPES[k].format(x=name, c=constant))
                    if "{c}" in SHAPES[k]:
                        inner.append((constant, i))
                if not inner and not offset:
                    continue
                scales = ["_a0"] if join == "*" else [f"_a{i}" for i in range(len(parts))]
                body = f"_a0*{'*'.join(f'({p})' for p in parts)}" if join == "*" else " + ".join(
                    f"_a{i}*({p})" for i, p in enumerate(parts))
                text = body + (" + _b" if offset else "")
                symbols = [*(c for c, _ in inner), *scales, *(["_b"] if offset else [])]
                if len(symbols) > max(len(y) // ROWS_PER_CONSTANT, 1):
                    continue
                try:
                    expr = sp.sympify(text, locals={n: sp.Symbol(n) for n in [*names, *symbols]})
                    fn = sp.lambdify([*(sp.Symbol(s) for s in symbols), *(sp.Symbol(n) for n in names)], expr, "numpy")
                except (sp.SympifyError, TypeError, SyntaxError, ValueError):
                    continue

                def resid(theta: np.ndarray, fn: Any = fn) -> np.ndarray:
                    with np.errstate(all="ignore"):
                        value = np.asarray(fn(*theta, *X.T), dtype=complex) * np.ones(len(y))
                    ok = np.isfinite(value.real) & (np.abs(value.imag) < 1e-12)
                    res: np.ndarray = np.where(ok, (value.real - y) * w, BAD)
                    return res

                best: tuple[float, np.ndarray] | None = None
                for first in starts:
                    start = np.array([*(first / sizes[i] if "exp" in SHAPES[choice[i]] or "log" in SHAPES[choice[i]]
                                        else abs(first) * sizes[i] for _, i in inner),
                                      *([1.0] * len(scales)), *([0.0] if offset else [])])
                    try:
                        got = least_squares(resid, start, max_nfev=80 * len(start))
                    except (ValueError, TypeError, FloatingPointError):
                        continue
                    final = resid(got.x)
                    if np.any(final >= BAD) or not np.all(np.isfinite(got.x)):
                        continue
                    cost = float(np.sum(final ** 2))
                    if best is None or cost < best[0]:
                        best = (cost, got.x)
                if best is None:
                    continue
                written = expr.xreplace({sp.Symbol(s): sp.Float(float(v)) for s, v in zip(symbols, best[1], strict=True)})
                out.append((str(written), "plain shapes"))
    return out


def _polynomials(names: Sequence[str]) -> list[tuple[str, str]]:
    line = " + ".join(f"1.0*{n}" for n in names) + " + 1.0"
    out = [(line, "a line")]
    if len(names) == 1:
        out.append((f"1.0*{names[0]}**2 + 1.0*{names[0]} + 1.0", "a parabola"))
        out.append((f"1.0*{names[0]}", "a line through the origin"))
    return out


def candidates(X: Mapping[str, np.ndarray], y: np.ndarray, *, units: Mapping[str, str] | None = None, proposer: Any = None,
               memory: bool = True, memory_seconds: float = MEMORY_SECONDS, w: np.ndarray | None = None
               ) -> tuple[list[Form], dict[str, Any]]:
    """Every source's candidates as forms, each shape once (the first source to give it keeps it)."""
    names = list(X)
    columns = np.column_stack([np.asarray(X[n], dtype=float) for n in names])
    w = np.ones(len(y)) if w is None else w
    found: list[tuple[str, str]] = [*_polynomials(names), *_power_laws(names, columns, y, units)]
    report: dict[str, Any] = {"failures": []}
    try:
        found += _shapes(names, columns, y, w)
    except Exception as exc:  # noqa: BLE001 - one source failing leaves the others; the record says which
        report["failures"].append(f"plain shapes: {type(exc).__name__}: {exc}"[:160])
    try:
        depth = 6 if len(names) == 1 else 5
        ranked = es.search(columns, y, names, max_complexity=depth)[:FIRST_PASS_KEEP]
        found += [(es.model_string(s), "first pass") for s in ranked]
    except Exception as exc:  # noqa: BLE001 - one source failing leaves the others; the record says which
        report["failures"].append(f"first pass: {type(exc).__name__}: {exc}"[:160])
    frame = pd.DataFrame({**{n: columns[:, i] for i, n in enumerate(names)}, TARGET: y})
    stages: list[tuple[str, Any]] = []
    if memory:
        from apedp_dyn.discovery_l2 import memory_stage
        from apedp_dyn.proposers.structure_memory import StructureMemoryProposer

        stages.append(("memory of confirmed laws", StructureMemoryProposer(memory_stage.recorded_memory(), max_proposals=3,
                                                                           seconds=memory_seconds, target_col=TARGET)))
    if proposer is not None:
        stages.append(("search stage", proposer))
    for label, stage in stages:
        try:
            found += [(str(p.formula), label) for p in stage.fit_and_propose(frame, target_col=TARGET)]
        except Exception as exc:  # noqa: BLE001 - one source failing leaves the others; the record says which
            report["failures"].append(f"{label}: {type(exc).__name__}: {exc}"[:160])
    forms: dict[str, Form] = {}
    allowed = max(len(y) // ROWS_PER_CONSTANT, 1)
    report["left_out"] = {"more_constants_than_the_rows_allow": 0, "one_function_inside_another": 0}
    for law, source in found:
        form = form_of(law, names, source, free_exponents=source == FITTED_EXPONENTS)
        if form is None or form.key in forms:
            continue
        if len(form.params) > allowed:
            report["left_out"]["more_constants_than_the_rows_allow"] += 1
        elif nesting(form.expr) > MAX_NESTING:
            report["left_out"]["one_function_inside_another"] += 1
        else:
            forms[form.key] = form
    report["by_source"] = {}
    for form in forms.values():
        report["by_source"][form.source] = report["by_source"].get(form.source, 0) + 1
    return list(forms.values()), report


# ---------------------------------------------------------------------- reading
def _r2(y: np.ndarray, errors: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """One minus the left-out error over the target's spread, both in the errors' own measure."""
    w = np.ones(len(y)) if w is None else w
    centre = float(np.sum(w * w * y) / np.sum(w * w))
    total = float(np.sum((w * (y - centre)) ** 2))
    return None if total <= 0 else 1.0 - float(np.sum(errors)) / total


def _entry(item: Scored, y: np.ndarray, w: np.ndarray | None = None) -> dict[str, Any]:
    score = _r2(y, item.errors, w)
    return {"law": str(item.form.written(item.theta)), "shape": item.form.key, "source": item.form.source,
            "constants": len(item.form.params), "bits": round(item.form.bits, 1),
            "left_out_error": item.mean, "left_out_standard_error": item.standard_error,
            "left_out_r2": None if score is None else round(score, 6)}


def read(X: Mapping[str, Any], y: Any, *, units: Mapping[str, str] | None = None,
         known: Sequence[Mapping[str, Any]] = (), proposer: Any = None, memory: bool = True, seed: int = 0,
         seconds: float = 300.0, noise: str = "auto", sigma: Any = None) -> dict[str, Any]:
    """What a small table supports. Never raises on bad data: the record carries the reason.

    ``sigma`` is each row's stated measurement error, where the table gives one; errors are then counted in units
    of it and ``noise`` is not used. Above ``LEAVE_ONE_OUT_UP_TO`` rows the choice is repeated with each of the ten
    folds removed in turn, not each row.
    """
    started = time.monotonic()
    record: dict[str, Any] = {"action": "not_run", "law": None, "reported": False, "tentative": False, "class": [],
                              "known_beforehand": [dict(s) for s in known]}
    try:
        names = [str(n) for n in X]
        columns = np.column_stack([np.asarray(X[n], dtype=float).ravel() for n in names])
        target = np.asarray(y, dtype=float).ravel()
        keep = np.all(np.isfinite(columns), axis=1) & np.isfinite(target)
        stated = None if sigma is None else np.asarray(sigma, dtype=float).ravel()
        if stated is not None:
            if len(stated) != len(target):
                raise ValueError("one stated error is needed for each row")
            keep &= np.isfinite(stated) & (stated > 0)
            stated = stated[keep]
        columns, target = columns[keep], target[keep]
        record["rows"] = len(target)
        if len(target) < MIN_ROWS:
            raise ValueError(f"{len(target)} usable rows; a reading needs at least {MIN_ROWS}")
        w = weights_of(target, noise) if stated is None else 1.0 / stated
        record["errors"] = "in units of each row's stated error" if stated is not None else (
            "relative" if not np.all(w == 1.0) else "absolute")
        forms, report = candidates({n: columns[:, i] for i, n in enumerate(names)}, target, units=units, proposer=proposer,
                                   memory=memory, w=w)
        record["candidates"] = {"forms": len(forms), **report}
        scored = score_all(forms, columns, target, seed, deadline=started + 0.5 * seconds, w=w)
        record["candidates"]["scored"] = len(scored)
        ranges = {n: (float(np.min(columns[:, i])), float(np.max(columns[:, i]))) for i, n in enumerate(names)}
        size = float(np.max(np.abs(target)))
        set_aside = []
        if known:
            agreeing = []
            for item in scored:
                checks = known_beforehand.check_all(item.form.written(item.theta), names, known, ranges, size)
                if known_beforehand.agrees(checks):
                    agreeing.append(item)
                else:
                    set_aside.append({**_entry(item, target, w), "contradicts": [c for c in checks if c.get("holds") is False]})
            scored = agreeing
        record["set_aside_by_what_is_known"] = {"forms": len(set_aside),
                                                "best": sorted(set_aside, key=lambda e: e["left_out_error"])[:SHOWN]}
        if not scored:
            raise ValueError("no candidate could be fitted with rows left out" if not set_aside
                             else "every candidate contradicts what is known beforehand")
        chosen, members = choose(scored)
        best = min(scored, key=lambda s: s.mean)
        # does the choice hold up with a row removed?
        ranked = sorted(scored, key=lambda s: s.mean)
        taking_part = dict(itertools.islice({s.form.key: s.form for s in [chosen, *members, *ranked]}.items(), STABILITY_FORMS))
        changed: list[dict[str, Any]] = []
        checked = 0
        removals = folds_of(len(target), seed)          # each row in turn; ten folds on a larger table
        for row, removed in enumerate(removals):
            if time.monotonic() > started + seconds:
                break
            rest: np.ndarray = np.ones(len(target), dtype=bool)
            rest[removed] = False
            again = score_all(list(taking_part.values()), columns[rest], target[rest], seed, w=w[rest])
            checked += 1
            if not again:
                changed.append({"row_removed": row, "chosen": None})
                continue
            other, _members = choose(again)
            if other.form.key != chosen.form.key:
                changed.append({"row_removed": row, "chosen": other.form.key})
        stable = checked == len(removals) and not changed
        score = _r2(target, chosen.errors, w)
        accurate = bool(score is not None and score >= DISCOVERY_ACCURACY_R2)
        reasons = []
        if not stable:
            reasons.append("the choice changes when a row is removed" if changed
                           else "the choice could not be repeated with every row removed in the time given")
        if not accurate:
            reasons.append("its error on left-out rows is above the accuracy line")
        record.update({
            "action": "read", "law": str(chosen.form.written(chosen.theta)), "reported": stable and accurate,
            "tentative": not (stable and accurate), "tentative_reasons": reasons, "chosen": _entry(chosen, target, w),
            "best_by_left_out_error": _entry(best, target, w), "class_size": len(members),
            "class": [_entry(m, target, w) for m in members[:SHOWN]],
            "stability": {"stable": stable, "rows_removed": checked, "changed": changed[:SHOWN], "changes": len(changed),
                          "forms_taking_part": len(taking_part)},
            "accuracy": {"left_out_r2": None if score is None else round(score, 6), "line": DISCOVERY_ACCURACY_R2,
                         "reached": accurate},
            "known_checks_on_the_law": known_beforehand.check_all(chosen.form.written(chosen.theta), names, known, ranges, size),
        })
    except Exception as exc:  # noqa: BLE001 - a reading that cannot be made is reported in the record, not raised
        record.update({"action": "not_run", "error": f"{type(exc).__name__}: {exc}"[:300]})
    record["seconds"] = round(time.monotonic() - started, 2)
    return record
