"""APEDP scientific layers: agnostic wrappers for all known equations.

Reuses existing engines (DimensionalAssuranceEngine, BuckinghamPiEngine,
AnalyticalUncertaintyPropagator, PVE, Classification) and fills gaps with
lightweight generic helpers. No domain hardcoding; fusion is one plugin.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)

#: Failures a best-effort wrapper may legitimately hit: a missing optional
#: dependency, malformed input to sympy/numpy, a missing mapping key, or a
#: numeric/operational error. ``numpy.linalg.LinAlgError`` and
#: ``sympy.SympifyError`` subclass ``ValueError``, and ``ZeroDivisionError`` /
#: ``OverflowError`` subclass ``ArithmeticError``. Anything outside this set is
#: a defect and must propagate instead of being reported as a failed probe
#: (mirrors ``gated_assurance._PROBE_ERRORS``; see
#: docs/INDEPENDENT_REVIEW_2026-09-23.md §4.2).
_PROBE_ERRORS = (RuntimeError, TypeError, ValueError, ArithmeticError,
                 KeyError, ImportError)


# 7-level validation hierarchy (reviewer item 8)
class ValidationLevel(Enum):
    L1_SYNTAX = auto()
    L2_MATH = auto()
    L3_DIMENSIONAL = auto()
    L4_NUMERICAL = auto()
    L5_PHYSICAL = auto()
    L6_EMPIRICAL = auto()
    L7_PREDICTIVE = auto()


# Relationship kind (reviewer item 18)
class RelationshipKind(Enum):
    LAW = auto()
    DERIVATION = auto()
    MODEL = auto()
    EMPIRICAL = auto()
    HEURISTIC = auto()
    FIGURE_OF_MERIT = auto()
    IDENTITY = auto()
    CANDIDATE = auto()


# Causal tag (reviewer item 11)
class RelationshipType(Enum):
    ALGEBRAIC = auto()
    STATISTICAL = auto()
    EMPIRICAL = auto()
    MECHANISTIC = auto()
    CAUSAL = auto()
    IDENTITY = auto()


@dataclass
class CandidateRecord:
    expression: str
    variables: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    envelope: dict = field(default_factory=dict)
    kind: RelationshipKind = RelationshipKind.CANDIDATE
    rel_type: RelationshipType = RelationshipType.ALGEBRAIC
    level: ValidationLevel = ValidationLevel.L1_SYNTAX
    provenance: dict = field(default_factory=dict)


def symmetry_check(expression: str, symbol: str, transform: str = "reflect") -> str:
    """Generic symmetry test: reflect / translate / scale invariance probe.

    Returns PASS / FAIL / UNKNOWN. Agnostic to domain.
    """
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expression)
        x = sp.Symbol(symbol)
        if x not in expr.free_symbols:
            return "UNKNOWN"
        if transform == "reflect":
            mirrored = expr.subs(x, -x)
        elif transform == "translate":
            a = sp.Symbol("_shift_tmp")
            mirrored = expr.subs(x, x + a)
        elif transform == "scale":
            mirrored = expr.subs(x, 2 * x)
        else:
            return "UNKNOWN"
        return "PASS" if bool(sp.simplify(expr - mirrored) == 0) else "FAIL"
    except _PROBE_ERRORS as exc:
        logger.debug("symmetry check inconclusive for %s: %s", expression, exc)
        return "UNKNOWN"


def pi_groups(variables: dict) -> list:
    """Agnostic Buckingham-Pi wrapper around core BuckinghamPiEngine."""
    try:
        from apedp.core import BuckinghamPiEngine
        return BuckinghamPiEngine.compute_pi_groups(variables)
    except _PROBE_ERRORS as exc:
        logger.debug("Pi-group computation unavailable: %s", exc)
        return []


def propagate_uncertainty(expr_str: str, values: dict, uncertainties: dict) -> dict:
    """Agnostic uncertainty wrapper around metrology AnalyticalUncertaintyPropagator."""
    try:
        from apedp.gated_assurance import safe_parse_expr
        from apedp.metrology import AnalyticalUncertaintyPropagator
        expr = safe_parse_expr(expr_str)
        rep = AnalyticalUncertaintyPropagator.propagate_uncertainty(
            expr, dict(values), dict(uncertainties)
        )
        return {"value": rep.evaluated_value,
                "uncertainty": rep.combined_uncertainty}
    except _PROBE_ERRORS as e:
        return {"value": None, "uncertainty": None, "error": str(e)}


def sensitivity_indices(expr_str: str, values: dict) -> dict:
    """Normalized sensitivity S_i = x_i/y * dy/dx_i. Agnostic."""
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
        y = float(expr.subs({sp.Symbol(k): v for k, v in values.items()}))
        out = {}
        for k, v in values.items():
            d = float(sp.diff(expr, sp.Symbol(k)).subs(
                {sp.Symbol(k2): v2 for k2, v2 in values.items()}))
            out[k] = (v / y * d) if y != 0 else 0.0
        return out
    except _PROBE_ERRORS as e:
        return {"error": str(e)}


def identifiability(expr_str: str, values: dict) -> dict:
    """Jacobian-rank identifiability via SVD. Agnostic to domain."""
    try:
        import numpy as np

        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
        syms = sorted(expr.free_symbols, key=str)
        subs = {s: values.get(str(s), 1.0) for s in syms}
        jac = [float(sp.diff(expr, s).subs(subs)) for s in syms]
        _u, svals, _ = np.linalg.svd([jac], full_matrices=False)
        rank = int((svals > 1e-10).sum())
        return {"rank": rank, "n_params": len(syms),
                "identifiable": rank == len(syms)}
    except _PROBE_ERRORS as e:
        return {"error": str(e)}


def out_of_sample(eval_fn, x_train, y_train, x_test, y_test) -> dict:
    """Generic train/test predictive check. eval_fn: X -> Y."""
    try:
        import numpy as np
        tr = np.asarray(eval_fn(np.asarray(x_train))).ravel()
        te = np.asarray(eval_fn(np.asarray(x_test))).ravel()
        yt_tr = np.asarray(y_train).ravel()
        yt_te = np.asarray(y_test).ravel()
        return {
            "train_mae": float(np.mean(abs(tr - yt_tr))),
            "test_mae": float(np.mean(abs(te - yt_te))),
        }
    except _PROBE_ERRORS as e:
        return {"error": str(e)}


@dataclass
class ProvenanceNode:
    kind: str  # entity/activity/agent
    label: str
    detail: dict = field(default_factory=dict)


@dataclass
class ProvenanceGraph:
    """W3C PROV-lite evidence chain."""
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    def add(self, kind: str, label: str, detail: dict | None = None) -> None:
        self.nodes.append(ProvenanceNode(kind, label, detail or {}))

    def link(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))


@dataclass
class FailureMemory:
    """Negative-knowledge base: rejected candidates with reasons."""
    records: list = field(default_factory=list)

    def add(self, expr: str, reason: str, meta: dict | None = None) -> None:
        self.records.append({"expression": expr, "reason": reason,
                             "meta": meta or {}})

    def count(self) -> int:
        return len(self.records)


@dataclass
class ReproducibilityBundle:
    """FAIR carrier for a discovery."""
    discovery_id: str = ""
    timestamp: str = ""
    software_version: str = ""
    corpus_version: str = ""
    dataset_hash: str = ""
    input_hash: str = ""
    random_seed: str = ""
    search_space: str = ""
    hyperparameters: dict = field(default_factory=dict)
    candidate_expression: str = ""
    validation_results: dict = field(default_factory=dict)


def scale_ratio(expr_a: str, expr_b: str, values: dict | None = None,
                small_threshold: float = 0.01) -> dict:
    """Dimensionless ratio R = A/B with a regime note. Agnostic.

    A small ratio explains why one term is a correction to the other in a
    given regime (e.g. E_kin/E_rest = v^2/2c^2 << 1 for v << c justifies the
    classical approximation). Returns ratio string, optional numeric value,
    and regime in {small-correction, comparable, dominant, unknown}.
    """
    try:
        from apedp.gated_assurance import safe_parse_expr
        ra = safe_parse_expr(expr_a)
        rb = safe_parse_expr(expr_b)
        ratio = sp.simplify(ra / rb)
        out: dict = {"ratio": str(ratio), "regime": "unknown",
                     "numeric": None}
        if values:
            subs = {sp.Symbol(k): v for k, v in values.items()}
            num = float(ratio.subs(subs).evalf())
            out["numeric"] = num
            a = abs(num)
            if a < small_threshold:
                out["regime"] = "small-correction"
            elif a <= 1.0 / small_threshold:
                out["regime"] = "comparable"
            else:
                out["regime"] = "dominant"
        return out
    except _PROBE_ERRORS as e:
        return {"ratio": "", "regime": "unknown", "numeric": None,
                "error": str(e)}


def domain_audit(expr_str: str, var_ranges: dict,
                 constraints: dict | None = None,
                 grid: int = 25) -> dict:
    """Scan a candidate over its declared domain for physical violations.

    constraints maps output/restriction names to bounds, e.g.
    {"output>0": True} or {"output_min": 0.0}. Returns PASS / CONDITIONAL /
    FAIL plus the violating points. Averages never excuse local impossibility:
    one violating grid point fails the domain.
    """
    import itertools
    import math
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
        syms = sorted(expr.free_symbols, key=str)
        axes = []
        for s in syms:
            name = str(s)
            lo, hi = var_ranges.get(name, (0.5, 2.0))
            axes.append([lo + (hi - lo) * i / max(grid - 1, 1)
                         for i in range(grid)])
        out_min = (constraints or {}).get("output_min", None)
        worst = None
        n_bad = 0
        n_total = 0
        for combo in itertools.product(*axes):
            subs = dict(zip(syms, combo, strict=False))
            try:
                val = float(expr.subs(subs).evalf())
            except (TypeError, ValueError, ArithmeticError, AttributeError):
                # Counted as a bad grid point below; the failure is disclosed
                # through n_bad/n_total rather than swallowed.
                n_bad += 1
                n_total += 1
                continue
            n_total += 1
            if math.isnan(val) or math.isinf(val):
                n_bad += 1
                if worst is None:
                    worst = (combo, val)
                continue
            if out_min is not None and val < out_min:
                n_bad += 1
                if worst is None:
                    worst = (combo, val)
        if n_total == 0:
            return {"status": "FAIL", "reason": "nothing evaluable",
                    "n_bad": 0, "n_total": 0}
        if n_bad == 0:
            return {"status": "PASS", "reason": "",
                    "n_bad": 0, "n_total": n_total}
        frac = n_bad / n_total
        return {"status": "CONDITIONAL" if frac < 0.5 else "FAIL",
                "reason": f"{n_bad}/{n_total} grid points violate constraints",
                "worst": (tuple(float(v) for v in worst[0]),
                          worst[1]) if worst else None,
                "n_bad": n_bad, "n_total": n_total}
    except _PROBE_ERRORS as e:
        return {"status": "FAIL", "reason": f"audit error: {e}",
                "n_bad": 0, "n_total": 0}


def interior_pole_scan(expr, observed: dict, *,
                       grid_points: int = 2001,
                       pad_fraction: float = 0.02,
                       spike_factor: float = 10.0,
                       jump_factor: float = 100.0,
                       jump_floor: float = 1e-9) -> str | None:
    """Flag candidate forms carrying a pole inside the observed domain.

    A form like ``1/(x - 2.5001)`` is well behaved at every sampled input
    {1, 2, 3, 4, 5} while being singular inside the range it is asked to
    model. Evaluating only where data exists cannot see that, and the fit then
    looks excellent. So the form is swept on a dense grid spanning the
    observed range (padded so a pole just outside is still caught).

    The comparison is scale-free: the reference is the peak the form already
    reached at the sampled inputs, so a smooth candidate scores about 1 and a
    straddled pole scores orders of magnitude higher.

    Returns a reason string when the form fails, else ``None``. A form with no
    evaluable free symbol is not this function's business and returns ``None``.
    """
    if expr is None:
        return None
    names = [str(sym) for sym in expr.free_symbols]
    if not names:
        return None
    # Only variables that actually have observations bound a domain. A free
    # parameter symbol (c0, __target__) has no data, so it is held at a neutral
    # value rather than being treated as an axis; columns of unequal length are
    # truncated to the common prefix rather than being stacked.
    columns: dict[str, np.ndarray] = {}
    for name in names:
        raw = observed.get(name)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size:
            columns[name] = arr
    if not columns:
        return None
    length = min(arr.size for arr in columns.values())
    columns = {name: arr[:length] for name, arr in columns.items()}
    axis_names = list(columns)
    finite_rows = np.logical_and.reduce(
        [np.isfinite(arr) for arr in columns.values()]
    )
    if not finite_rows.any():
        return "no finite input rows to bound the domain"
    lows = [float(np.min(columns[name][finite_rows])) for name in axis_names]
    highs = [float(np.max(columns[name][finite_rows])) for name in axis_names]
    spans = [high - low for low, high in zip(lows, highs, strict=True) if high > low]
    pad = pad_fraction * (max(spans) if spans else 1.0)
    axis = np.linspace(min(lows) - pad, max(highs) + pad, grid_points)
    base = {name: float(np.mean(columns[name][finite_rows]))
            for name in axis_names}
    # Symbols with no observations are held at 1.0: neutral, and the sweep is
    # about the geometry of the observed domain, not about parameter values.
    held = {name: 1.0 for name in names if name not in columns}

    def _call(cols_by_name: dict[str, np.ndarray]) -> np.ndarray:
        ordered = [cols_by_name.get(n, held.get(n, 1.0)) for n in names]
        return np.atleast_1d(
            np.asarray(func(*ordered), dtype=float)
        ).reshape(-1)

    try:
        func = sp.lambdify(names, expr, modules="numpy")
        if len(axis_names) == 1:
            only = axis_names[0]
            sweeps = [_call({only: axis})]
        else:
            parts = []
            for swept_name in axis_names:
                cols = {name: base[name] for name in axis_names}
                cols[swept_name] = axis
                parts.append(_call(cols))
            sweeps = parts
    except _PROBE_ERRORS:
        # Not symbolically evaluable on a sweep; other gates still apply.
        return None
    # Each axis sweep is kept separate. Concatenating them and taking
    # successive differences would read the junction between two sweeps as an
    # enormous step and condemn every smooth multi-variable form.
    sweeps = [np.atleast_1d(part).reshape(-1) for part in sweeps]
    sweeps = [part for part in sweeps if part.size]
    if not sweeps:
        return None
    swept = np.concatenate(sweeps)
    if not np.all(np.isfinite(swept)):
        return "form is non-finite inside the observed domain (interior pole)"
    try:
        sampled = _call({name: columns[name] for name in axis_names})
    except _PROBE_ERRORS:
        sampled = None
    if sampled is not None:
        sampled = np.atleast_1d(sampled)
        finite_sampled = sampled[np.isfinite(sampled)]
        reference = (float(np.max(np.abs(finite_sampled)))
                     if finite_sampled.size else 0.0)
    else:
        reference = 0.0
    reference = max(reference, 1e-12)
    peak = float(np.max(np.abs(swept)))
    if peak > spike_factor * reference:
        return (
            f"interior singularity: peak |f| = {peak:.3g} on a dense sweep of the "
            f"observed domain exceeds {spike_factor:g}x the {reference:.3g} reached "
            f"at the sampled inputs, which do not resolve the pole"
        )
    # A pole small enough to hide inside a large background is invisible to the
    # peak test: 0.001/(x - 1.5) added to 2x**2 + 1 never exceeds the
    # polynomial. What it does produce is one enormous step in an otherwise
    # smooth sequence, so the local jump is compared against the typical step.
    for part in sweeps:
        if part.size <= 4:
            continue
        steps = np.abs(np.diff(part))
        typical = float(np.median(steps))
        largest = float(np.max(steps))
        if largest > jump_factor * typical and largest > jump_floor:
            return (
                f"interior singularity: the form jumps by {largest:.3g} between "
                f"adjacent sweep points while the typical step is {typical:.3g}, "
                f"a local blow-up the sampled inputs do not resolve"
            )
    return None


def extrapolation_audit(expr_str: str, var_ranges: dict,
                        y_scale: float, factor: float = 10.0,
                        tol: float = 1e4, grid: int = 25) -> dict:
    """Flag super-polynomial blowup just outside the data range.

    Evaluates the expression on [lo/factor, hi*factor] per variable (log
    grid for positive ranges, linear extension otherwise), holding other
    variables at mid-range. FAILS on non-finite values or |pred] exceeding
    ``tol`` times the data scale -- this kills coincidental in-window fits
    like exp(+c/x) for decaying data (ratio ~1e8 at 0.1*xmin) while true
    power laws pass (1/x^2 gives ratio exactly 100). Parse failures are
    CONDITIONAL (never reject on missing evidence). Pure function.
    """
    import math
    try:
        from apedp.gated_assurance import safe_parse_expr
        expr = safe_parse_expr(expr_str)
        syms = sorted(expr.free_symbols, key=str)
        if not syms or not math.isfinite(y_scale) or y_scale <= 0:
            return {"status": "CONDITIONAL",
                    "reason": "no symbols or non-positive y scale"}
        import numpy as np
        axes = {}
        for s in syms:
            lo, hi = var_ranges.get(str(s), (0.5, 2.0))
            lo, hi = float(lo), float(hi)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                return {"status": "CONDITIONAL",
                        "reason": f"bad range for {s}"}
            if lo > 0:
                axes[str(s)] = np.logspace(np.log10(lo / factor),
                                           np.log10(hi * factor), grid)
            else:
                span = hi - lo
                axes[str(s)] = np.linspace(lo - span * (factor - 1),
                                           hi + span * (factor - 1), grid)
        mids = {k: float((v[0] + v[-1]) / 2) for k, v in axes.items()}
        worst_pt: tuple[str, float] | None = None
        max_ratio = 0.0
        for s in [str(x) for x in syms]:
            for val in axes[s]:
                subs = dict(mids)
                subs[s] = float(val)
                try:
                    out = float(expr.subs(
                        {sp.Symbol(k): v for k, v in subs.items()}).evalf())
                except (TypeError, ValueError, ArithmeticError, AttributeError):
                    return {"status": "FAIL",
                            "reason": f"non-evaluable at {s}={val:.3g}",
                            "max_ratio": None}
                if math.isnan(out) or math.isinf(out):
                    return {"status": "FAIL",
                            "reason": f"non-finite at {s}={val:.3g}",
                            "max_ratio": None}
                ratio = abs(out) / y_scale
                if ratio > max_ratio:
                    max_ratio, worst_pt = ratio, (s, float(val))
        if max_ratio > tol:
            where = (f"{worst_pt[0]}={worst_pt[1]:.3g}" if worst_pt is not None
                     else "an unidentified probe point")
            return {"status": "FAIL",
                    "reason": f"blowup {max_ratio:.2g}x data scale at {where}",
                    "max_ratio": round(max_ratio, 3)}
        return {"status": "PASS", "max_ratio": round(max_ratio, 3),
                "n_total": len(syms) * grid}
    except _PROBE_ERRORS as e:
        return {"status": "CONDITIONAL",
                "reason": f"audit error: {e}"}


def extrapolation_split(x: dict, y, train_frac: float = 0.8,
                        axis: str | None = None) -> dict:
    """Split training-interior vs exterior bands (extrapolation test).

    Sorts by `axis` (default: first key) and holds out both tails, so the
    candidate must predict outside its training range, not just between points.
    """
    import numpy as np
    keys = list(x.keys())
    ax = axis or keys[0]
    order = np.argsort(np.asarray(x[ax])).ravel()
    n = len(order)
    n_tr = max(1, int(n * train_frac))
    lo_n = (n - n_tr) // 2
    tr_idx = order[lo_n:lo_n + n_tr]
    te_idx = [i for i in range(n) if i not in set(tr_idx.tolist())]
    return {
        "x_train": {k: np.asarray(v)[tr_idx] for k, v in x.items()},
        "y_train": np.asarray(y)[tr_idx],
        "x_test": {k: np.asarray(v)[te_idx] for k, v in x.items()},
        "y_test": np.asarray(y)[te_idx],
    }


def candidate_report(name: str, expression: str, reference: str,
                     domain: dict, held_out_n: int, mae: float,
                     baseline_mae: float, dim_valid: str,
                     domain_status: dict, reference_status: str,
                     experimental: str, novelty: str,
                     ab_p: float | None = None,
                     ab_verdict: str | None = None) -> str:
    """Render the rigorous 11-field candidate block. Agnostic to domain."""
    rel = (baseline_mae - mae) / baseline_mae * 100.0 if baseline_mae else 0.0
    lines = [
        f"REFERENCE: {reference}",
        f"CANDIDATE: {name}: {expression}",
        "DOMAIN: " + ", ".join(f"{k} in [{v[0]}, {v[1]}]"
                               for k, v in domain.items()),
        f"HELD-OUT: n = {held_out_n}",
        f"MAE: {mae:.3f}",
        f"BASELINE MAE: {baseline_mae:.3f}",
        f"RELATIVE IMPROVEMENT: {rel:.2f}% lower held-out MAE",
        f"DIMENSIONAL VALIDITY: {dim_valid}",
        f"PHYSICAL DOMAIN: {domain_status.get('status', '?')}",
        f"REASON: {domain_status.get('reason', '')}",
        f"REFERENCE STATUS: {reference_status}",
        f"EXPERIMENTAL VALIDATION: {experimental}",
        f"LITERATURE NOVELTY: {novelty}",
    ]
    if ab_p is not None or ab_verdict is not None:
        lines.append(f"A/B SIGNIFICANCE: p={ab_p} verdict={ab_verdict} "
                     f"(gap without significance is noise, not victory)")
    return "\n".join(lines)


def ab_compare(err_a: list, err_b: list, name_a: str = "A",
               name_b: str = "B") -> dict:
    """Paired comparison of absolute-error vectors. Agnostic.

    Evaluates whether observed paired MAE differences are statistically
    distinguishable from zero; otherwise the difference is reported as
    non-significant rather than as a model advantage. Reports MAEs, paired
    t-test and Wilcoxon p-values, and per-point win fractions. p-values
    above the threshold are not "almost significant" — effect sizes stand
    as separate information.
    """
    import numpy as np
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError as exc:
        logger.debug("scipy unavailable; paired significance tests skipped: %s",
                     exc)
        ttest_rel = wilcoxon = None
    a = np.asarray(err_a, float).ravel()
    b = np.asarray(err_b, float).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out: dict = {"n": n,
                 f"mae_{name_a}": float(np.mean(a)),
                 f"mae_{name_b}": float(np.mean(b)),
                 f"win_frac_{name_a}": float((a < b).mean()),
                 f"win_frac_{name_b}": float((b < a).mean())}
    if ttest_rel is not None:
        try:
            out["paired_t_p"] = float(ttest_rel(a, b).pvalue)
        except (ValueError, TypeError, ArithmeticError) as exc:
            logger.debug("paired t-test unavailable: %s", exc)
            out["paired_t_p"] = None
    if wilcoxon is not None:
        try:
            out["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
        except (ValueError, TypeError, ArithmeticError) as exc:
            logger.debug("wilcoxon test unavailable: %s", exc)
            out["wilcoxon_p"] = None
    diff = float(np.mean(a) - np.mean(b))
    sig = out.get("wilcoxon_p")
    out["verdict"] = ("significant" if sig is not None and sig < 0.05
                      else "not-significant")
    out["winner"] = name_a if diff < 0 else name_b
    return out


def bakeoff(x_train: dict, y_train, x_test: dict, y_test,
            candidate_fn, extra_forms: dict | None = None) -> dict:
    """Fit a library of standard forms on train, rank ALL on test with A/B.

    The library (constant, additive-linear, log-linear power product, plus
    caller-supplied forms) removes the hand-picked-baseline weakness: a
    candidate must beat the best library form, not a chosen rival.
    Returns ranked rows with MAE and pairwise significance vs the best.
    """
    import numpy as np

    xtr = {k: np.asarray(v, float).ravel() for k, v in x_train.items()}
    ytr = np.asarray(y_train, float).ravel()
    xte = {k: np.asarray(v, float).ravel() for k, v in x_test.items()}
    yte = np.asarray(y_test, float).ravel()
    names = list(xtr.keys())
    forms: dict = {}

    def _const(Xt, Xte):
        c = float(np.mean(ytr))
        return np.full(len(yte), c), {"c": c}

    def _additive(Xt, Xte):
        A = np.column_stack([Xt[k] for k in names] + [np.ones(len(ytr))])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        B = np.column_stack([Xte[k] for k in names] + [np.ones(len(yte))])
        return B @ coef, {"coef": [round(float(v), 4) for v in coef]}

    def _power_product(Xt, Xte):
        if not bool(np.all(ytr > 0)):
            raise ValueError("power-product needs positive y")
        L = np.log(np.abs(np.column_stack([Xt[k] for k in names])) + 1e-300)
        A = np.column_stack([L, np.ones(len(ytr))])
        coef, *_ = np.linalg.lstsq(A, np.log(ytr), rcond=None)
        Le = np.log(np.abs(np.column_stack([Xte[k] for k in names])) + 1e-300)
        B = np.column_stack([Le, np.ones(len(yte))])
        return np.exp(B @ coef), {"exponents": [round(float(v), 4)
                                                for v in coef[:-1]]}

    forms["constant"] = _const
    forms["additive-linear"] = _additive
    forms["power-product"] = _power_product
    for fname, fn in (extra_forms or {}).items():
        forms[fname] = fn
    rows = []
    for fname, fn in forms.items():
        try:
            pred, detail = fn(xtr, xte)
            err = np.abs(np.asarray(pred, float).ravel() - yte)
            rows.append({"form": fname, "mae": float(err.mean()),
                         "errors": err, "detail": detail, "failed": False})
        except _PROBE_ERRORS as e:
            rows.append({"form": fname, "mae": None, "errors": None,
                         "detail": {"error": str(e)[:100]}, "failed": True})
    try:
        cand = np.asarray(candidate_fn(x_test), float).ravel()
        cerr = np.abs(cand - yte)
    except _PROBE_ERRORS as exc:
        logger.debug("candidate evaluation failed on the test split: %s", exc)
        cerr = None
    rows.append({"form": "candidate", "mae": (float(cerr.mean())
                                              if cerr is not None else None),
                 "errors": cerr, "detail": {}, "failed": cerr is None})
    ok = [r for r in rows if not r["failed"]]
    ok.sort(key=lambda r: r["mae"])
    best = ok[0] if ok else None
    for r in ok:
        if best is not None and r["form"] != best["form"]:
            cmp = ab_compare(list(r["errors"]), list(best["errors"]),
                             r["form"][:8], best["form"][:8])
            r["vs_best_p"] = cmp.get("wilcoxon_p")
            r["vs_best"] = cmp.get("verdict")
        else:
            r["vs_best_p"] = None
            r["vs_best"] = "best"
    return {"rows": [{k: v for k, v in r.items() if k != "errors"}
                     for r in ok],
            "failed": [r["form"] for r in rows if r["failed"]]}


def stability_report(runs: list) -> dict:
    """Seed-stability across repeated runs: each run is
    {"expression": str, "mae": float, "seed": ...}.

    Reports pairwise structural agreement (same_relationship), MAE spread,
    and a stability verdict. A winner that changes with the seed is
    undecided, not victorious.
    """
    try:
        from apedp.physics_validity_engine import same_relationship as _sr
        same_relationship = _sr
    except ImportError as exc:
        logger.debug("same_relationship unavailable: %s", exc)
        same_relationship = None  # type: ignore
    n = len(runs)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            try:
                if (same_relationship is not None
                        and same_relationship(runs[i]["expression"],
                                              runs[j]["expression"])):
                    agree += 1
            except (TypeError, ValueError, ArithmeticError, KeyError,
                    AttributeError) as exc:
                logger.debug("relationship comparison skipped: %s", exc)
                continue
    maes = [r["mae"] for r in runs
            if isinstance(r.get("mae"), (int, float))]
    spread = (max(maes) - min(maes)) / (min(maes) + 1e-300) if maes else None
    seeds = [r.get("seed") for r in runs]
    return {"n_runs": n, "pairwise_agreement": agree,
            "pairwise_total": max(total, 1),
            "mae_spread_frac": spread,
            "seeds_prespecified": seeds,
            "selection_note": (
                f"best-of-{n} selected post hoc: reported best error is "
                "optimistically biased; do not compare it against "
                "single-shot baselines without adjustment"),
            "verdict": ("stable" if total > 0 and agree == total
                        else "unstable")}


@dataclass(frozen=True)
class ExperimentDesign:
    """A reproducible, noise-aware set of candidate-discrimination points."""
    selected_points: list
    scores: list[float]
    raw_spread: list[float]
    candidate_count: int
    objective: str = "prediction spread / measurement uncertainty"

    def to_dict(self) -> dict:
        return {
            "selected_points": self.selected_points,
            "scores": self.scores,
            "raw_spread": self.raw_spread,
            "candidate_count": self.candidate_count,
            "objective": self.objective,
        }


def design_experiments(
    candidates: list[Callable],
    xs,
    *,
    uncertainty_fn: Callable | None = None,
    feasible_fn: Callable | None = None,
    n_points: int = 1,
    min_distance: float = 0.0,
) -> ExperimentDesign:
    """Greedily select multiple feasible, separated discrimination points.

    The objective is model-prediction spread divided by measurement uncertainty.
    ``feasible_fn`` may return a boolean mask; ``min_distance`` prevents a batch
    from spending every measurement at the same location. No experimental data
    are fabricated—the result is a design proposal only.
    """
    import numpy as np

    if not candidates:
        raise ValueError("at least one candidate model is required")
    if n_points < 1:
        raise ValueError("n_points must be positive")
    grid = np.asarray(xs, dtype=float).reshape(-1)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("xs must contain finite candidate points")
    predictions = np.asarray([np.asarray(fn(grid), dtype=float).reshape(-1)
                              for fn in candidates])
    if predictions.shape != (len(candidates), grid.size):
        raise ValueError("candidate outputs must match the xs shape")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("candidate predictions must be finite")
    feasible = np.ones(grid.size, dtype=bool)
    if feasible_fn is not None:
        feasible = np.asarray(feasible_fn(grid), dtype=bool).reshape(-1)
        if feasible.shape != (grid.size,):
            raise ValueError("feasible_fn must return one boolean per candidate point")
    spread = np.std(predictions, axis=0)
    if uncertainty_fn is None:
        sigma = np.ones(grid.size, dtype=float)
    else:
        sigma = np.asarray(uncertainty_fn(grid), dtype=float).reshape(-1)
        if sigma.shape != (grid.size,) or not np.all(np.isfinite(sigma)):
            raise ValueError("uncertainty_fn must return finite values matching xs")
        if np.any(sigma < 0.0):
            raise ValueError("measurement uncertainty cannot be negative")
        sigma = np.maximum(sigma, 1e-12)
    scores = spread / sigma
    scores = np.where(feasible, scores, -np.inf)
    selected: list[int] = []
    for _ in range(min(n_points, int(feasible.sum()))):
        ranked = np.argsort(scores)[::-1]
        choice = next((int(index) for index in ranked
                       if np.isfinite(scores[index])
                       and all(abs(grid[index] - grid[old]) >= min_distance
                               for old in selected)), None)
        if choice is None:
            break
        selected.append(choice)
        scores[choice] = -np.inf
    return ExperimentDesign(
        selected_points=[float(grid[index]) for index in selected],
        scores=[float(spread[index] / sigma[index]) for index in selected],
        raw_spread=[float(spread[index]) for index in selected],
        candidate_count=len(candidates),
    )


def suggest_experiment(candidates: list, xs) -> dict:
    """Backward-compatible single-point wrapper around :func:`design_experiments`."""
    try:
        design = design_experiments(candidates, xs)
        if not design.selected_points:
            return {"error": "no feasible candidate points"}
        return {"x_star": design.selected_points[0], "spread": design.raw_spread[0]}
    except (TypeError, ValueError, ArithmeticError) as e:
        return {"error": str(e)}


DISCOVERY_TIERS = ("A_rediscovery", "B_transformations", "C_deceptive",
                   "D_noisy", "E_extrapolation", "F_withheld")


def benchmark_tier_summary() -> dict:
    """Tier names for discovery benchmark (reviewer item 17)."""
    return dict.fromkeys(DISCOVERY_TIERS, "pending")
