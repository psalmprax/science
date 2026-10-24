"""DE arm, layers B/C: Level-1 dependency screening + ODE system discovery.

Implements the reviewed architecture inside a SEPARATE arm (RIME's core
stays untouched; the dense-library engine ``discover_coupled_system`` is
reused, not reimplemented):

Ranking evidence (Level 1). For each declared state, other states are
ranked by their incremental predictive gain on that state's derivative
(baseline autoregression in the state's own lags vs baseline + candidate
parent lags, TRAINING partition only, deterministic relative-RSS statistic
— a predictive-increment ordering, NOT a causal-topology verdict; review
correction B). It only prunes candidate libraries and can never be a
mandatory gate.

Mandatory escape hatch (review correction: leave a route for what
screening misses). The DENSE engine result always runs alongside in the
same outcome, so a parent the screening excluded can never silently vanish
from the artifact the referee sees; the arm records both and decides
nothing (no acceptance, promotion, or verdict fields — EVER).

The screened fit runs a Python STRidge mirror (ridge + iterative prune,
documented constants) over the REDUCED library: own terms + own powers +
kept parents + own·kept products + kept·kept products. Kernel parity is
NOT claimed — provenance names the implementation and the library kind,
so the referee never compares two engines as if they were one. The
kept·kept columns exist for representational completeness: without them a
reduced library could not express parent×parent structure the dense
engine demonstrably recovers (e.g. Lorenz ẏ = 28x − y − xz needs the
x·z monomial, which couples two parents and cannot arise from any
own·parent product).

Status: experimental (DE-discovery arm phase 1).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from apedp_dyn.epistemic.coupled_discovery import (
    CoupledSystemDiscoveryResult,
    discover_coupled_system,
)

__all__ = [
    "OdeArmOutcome",
    "ScreenedStateFit",
    "StateParentRanking",
    "discover_ode_system",
    "granger_screen",
]

# STRidge-mirror + partition constants (documented; NOT referee gates).
_PRUNE_ITERS = 10
_PRUNE_THRESHOLD = 0.02   # relative to max |w| per iteration (non-intercept)
_RIDGE_Z_ALPHA = 0.05     # z-space ridge penalty
_MIN_INTERIOR_ROWS = 16
_MIN_STATES = 2


def _interior_derivative(s: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Interior values + 4th-order central d/dt (kernel-mirrored indexing)."""
    n = len(s)
    if n < 7:
        raise ValueError(
            f"series too short for a 4th-order central derivative ({n} rows; "
            "the interior needs the same stencil the kernel enforces)")
    d = (-s[4:] + 8.0 * s[3:n - 1] - 8.0 * s[1:n - 3] + s[0:n - 4]) / (12.0 * float(dt))
    vals = s[2:n - 2]
    if len(d) != len(vals):
        raise ValueError("interior derivative/value sizing mismatch")
    return vals, d


def _interior_vals(s: np.ndarray) -> np.ndarray:
    return s[2:len(s) - 2]


@dataclass(frozen=True)
class StateParentRanking:
    """Predictive-increment ranking of candidate parents for ONE state."""

    state: str
    ranked: tuple[tuple[str, float], ...]   # (parent, relative-RSS gain), descending
    kept: tuple[str, ...]
    excluded: tuple[str, ...]
    baseline_rss: float
    disclaimer: str = (
        "predictive-increment ordering on the training partition; NOT a causal "
        "topology verdict — the screened library always runs beside the dense "
        "engine result in the same outcome")


@dataclass(frozen=True)
class ScreenedStateFit:
    """One reduced-library STRidge-mirror fit for ONE state."""

    state: str
    library: tuple[str, ...]
    active_terms: tuple[str, ...]
    coefficients: dict[str, float]   # raw-unit coefficient per active term
    train_r2: float
    train_rmse: float
    excluded_parents: tuple[str, ...]
    implementation: str = "python-ridge-prune-mirror (z alpha=0.05, prune=0.02)"
    library_kind: str = "screened"


@dataclass
class OdeArmOutcome:
    """Evidence-only ODE-arm artifact (referee-facing; never a verdict)."""

    dense_result: CoupledSystemDiscoveryResult
    ranking: dict[str, StateParentRanking]
    screened: dict[str, ScreenedStateFit]
    meta: dict[str, float] = field(default_factory=dict)
    referee_note: str = (
        "arm supplies evidence only: dense kernel result + Level-1 screening "
        "rankings + screened-library mirror fits; acceptance belongs to the "
        "frozen referee protocol")


def granger_screen(
    series: Mapping[str, Sequence[float]],
    dt: float,
    *,
    top_k: int = 2,
    max_lag: int = 2,
    train_fraction: float = 0.7,
) -> dict[str, StateParentRanking]:
    """Rank candidate parents per state by incremental predictive gain.

    Deterministic and dependency-free: relative RSS reduction when a
    parent's lags join the state's own-lag baseline. Relative gains are
    RANKED, never p-value certified (no distribution is assumed).
    """
    if len(series) < _MIN_STATES:
        raise ValueError(
            "dependency screening is a COUPLED-system pass; fewer than two "
            "states is a single-series route (static/ODE path)")
    names = sorted(series)
    arr = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    lens = {len(arr[k]) for k in names}
    if len(lens) != 1:
        raise ValueError("state series must share one time axis (equal length)")
    n = lens.pop()
    interior = n - 4
    n_train = int(interior * float(train_fraction))
    if n_train < _MIN_INTERIOR_ROWS:
        raise ValueError(
            f"training partition too small for ranking evidence ({n_train} rows)")

    derivs = {k: _interior_derivative(arr[k], float(dt)) for k in names}
    out: dict[str, StateParentRanking] = {}
    for target in names:
        y, _ = derivs[target]
        y_tr = y[:n_train]
        own_cols = [np.ones(n_train)]
        for lag in range(1, max_lag + 1):
            col = _lagged(_interior_vals(arr[target]), lag, n_train)
            if col is not None:
                own_cols.append(col)
        base_rss = _rss(y_tr, own_cols)
        scored: list[tuple[str, float]] = []
        for parent in names:
            if parent == target:
                continue
            p_cols = list(own_cols)
            for lag in range(1, max_lag + 1):
                col = _lagged(_interior_vals(arr[parent]), lag, n_train)
                if col is not None:
                    p_cols.append(col)
            inc = (base_rss - _rss(y_tr, p_cols)) / max(base_rss, 1e-300)
            scored.append((parent, float(max(inc, 0.0))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        k = max(0, min(int(top_k), len(scored)))
        out[target] = StateParentRanking(
            state=target,
            ranked=tuple(scored),
            kept=tuple(p for p, _ in scored[:k]),
            excluded=tuple(p for p, _ in scored[k:]),
            baseline_rss=float(base_rss),
        )
    return out


def _lagged(vals: np.ndarray, lag: int, n_train: int) -> np.ndarray | None:
    """Aligned lag vector for the first ``n_train`` interior rows.

    Warm-up rows are honest zeros (no circular wrap): a lagged parent never
    carries fabricated pre-history.
    """
    if lag < 0 or lag > len(vals) - 1:
        return None
    lagged = np.zeros(len(vals), dtype=float)
    lagged[lag:] = vals[:len(vals) - lag]
    return lagged[:n_train]


def _rss(y: np.ndarray, cols: list[np.ndarray]) -> float:
    X = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ w
    return float(r @ r)


def discover_ode_system(
    series: Mapping[str, Sequence[float]],
    dt: float,
    *,
    labels: Sequence[str] | None = None,
    top_k: int = 2,
    max_lag: int = 2,
    train_fraction: float = 0.7,
    max_polynomial_degree: int = 2,
    include_cross_interactions: bool = True,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
) -> OdeArmOutcome:
    """Dense kernel discovery + Level-1 screening + screened mirror fits.

    Both library kinds leave the arm in ONE outcome; selection between them
    is adjudication work, so it never happens inside this package.
    """
    if labels is not None and len(labels) != len(series):
        raise ValueError("declared state labels count ≠ series count")
    if labels is None:
        state_labels = [str(k) for k in series]
        state_data = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    else:
        state_labels = [str(k) for k in labels]
        state_data = {}
        for name, (_, raw) in zip(state_labels, enumerate(series.values()), strict=True):
            state_data[name] = np.asarray(raw, dtype=float)
    lens = {len(v) for v in state_data.values()}
    if len(lens) != 1:
        raise ValueError("state series must share one time axis (equal length)")
    if len(state_labels) < _MIN_STATES:
        raise ValueError(
            "the ODE arm handles COUPLED systems (>= 2 states); single-series "
            "routes stay on the static/ODE path")
    if any(not np.isfinite(v).all() for v in state_data.values()):
        raise ValueError("state series contain non-finite values; refusing to fit")

    dense = discover_coupled_system(
        {k: state_data[k] for k in state_labels}, float(dt),
        max_polynomial_degree=max_polynomial_degree,
        include_cross_interactions=include_cross_interactions,
        sparsity_threshold=sparsity_threshold,
        ridge_alpha=ridge_alpha,
    )
    ranking = granger_screen(state_data, float(dt), top_k=top_k,
                             max_lag=max_lag, train_fraction=train_fraction)

    n = next(iter(lens))
    interior = n - 4
    n_train = int(interior * float(train_fraction))
    screened = _screened_fits(state_data, ranking, float(dt), n_train)

    return OdeArmOutcome(
        dense_result=dense,
        ranking=ranking,
        screened=screened,
        meta={"dt": float(dt), "n_rows": float(n),
              "n_states": float(len(state_labels)),
              "train_rows": float(n_train)},
    )


def _screened_fits(
    state_data: dict[str, np.ndarray],
    ranking: dict[str, StateParentRanking],
    dt: float,
    n_train: int,
) -> dict[str, ScreenedStateFit]:
    """Reduced-library mirror fit per state from the ranking's KEPT parents.

    Library grammar: own + own^2 + own^3 + kept parents + own·kept +
    kept·kept (representational completeness over the kept set). Every
    column uses ONLY kept parents — an excluded parent can never leak in
    (the dense escape hatch is the sole route back for it).
    """
    fits: dict[str, ScreenedStateFit] = {}
    for state in sorted(state_data):
        vals, d = _interior_derivative(state_data[state], dt)
        y_tr = d[:n_train]
        own = vals[:n_train]
        lib: dict[str, np.ndarray] = {"1": np.ones(n_train), state: own}
        spread = float(np.ptp(own)) if n_train > 1 else 0.0
        if spread > 0:
            lib[f"{state}^2"] = own ** 2
            lib[f"{state}^3"] = own ** 3
        for parent in ranking[state].kept:
            pv = _interior_vals(state_data[parent])[:n_train]
            lib[f"{parent}"] = pv
            lib[f"{state}*{parent}"] = own * pv
        kept_vals = {
            p: _interior_vals(state_data[p])[:n_train]
            for p in ranking[state].kept
        }
        kept_names = sorted(kept_vals)
        for a, b in pairwise(kept_names):
            lib[f"{a}*{b}"] = kept_vals[a] * kept_vals[b]
        terms, coefs, r2, rmse = _stridge_fit(y_tr, lib)
        fits[state] = ScreenedStateFit(
            state=state,
            library=tuple(sorted(lib)),
            active_terms=terms,
            coefficients=coefs,
            train_r2=r2,
            train_rmse=rmse,
            excluded_parents=ranking[state].excluded,
        )
    return fits


def _stridge_fit(
    y: np.ndarray,
    lib: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], dict[str, float], float, float]:
    """Deterministic ridge + iterative-prune mirror over a reduced library.

    Z-space standardization per column (a degenerate column is REFUSED, not
    silently averaged); iterative prune relative to max|w| over active
    non-intercept terms; the intercept column "1" is never pruned.
    Coefficients are mapped back to raw units for interpretability.
    """
    names = sorted(lib)
    raw_cols = np.column_stack([lib[k] for k in names])
    mu = raw_cols.mean(axis=0)
    sd = raw_cols.std(axis=0)
    interp = names.index("1")

    z = np.empty_like(raw_cols)
    col_scale: dict[int, float] = {}
    for j in range(len(names)):
        if j == interp:
            z[:, j] = raw_cols[:, j]      # intercept stays in raw scale
            col_scale[j] = 1.0
        else:
            if not sd[j] > 1e-12:
                raise ValueError(
                    f"degenerate library column {names[j]!r} (zero variance): "
                    "the screen refuses to carry an uninformative term")
            z[:, j] = (raw_cols[:, j] - mu[j]) / sd[j]
            col_scale[j] = float(sd[j])

    y_mu, y_sd = float(y.mean()), float(y.std())
    if not y_sd > 1e-12:
        raise ValueError("target derivative is (near-)constant: nothing to fit")
    yz = (y - y_mu) / y_sd

    active = list(range(len(names)))
    w = np.zeros(len(names))
    for _ in range(_PRUNE_ITERS):
        za = z[:, active]
        G = za.T @ za + _RIDGE_Z_ALPHA * np.eye(len(active))
        w_a = np.linalg.solve(G, za.T @ yz)
        w = np.zeros(len(names))
        w[active] = w_a
        non_int = [j for j in active if j != interp]
        ref = max((abs(float(w[j])) for j in non_int), default=0.0)
        cut = ref * _PRUNE_THRESHOLD
        retained = [j for j in active if j == interp or abs(float(w[j])) > cut]
        if retained == active:
            break
        active = retained

    pred_z = z[:, active] @ w[active]
    resid_z = yz - pred_z
    ss_res = float(resid_z @ resid_z)
    ss_tot = float(((yz - yz.mean()) ** 2).sum())
    train_r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    raw_pred = y_mu + y_sd * pred_z
    train_rmse = float(np.sqrt(float(np.mean((y - raw_pred) ** 2))))

    active_names = tuple(names[j] for j in active)
    coefs: dict[str, float] = {}
    for j in active:
        if j == interp:
            coefs["1"] = float(y_mu + y_sd * float(w[j]))
        else:
            coefs[names[j]] = float(w[j]) * y_sd / col_scale[j]
    return active_names, coefs, train_r2, train_rmse
