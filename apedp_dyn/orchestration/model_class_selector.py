"""EHI Phase 1: Probabilistic Model-Class Selector (Pillar 1 redesign).

The deterministic sieve verdicts are routing *evidence*, not conclusions;
single statistics misclassify systematically (Fano > 1 is overdispersion,
not self-excitation; path divergence is not a data classifier). This layer
performs the Bayesian model-class comparison the external critique calls
for:

    P(M | D) ∝ P(D | M) · P(M)

over competing data-generating classes with DOCUMENTED per-class diagnostic
profiles (Gaussian likelihoods on each diagnostic statistic; a class only
updates on diagnostics its data can actually produce — no diagnostic, no
update, no silent favoritism), uniform priors, and explicit
ambiguity-preserved outcomes:

  ODE_PLAUSIBLE | PDE_PLAUSIBLE | STATIC_RELATION_SUPPORTED |
  HAWKES_PLAUSIBLE | POISSON_RENEWAL_PLAUSIBLE |
  MULTIPLE_CLASSES_PLAUSIBLE | INSUFFICIENT_METADATA

A decisive outcome requires post_max ≥ DECISION_THRESHOLD; anything weaker
stays visible as ambiguity (the resolution default is the caller's problem,
recorded here, never a silent route). Priors and profile means/widths are
documented module constants of the SELECTION layer: they shape routing, not
adjudication; the referee remains the acceptance authority.

Status: experimental (EHI Phase 1 deliverable: an unannotated dataset is
converted into a structured problem specification with routing evidence
AND uncertainty recorded).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apedp_dyn.orchestration.modality_sieve import _numeric_time_axis

__all__ = [
    "OUTCOME_INSUFFICIENT_METADATA",
    "OUTCOME_MULTIPLE_CLASSES",
    "ModelClassProfile",
    "ModelClassSelection",
    "select_model_class",
]

# Outcome vocabulary (ambiguity is an outcome, never a hidden default).
OUTCOME_MULTIPLE_CLASSES = "MULTIPLE_CLASSES_PLAUSIBLE"
OUTCOME_INSUFFICIENT_METADATA = "INSUFFICIENT_METADATA"

#: A decisive routing label needs this posterior share; below it the
#: selection reports ambiguity (documented selection-layer constant).
DECISION_THRESHOLD = 0.75
#: Top-vs-runner-up Bayes-factor floor for even a threshold-passing decision.
BAYES_FACTOR_FLOOR = 4.0
#: Log-likelihood penalty for a class whose only fired diagnostics are
#: shared across the class table (documented selection-layer constant).
NO_DISTINGUISHING_PENALTY = 2.0

CLASS_STATIC = "static_relation"
CLASS_ODE = "ode_kinetic"
CLASS_PDE = "pde_field"
CLASS_HAWKES = "hawkes_point_process"
CLASS_RENEWAL = "poisson_renewal"

_UNIFORM_PRIOR = 1.0 / 5.0


@dataclass(frozen=True)
class ModelClassProfile:
    """Documented diagnostic profile of one data-generating class.

    ``expected``/``width`` are keyed by diagnostic name; classes without a
    relevant diagnostic simply have no entry and never take updates from it.
    """

    diagnostics: Mapping[str, tuple[float, float]] = field(default_factory=dict)


#: Profiles over the diagnostic statistics computed by this module:
#    "time_axis" in {0,1}   (declarative; width covers near-miss detection)
#    "rel_curvature"        (kinetic velocity continuity, ODE diagnostic)
#    "mixed_energy"         (2D spectral dispersion, PDE diagnostic)
#    "fano"                 (count-series burstiness, Hawkes diagnostic)
#    "inter_arrival_acorr"  (clustering, Hawkes diagnostic)
#: A class lacking a diagnostic in its profile takes NO update from it.
_MODEL_PROFILES: dict[str, ModelClassProfile] = {
    CLASS_STATIC: ModelClassProfile(diagnostics={"time_axis": (0.0, 0.10)}),
    CLASS_ODE: ModelClassProfile(diagnostics={
        "time_axis": (1.0, 0.20), "rel_curvature": (0.02, 0.30)}),
    CLASS_PDE: ModelClassProfile(diagnostics={
        "time_axis": (1.0, 0.30), "mixed_energy": (0.90, 0.25)}),
    CLASS_HAWKES: ModelClassProfile(diagnostics={
        "time_axis": (1.0, 0.40), "fano": (3.0, 1.5), "inter_arrival_acorr": (0.35, 0.30)}),
    CLASS_RENEWAL: ModelClassProfile(diagnostics={
        "time_axis": (1.0, 0.40), "fano": (1.0, 0.60), "inter_arrival_acorr": (0.0, 0.30)}),
}


def _gauss_loglik(stat: float, expected: float, width: float) -> float:
    return float(-0.5 * ((stat - expected) / width) ** 2)


def _diagnostics(
    frame: pd.DataFrame,
    data_kind: str,
    value_cols: list[str] | None,
    time_col: str | None,
    coord_col: str | None,
) -> dict[str, float] | None:
    """Compute the diagnostics a frame can support (None: nothing measurable)."""
    if data_kind == "events":
        stamps = np.asarray(pd.to_numeric(frame[time_col] if time_col else frame.iloc[:, 0],
                                          errors="coerce").to_numpy(), dtype=float)
        stamps = stamps[np.isfinite(stamps)]
        if len(stamps) < 12:
            return None
        d = np.diff(np.sort(stamps))
        edges = np.linspace(stamps.min(), stamps.max(), 21)
        counts, _ = np.histogram(stamps, bins=edges)
        mean_c = float(np.mean(counts))
        fano = float(np.var(counts, ddof=1) / mean_c) if mean_c > 0 else 1.0
        var_d = float(np.var(d, ddof=1))
        acorr = (float(np.corrcoef(d[:-1], d[1:])[0, 1])
                 if len(d) > 4 and var_d > 0 else 0.0)
        return {"time_axis": 1.0, "fano": fano, "inter_arrival_acorr": acorr}

    vals, time_name = _numeric_time_axis(frame)
    # The sieve sentinel for "no isochronous axis" is an EMPTY string (plus a
    # None array); treating "" as a found axis stamped time_axis=1.0 onto pure
    # value-value frames (isotherms routed to ode_kinetic — the defect the
    # known-label routing pin caught). Normalize the sentinel once, here.
    if not time_name:
        vals, time_name = None, None
    if time_col is not None:
        t_arr = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=float)
        fine = np.isfinite(t_arr)
        vals, time_name = (t_arr[fine], time_col)
    diag: dict[str, float] = {}
    if time_name is None and time_col is None:
        if coord_col is None:
            diag["time_axis"] = 0.0
            return diag
        # Spatial axis declared but no time: a single-sweep field (no ω axis)
        diag["time_axis"] = 0.0
        return diag
    diag["time_axis"] = 1.0
    t = np.asarray(vals, dtype=float)
    vcols = value_cols or [c for c in frame.columns
                           if c not in (time_name, coord_col)
                           and np.isfinite(pd.to_numeric(frame[c], errors="coerce")).mean() > 0.9]
    series = np.column_stack([pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
                              for c in vcols]) if vcols else np.empty((len(frame), 0))
    # Kinetic diagnostic: rel curvature on the first value column.
    # Scalar-trajectory diagnostic: the consumed (t, v) rows must form a
    # strictly increasing time axis — raveled (t, x) grids carry repeated
    # stamps whose zero steps poison v with inf/nan and would launder a nan
    # likelihood into a decisive route. On a GRID (repeated stamps), the
    # kinetic diagnostic aggregates the value column to per-time MEANS first
    # (a legitimately scalar conditioning of the field), so a kinetic field
    # can still contest ODE against a firing PDE diagnostic — ambiguity
    # between classes stays attainable instead of being forbidden by
    # construction. If aggregation cannot yield a strictly increasing axis,
    # no curvature update is emitted (absence of evidence, not evidence).
    ok = np.isfinite(t) & np.isfinite(series[:, 0]) if series.shape[1] else np.zeros(len(t), bool)
    t_ok, v_ok = t[ok], (series[ok, 0] if series.shape[1] else np.array([]))
    if ok.sum() >= 10 and not bool(np.all(np.diff(t_ok) > 0)):
        uniq, inv = np.unique(t_ok, return_inverse=True)
        means = np.asarray([v_ok[inv == j].astype(float).mean() for j in range(uniq.size)])
        spreads = np.asarray([np.ptp(v_ok[inv == j]) if v_ok[inv == j].size > 1 else 0.0
                              for j in range(uniq.size)])
        # Aggregation validity: a kinetic mean field is representative only if
        # its amplitude is comparable to the field's own spatial spread. A
        # traveling wave's per-time mean is a negligible ripple (well below
        # the field amplitude); treating it as a kinetic ODE signal mis-badges
        # a PDE as ambiguous — the defect the routing pin caught.
        mean_amp = float(np.max(np.abs(means))) if means.size else 0.0
        spread_med = float(np.median(spreads)) if spreads.size else 0.0
        representative = mean_amp >= (0.25 * spread_med)
        if (np.all(np.diff(uniq) > 0) and np.isfinite(means).all() and len(uniq) >= 10
                and representative):
            t_ok, v_ok = uniq, means
        else:
            t_ok, v_ok = t_ok[:0], v_ok[:0]
    if len(t_ok) >= 10:
        dt = np.diff(t_ok)
        v = np.diff(v_ok) / dt
        span = float(np.median(np.abs(v)))
        if np.all(np.isfinite(v)) and span > 0 and len(dt) >= 2:
            dtv = (dt[1:] + dt[:-1]) / 2.0
            rel = np.abs(np.diff(v) / dtv) * dtv ** 2 / span
            res = rel[np.isfinite(rel)]
            if res.size:
                diag["rel_curvature"] = float(np.median(res))
    # PDE diagnostic: complete (coord, time) grid required.
    if coord_col is not None and vcols:
        xs = pd.to_numeric(frame[coord_col], errors="coerce").to_numpy(dtype=float)
        u = pd.to_numeric(frame[vcols[0]], errors="coerce").to_numpy(dtype=float)
        xs_u, ts_u = np.unique(xs), np.unique(t)
        if np.isfinite(u).all() and xs_u.size * ts_u.size == len(u):
            grid = u.reshape(ts_u.size, xs_u.size)
            spec = np.abs(np.fft.fftshift(np.fft.fft2(grid))) ** 2
            total = float(spec.sum())
            kx = np.fft.fftshift(np.fft.fftfreq(grid.shape[1]))
            w = np.fft.fftshift(np.fft.fftfreq(grid.shape[0]))
            K, W = np.meshgrid(kx, w)
            moving = float(spec[(np.abs(K) > 1e-12) & (np.abs(W) > 1e-12)].sum())
            diag["mixed_energy"] = moving / total if total > 0 else 0.0
    return diag or None


@dataclass(frozen=True)
class ModelClassSelection:
    outcome: str
    posteriors: dict[str, float]
    dominant_class: str
    bayes_factor: float
    diagnostics: dict[str, float]
    reason: str = ""


def select_model_class(
    frame: pd.DataFrame,
    value_cols: list[str] | None = None,
    time_col: str | None = None,
    coord_col: str | None = None,
    data_kind: str = "frames",
) -> ModelClassSelection:
    """P(M|D) over documented profiles with ambiguity-preserved outcomes."""
    diag = _diagnostics(frame, data_kind, value_cols, time_col, coord_col)
    # The DECLARED observation kind narrows the class table: rows that ARE
    # event stamps cannot be kinetic series or spatial fields — those classes
    # are not updated at all rather than left sitting at their priors. A
    # misdeclared kind is caller error, caught downstream by the referee.
    if data_kind == "events":
        profiles = {CLASS_HAWKES: _MODEL_PROFILES[CLASS_HAWKES],
                    CLASS_RENEWAL: _MODEL_PROFILES[CLASS_RENEWAL]}
    else:
        profiles = {CLASS_STATIC: _MODEL_PROFILES[CLASS_STATIC],
                    CLASS_ODE: _MODEL_PROFILES[CLASS_ODE],
                    CLASS_PDE: _MODEL_PROFILES[CLASS_PDE]}
    if diag is None:
        priors = dict.fromkeys(profiles, _UNIFORM_PRIOR)
        return ModelClassSelection(
            OUTCOME_INSUFFICIENT_METADATA, priors, "", 0.0, {},
            "no diagnostics computable from the declared frame (declare axes/kind)")
    loglik: dict[str, float] = {}
    lacks_defining: dict[str, bool] = {}
    for cls, profile in profiles.items():
        ll = 0.0
        fired_names: list[str] = []
        for name, (mu, sigma) in profile.diagnostics.items():
            if name in diag:
                ll += _gauss_loglik(diag[name], mu, sigma)
                fired_names.append(name)
        # A class whose defining diagnostic did NOT fire is penalized so a
        # missing update cannot win by staying at the prior — and "fired"
        # means a DISTINGUISHING diagnostic, not a shared one: time_axis≈1
        # is carried by every dynamic class, so letting it satisfy firing
        # let any time-indexed frame ride into a PDE/ODE tie, defeating the
        # routing pin. (For the STATIC class an ABSENT axis — time_axis≈0 —
        # is exactly what distinguishes it, so it counts as fired.) Riding
        # only shared evidence costs 2 nats: the floor must clear the
        # documented Bayes-factor barrier with a real margin.
        distinguished = [n for n in fired_names if n != "time_axis"]
        if cls == CLASS_STATIC and "time_axis" in fired_names:
            distinguished = fired_names
        if not distinguished:
            ll = ll - NO_DISTINGUISHING_PENALTY
            ll_lacks_defining = True
        else:
            ll_lacks_defining = False
        loglik[cls] = ll
        lacks_defining[cls] = ll_lacks_defining
    m = max(loglik.values())
    exp_shift = {c: float(np.exp(v - m)) for c, v in loglik.items()}
    z = sum(exp_shift.values())
    post = {c: float(v / z) for c, v in exp_shift.items()}
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    top_cls, top_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
    bf = top_p / runner_p if runner_p > 0 else float("inf")
    if top_p < DECISION_THRESHOLD or bf < BAYES_FACTOR_FLOOR:
        return ModelClassSelection(
            OUTCOME_MULTIPLE_CLASSES, post, top_cls, bf, diag,
            f"ambiguous evidence: top posterior {top_p:.3f} (bf {bf:.2f}) "
            "below the documented decision floor; ambiguity preserved")
    if lacks_defining[top_cls]:
        # FABRICATION GUARD: the winner is decisively ahead ONLY because its
        # competitors misfit — its own defining diagnostic never fired (the
        # −2 nat penalty confirms it). A decisive route on evidence the
        # winner did not itself produce would let pure noise on a time axis
        # fabricate a PDE claim (mixed_energy never fired). Honest outcome:
        # ambiguity is preserved and the caller is told what is missing.
        return ModelClassSelection(
            OUTCOME_MULTIPLE_CLASSES, post, top_cls, bf, diag,
            f"decisive routing refused: top class '{top_cls}' is ahead "
            f"(posterior {top_p:.3f}, bf {bf:.2f}) but its defining "
            "diagnostic never fired — evidence is competitive-misfit, not "
            "positive support; ambiguity preserved")
    outcome = {
        CLASS_STATIC: "STATIC_RELATION_SUPPORTED",
        CLASS_ODE: "ODE_PLAUSIBLE",
        CLASS_PDE: "PDE_PLAUSIBLE",
        CLASS_HAWKES: "HAWKES_PLAUSIBLE",
        CLASS_RENEWAL: "POISSON_RENEWAL_PLAUSIBLE",
    }[top_cls]
    return ModelClassSelection(outcome, post, top_cls, bf, diag,
                               f"decisive routing (posterior {top_p:.3f}, bf {bf:.2f})")
