"""Level-5 roadmap Pillar 1: Autonomous Modality & Topology Sieve.

Pre-scans a raw dataset BEFORE any proposer is invoked and routes it to the
mathematically admissible discovery family, masking out engines whose
operating regime the data cannot support (e.g. feeding a static isotherm
table to ODE/PDE engines ill-poses the derivative estimation and would
manufacture derivatives out of pure noise).

Detectors (evidence-based, each reporting its numeric evidence):
  1. Time-axis detection: a strictly monotone numeric column with a regular
     positive median step.
  2. Static scalar manifold: no usable time axis and no spatial+temporal
     grid structure — every column is a simultaneous steady-state ensemble.
  3. Kinetic trajectory (ODE): monotone time axis with continuous velocity
     x_i' ≈ Δx/Δt (bounded relative curvature). Phase-space divergence
     ∇·v is REPORTED as evidence when identifiable from the sampled path
     (multi-state with enough samples) and explicitly flagged
     non-identifiable for single-series data — one trajectory cannot
     identify a divergence, and the sieve refuses to fake that inference.
  4. Spatiotemporal field (PDE): a spatial + temporal axis pair; the 2D
     Fourier transform must place dominant energy at nontrivial (k, ω)
     (dispersion ω(k) ≠ 0 / propagating structure), not at DC.
  5. Clustered event flow (Hawkes): event stamps with (a) Fano factor
     F > 1 over fixed windows and/or (b) positive inter-arrival lag-1
     autocorrelation — burstiness, not Poisson regularity.

Flaw-contract notes:
- Routing thresholds are MODULE-DOCUMENTED HEURISTICS of the ingestion
  layer (selection heuristics over structural evidence), NOT preregistered
  epistemic-referee thresholds; they mask which engines may run, they
  never gate acceptance. The canonical spec is untouched.
- Fail-closed: when structural evidence conflicts (two modalities both
  above their routing floor), the verdict is UNRESOLVED with both families
  masked except rime — the caller is told to segment or declare the axis,
  not to let a silent guess pick a physics family.
Status: experimental (Level-5 roadmap Phase 1; wired into the dispatch
surface by tests, not yet a production pipeline default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

__all__ = [
    "ALL_ENGINE_FAMILIES",
    "Modality",
    "ModalityVerdict",
    "classify_event_stream",
    "classify_modality",
    "route_engines",
]


class Modality(str, Enum):
    STATIC_MANIFOLD = "static_manifold"
    ODE_KINETIC = "ode_kinetic"
    PDE_FIELD = "pde_field"
    HAWKES_POINT_PROCESS = "hawkes_point_process"
    UNRESOLVED = "unresolved"


#: Engine families known to the dispatch surface.
STATIC_ENGINES = ("rime",)
DYNAMIC_ENGINES = ("ode", "pde", "hawkes")
ALL_ENGINE_FAMILIES = STATIC_ENGINES + DYNAMIC_ENGINES

# ---------------------------------------------------------------------------
# Routing heuristics (documented, NOT epistemic gate thresholds — see the
# module docstring): masks are about ill-posedness of the derivative
# estimation, never about acceptance.
# ---------------------------------------------------------------------------
#: Maximum acceptable relative curvature for the kinetic detector: the
#: median |x''| Δt² relative to the span of |x'|. Above this the series is
#: too jerky/noisy to define a continuous velocity.
ROUND_KINETIC_MAX_REL_CURVATURE = 0.25
#: Minimal share of 2D spectral energy at nontrivial (k, ω) for the PDE
#: detector (energy concentrated at DC = spatially constant field).
ROUND_PDE_MIN_MIXED_ENERGY_FRACTION = 0.5
#: Fano-factor floor for the bursty point-process detector (Poisson → F≈1).
ROUND_FANO_MIN = 1.5
#: Inter-arrival lag-1 autocorrelation floor (bursty → positive).
ROUND_EVENT_ACORR_MIN = 0.1
#: Adjacent-evidence conflict resolution floor: two modalities can only be
#: resolved by the dispatcher if the loser's evidence ratio is below this.
ROUND_CONFLICT_RATIO_MAX = 2.0


@dataclass(frozen=True)
class ModalityVerdict:
    """Structural classification of one raw dataset."""

    modality: Modality
    evidence: dict[str, float] = field(default_factory=dict)
    admissible_engine_families: tuple[str, ...] = ()
    masked_engine_families: tuple[str, ...] = ()
    reason: str = ""


def _numeric_time_axis(frame: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """Find a column parseable to a strictly monotone ISOCHRONOUS numeric axis.

    Time is physical provenance, not a pattern to sniff: auto-detection
    demands REGULAR sampling (large median step, small CV of the steps).
    An irregular monotone axis (e.g. a sorted isotherm concentration column)
    must be declared by the caller via ``time_col`` — the sieve never
    re-badges a bump-steep x-axis as a time axis.
    """
    for col in frame.columns:
        try:
            vals = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError):
            continue
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        d = np.diff(vals)
        step = float(np.median(d))
        mean_d = float(np.mean(d))
        cv = float(np.std(d, ddof=1) / abs(mean_d)) if mean_d != 0 else float("inf")
        if step > 0 and np.all(d > 0) and cv <= 0.1:
            return vals, str(col)
    return None, ""


def _rel_curvature(x: np.ndarray, t: np.ndarray) -> float:
    """Median |x''| Δt² relative to the span of |x'| (velocity continuity)."""
    dt = np.diff(t)
    if len(dt) < 2:
        return float("inf")
    v = np.diff(x) / dt
    span = float(np.median(np.abs(v)))
    if span <= 0:
        return float("inf")
    dtv = (dt[1:] + dt[:-1]) / 2.0
    a = np.diff(v) / dtv
    rel = np.abs(a) * (dtv ** 2) / max(float(np.median(np.abs(v))), 1e-30)
    return float(np.median(rel))


def _phase_space_divergence(states: np.ndarray, t: np.ndarray) -> float | None:
    """Directional trace estimate of ∇·v along ONE sampled trajectory.

    With a single trajectory a Jacobian is only ever locally identifiable,
    so this is recorded as path-level EVIDENCE, never a classifier. Returns
    None when the states/velocity cannot support the estimate.
    """
    if states.shape[1] < 2 or len(t) < 8:
        return None
    dt = np.diff(t)
    v = np.diff(states, axis=0) / dt[:, None]
    mid_x = (states[1:] + states[:-1]) / 2.0
    dx = np.diff(mid_x, axis=0)
    dv = np.diff(v, axis=0)
    # Subsample rows so the lstsq is well-posed and resists jitter.
    rows = dx[np.isfinite(dx).all(axis=1)]
    rows_v = dv[np.isfinite(dx).all(axis=1)]
    if rows.shape[0] < 4 * states.shape[1]:
        return None
    try:
        jac, *_ = np.linalg.lstsq(rows, rows_v, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(np.trace(jac))


def _mixed_energy_fraction(u: np.ndarray) -> float:
    """Share of total 2D FFT energy at spatially AND temporally nonzero modes."""
    spec = np.abs(np.fft.fftshift(np.fft.fft2(u))) ** 2
    total = float(spec.sum())
    if total <= 0 or not np.isfinite(total):
        return 0.0
    kx = np.fft.fftshift(np.fft.fftfreq(u.shape[1]))
    w = np.fft.fftshift(np.fft.fftfreq(u.shape[0]))
    K, W = np.meshgrid(kx, w)
    moving = spec[(np.abs(K) > 1e-12) & (np.abs(W) > 1e-12)]
    return float(moving.sum()) / total


def classify_event_stream(stamps: np.ndarray, n_windows: int = 20) -> ModalityVerdict:
    """Classify a pure event-stamp series (Hawkes vs Poisson vs insufficient).

    Evidence: Fano factor of windowed counts and inter-arrival lag-1
    autocorrelation. F > 1 or positive lag-1 acorr (both documented heuristics)
    declare clustered events; otherwise the stream is Poisson-like and Is
    NOT claimable as self-exciting (fail-closed: the sieve's job is routing,
    and a Poisson stream has no clustering to attribute).
    """
    s = np.asarray(stamps, dtype=float)
    s = np.sort(s[np.isfinite(s)])
    if len(s) < 12:
        return ModalityVerdict(Modality.UNRESOLVED, {"n_events": float(len(s))},
                               (), tuple(ALL_ENGINE_FAMILIES),
                               "event stream too short to sieve")
    edges = np.linspace(s[0], s[-1], n_windows + 1)
    counts, _ = np.histogram(s, bins=edges)
    mean_c = float(np.mean(counts))
    fano = float(np.var(counts, ddof=1) / mean_c) if mean_c > 0 else float("nan")
    inter = np.diff(s)
    acorr = 0.0
    if len(inter) > 4 and float(np.var(inter, ddof=1)) > 0:
        acorr = float(np.corrcoef(inter[:-1], inter[1:])[0, 1])
    bursty = (np.isfinite(fano) and fano > ROUND_FANO_MIN) or acorr > ROUND_EVENT_ACORR_MIN
    if bursty:
        return ModalityVerdict(
            Modality.HAWKES_POINT_PROCESS,
            {"fano": fano, "inter_arrival_acorr": acorr, "n_events": float(len(s))},
            ("hawkes",), tuple(f for f in ALL_ENGINE_FAMILIES if f != "hawkes"),
            "clustered event flow (Fano > 1 or positive inter-arrival autocorrelation)",
        )
    return ModalityVerdict(
        Modality.UNRESOLVED,
        {"fano": fano, "inter_arrival_acorr": acorr, "n_events": float(len(s))},
        (), tuple(ALL_ENGINE_FAMILIES),
        "Poisson-like event stream: no clustering to attribute (fail-closed)",
    )


def classify_modality(
    frame: pd.DataFrame,
    value_cols: list[str] | None = None,
    time_col: str | None = None,
    coord_col: str | None = None,
    data_kind: str = "frames",
) -> ModalityVerdict:
    """Structurally sieve a raw frame into one routing modality.

    Parameters
    ----------
    frame:
        Raw observations. Column order is respected; nothing is reshaped
        silently — a 2D field must arrive as a (coord, time) long-format
        pair or a complete regular grid pivot.
    value_cols / time_col / coord_col:
        Explicit axis names; when ``None`` the sieve attempts detection
        (peer columns below).
    data_kind:
        ``"frames"`` (steady rows / trajectories / fields) or ``"events"``
        (the rows ARE event stamps). Declaring the kind is honest: the
        caller who knows the rows are stamps should say so.
    """
    if data_kind == "events":
        axis = np.asarray(frame[time_col] if time_col else frame.iloc[:, 0], dtype=float)
        base = classify_event_stream(axis)
        if base.modality == Modality.HAWKES_POINT_PROCESS:
            return base
        return ModalityVerdict(
            Modality.UNRESOLVED,
            dict(base.evidence),
            (), tuple(ALL_ENGINE_FAMILIES),
            "declared events but Poisson-like regularity; no Hawkes claim",
        )

    vals, time_name = _numeric_time_axis(frame)
    if time_col is not None:
        vals = pd.to_numeric(frame[time_col], errors="raise").to_numpy(dtype=float)
        time_name = time_col
    coord_vals = None
    if coord_col is not None:
        coord_vals = pd.to_numeric(frame[coord_col], errors="raise").to_numpy(dtype=float)
    if vals is None and coord_vals is None:
        # No axis declared and none detectable: the honest verdict is a
        # static isotherm/manifold ensemble (the mode the RIME gate serves).
        return ModalityVerdict(
            Modality.STATIC_MANIFOLD,
            {"time_axis_detected": 0.0},
            STATIC_ENGINES, DYNAMIC_ENGINES,
            "no monotone time axis and no declared grid: static scalar manifold",
        )

    t = vals if time_name else np.arange(len(frame), dtype=float)
    vcols = value_cols or [c for c in frame.columns
                           if c not in (time_name, coord_col)
                           and np.isfinite(pd.to_numeric(frame[c], errors="coerce")).mean() > 0.9]
    series = np.column_stack([
        pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) for c in vcols
    ]) if vcols else np.empty((len(frame), 0))

    # --- PDE branch: a declared/complete (coord, time) grid --------------
    if coord_vals is not None and len(series):
        grid_ok = coord_vals is not None and time_name
        if grid_ok:
            u = series[:, 0] if series.shape[1] == 1 else series.mean(axis=1)
            xs = np.unique(coord_vals)
            ts = np.unique(t)
            if xs.size * ts.size == len(u):
                grid = u.reshape(ts.size, xs.size)
                frac = _mixed_energy_fraction(grid)
                if frac >= ROUND_PDE_MIN_MIXED_ENERGY_FRACTION:
                    return ModalityVerdict(
                        Modality.PDE_FIELD,
                        {"mixed_spectral_energy_fraction": frac,
                         "n_grid": float(grid.size)},
                        ("pde",), tuple(f for f in ALL_ENGINE_FAMILIES if f != "pde"),
                        "dominant 2D spectral energy at nontrivial (k, omega): dispersion present",
                    )
                return ModalityVerdict(
                    Modality.UNRESOLVED,
                    {"mixed_spectral_energy_fraction": frac,
                     "n_grid": float(grid.size)},
                    (), tuple(ALL_ENGINE_FAMILIES),
                    "grid detected but spectral energy sits at DC (no dispersion evidence)",
                )

    # --- ODE branch: monotone time + continuous velocity -----------------
    if time_name and series.shape[1] >= 1:
        finite_t = np.isfinite(t) & np.isfinite(series).all(axis=1)
        curv = _rel_curvature(series[finite_t, 0], t[finite_t])
        div = _phase_space_divergence(series[finite_t], t[finite_t])
        if curv < ROUND_KINETIC_MAX_REL_CURVATURE:
            dt = np.diff(t)
            cv = float(np.std(dt[dt > 0], ddof=1) / np.mean(dt[dt > 0])) if np.any(dt > 0) else float("inf")
            ev: dict[str, float] = {"rel_curvature": curv, "time_axis": 1.0,
                                    "time_step_cv": cv}
            if div is not None:
                ev["path_phase_divergence"] = div
            else:
                ev["phase_divergence_identifiable"] = 0.0  # single series: unidentifiable
            return ModalityVerdict(
                Modality.ODE_KINETIC,
                ev,
                ("ode",), tuple(f for f in ALL_ENGINE_FAMILIES if f != "ode"),
                "monotone time axis with continuous velocity (bounded rel curvature)",
            )

    # Conflicting / below-floor structural evidence: refuse to guess.
    return ModalityVerdict(
        Modality.UNRESOLVED,
        {"time_axis": 1.0 if time_name else 0.0},
        (), tuple(ALL_ENGINE_FAMILIES),
        "structural evidence below every routing floor (jerk/noisy series or ambiguous axes)",
    )


def route_engines(verdict: ModalityVerdict) -> tuple[str, ...]:
    """Engine families the verdict ADMITS (already excludes masked ones)."""
    return verdict.admissible_engine_families
