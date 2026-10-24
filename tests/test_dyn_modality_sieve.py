"""Level-5 Pillar 1 pins: the modality sieve routes by structural evidence.

Each pin carries an inline negative case so the dispatcher's masking is
observable: a static table must NOT be silently classified as kinetic (and
vice versa). These are routing pins (which engines may run), never
adjudication pins (which candidates pass).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apedp_dyn.orchestration.modality_sieve import (
    ALL_ENGINE_FAMILIES,
    Modality,
    classify_event_stream,
    classify_modality,
    route_engines,
)


def test_static_isotherm_table_routes_to_rime_and_masks_dynamic_engines():
    """A steady-state (p, q) isotherm table has no time axis: the ODE/PDE
    engines would ill-pose the derivative estimation, so they are masked."""
    rng = np.random.default_rng(21)
    p = np.sort(rng.uniform(0.01, 5.0, 60))
    q = 5.0 * p / (2.0 + p) + 0.01 * rng.standard_normal(60)
    frame = pd.DataFrame({"p": p, "q": q})
    verdict = classify_modality(frame)
    assert verdict.modality == Modality.STATIC_MANIFOLD
    assert route_engines(verdict) == ("rime",)
    assert set(verdict.masked_engine_families) == {"ode", "pde", "hawkes"}


def test_smooth_kinetic_trajectory_routes_to_ode():
    t = np.linspace(0.0, 12.0, 240) + 1e-3
    x = np.sin(t) * np.exp(-0.2 * t)
    frame = pd.DataFrame({"time": t, "x": x})
    verdict = classify_modality(frame, value_cols=["x"])
    assert verdict.modality == Modality.ODE_KINETIC, verdict.reason
    assert route_engines(verdict) == ("ode",)
    curv = verdict.evidence["rel_curvature"]
    assert 0.0 <= curv <= 0.25
    # The honest single-series disclosure: a phase-space divergence is NOT
    # identifiable from one trajectory and the verdict says so.
    assert verdict.evidence.get("phase_divergence_identifiable") == 0.0


def test_travelling_wave_grid_routes_to_pde_on_dispersion_evidence():
    xs = np.linspace(0.0, 2.0, 64)
    ts = np.linspace(0.0, 2.0, 64)
    uu = np.sin(2.0 * np.pi * (2.0 * xs[None, :] - 1.0 * ts[:, None]))
    long_rows = []
    for ti, tv in enumerate(ts):
        for xi, xv in enumerate(xs):
            long_rows.append((xv, tv, uu[ti, xi]))
    frame = pd.DataFrame(long_rows, columns=["x", "t", "u"])
    verdict = classify_modality(frame, time_col="t", coord_col="x", value_cols=["u"])
    assert verdict.modality == Modality.PDE_FIELD, verdict.reason
    assert verdict.evidence["mixed_spectral_energy_fraction"] >= 0.5
    assert route_engines(verdict) == ("pde",)


def test_dc_dominated_grid_fails_closed_unresolved():
    xs = np.linspace(0.0, 1.0, 48)
    ts = np.linspace(0.0, 1.0, 48)
    rows = [(xv, tv, 1.0 + 0.001 * (xi + ti)) for ti, tv in enumerate(ts) for xi, xv in enumerate(xs)]
    frame = pd.DataFrame(rows, columns=["x", "t", "u"])
    verdict = classify_modality(frame, time_col="t", coord_col="x", value_cols=["u"])
    assert verdict.modality == Modality.UNRESOLVED
    assert route_engines(verdict) == ()
    assert verdict.masked_engine_families == tuple(ALL_ENGINE_FAMILIES)


def test_bursty_event_stream_routes_to_hawkes_and_poisson_refuses():
    rng = np.random.default_rng(11)
    stamps = []
    cursor = 0.0
    for _ in range(40):
        cursor += float(rng.exponential(10.0))  # long quiet gap
        for _ in range(int(rng.integers(4, 9))):  # dense burst
            cursor += float(rng.exponential(0.02))
            stamps.append(cursor)
    verdict = classify_event_stream(np.asarray(stamps))
    assert verdict.modality == Modality.HAWKES_POINT_PROCESS, verdict.evidence
    assert verdict.evidence["fano"] > 1.5 or verdict.evidence["inter_arrival_acorr"] > 0.1

    poisson = np.sort(np.random.default_rng(3).uniform(0.0, 1000.0, 400))
    verdict2 = classify_event_stream(poisson)
    assert verdict2.modality == Modality.UNRESOLVED
    assert verdict2.reason and "no clustering" in verdict2.reason
    assert route_engines(verdict2) == ()


def test_declared_event_stream_with_poisson_regularity_never_claims_hawkes():
    stamps = np.sort(np.random.default_rng(9).uniform(0.0, 200.0, 300))
    frame = pd.DataFrame({"stamp": stamps})
    verdict = classify_modality(frame, time_col="stamp", data_kind="events")
    assert verdict.modality == Modality.UNRESOLVED
    assert route_engines(verdict) == ()
