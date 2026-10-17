"""Shape-inference acceptance tests: it must FIND steps and STAY SILENT on smooth.

Acceptance criteria (preregistered, both directions, fixed seeds):
  1. Staircase -> family "piecewise_constant", change-points estimated near
     the true steps, strong improvement over the line.
  2. Straight line -> family "smooth" (NOT piecewise_constant), few segments.
  3. Smooth saturating curve (QEC Pade form) -> NOT piecewise_constant.
     A saturating curve is smooth, not stepped; claiming steps here would be
     a false structural claim.
  4. Pure noise -> no structure.
  5. Real BTC forward returns -> honest report, no fabricated steps.
  6. Step count is data-estimated: a coarser staircase must report FEWER
     change-points, so k is not a free parameter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.epistemic.shape_inference import (
    FAMILY_NO_STRUCTURE,
    FAMILY_PIECEWISE_CONSTANT,
    FAMILY_SMOOTH,
    fit_step_family,
    infer_shape_prior,
    shape_vocabulary,
    staircase_prior,
    step_family_basis,
)

# Explicit, test-local cutoffs (the module takes them as arguments).
PENALTY = 1.0
MARGIN = 0.05
CUTS = {"penalty_per_segment": PENALTY, "min_segment_len": 5,
            "max_segments": 32, "improvement_margin": MARGIN, "min_rows": 40,
            "flat_tol": 0.05, "min_flat_fraction": 0.5, "min_run_length": 5.0, "min_pc_r2": 0.05,
            "sse_target": 0.01}


def _staircase(vg, edges):
    g = np.zeros_like(vg)
    for e in edges:
        g += (vg >= float(e)).astype(float)
    return g


def test_staircase_is_inferred_piecewise_constant_with_estimated_steps():
    rng = np.random.RandomState(42)
    vg = np.sort(rng.uniform(0.1, 5.9, 200))
    g = _staircase(vg, (1.0, 2.0, 3.0, 4.0, 5.0))
    p = infer_shape_prior(vg, g, **CUTS)
    print("\n[staircase]", p.to_dict())
    assert p.family == FAMILY_PIECEWISE_CONSTANT, p.note
    # Change-points are estimated; each true step must be within one bin.
    assert 4 <= p.n_change_points <= 7, p.change_points
    for e in (1.0, 2.0, 3.0, 4.0, 5.0):
        assert any(abs(c - e) < 0.15 for c in p.change_points), (e, p.change_points)


def test_straight_line_is_not_inferred_as_stepped():
    rng = np.random.RandomState(1)
    x = rng.uniform(0.1, 6.0, 200)
    y = 2.5 * x + 10.0 + 0.02 * rng.randn(200)
    p = infer_shape_prior(x, y, **CUTS)
    print("\n[line]", p.to_dict())
    assert p.family != FAMILY_PIECEWISE_CONSTANT, p.note
    assert p.improvement < MARGIN, p.improvement
    assert p.n_change_points == 0, p.change_points


def test_saturating_curve_is_not_inferred_as_stepped():
    rng = np.random.RandomState(3)
    x = np.linspace(0.001, 0.025, 200)
    u = (x / 0.01) ** 2
    y = (1000.0 * u) / (1.0 + 2000.0 * u) + 0.001 * rng.randn(200)
    p = infer_shape_prior(x, y, **CUTS)
    print("\n[saturating]", p.to_dict())
    assert p.family != FAMILY_PIECEWISE_CONSTANT, p.note
    # The discriminator must be decisive here, not marginal.
    # Either the constant family cannot reach the fit target at all, or slopes
    # reach it with FEWER parameters. Both mean "not plateaus".
    assert p.params_piecewise_constant < 0 or (
        p.params_piecewise_linear < p.params_piecewise_constant), (
        p.params_piecewise_constant, p.params_piecewise_linear, p.note)


def test_drift_ratio_separates_plateaus_from_curves():
    """The decisive statistic, measured across families, not asserted loosely.

    A staircase's within-segment leftover is noise, so adding a slope buys
    nothing (drift ratio ~1). A curve's leftover IS a ramp, so a slope buys
    nearly all of it (drift ratio ~0). This is the property that replaced three
    earlier statistics that were measured wrong (see CHANGELOG).
    """
    rng = np.random.RandomState(42)
    vg = np.sort(rng.uniform(0.1, 5.9, 200))
    stair = np.zeros_like(vg)
    for e in (1.0, 2.0, 3.0, 4.0, 5.0):
        stair += (vg >= float(e))
    px = np.linspace(0.1, 6.0, 200)
    line = 2.5 * px + 10.0
    sat_x = np.linspace(0.001, 0.025, 200)
    u = (sat_x / 0.01) ** 2
    sat = (1000.0 * u) / (1.0 + 2000.0 * u)
    ps = infer_shape_prior(vg, stair, **CUTS)
    pl = infer_shape_prior(px, line, **CUTS)
    pq = infer_shape_prior(sat_x, sat, **CUTS)
    print(f"\n[parsimony] stair={(ps.params_piecewise_constant, ps.params_piecewise_linear)} line={(pl.params_piecewise_constant, pl.params_piecewise_linear)} pade={(pq.params_piecewise_constant, pq.params_piecewise_linear)}")
    assert ps.family == FAMILY_PIECEWISE_CONSTANT, ps.note
    assert pl.family != FAMILY_PIECEWISE_CONSTANT, pl.note
    assert pq.family != FAMILY_PIECEWISE_CONSTANT, pq.note
    # Slopes are CHEAPER on a curve, and NOT cheaper on a staircase.
    assert pl.params_piecewise_linear < pl.params_piecewise_constant
    assert ps.params_piecewise_linear > ps.params_piecewise_constant


def test_pure_noise_has_no_structure():
    rng = np.random.RandomState(7)
    x = rng.uniform(0.1, 6.0, 200)
    y = rng.randn(200)
    p = infer_shape_prior(x, y, **CUTS)
    print("\n[noise]", p.to_dict())
    assert p.family in (FAMILY_SMOOTH, FAMILY_NO_STRUCTURE), p.note


def test_step_count_is_estimated_not_free():
    rng = np.random.RandomState(4)
    vg = np.sort(rng.uniform(0.1, 5.9, 200))
    coarse = _staircase(vg, (1.0, 3.0))
    fine = _staircase(vg, (1.0, 2.0, 3.0, 4.0, 5.0))
    c = infer_shape_prior(vg, coarse, **CUTS)
    f = infer_shape_prior(vg, fine, **CUTS)
    print(f"\n[step-count] coarse={c.n_change_points} steps fine={f.n_change_points} steps")
    assert c.n_change_points < f.n_change_points, (c.n_change_points, f.n_change_points)


def test_real_market_forwards_do_not_fabricate_steps():
    path = Path(__file__).resolve().parent.parent / "data" / "market" / "crypto_top15_orderflow.csv"
    if not path.exists():
        pytest.skip(f"market data absent: {path}")
    import pandas as pd

    df = pd.read_csv(path, usecols=["observation_date", "BTC_CLOSE"])
    px = df["BTC_CLOSE"].to_numpy(float)
    r = np.append(0.0, np.diff(np.log(px)))
    x = px[:-1]
    y = r[1:]
    p = infer_shape_prior(x, y, **CUTS)
    print("\n[btc]", p.to_dict())
    assert p.family != FAMILY_PIECEWISE_CONSTANT, p.note
    assert p.n_change_points == 0, p.change_points


def test_too_few_rows_reports_absent():
    rng = np.random.RandomState(5)
    x = rng.uniform(0, 1, 10)
    y = rng.randn(10)
    p = infer_shape_prior(x, y, **CUTS)
    assert p.family == FAMILY_NO_STRUCTURE, p.note


def test_vocabulary_is_frozen():
    assert set(shape_vocabulary()) == {"piecewise_constant", "smooth", "no_structure"}


# ---------------------------------------------------------------------------
# Step 2: step-family basis at data-estimated change-points.
# ---------------------------------------------------------------------------
CUTS2 = {"min_segment_len": 5, "max_segments": 32, "flat_tol": 0.05, "min_flat_fraction": 0.5,
             "min_run_length": 5.0, "min_pc_r2": 0.05, "min_rows": 40,
             "sse_target": 0.01}


def test_staircase_recovered_exactly_with_estimated_thresholds():
    """The point of the whole exercise: staircase recovered with NO declared step op.

    Verified against the known generating form y = 2.5 + 0.5*sum_i sign(x - e_i)
    with e = 1..5. Coefficients and thresholds both come from the data.
    """
    rng = np.random.RandomState(42)
    vg = np.sort(rng.uniform(0.1, 5.9, 200))
    g = np.zeros_like(vg)
    for e in (1.0, 2.0, 3.0, 4.0, 5.0):
        g += (vg >= float(e))
    prior, seeds = staircase_prior(vg, g, **CUTS2)
    print(f"\n[step2] r2={seeds.r2!r} coeffs={seeds.coefficients!r}")
    assert prior.family == FAMILY_PIECEWISE_CONSTANT
    assert seeds is not None and seeds.identifiable
    assert seeds.r2 == pytest.approx(1.0, abs=1e-9)
    # Coefficients match the true generating values.
    assert seeds.intercept == pytest.approx(2.5, abs=0.05)
    step_coeffs = seeds.coefficients[1:]  # coefficients[0] is the intercept
    assert all(c == pytest.approx(0.5, abs=0.05) for c in step_coeffs)
    # An edge estimate is a MIDPOINT between two adjacent rows, so its error is
    # bounded by half the local gap. Use the largest gap in x as the honest
    # bound: measured max error 0.027 vs bound 0.086.
    max_gap = float(np.max(np.diff(vg)))
    for e in (1.0, 2.0, 3.0, 4.0, 5.0):
        assert any(abs(c - e) <= 1.01 * max_gap / 2 for c in seeds.thresholds), \
            (e, seeds.thresholds, max_gap / 2)
    # The emitted form is exactly representable in the existing node vocabulary.
    assert set(seeds.basis_names) == {"1"} | {f"sign(x - {t:.6g})" for t in seeds.thresholds}


def test_smooth_line_is_never_handed_a_step_family():
    """Gate dependence, the critical negative control.

    `fit_step_family` in isolation fits a STRAIGHT LINE at R2 ~0.979 with the
    sign basis (6 constants approximate a line over a narrow range), and
    reports identifiable=True. So R2 from this basis is NOT admissible
    evidence: only the shape prior's independent verdict makes it admissible.
    `staircase_prior` must therefore decline to emit seeds on a line, and this
    test asserts exactly that the raw R2 would have been misleading.
    """
    rng = np.random.RandomState(1)
    x = rng.uniform(0.1, 6.0, 200)
    y = 2.5 * x + 10.0 + 0.02 * rng.randn(200)
    prior, seeds = staircase_prior(x, y, **CUTS2)
    print(f"\n[gate] prior={prior.family} seeds={seeds}")
    assert prior.family != FAMILY_PIECEWISE_CONSTANT
    assert seeds is None, "a smooth line must never be handed a step family"
    # Proof that the gate is doing real work, not that the numbers were small.
    raw = fit_step_family(x, y, (1.0, 2.0, 3.0, 4.0, 5.0))
    print(f"[gate] raw R2 on a straight line = {round(raw.r2, 4)} identifiable={raw.identifiable}")
    assert raw.r2 > 0.9, raw.r2


def test_noise_is_never_handed_a_step_family():
    rng = np.random.RandomState(7)
    x = rng.uniform(0.1, 6.0, 200)
    y = rng.randn(200)
    prior, seeds = staircase_prior(x, y, **CUTS2)
    print("\n[noise-step2]", prior.family, seeds)
    assert seeds is None


def test_noisy_staircase_still_identifiable():
    """2% observation noise must not destroy step identification."""
    rng = np.random.RandomState(11)
    vg = np.sort(rng.uniform(0.1, 5.9, 200))
    g = np.zeros_like(vg)
    for e in (1.0, 2.0, 3.0, 4.0, 5.0):
        g += (vg >= float(e))
    g = g + 0.02 * g.std() * rng.randn(200)
    prior, seeds = staircase_prior(vg, g, **CUTS2)
    print("\n[noisy-2pct]", prior.family, None if seeds is None else round(seeds.r2, 4))
    assert seeds is not None and seeds.identifiable
    assert seeds.r2 > 0.95, seeds.r2


def test_too_few_rows_for_step_fit():
    rng = np.random.RandomState(12)
    x = rng.uniform(0, 1, 20)
    y = np.zeros(20)
    seeds = fit_step_family(x, y, (0.2, 0.5, 0.8))
    assert not seeds.identifiable


def test_step_family_basis_names_and_shape():
    names, M = step_family_basis([0.0, 1.0, 2.0], (1.0,))
    assert names[0] == "1" and len(names) == 2
    assert M.shape == (3, 2)
    assert list(M[:, 1]) == [-1.0, 1.0, 1.0]
