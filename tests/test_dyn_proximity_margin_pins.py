"""Proximity-margin pins: the non-neighbor barrier argument, machine-checked.

The pitch lives in ``apedp_dyn/robotics/proximity_margin.py`` — pairs at
barrier height h >= h* = 2·v_max/γ need NO explicit constraint: the
worst-case box-bounded relative velocity (−2·v_max) can never violate
hdot ≥ −γ·h there. These pins enforce that the pitch's algebra and its
solver stay honest:

1. **Margin formula** — h* = 2·v_max/γ exactly; defaults give h* = 4.0
   length units (r_safe excluded — h measures distance BEYOND r_safe).
2. **Midpoint saturation** — a pair at barrier bottom EXACTLY h*, with
   both agents closing at top speed (worst case), saturates the paired
   law:  (v_i − v_j)·n_ij = −2·v_max = −γ·h* — equality, never violation.
3. **Restricted solve = certified honesty** — ``solve_cbf_qp_proximity``
   constrains only pairs with h < h*, but its release passes the FULL
   all-pairs fail-closed re-check on randomized fixtures (any defect >
   tol raises).
4. **Epistemic bookkeeping** — the outcome records constrained/free/all
   pairs and the measured constraint ratio; on the A3-guarded grid
   fixture the ratio is FAR below 1 (the countable evidence the N≥256
   gate needed), and identical-agent commands (factory lemma) remain
   valid with a reduced set too.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    CertificateViolation,
    _pair_barriers,
)
from apedp_dyn.robotics.proximity_margin import (
    free_margin,
    indexed_barrier_components,
    indexed_proximity_pairs,
    model_check_unicycle_horizon_proximity,
    solve_cbf_qp_proximity,
)


def _assumptions(max_agents: int = 256) -> CBFQPCertificateAssumptions:
    return CBFQPCertificateAssumptions(
        v_max=1.0, r_safe=0.5, gamma=0.5, dt=0.05, max_agents=max_agents)


def _grid_fixture(n: int, seed: int, spacing: float = 3.0):
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n)) + 1
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * spacing
    pts = pts + rng.uniform(-0.4, 0.4, pts.shape)
    vel = rng.uniform(-1.0, 1.0, (n, 2))
    # A3 guard: min inter-agent gap > r_safe
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d[np.diag_indices(n)] = np.inf
    if not d.min() > 0.5:
        raise RuntimeError("fixture violates A3")
    return pts, vel


def test_margin_formula_exact():
    a = _assumptions()
    assert free_margin(a) == pytest.approx(2.0 * a.v_max / a.gamma)
    # documented default regime: v_max=1, gamma=0.5 → h* = 4.0 length units
    assert free_margin(a) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        free_margin(CBFQPCertificateAssumptions(
            v_max=1.0, r_safe=0.5, gamma=0.0, dt=0.05))


def test_midpoint_saturation_at_exact_h_star():
    """Worst case at h = h* exactly: anti-aligned top-speed closing
    SATURATES the paired law (equality), never violates it."""
    a = _assumptions()
    h_star = free_margin(a)
    p0 = np.array([[0.0, 0.0], [a.r_safe + h_star, 0.0]])
    h = p0[1, 0] - p0[0, 0] - a.r_safe
    assert h == pytest.approx(h_star)
    v = np.array([[a.v_max, 0.0], [-a.v_max, 0.0]])  # both closing at top speed
    unit_pi_minus_pj = np.array([-1.0, 0.0])         # (p_i − p_j)/d
    lhs = float((v[0] - v[1]) @ unit_pi_minus_pj)
    rhs = -a.gamma * h
    assert lhs >= rhs
    assert lhs == pytest.approx(rhs)                 # saturated exactly


def test_restricted_solve_passes_full_all_pairs_recheck():
    a = _assumptions()
    for seed in range(5):
        pts, vel = _grid_fixture(12, seed)
        out = solve_cbf_qp_proximity(pts, vel, a)
        # the solver itself ran the full all-pairs fail-closed re-check;
        # assert the released commands still satisfy every pair here too
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        n = len(pts)
        idxi, idxj = np.triu_indices(n, k=1)
        unit = diff[idxi, idxj] / np.maximum(dist[idxi, idxj], 1e-12)[:, None]
        lhs = np.einsum("pk,pk->p", unit, out.commands[idxi] - out.commands[idxj])
        assert float(((-a.gamma * (dist[idxi, idxj] - a.r_safe)) - lhs).max()) <= a.tol


def test_restricted_solve_refuses_box_violations_fail_closed():
    """max_iter=0 asks for an UNSOLVED command: a provably-violated nominal
    (two agents just outside r_safe, closing at top speed) must reach the
    full all-pairs re-check and raise — never be silently released."""
    a = _assumptions()
    # pair gap barely outside r_safe: h tiny → constrained pair, and the
    # zero-sweep nominal deficit is −γ·h − (−2·v_max) = 2·v_max − γ·h >> tol
    gap = a.r_safe + 1e-3
    pts = np.array([[0.0, 0.0], [gap, 0.0]])
    vel = np.array([[a.v_max, 0.0], [-a.v_max, 0.0]])   # both closing at top speed
    h = gap - a.r_safe
    nominal_deficit = a.gamma * h - (-2.0 * a.v_max)    # rhs − lhs with lhs = −2·v_max
    assert nominal_deficit > 1e-3                       # provably violated pre-solve
    with pytest.raises(CertificateViolation):
        solve_cbf_qp_proximity(pts, vel, a, max_iter=0)


def test_constrained_pair_bookkeeping_and_ratio():
    a = _assumptions()
    pts, vel = _grid_fixture(64, seed=3)
    out = solve_cbf_qp_proximity(pts, vel, a)
    assert out.all_pairs == 64 * 63 // 2
    assert out.constrained_pairs + out.free_pairs == out.all_pairs
    # the grid fixture is sparse relative to h* = 4.0: most pairs are FREE
    assert out.constraint_ratio < 0.5
    assert out.h_star == pytest.approx(free_margin(a))
    # identical-agent commands (factory lemma): no deficits anywhere, the
    # reduced set remains satisfiable trivially
    v0 = np.full((64, 2), 0.5)
    out0 = solve_cbf_qp_proximity(pts, v0, a)
    assert np.allclose(out0.commands, v0, atol=1e-12)


# ── spatial index: the dense floor can never hide a constrained pair ─────
def test_index_candidates_superset_and_excluded_provably_free(seed=5, n=128):
    a = _assumptions()
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n)) + 1
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * 2.0
    pts = pts + rng.uniform(-0.4, 0.4, pts.shape)

    pairs, diag = indexed_proximity_pairs(pts, a)
    bar = _pair_barriers(pts, a.r_safe)
    h_star = free_margin(a)
    idxi, idxj = np.triu_indices(n, k=1)
    cand = {(int(p), int(q)) for p, q in pairs}
    constrained = {(int(i), int(j)) for i, j in
                   zip(idxi[bar.h[idxi, idxj] < h_star],
                       idxj[bar.h[idxi, idxj] < h_star], strict=True)}
    # 1. the candidate set covers the ENTIRE constrained band
    assert constrained <= cand
    # 2. every EXCLUDED pair provably free: d ≥ r_safe + h* (exact margin)
    dist_all = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    excluded = {(int(i), int(j)) for i, j in zip(idxi, idxj, strict=True)} - cand
    for p, q in excluded:
        assert dist_all[p, q] >= a.r_safe + h_star - 1e-9, (p, q)
    assert diag["pair_reduction"] > 0.8   # the counted evidence, measured


def test_boundary_equality_pair_is_free_and_still_certified():
    """d == r_safe + h* exactly: scipy's `at most r` candidates include it,
    the dense h < h* filter does NOT (h = h* → free). Either way the
    release must pass the all-pairs re-check."""
    a = _assumptions()
    gap = a.r_safe + free_margin(a)
    pts = np.array([[0.0, 0.0], [gap, 0.0]])
    pairs, _diag = indexed_proximity_pairs(pts, a)
    assert (0, 1) in {(int(p), int(q)) for p, q in pairs}   # candidate (free)
    vel = np.array([[a.v_max, 0.0], [-a.v_max, 0.0]])       # worst-case closing
    out = solve_cbf_qp_proximity(pts, vel, a)               # must certify, not raise
    assert out.constrained_pairs == 0                       # free at the exact boundary


def test_indexed_barrier_components_match_dense_exactly(seed=5, n=128):
    a = _assumptions()
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n)) + 1
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * 2.0
    pts = pts + rng.uniform(-0.4, 0.4, pts.shape)

    pairs, unit, h, _diag = indexed_barrier_components(pts, a)
    bar = _pair_barriers(pts, a.r_safe)
    assert np.array_equal(unit, bar.unit[pairs[:, 0], pairs[:, 1]])
    assert np.array_equal(h, bar.h[pairs[:, 0], pairs[:, 1]])


@pytest.mark.parametrize("seed", [9, 11])
def test_horizon_model_check_proximity_collision_free(seed):
    """The proximity-LOCKED realization through the production arc stepper
    must hold h > 0 on adversarial horizons — measured evidence, explicitly
    not a formal claim."""
    a = _assumptions()
    # adversarial head-on courses: both agents top-speed toward each other
    p0 = np.array([[0.0, 0.0], [3.0, 0.0]])
    th0 = np.array([0.0, np.pi])
    res = model_check_unicycle_horizon_proximity(p0, th0, a, steps=60, seed=seed)
    assert res["min_h_observed"] > 0.0
    assert res["min_h_observed"] >= 1e-3   # realized margin never approaches r_safe

    # random multi-agent grid closing under adversarial headings
    rng = np.random.default_rng(seed)
    n = 16
    side = int(np.sqrt(n)) + 1
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * 1.6
    pts = pts + rng.uniform(-0.3, 0.3, pts.shape)
    th = rng.uniform(-np.pi, np.pi, n)
    res2 = model_check_unicycle_horizon_proximity(pts, th, a, steps=80, seed=seed)
    assert res2["min_h_observed"] > 0.0
    assert res2["min_h_observed"] >= 1e-3
