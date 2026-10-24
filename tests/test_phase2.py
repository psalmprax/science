"""Regression tests for Phase 2 delay, integral, and 3D meshing APIs."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from apedp import (
    delaunay_triangulate_3d,
    evaluate_delay_ivp,
    evaluate_delay_system,
    solve_nystrom_fredholm,
    solve_nystrom_volterra,
)


def test_delay_scalar_respects_constant_history():
    xs = np.linspace(0.0, 2.0, 41)
    values = evaluate_delay_ivp("-y + yd", 1.0, xs, delay=0.5, history=1.0)
    assert values.shape == xs.shape
    assert np.max(np.abs(values - 1.0)) < 1e-12
    sparse = evaluate_delay_ivp("-y + yd", 1.0, [0.0, 1.0, 2.0], delay=0.5, history=1.0)
    assert np.max(np.abs(sparse - 1.0)) < 1e-12


def test_delay_system_uses_component_history():
    xs = np.asarray([0.0, 0.5, 1.0])
    values = evaluate_delay_system(
        ["-y1 + d1", "-y2 + d2"],
        [1.0, 2.0],
        xs,
        delay=0.25,
        history=[1.0, 2.0],
    )
    assert values.shape == (len(xs), 2)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values - np.asarray([1.0, 2.0]))) < 1e-12


def test_nystrom_fredholm_constant_kernel_has_closed_form():
    result = solve_nystrom_fredholm("1", "1", 0.0, 1.0, nodes=24, coupling=0.5)
    assert np.max(np.abs(result["solution"] - 2.0)) < 1e-10
    assert result["residual_norm"] < 1e-10
    assert result["condition_number"] < 10.0


def test_nystrom_volterra_matches_exponential_solution():
    result = solve_nystrom_volterra("1", "1", 0.0, 1.0, nodes=128)
    assert result["solution"][-1] == pytest.approx(np.e, rel=2e-4)
    assert result["residual_norm"] < 1e-10


def test_delaunay_3d_cube_is_manifold_and_covers_all_points():
    points = np.asarray([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    mesh = delaunay_triangulate_3d(points)
    tetrahedra = np.asarray(mesh["tetrahedra"], dtype=int)
    assert mesh["num_nodes"] == 8
    assert mesh["num_elements"] >= 5
    assert np.array_equal(np.unique(tetrahedra), np.arange(8))
    faces = Counter(tuple(sorted(face)) for tetra in tetrahedra for face in (
        (tetra[0], tetra[1], tetra[2]),
        (tetra[0], tetra[3], tetra[1]),
        (tetra[0], tetra[2], tetra[3]),
        (tetra[1], tetra[3], tetra[2]),
    ))
    assert max(faces.values()) <= 2
    volume = sum(
        abs(np.linalg.det(np.vstack((points[t[1]] - points[t[0]], points[t[2]] - points[t[0]], points[t[3]] - points[t[0]])))) / 6
        for t in tetrahedra
    )
    assert volume == pytest.approx(1.0, rel=1e-12)


def test_delaunay_3d_refuses_coplanar_points():
    with pytest.raises(ValueError, match="coplanar"):
        delaunay_triangulate_3d([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
