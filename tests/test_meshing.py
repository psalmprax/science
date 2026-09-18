"""Tests for mesh generation, smoothing, and refinement."""
from __future__ import annotations

import numpy as np
import pytest

from apedp.meshing import (
    delaunay_triangulate,
    refine_mesh_2d,
    smooth_mesh_2d,
    uniform_mesh_2d,
)
from apedp.pde_solvers import solve_poisson_fem


def _sine_source() -> str:
    return "2*9.8696*sin(3.14159*x)*sin(3.14159*y)"


def _sine_error(nodes, u) -> float:
    pts = np.asarray(nodes)
    exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    return float(np.max(np.abs(np.asarray(u) - exact)))


def test_uniform_mesh_counts():
    m = uniform_mesh_2d(6, 5)
    assert m["num_nodes"] == 30
    assert m["num_elements"] == 40


def test_delaunay_random_validates_and_solves():
    rng = np.random.RandomState(0)
    m = delaunay_triangulate(rng.uniform(0, 1, size=(150, 2)))
    assert m["num_elements"] > 200
    r = solve_poisson_fem(m["nodes"], m["triangles"], source=_sine_source())
    assert bool(np.all(np.isfinite(r["u"])))


def test_delaunay_refuses_collinear():
    with pytest.raises(ValueError, match="collinear"):
        delaunay_triangulate([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


def test_clipped_jitter_fem_after_snap():
    """Boundary-hugging clusters snap onto hull edges: error < 0.1."""
    rng = np.random.RandomState(2)
    gx, gy = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 1, 12))
    pts = np.clip(
        np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.02, 0.02, size=(144, 2)),
        0, 1,
    )
    m = delaunay_triangulate(pts)
    r = solve_poisson_fem(m["nodes"], m["triangles"], source=_sine_source())
    assert _sine_error(r["nodes"], r["u"]) < 0.1


def test_smooth_keeps_counts_and_stays_finite():
    m = uniform_mesh_2d(9, 9)
    s = smooth_mesh_2d(m["nodes"], m["triangles"], iterations=5)
    assert s["num_nodes"] == m["num_nodes"]
    assert s["num_elements"] == m["num_elements"]
    r = solve_poisson_fem(
        s["nodes"], s["triangles"],
        source="2*9.869604401089358*sin(3.141592653589793*x)*sin(3.141592653589793*y)",
    )
    assert _sine_error(r["nodes"], r["u"]) < 0.1


def test_refine_returns_valid_mesh():
    rng = np.random.RandomState(4)
    gx, gy = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    pts = np.clip(np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.01, 0.01, size=(64, 2)), 0, 1)
    m = delaunay_triangulate(pts)
    rm = refine_mesh_2d(m["nodes"], m["triangles"], max_angle_deg=150.0)
    r = solve_poisson_fem(rm["nodes"], rm["triangles"], source=_sine_source())
    assert _sine_error(r["nodes"], r["u"]) < 1.0


def test_refine_kills_hand_built_obtuse_corner():
    from apedp.meshing import _triangle_max_angle

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.02]])
    tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    before = max(_triangle_max_angle(*nodes[t]) for t in tris)
    assert before > 150.0
    rm = refine_mesh_2d(nodes, tris, max_angle_deg=150.0)
    after = max(
        _triangle_max_angle(*np.asarray(rm["nodes"])[t]) for t in np.asarray(rm["triangles"])
    )
    assert after < before
