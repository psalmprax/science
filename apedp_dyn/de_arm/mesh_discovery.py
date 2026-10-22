"""DE arm, PDE prototype layer: cotangent-Laplacian discovery on meshes.

This ACTIVATES the ``pde_mesh_backlog`` gate as an EXPERIMENTAL PROTOTYPE
(the prototype itself is the evidence base for the frozen-benchmark
decision the gate demands — it makes no production claim):

  Layer A (typing)  — every candidate is declared through the arm's
                      ``PdeMathModel`` (fields, independent variables,
                      positive dt, boundary treatment); incomplete or
                      unsupported declarations are fail-closed refusals.
  Layer B (physics) — Discrete-cotangent Laplace-Beltrami assembly on a
                      2D Delaunay mesh (the operator carries GEOMETRY:
                      cotangent weights + barycentric vertex areas, NOT a
                      plain graph Laplacian D − A — the reviewed trap).
  Layer C (discover)— time-derivative residual regression per the arm's
                      documented python ridge+prune STRidge mirror
                      (``dependency_screen._stridge_fit``; kernel parity
                      never claimed) over the mesh library
                      {1, u, u^2, u^3, Δu (, u·Δu)} on INTERIOR vertices.

  Escape route      — the structured-grid Fornberg engine
                      (``epistemic.pde_discovery.discover_pde``) remains
                      the established route for grid-runnable data; the
                      mesh prototype covers irregular domains and never
                      competes silently with it.

Honest scope (the prototype's limitations, stated where a referee will
read them): one 2D domain class (jittered planar meshes), one temporal
derivative, DELAUNAY-planar meshes only (no intrinsic remeshing), the
term library above (no advection field, no higher-order spatial
operators), boundary handling at 'dirichlet' (interior-only fitting on a
zero boundary) and 'none' (honest note: boundary triangle weights are
single-cotangent, so a 2-ring interior mask is used there too).
'periodic' on a triangle mesh is REFUSED — mesh periodicity is a
different operator (flat torus) this prototype does not pretend to
assemble.

Evidence-only: the outcome carries NO acceptance, promotion or verdict
fields — the frozen referee protocol owns those decisions.

Status: experimental (PDE arm prototype phase 1, behind the
``pde_mesh_backlog`` gate — see docs/pde_mesh_discovery_prototype.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # scipy is the planar-Delaunay/sparse dependency of this prototype
    import scipy.sparse as sp
    from scipy.spatial import Delaunay
except ImportError as exc:  # pragma: no cover - env-dependent
    raise ImportError(
        "mesh_discovery needs scipy (Delaunay + sparse); the structured-grid "
        "PDE path does not") from exc

from apedp_dyn.de_arm.dependency_screen import _stridge_fit
from apedp_dyn.de_arm.typed_model import PdeMathModel

__all__ = [
    "MeshPdeOutcome",
    "assemble_cotangent_laplacian",
    "discover_mesh_pde",
]

_MIN_INTERIOR_VERTICES = 25
_MIN_FRAMES = 5
_MAX_ABS_COT = 1e6          # degenerate-angle guard (sliver refusal)
_MIN_TRIANGLE_AREA = 1e-14


@dataclass(frozen=True)
class MeshPdeOutcome:
    """Evidence-only mesh PDE discovery artifact (never a verdict)."""

    active_terms: tuple[str, ...]
    coefficients: dict[str, float]
    train_r2: float
    train_rmse: float
    mesh: dict[str, float]              # vertices/triangles/area/edge stats
    boundary: str
    typed_model: dict[str, Any]
    library: tuple[str, ...]
    referee_note: str = (
        "evidence only: cotangent-Laplacian mesh library terms + python "
        "ridge-prune mirror coefficients; the structured-grid Fornberg "
        "engine (epistemic.pde_discovery.discover_pde) is the escape route "
        "for grid-runnable data; acceptance belongs to the frozen referee "
        "protocol")


def _edge_boundary_mask(triangles: np.ndarray, n: int) -> np.ndarray:
    """Vertices on edges owned by exactly ONE triangle = mesh boundary."""
    counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e in ((a, b), (b, c), (c, a)):
            key = (min(e), max(e))
            counts[key] = counts.get(key, 0) + 1
    boundary = np.zeros(n, dtype=bool)
    for (p, q), c in counts.items():
        if c == 1:
            boundary[p] = True
            boundary[q] = True
    return boundary


def assemble_cotangent_laplacian(
    vertices: np.ndarray,
    triangles: np.ndarray | None = None,
    area_kind: str = "barycentric",
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    """Cotangent Laplace-Beltrami (Δu)_i = (1/(2A_i)) Σ_j w_ij (u_j − u_i).

    w_ij = cot(α) + cot(β) with α, β the angles OPPOSITE edge (i, j) in
    its (≤2) adjacent triangles — the operator carries the mesh geometry,
    unlike a plain graph Laplacian D − A.

    Fail-closed: any non-positive or sliver triangle area
    (≤ min area / cotangent magnitude > guard) is a REFUSAL — a
    numerically unsafe cotangent operator is never handed to discovery.

    Returns (L csr [N,N], vertex_areas [N], interior_mask [N], diag).
    ``interior_mask`` is False on boundary vertices (edges owned by one
    triangle), False-safe for dirichlet masking.
    """
    if area_kind != "barycentric":
        raise ValueError(
            f"unknown area convention {area_kind!r}: only 'barycentric' "
            "(1/3 of each incident triangle) is validated in this prototype")
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("vertices must be [N, 2] planar coordinates")
    n = v.shape[0]
    if triangles is None:
        tri = Delaunay(v)
        triangles = np.asarray(tri.simplices, dtype=int)
    triangles = np.asarray(triangles, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be [T, 3] vertex indices")

    pa, pb, pc = v[triangles[:, 0]], v[triangles[:, 1]], v[triangles[:, 2]]
    cross_z = ((pb - pa)[:, 0] * (pc - pa)[:, 1]
               - (pb - pa)[:, 1] * (pc - pa)[:, 0])
    tri_area = 0.5 * np.abs(cross_z)
    if not np.all(tri_area > _MIN_TRIANGLE_AREA):
        bad = int(np.argmin(tri_area))
        raise ValueError(
            f"sliver/degenerate triangle refused (index {bad}, area "
            f"{tri_area[bad]:.3e} ≤ {_MIN_TRIANGLE_AREA}): a cotangent "
            "operator on this triangle is numerically unsafe")

    # Cotangent of the angle OPPOSITE each edge, per incident triangle:
    # cot = ((a−o)·(b−o)) / |cross(a−o, b−o)| for the opposite vertex o.
    weights_sum: dict[tuple[int, int], float] = {}
    for k, tri in enumerate(np.asarray(triangles, dtype=int)):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        # edge (a,b): opposite vertex c; etc.
        pairs = ((a, b, c), (b, c, a), (c, a, b))
        for (i, jj, opp) in pairs:
            if i == jj:
                raise ValueError(f"degenerate triangle {k}: repeated vertex")
            u_ = v[i] - v[opp]
            w_ = v[jj] - v[opp]
            den = abs(u_[0] * w_[1] - u_[1] * w_[0])
            if den <= 0.0:
                raise ValueError(
                    f"degenerate angle at edge ({i},{jj}) of triangle {k}: "
                    "cross product zero — sliver refusal")
            cot = float((u_ @ w_) / den)
            if not (np.isfinite(cot) and abs(cot) <= _MAX_ABS_COT):
                raise ValueError(
                    f"sliver refusal: |cotangent| {abs(cot):.3e} exceeds "
                    f"{_MAX_ABS_COT:.0e} at edge ({i},{jj}) of triangle {k}")
            key = (min(i, jj), max(i, jj))
            weights_sum[key] = weights_sum.get(key, 0.0) + float(cot)

    vertex_areas = np.zeros(n)
    for tri_idx in range(len(triangles)):
        cnt_area = tri_area[tri_idx] / 3.0
        for vtx in triangles[tri_idx]:
            vertex_areas[int(vtx)] += cnt_area
    if np.any(vertex_areas <= 0.0):
        raise ValueError("isolated vertex with zero incident area: mesh inconsistent")

    L = sp.lil_matrix((n, n))
    for (i, j), w in weights_sum.items():
        L[i, j] = w / (2.0 * vertex_areas[i])
        L[j, i] = w / (2.0 * vertex_areas[j])
    # Laplacian diagonal: L[i,i] = -Σ_{j≠i} w_ij / (2 A_i) — this is what
    # makes the operator a Laplacian (constants sit in its nullspace and
    # the Dirichlet energy is positive; without it L@1 ≠ 0 and discovery
    # compensates with enormous bogus u/constant coefficients).
    for i in range(n):
        acc = -sum(w for (p, q), w in weights_sum.items() if p == i or q == i)
        L[i, i] = acc / (2.0 * vertex_areas[i])
    diag = {
        "n_vertices": int(n),
        "n_triangles": len(triangles),
        "min_edge_len": None,  # filled by the caller with vertex info if needed
        "area_kind": area_kind,
        "guards": {"min_triangle_area": _MIN_TRIANGLE_AREA,
                   "max_abs_cot": _MAX_ABS_COT},
    }
    return L.tocsr(), vertex_areas, ~_edge_boundary_mask(triangles, n), diag


def discover_mesh_pde(
    frames: np.ndarray,
    vertices: np.ndarray,
    dt: float,
    boundary: str = "dirichlet",
    include_cross_lap: bool = True,
    interior_ring_from_boundary: int = 2,
) -> MeshPdeOutcome:
    """Discover u_t = F(u, Δu) on an irregular 2D mesh (evidence-only).

    Rows: (interior vertex, interior time) panel. Target: 2nd-order
    central time difference. Library: {1, u, u^2, u^3, laplacian(=Δu via
    the cotangent operator), (optional) u*laplacian}. Fitting is the
    arm's documented ridge+prune mirror (kernel parity NOT claimed).
    Boundary 'dirichlet'/'none' both fit on interior vertices ≥
    ``interior_ring_from_boundary`` hops away from the boundary (honest
    note recorded); 'periodic' on a triangle mesh is a REFUSAL.
    """
    if boundary == "periodic":
        raise ValueError(
            "mesh periodicity is a different operator (flat-torus wrap); "
            "this prototype refuses rather than fabricating it — use "
            "boundary='dirichlet' or 'none'")
    # fail-closed typing EARLY (dt/boundary validation before any search);
    # the final typed model is rebuilt below with the real median edge len
    PdeMathModel(
        fields=("u",), independent_variables=("t", "x"),
        dt=float(dt), dx={"x": 1.0}, boundary=boundary)
    fr = np.asarray(frames, dtype=float)
    v = np.asarray(vertices, dtype=float)
    if fr.ndim != 2 or fr.shape[1] != v.shape[0]:
        raise ValueError("frames must be [T, N] with N = len(vertices)")
    T, n = fr.shape
    if T < _MIN_FRAMES:
        raise ValueError(f"refusing: {T} frames < minimum {_MIN_FRAMES} "
                         "for a central time derivative")
    if not (np.isfinite(fr).all() and np.isfinite(v).all()):
        raise ValueError("mesh panels contain non-finite values: refusing to fit")
    if boundary not in ("dirichlet", "none"):
        raise ValueError(f"unsupported boundary treatment {boundary!r}")

    L, areas, interior0, diag = assemble_cotangent_laplacian(v)
    # 2-ring interior mask (edge-hop distance from the boundary)
    rings = _ring_distance_from_boundary(v, L, ~interior0)
    mask = rings >= interior_ring_from_boundary
    # median edge length (typed-model dx on the mesh) replaces the placeholder
    if int(mask.sum()) < _MIN_INTERIOR_VERTICES:
        raise ValueError(
            f"refusing: {int(mask.sum())} time-interior masked vertices < "
            f"{_MIN_INTERIOR_VERTICES} (mesh too small near its boundary)")

    lap_all = np.asarray(L @ fr.T).T          # [T, N]: Δu per (frame, vertex)
    t_int = np.arange(1, T - 1)
    rows_t = np.repeat(t_int, int(mask.sum()))
    rows_v = np.tile(np.nonzero(mask)[0], T - 2)
    ut = (fr[rows_t + 1, rows_v] - fr[rows_t - 1, rows_v]) / (2.0 * dt)
    u_rows = fr[rows_t, rows_v]
    lap_rows = lap_all[rows_t, rows_v]
    if not float(np.std(ut)) > 1e-12:
        raise ValueError("target time-derivative is (near-)constant: nothing to fit")

    lib: dict[str, np.ndarray] = {
        "1": np.ones_like(ut),
        "u": u_rows,
        "u^2": u_rows ** 2,
        "u^3": u_rows ** 3,
        "laplacian": lap_rows,
    }
    names = ["1", "u", "u^2", "u^3", "laplacian"]
    if include_cross_lap:
        lib["u*laplacian"] = u_rows * lap_rows
        names.append("u*laplacian")

    active, coefs, r2, rmse = _stridge_fit(ut, lib)

    edges_len = _median_edge_length(v, diag)
    typed = PdeMathModel(
        fields=("u",),
        independent_variables=("t", "x"),
        dt=float(dt),
        dx={"x": float(edges_len)},
        boundary=boundary,
        extra={
            "note": "dx is the MEDIAN EDGE LENGTH — the mesh's characteristic spacing",
            "n_interior_fitted": int(mask.sum()),
            "assembly_guard": diag["guards"],
        },
    )
    return MeshPdeOutcome(
        active_terms=tuple(active),
        coefficients={k: float(c) for k, c in coefs.items()},
        train_r2=float(r2),
        train_rmse=float(rmse),
        mesh={
            "n_vertices": float(n),
            "n_triangles": float(diag["n_triangles"]),
            "min_vertex_area": float(areas.min()),
            "median_edge_len": float(edges_len),
            "n_frames": float(T),
            "dt": float(dt),
        },
        boundary=boundary,
        typed_model={
            "kind": typed.kind,
            "fields": list(typed.fields),
            "independent_variables": list(typed.independent_variables),
            "dt": typed.dt,
            "dx": dict(typed.dx),
            "boundary": typed.boundary,
        },
        library=tuple(names),
    )


def _median_edge_length(v: np.ndarray, diag: dict[str, Any]) -> float:
    """Delaunay edges (unique triangle edges) median length — the typed dx."""
    try:
        tri = Delaunay(v)
        sims = np.asarray(tri.simplices, dtype=int)
    except Exception as exc:  # pragma: no cover - degenerate point sets raise
        raise ValueError(f"Delaunay failed on this mesh: {exc}") from exc
    edges = set()
    for a, b, c in sims:
        for e in ((a, b), (b, c), (c, a)):
            edges.add((min(e), max(e)))
    idx = np.array(sorted(edges))
    lens = np.linalg.norm(v[idx[:, 0]] - v[idx[:, 1]], axis=1)
    return float(np.median(lens))


def _ring_distance_from_boundary(
    v: np.ndarray, L: sp.csr_matrix, boundary_mask: np.ndarray
) -> np.ndarray:
    """Edge-hop (BFS) distance from the nearest boundary vertex."""
    n = len(v)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    Lc = L.tocoo()
    for i, j in zip(Lc.row, Lc.col, strict=True):
        if i != j:
            adj[int(i)].append(int(j))
    dist = np.full(n, np.inf)
    frontier = [int(i) for i in np.nonzero(boundary_mask)[0]]
    for i in frontier:
        dist[i] = 0
    d = 0
    while frontier:
        nxt: list[int] = []
        for i in frontier:
            for j in adj[i]:
                if dist[j] == np.inf:
                    dist[j] = d + 1
                    nxt.append(int(j))
        frontier = nxt
        d += 1
    return dist
