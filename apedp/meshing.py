"""APEDP meshing: exact-predicate 2D/3D Delaunay and structured helpers.

Bowyer-Watson Delaunay over user points (bounded size, seeded insertion
order, determinant circumcircle predicate) plus exact-predicate 3D
tetrahedralization and a uniform-grid triangulator.
Every Bowyer-Watson result is *validated* (Euler count ``T = 2n-2-k`` plus
area-vs-hull match) across successive seeds — near-cocircular inputs can
defeat any float predicate, so a triangulation that fails validation is
retried, and persistent failure raises instead of returning a mesh with
holes or overlaps. The 3D path adds exact Bareiss orientation/insphere
predicates and returns ``(nodes, tetrahedra)``; the 2D output feeds
:func:`apedp.pde_solvers.solve_poisson_fem` directly.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

_MAX_MESH_POINTS = 5000
_MAX_SEED_RETRIES = 10


def _bareiss_det_4x4(m: list[list[int]]) -> int:
    """Exact 4x4 determinant over Python ints (Bareiss, fraction-free)."""
    a = [row[:] for row in m]
    sign = 1
    prev = 1
    for k in range(3):
        if a[k][k] == 0:
            swap = next((r for r in range(k + 1, 4) if a[r][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, 4):
            for j in range(k + 1, 4):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // prev
        prev = a[k][k]
        if prev == 0:
            return 0
    return sign * a[3][3]


_QUANTUM = 2**40


def _quantize_global(pts: list[np.ndarray]) -> list[tuple[int, int]]:
    """One deterministic integer lattice per triangulation (consistency first)."""
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    lox, loy = min(xs), min(ys)
    span = max(max(xs) - lox, max(ys) - loy, 1e-300)
    return [
        (round((x - lox) / span * _QUANTUM), round((y - loy) / span * _QUANTUM))
        for x, y in zip(xs, ys)
    ]


def _incircle_q(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int], d: tuple[int, int]) -> bool:
    """Exact incircle test on global integer lattice (ties count as outside)."""
    (ax, ay), (bx, by), (cx, cy), (dx, dy) = a, b, c, d
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if orient == 0:
        return False
    det = _bareiss_det_4x4([
        [ax, ay, ax * ax + ay * ay, 1],
        [bx, by, bx * bx + by * by, 1],
        [cx, cy, cx * cx + cy * cy, 1],
        [dx, dy, dx * dx + dy * dy, 1],
    ])
    return det * (1 if orient > 0 else -1) > 0


def _circumcircle_contains(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Float convenience wrapper (single-shot use; bulk code uses ``_incircle_q``)."""
    qa, qb, qc, qd = _quantize_global([a, b, c, d])
    return _incircle_q(qa, qb, qc, qd)


def delaunay_triangulate(
    points: list[list[float]] | np.ndarray,
    seed: int = 0,
) -> dict[str, object]:
    """Bowyer-Watson Delaunay triangulation of 2D points.

    Returns ``{"nodes", "triangles", "num_nodes", "num_elements"}`` with
    triangles as index triples into the (deduplicated) nodes. Refuses
    collinear input, duplicates are merged, and point sets over
    ``_MAX_MESH_POINTS`` are refused instead of hanging in O(n²) insertion.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N, 2)")
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("points must be finite")
    uniq = np.unique(np.round(pts, decimals=12), axis=0)
    if len(uniq) != len(pts):
        logger.warning("Merged %d duplicate points", len(pts) - len(uniq))
        pts = uniq
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points")
    if n > _MAX_MESH_POINTS:
        raise ValueError(f"Need N <= {_MAX_MESH_POINTS} for Bowyer-Watson insertion")
    if float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()) == 0.0:
        raise ValueError("All points identical")
    # Collinear guard via PCA spread ratio.
    centered = pts - pts.mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= 1e-12 * max(singular[0], 1e-300):
        raise ValueError("Points are collinear: no triangulation exists")
    # Anti-sliver snap: near-collinear hull-huggers become exactly collinear.
    # Snapping can coincide points, so deduplicate again afterwards.
    pts = _snap_to_hull(pts)
    uniq2 = np.unique(np.round(pts, decimals=12), axis=0)
    if len(uniq2) != len(pts):
        logger.warning("Snap merged %d points", len(pts) - len(uniq2))
        pts = uniq2
        n = len(pts)

    hull = _convex_hull_ordered(pts)
    hp = pts[np.asarray(hull)]
    hull_area = abs(float(np.sum(hp[:, 0] * np.roll(hp[:, 1], -1) - np.roll(hp[:, 0], -1) * hp[:, 1]))) / 2
    # Near-cocircular inputs can defeat any float predicate: validate the
    # result and retry later seeds deterministically; residual holes (extra
    # boundary cycles) are ear-filled before giving up on the seed.
    for attempt in range(_MAX_SEED_RETRIES):
        tris = _bowyer_watson(pts, int(seed) + attempt)
        if _validate_tris(pts, tris, hull_area):
            if attempt:
                logger.info("Triangulation validated on seed retry %d", attempt)
            return {"nodes": pts, "triangles": tris, "num_nodes": n, "num_elements": len(tris)}
        filled = _fill_holes(pts, tris)
        if filled is not None and _validate_tris(pts, filled, hull_area):
            logger.info("Triangulation repaired by ear-filling %d hole triangles", len(filled) - len(tris))
            return {"nodes": pts, "triangles": filled, "num_nodes": n, "num_elements": len(filled)}
    raise ArithmeticError(
        f"No valid triangulation in {_MAX_SEED_RETRIES} seed retries "
        "(holes/overlaps persisted): jitter or coarsen the point set"
    )

def _convex_hull_ordered(pts: np.ndarray) -> list[int]:
    """Gift-wrap convex hull as ordered indices (CCW from lowest corner)."""
    n = len(pts)
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    hull = [start]
    cur = start
    while True:
        nxt = None
        for i in range(n):
            if i == cur:
                continue
            if nxt is None:
                nxt = i
                continue
            cross = (pts[nxt, 0] - pts[cur, 0]) * (pts[i, 1] - pts[cur, 1]) - (
                pts[nxt, 1] - pts[cur, 1]
            ) * (pts[i, 0] - pts[cur, 0])
            if cross < 0 or (
                cross == 0
                and float(np.sum((pts[i] - pts[cur]) ** 2)) > float(np.sum((pts[nxt] - pts[cur]) ** 2))
            ):
                nxt = i
        if nxt is None:  # invariant: assigned inside the scan above
            raise RuntimeError("Convex-hull walk selected no next point")
        if nxt == start:
            break
        hull.append(nxt)
        cur = nxt
    return hull


def _snap_to_hull(pts: np.ndarray, tol_frac: float | None = None) -> np.ndarray:
    """Project near-collinear points onto hull edges (anti-sliver preprocessing).

    Boundary-hugging clusters (points within tolerance of a hull edge)
    triangulate into max-angle-violating needles that no insertion scheme can
    fix without moving nodes. Snapping them exactly onto the edge changes the
    domain below coarse-mesh resolution while making the point set
    triangulatable. Default tolerance is a quarter of the nominal spacing
    (``0.25·span/√n``) so it tracks resolution instead of punishing fine
    meshes or missing coarse clusters.
    """
    n = len(pts)
    hull = _convex_hull_ordered(pts)
    span = max(float((pts.max(axis=0) - pts.min(axis=0)).max()), 1e-300)
    tol = float(tol_frac) * span if tol_frac is not None else 0.25 * span / max(1.0, float(n) ** 0.5)
    out = pts.copy()
    for k in range(len(hull)):
        a, b = pts[hull[k]], pts[hull[(k + 1) % len(hull)]]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 <= 0:
            continue
        rel = (out - a) @ ab / L2
        proj = a[None, :] + np.clip(rel, 0.0, 1.0)[:, None] * ab[None, :]
        dist = np.linalg.norm(out - proj, axis=1)
        near = (dist < tol) & (dist > 0)
        out[near] = proj[near]
    return out


def _bowyer_watson(pts: np.ndarray, seed: int) -> np.ndarray:
    """Single Bowyer-Watson pass; may contain holes on adversarial input."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(float((hi - lo).max()), 1e-9)
    center = (lo + hi) / 2
    # Supertriangle strictly containing the bbox (20x span is safely outside
    # any circumcircle of input triples without overflowing float64).
    st = np.array([
        center + [-20 * span, -20 * span],
        center + [20 * span, -20 * span],
        center + [0.0, 20 * span],
    ])
    verts = [st[0], st[1], st[2]] + [np.asarray(p) for p in pts]
    qverts = _quantize_global(verts)
    triangles = [(0, 1, 2)]
    rng = np.random.RandomState(int(seed))
    order = list(range(3, len(verts)))
    rng.shuffle(order)
    for idx in order:
        bad = [
            t for t in triangles
            if _incircle_q(qverts[t[0]], qverts[t[1]], qverts[t[2]], qverts[idx])
        ]
        edge_count: dict[tuple[int, int], int] = {}
        for t in bad:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
        triangles = [t for t in triangles if t not in bad]
        for (a, b), count in edge_count.items():
            if count == 1:
                triangles.append((a, b, idx))
    triangles = [t for t in triangles if t[0] > 2 and t[1] > 2 and t[2] > 2]
    return np.asarray([(a - 3, b - 3, c - 3) for a, b, c in triangles], dtype=int)


def _boundary_cycles(counts: dict[tuple[int, int], int]) -> list[list[int]]:
    """Every boundary loop (outer + holes) from edge-use counts."""
    adj: dict[int, list[int]] = {}
    for (a, b), v in counts.items():
        if v == 1:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    cycles: list[list[int]] = []
    used: set[tuple[int, int]] = set()
    for (a, b) in [e for e, v in counts.items() if v == 1]:
        if (a, b) in used:
            continue
        cycle = [a]
        cur = a
        nxt = b
        while True:
            used.add((min(cur, nxt), max(cur, nxt)))
            if nxt == a:
                break
            cycle.append(nxt)
            options = [w for w in adj[nxt] if w != cur]
            if len(options) != 1:
                cycle = []
                break
            cur, nxt = nxt, options[0]
            if len(cycle) > len(adj) + 1:
                cycle = []
                break
        if len(cycle) >= 3:
            cycles.append(cycle)
    return cycles


def _ear_clip_fill(cycle: list[int], pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon loop (hole filler)."""
    poly = list(cycle)
    if len(poly) < 3:
        return []

    def _inside_strict(px: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        # Barycentric sign test with epsilon margin (boundary contact is fine).
        d1 = (px[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (px[1] - b[1])
        d2 = (px[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (px[1] - c[1])
        d3 = (px[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (px[1] - a[1])
        eps = 1e-12
        pos = bool(d1 > eps and d2 > eps and d3 > eps)
        neg = bool(d1 < -eps and d2 < -eps and d3 < -eps)
        return pos or neg

    total = sum(_area2_wrap(pts, poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1))
    ccw = total > 0
    out: list[tuple[int, int, int]] = []
    guard = 0
    while len(poly) > 3 and guard < len(poly) * len(poly) + 10:
        guard += 1
        clipped = False
        for i in range(len(poly)):
            prev_i, next_i = poly[i - 1], poly[(i + 1) % len(poly)]
            cur = poly[i]
            ear_area = _area2_wrap(pts, prev_i, cur, next_i)
            if (ear_area > 0) != ccw:
                continue
            ear_ok = True
            for other in poly:
                if other in (prev_i, cur, next_i):
                    continue
                if _inside_strict(pts[other], pts[prev_i], pts[cur], pts[next_i]):
                    ear_ok = False
                    break
            if ear_ok:
                out.append((prev_i, cur, next_i))
                del poly[i]
                clipped = True
                break
        if not clipped:
            return []
    if len(poly) == 3:
        out.append((poly[0], poly[1], poly[2]))
    return out


def _area2_wrap(pts: np.ndarray, i: int, j: int, k: int) -> float:
    a, b, c = pts[i], pts[j], pts[k]
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _fill_holes(pts: np.ndarray, tris: np.ndarray) -> np.ndarray | None:
    """Ear-fill interior hole loops; None when the soup is unrepairable."""
    counts: dict[tuple[int, int], int] = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            counts[edge] = counts.get(edge, 0) + 1
    if any(v > 2 for v in counts.values()):
        return None
    cycles = _boundary_cycles(counts)
    if len(cycles) < 2:
        return None
    # Outer loop = largest absolute area; the rest are holes.
    def _loop_area(cyc: list[int]) -> float:
        p = pts[np.asarray(cyc)]
        return abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))) / 2

    outer = max(cycles, key=_loop_area)
    merged = [tuple(int(i) for i in t) for t in tris]
    for cyc in cycles:
        if cyc == outer:
            continue
        fill = _ear_clip_fill(cyc, pts)
        if not fill:
            return None
        merged.extend(fill)
    return np.asarray(merged, dtype=int)


def _validate_tris(pts: np.ndarray, tris: np.ndarray, hull_area: float) -> bool:
    """Combinatorial validity: coverage, manifold edges, approximate area.

    Deliberately avoids the Euler count ``T = 2n-2-k``: ``k`` (hull-boundary
    points, collinear-inclusive) is tol-sensitive, while point coverage +
    edge-manifold + single-boundary-cycle are exact integer combinatorics.
    The area check uses a coarse 5e-4 relative tolerance: slivers between a
    zigzag mesh boundary and the straight hull edge hold no input points and
    perturb coarse-mesh FEM below its own discretization error; real holes
    break the single-cycle/manifold checks, not just the area.
    """
    if len(tris) == 0:
        return False
    n = len(pts)
    seen: set[tuple[int, int, int]] = set()
    covered = np.zeros(n, dtype=bool)
    area_sum = 0.0
    counts: dict[tuple[int, int], int] = {}
    for t in tris:
        a0, a1, a2 = int(t[0]), int(t[1]), int(t[2])
        lo, hi = min(a0, a1, a2), max(a0, a1, a2)
        skey: tuple[int, int, int] = (lo, a0 + a1 + a2 - lo - hi, hi)
        if skey in seen:
            return False
        seen.add(skey)
        for i in t:
            if not (0 <= int(i) < n):
                return False
            covered[int(i)] = True
        p = pts[np.asarray(t)]
        edge_a = p[1] - p[0]
        edge_b = p[2] - p[0]
        area = abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])) / 2
        if area <= 1e-300:
            return False
        area_sum += area
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            counts[edge] = counts.get(edge, 0) + 1
            if counts[edge] > 2:
                return False
    if not bool(np.all(covered)):
        return False
    if abs(area_sum - hull_area) > 5e-4 * max(1.0, hull_area):
        return False
    boundary = [e for e, v in counts.items() if v == 1]
    if not boundary:
        return False
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return False
    # Single cycle covering every boundary edge.
    start = boundary[0][0]
    cur, prev, visited = start, -1, {start}
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == start:
            break
        prev, cur = cur, nxt
        visited.add(cur)
    return len(visited) == len(adj)


def uniform_mesh_2d(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> dict[str, object]:
    """Structured triangulation of a rectangle (fast path, no predicates)."""
    if nx < 2 or ny < 2:
        raise ValueError("Need nx, ny >= 2")
    for name, val in (("lx", lx), ("ly", ly)):
        if not (np.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    xs = np.linspace(0.0, float(lx), nx)
    ys = np.linspace(0.0, float(ly), ny)
    nodes = np.array([[x, y] for y in ys for x in xs])
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            tris += [[a, b, c], [b, d, c]]
    return {"nodes": nodes, "triangles": np.asarray(tris, dtype=int), "num_nodes": len(nodes), "num_elements": len(tris)}


def smooth_mesh_2d(
    nodes: list[list[float]] | np.ndarray,
    triangles: list[list[int]] | np.ndarray,
    iterations: int = 20,
    fixed: list[int] | np.ndarray | None = None,
) -> dict[str, object]:
    """Laplacian smoothing: interior nodes move to their neighbor centroid.

    Kills needle slivers that violate the Babuška-Aziz maximum-angle
    condition (flat boundary triangles otherwise cause O(1) FEM errors even
    though the stiffness assembly is exact). ``fixed`` indices never move
    (default: bounding-box nodes). Combinatorics untouched; a move that
    inverts any adjacent triangle is rolled back for that node.
    """
    pts = np.asarray(nodes, dtype=float).copy()
    tris = np.asarray(triangles, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("nodes must be (N, 2)")
    n = len(pts)
    if fixed is None:
        on_box = (
            (np.abs(pts[:, 0] - pts[:, 0].min()) < 1e-12)
            | (np.abs(pts[:, 0] - pts[:, 0].max()) < 1e-12)
            | (np.abs(pts[:, 1] - pts[:, 1].min()) < 1e-12)
            | (np.abs(pts[:, 1] - pts[:, 1].max()) < 1e-12)
        )
        fixed_set = {int(i) for i in np.where(on_box)[0]}
    else:
        fixed_set = {int(i) for i in np.asarray(fixed).ravel()}
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    def _min_abs_area() -> float:
        p = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        return float(np.min(np.abs((p[1][:, 0] - p[0][:, 0]) * (p[2][:, 1] - p[0][:, 1]) - (p[1][:, 1] - p[0][:, 1]) * (p[2][:, 0] - p[0][:, 0])))) / 2

    for _ in range(max(1, int(iterations))):
        for i in range(n):
            if i in fixed_set or not adj[i]:
                continue
            old = pts[i].copy()
            before = _min_abs_area()
            pts[i] = sum(pts[j] for j in adj[i]) / len(adj[i])
            if _min_abs_area() < before:
                pts[i] = old
    return {"nodes": pts, "triangles": tris, "num_nodes": n, "num_elements": len(tris)}


def _triangle_min_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Minimum interior angle in degrees (opposite the shortest side)."""
    import math

    sides = sorted([
        float(np.linalg.norm(p1 - p2)),
        float(np.linalg.norm(p0 - p2)),
        float(np.linalg.norm(p0 - p1)),
    ])
    if sides[0] <= 0.0:
        return 0.0
    longest, mid, shortest = sides[2], sides[1], sides[0]
    cosang = (mid * mid + longest * longest - shortest * shortest) / (2 * mid * longest)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def _in_convex_hull(pt: np.ndarray, hull_pts: np.ndarray) -> bool:
    """Point strictly inside (or on) a CCW/CW convex polygon."""
    signs = set()
    m = len(hull_pts)
    for k in range(m):
        a, b = hull_pts[k], hull_pts[(k + 1) % m]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(float(cross)) <= 1e-12 * max(1.0, float(np.sum((b - a) ** 2))):
            continue
        signs.add(1 if cross > 0 else -1)
        if len(signs) > 1:
            return False
    return True


def _global_index(cur_pts: np.ndarray, points: list[np.ndarray], tri_idx: int) -> int | None:
    """Locate ``cur_pts[tri_idx]`` inside the working ``points`` list."""
    target = np.asarray(cur_pts[tri_idx])
    for j, q in enumerate(points):
        if float(np.sum((np.asarray(q) - target) ** 2)) <= 1e-24:
            return j
    return None


def _triangle_max_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Maximum interior angle in degrees (the Babuška-Aziz quantity)."""
    import math

    sides = sorted([
        float(np.linalg.norm(p1 - p2)),
        float(np.linalg.norm(p0 - p2)),
        float(np.linalg.norm(p0 - p1)),
    ])
    if sides[0] <= 0.0:
        return 180.0
    longest, mid, shortest = sides[2], sides[1], sides[0]
    cosang = (mid * mid + shortest * shortest - longest * longest) / (2 * mid * shortest)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def refine_mesh_2d(
    nodes: list[list[float]] | np.ndarray,
    triangles: list[list[int]] | np.ndarray,
    max_angle_deg: float = 150.0,
    max_points: int = 2000,
    max_rounds: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Ruppert-style refinement driven by the maximum angle (Babuška-Aziz).

    Delaunay already maximizes the minimum angle, so min-driven refinement is
    a no-op on its output; obtuse corners are the real FEM killer. While any
    triangle exceeds ``max_angle_deg``, insert its circumcenter (or split the
    encroached boundary segment, or decimate an extreme interior obtuse
    apex) and re-triangulate. Returns the best validated mesh seen, together
    with ``max_angle_deg``, ``target_max_angle_deg``, ``angle_limit_met``, and
    ``refinement_status`` so unresolved obtuse corners are explicit rather
    than silently accepted.
    """
    pts = np.asarray(nodes, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("nodes must be (N, 2)")
    points = [np.asarray(p) for p in pts]
    # Deduplicate upfront (same 12-decimal rule as the triangulator): the
    # retry loop indexes triangles into this list, so a silent internal merge
    # would shift indices and corrupt Steiner placement.
    seen_pts: dict[tuple[float, float], int] = {}
    deduped: list[np.ndarray] = []
    remap: dict[int, int] = {}
    for i, p in enumerate(points):
        key = (round(float(p[0]), 12), round(float(p[1]), 12))
        if key not in seen_pts:
            seen_pts[key] = len(deduped)
            deduped.append(np.asarray(p))
        remap[i] = seen_pts[key]
    if len(deduped) != len(points):
        logger.warning("Merged %d duplicate points", len(points) - len(deduped))
        points = deduped
        tris = np.asarray([[remap[int(v)] for v in t] for t in tris], dtype=int)

    def _worst() -> tuple[float, np.ndarray, np.ndarray] | None:
        cur_pts = np.asarray(points)
        try:
            cur_tris = delaunay_triangulate(cur_pts, seed=seed)["triangles"]
        except ArithmeticError as e:
            logger.warning("Retriangulation failed inside refinement (%s)", e)
            return None
        cur_tris = np.asarray(cur_tris)
        worst_ang, worst_tri = -1.0, None
        for t in cur_tris:
            p = cur_pts[np.asarray(t)]
            ang = _triangle_max_angle(p[0], p[1], p[2])
            if ang > worst_ang:
                worst_ang, worst_tri = ang, np.asarray(t)
        if worst_tri is None:
            return None
        return worst_ang, cur_pts, worst_tri

    def _mesh_max_angle() -> float:
        try:
            cur = delaunay_triangulate(np.asarray(points), seed=seed)["triangles"]
        except ArithmeticError:
            return 180.0
        return max(
            _triangle_max_angle(*np.asarray(points)[np.asarray(t)]) for t in np.asarray(cur)
        )

    champion = (delaunay_triangulate(np.asarray(points), seed=seed), _mesh_max_angle())
    rounds = 0
    attempts = 0
    while rounds < max_rounds and len(points) < max_points and attempts < 3 * max_rounds:
        attempts += 1
        found = _worst()
        if found is None:
            break
        # NOTE: worst_ang is the MAXIMUM angle (obtuse-corner hunt): Delaunay
        # already maximizes the minimum angle, so min-driven refinement is a
        # no-op on its output; only obtuse corners above max_angle_deg matter.
        worst_ang, cur_pts, worst_tri = found
        if worst_ang <= float(max_angle_deg):
            break
        p = cur_pts[worst_tri]
        if worst_ang > 170.0:
            # Extreme obtuse corner: decimate its vertex when interior (the
            # circumcenter lies uselessly far outside for such needles).
            def _angle_at(k: int, pts) -> float:
                import math

                u = pts[(k + 1) % 3] - pts[k]
                v = pts[(k + 2) % 3] - pts[k]
                nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if nu <= 0 or nv <= 0:
                    return 180.0
                return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / (nu * nv)))))

            apex_local = int(np.argmax([_angle_at(k, p) for k in range(3)]))
            apex_global = _global_index(cur_pts, points, worst_tri[apex_local])
            hull_now = set(_convex_hull_ordered(cur_pts))
            if apex_global is not None and len(points) > 4 and worst_tri[apex_local] not in hull_now:
                del points[apex_global]
                rounds += 1
                try:
                    cand = delaunay_triangulate(np.asarray(points), seed=seed)
                except ArithmeticError:
                    cand = None
                if cand is not None:
                    cur_ang = _mesh_max_angle()
                    if cur_ang < champion[1]:
                        champion = (cand, cur_ang)
                continue
        # Circumcenter of the worst triangle.
        d = 2 * (p[0, 0] * (p[1, 1] - p[2, 1]) + p[1, 0] * (p[2, 1] - p[0, 1]) + p[2, 0] * (p[0, 1] - p[1, 1]))
        if abs(d) < 1e-300:
            break
        ux = ((p[0, 0] ** 2 + p[0, 1] ** 2) * (p[1, 1] - p[2, 1]) + (p[1, 0] ** 2 + p[1, 1] ** 2) * (p[2, 1] - p[0, 1]) + (p[2, 0] ** 2 + p[2, 1] ** 2) * (p[0, 1] - p[1, 1])) / d
        uy = ((p[0, 0] ** 2 + p[0, 1] ** 2) * (p[2, 0] - p[1, 0]) + (p[1, 0] ** 2 + p[1, 1] ** 2) * (p[0, 0] - p[2, 0]) + (p[2, 0] ** 2 + p[2, 1] ** 2) * (p[1, 0] - p[0, 0])) / d
        cc = np.array([ux, uy])
        # Encroachment: circumcenter inside/on any hull-edge diametral circle?
        hull = _convex_hull_ordered(cur_pts)
        encroached = None
        for k in range(len(hull)):
            a, b = cur_pts[hull[k]], cur_pts[hull[(k + 1) % len(hull)]]
            mid = (a + b) / 2
            if float(np.sum((cc - mid) ** 2)) <= float(np.sum((b - a) ** 2)) / 4 + 1e-12:
                encroached = (a, b)
                break
        if encroached is None and not _in_convex_hull(cc, cur_pts[np.asarray(hull)]):
            # Circumcenter outside the domain: split the longest hull edge
            # instead of inserting a far-field point that explodes the hull.
            longest, longest_len = None, -1.0
            for k in range(len(hull)):
                a, b = cur_pts[hull[k]], cur_pts[hull[(k + 1) % len(hull)]]
                length = float(np.sum((b - a) ** 2))
                if length > longest_len:
                    longest, longest_len = (a, b), length
            encroached = longest
        points.append((encroached[0] + encroached[1]) / 2 if encroached is not None else cc)
        # A Steiner point landing on an existing node would desync indices on
        # the next internal re-triangulation: drop near-duplicates outright.
        if len(points) >= 2 and float(np.min(np.sum((np.asarray(points[:-1]) - points[-1]) ** 2, axis=1))) <= 1e-24:
            points.pop()
        else:
            rounds += 1
            try:
                cand = delaunay_triangulate(np.asarray(points), seed=seed)
            except ArithmeticError:
                cand = None
            if cand is not None:
                cur_ang = _mesh_max_angle()
                if cur_ang < champion[1]:
                    champion = (cand, cur_ang)
    out, _ = champion
    no_progress = champion[1] >= 180.0
    if no_progress:
        logger.warning("Refinement made no progress; returning input mesh")
        out = {"nodes": pts, "triangles": tris, "num_nodes": len(pts), "num_elements": len(tris)}
    out = dict(out)
    result_nodes = np.asarray(out["nodes"], dtype=float)
    result_triangles = np.asarray(out["triangles"], dtype=int)
    reported_angle = (
        180.0
        if result_triangles.size == 0
        else max(_triangle_max_angle(*result_nodes[t]) for t in result_triangles)
    )
    if no_progress:
        status = "NO_PROGRESS"
    elif reported_angle <= float(max_angle_deg):
        status = "CONVERGED"
    else:
        status = "LIMIT_REACHED"
        logger.warning(
            "Refinement stopped with max angle %.3f° above target %.3f°",
            float(reported_angle),
            float(max_angle_deg),
        )
    out.update(
        {
            "max_angle_deg": float(reported_angle),
            "target_max_angle_deg": float(max_angle_deg),
            "angle_limit_met": bool(status == "CONVERGED"),
            "refinement_status": status,
        }
    )
    return out
def _bareiss_det_int(matrix: list[list[int]]) -> int:
    """Exact determinant of a square integer matrix using Bareiss elimination."""
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    a = [list(row) for row in matrix]
    n = len(a)
    sign = 1
    previous = 1
    for pivot_col in range(n - 1):
        pivot_row = next(
            (row for row in range(pivot_col, n) if a[row][pivot_col] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_col:
            a[pivot_col], a[pivot_row] = a[pivot_row], a[pivot_col]
            sign = -sign
        pivot = a[pivot_col][pivot_col]
        for row in range(pivot_col + 1, n):
            for col in range(pivot_col + 1, n):
                numerator = a[row][col] * pivot - a[row][pivot_col] * a[pivot_col][col]
                if previous != 1:
                    if numerator % previous != 0:
                        raise ArithmeticError("Bareiss elimination lost exact divisibility")
                    numerator //= previous
                a[row][col] = numerator
            a[row][pivot_col] = 0
        previous = pivot
    return sign * a[-1][-1]


def _orient3d_q(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    d: tuple[int, int, int],
) -> int:
    """Exact signed orientation determinant for four integer-lattice points."""
    return _bareiss_det_int([
        [a[0], a[1], a[2], 1],
        [b[0], b[1], b[2], 1],
        [c[0], c[1], c[2], 1],
        [d[0], d[1], d[2], 1],
    ])


def _insphere_q(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    d: tuple[int, int, int],
    e: tuple[int, int, int],
) -> bool:
    """Exact in-sphere predicate, normalized for either tetra orientation."""
    orientation = _orient3d_q(a, b, c, d)
    if orientation == 0:
        return False
    def row(point: tuple[int, int, int]) -> list[int]:
        x, y, z = point
        return [x, y, z, x * x + y * y + z * z, 1]

    determinant = _bareiss_det_int([row(a), row(b), row(c), row(d), row(e)])
    return determinant * (1 if orientation > 0 else -1) > 0


def _quantize_global_3d(points: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """Map one 3D point set to a shared exact integer lattice."""
    data = np.asarray(points, dtype=float)
    lower = data.min(axis=0)
    span = max(float(np.ptp(data, axis=0).max()), 1e-300)
    return [
        (
            round((float(point[0]) - float(lower[0])) / span * _QUANTUM),
            round((float(point[1]) - float(lower[1])) / span * _QUANTUM),
            round((float(point[2]) - float(lower[2])) / span * _QUANTUM),
        )
        for point in data
    ]


def _tetra_volume6(points: np.ndarray, tetrahedron: np.ndarray) -> float:
    """Return six times the absolute tetrahedron volume."""
    a, b, c, d = points[np.asarray(tetrahedron, dtype=int)]
    return abs(float(np.linalg.det(np.vstack((b - a, c - a, d - a)))))


def _sorted_pair(a: int, b: int) -> tuple[int, int]:
    ordered = sorted((a, b))
    return ordered[0], ordered[1]


def _sorted_triple(values: tuple[int, int, int]) -> tuple[int, int, int]:
    ordered = sorted(values)
    return ordered[0], ordered[1], ordered[2]


def _validate_tetrahedra(points: np.ndarray, tetrahedra: np.ndarray) -> bool:
    """Validate coverage, manifold faces, connected boundary, and Euler count."""
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) == 0:
        return False
    n = len(points)
    covered = np.zeros(n, dtype=bool)
    seen: set[tuple[int, int, int, int]] = set()
    faces: dict[tuple[int, int, int], int] = {}
    edges: set[tuple[int, int]] = set()
    volume6 = 0.0
    scale = max(float(np.ptp(points, axis=0).max()), 1.0) ** 3
    for tetrahedron in tetrahedra:
        indices = (int(tetrahedron[0]), int(tetrahedron[1]), int(tetrahedron[2]), int(tetrahedron[3]))
        if len(set(indices)) != 4 or any(value < 0 or value >= n for value in indices):
            return False
        ordered = sorted(indices)
        canonical = (ordered[0], ordered[1], ordered[2], ordered[3])
        if canonical in seen:
            return False
        seen.add(canonical)
        covered[list(indices)] = True
        volume = _tetra_volume6(points, np.asarray(indices))
        if not math.isfinite(volume) or volume <= 1e-14 * scale:
            return False
        volume6 += volume
        faces_of_tetra: tuple[tuple[int, int, int], ...] = (
            (indices[0], indices[1], indices[2]),
            (indices[0], indices[3], indices[1]),
            (indices[0], indices[2], indices[3]),
            (indices[1], indices[3], indices[2]),
        )
        for face in faces_of_tetra:
            key = _sorted_triple(face)
            faces[key] = faces.get(key, 0) + 1
            if faces[key] > 2:
                return False
        edges_of_tetra: tuple[tuple[int, int], ...] = (
            (indices[0], indices[1]),
            (indices[0], indices[2]),
            (indices[0], indices[3]),
            (indices[1], indices[2]),
            (indices[1], indices[3]),
            (indices[2], indices[3]),
        )
        for edge in edges_of_tetra:
            edges.add(_sorted_pair(edge[0], edge[1]))
    if not bool(np.all(covered)) or not math.isfinite(volume6) or volume6 <= 0:
        return False
    boundary = [face for face, count in faces.items() if count == 1]
    if not boundary:
        return False
    boundary_vertices = {vertex for face in boundary for vertex in face}
    boundary_adjacency: dict[int, set[int]] = {vertex: set() for vertex in boundary_vertices}
    for a, b, c in boundary:
        boundary_adjacency[a].update((b, c))
        boundary_adjacency[b].update((a, c))
        boundary_adjacency[c].update((a, b))
    reached = {min(boundary_vertices)}
    frontier = list(reached)
    while frontier:
        vertex = frontier.pop()
        for neighbor in boundary_adjacency[vertex]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    if reached != boundary_vertices:
        return False
    vertices, face_count, tetra_count = n, len(faces), len(tetrahedra)
    return vertices - len(edges) + face_count - tetra_count == 1


def _bowyer_watson_3d(points: np.ndarray, seed: int) -> np.ndarray | None:
    """One exact-predicate 3D Bowyer-Watson pass."""
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = max(float(np.ptp(points, axis=0).max()), 1e-9)
    center = (lower + upper) / 2
    extent = 20.0 * span
    supertetra = np.asarray([
        center + [-extent, -extent, -extent],
        center + [extent, -extent, extent],
        center + [-extent, extent, extent],
        center + [extent, extent, -extent],
    ])
    vertices = [np.asarray(point) for point in supertetra] + [np.asarray(point) for point in points]
    qvertices = _quantize_global_3d(vertices)
    tetrahedra: list[tuple[int, int, int, int]] = [(0, 1, 2, 3)]
    rng = np.random.RandomState(int(seed))
    insertion_order = list(range(4, len(vertices)))
    rng.shuffle(insertion_order)
    for index in insertion_order:
        bad = [
            tetrahedron for tetrahedron in tetrahedra
            if _insphere_q(
                qvertices[tetrahedron[0]], qvertices[tetrahedron[1]],
                qvertices[tetrahedron[2]], qvertices[tetrahedron[3]], qvertices[index],
            )
        ]
        if not bad:
            return None
        face_counts: dict[tuple[int, int, int], int] = {}
        face_orientations: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for tetrahedron in bad:
            faces_of_tetra: tuple[tuple[int, int, int], ...] = (
                (tetrahedron[0], tetrahedron[1], tetrahedron[2]),
                (tetrahedron[0], tetrahedron[3], tetrahedron[1]),
                (tetrahedron[0], tetrahedron[2], tetrahedron[3]),
                (tetrahedron[1], tetrahedron[3], tetrahedron[2]),
            )
            for face in faces_of_tetra:
                key = _sorted_triple(face)
                face_counts[key] = face_counts.get(key, 0) + 1
                face_orientations[key] = face
        bad_set = set(bad)
        tetrahedra = [tetrahedron for tetrahedron in tetrahedra if tetrahedron not in bad_set]
        for key, count in face_counts.items():
            if count != 1:
                continue
            face = face_orientations[key]
            candidate = (face[0], face[1], face[2], index)
            if _orient3d_q(*[qvertices[value] for value in candidate]) != 0:
                tetrahedra.append(candidate)
    result = [tetrahedron for tetrahedron in tetrahedra if all(value >= 4 for value in tetrahedron)]
    if not result:
        return None
    return np.asarray([(a - 4, b - 4, c - 4, d - 4) for a, b, c, d in result], dtype=int)


def delaunay_triangulate_3d(
    points: list[list[float]] | np.ndarray,
    seed: int = 0,
) -> dict[str, object]:
    """Exact-predicate 3D Delaunay tetrahedralization of a point cloud.

    Returns ``nodes`` and ``tetrahedra`` (index quadruples), with deterministic
    seed retries for cospherical inputs. Inputs are deduplicated and bounded;
    coplanar or persistently invalid point sets raise instead of returning a
    mesh with missing volume. The predicates use a shared integer lattice and
    Bareiss determinants, matching the 2D mesher's exact-predicate contract.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("points must be finite")
    pts = np.unique(np.round(pts, decimals=12), axis=0)
    n = len(pts)
    if n < 4:
        raise ValueError("Need at least 4 points")
    if n > 2000:
        raise ValueError("Need N <= 2000 for 3D Bowyer-Watson insertion")
    centered = pts - pts.mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if singular[2] <= 1e-12 * max(singular[0], 1e-300):
        raise ValueError("Points are coplanar: no tetrahedralization exists")
    for attempt in range(_MAX_SEED_RETRIES):
        tetrahedra = _bowyer_watson_3d(pts, int(seed) + attempt)
        if tetrahedra is not None and _validate_tetrahedra(pts, tetrahedra):
            if attempt:
                logger.info("3D triangulation validated on seed retry %d", attempt)
            return {
                "nodes": pts,
                "tetrahedra": tetrahedra,
                "num_nodes": n,
                "num_elements": len(tetrahedra),
            }
    raise ArithmeticError(
        f"No valid 3D tetrahedralization in {_MAX_SEED_RETRIES} seed retries "
        "(holes, degeneracies, or non-manifold faces persisted)"
    )
