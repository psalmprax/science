"""CBF O(N²) scaling profile — proximity-graph backlog activation gate.

The 2026-10-22 graph census lists the robotics proximity interaction graph
(k-NN / dynamic G(V,E(t))) as NOT implemented and profile-gated: dense
all-pairs broadcasting is exact and validated; a proximity graph needs its
own soundness re-verification of the barrier certificates at reduced
connectivity. This harness produces that gate's measurements.

Two distinct O(N²) paths are profiled separately (they have different
constants and different decision moments):

  CERTIFIED ``solve_cbf_qp`` (``apedp_dyn/robotics/cbf_certificate.py``) —
  vectorized einsum over ALL P = N(N-1)/2 pair constraints EVERY iteration,
  plus the [N, N, 2] barrier build. Nothing is distance-filtered: the
  certified contract requires hdot >= -gamma*h for every pair, so a
  proximity graph would change the certified-set definition itself and
  needs a soundness argument (see below). Pairs are reduced strictly by
  the assumption box bounds, never ignored.

  HEURISTIC ``multi_agent_step`` (``apedp_dyn/robotics/multi_agent_transport.py``) —
  a Python-level FULL N² per-agent pair loop (no upper-triangular halving)
  whose callee ``compute_cbf_repulsive_force`` early-returns zeros beyond
  r_warning — i.e. the distance filter a proximity graph would replicate
  already exists INSIDE each pair call; the O(N²) is the loop invocation
  structure itself.

The k-NN counterfactual is COMPUTATIONAL ONLY, measured as a prototype
estimate and labeled as such: it quantifies pair-count reduction
(P_knn/P_dense) and restricted-structure setup cost. It is NOT a soundness
claim — a proximity-restricted certified set would require proving that
non-neighbor pairs cannot violate their barriers within the assumption
box (e.g. a guaranteed-separation margin argument) before product use.

Gate criterion (preregistered here, before reading any numbers): the
backlog ACTIVATES for a path when its measured per-call wall time exceeds
the real-time control budget B = 5 ms (module operates at dt=0.05 → 20 Hz
control; 5 ms leaves 90% headroom for the rest of the pipeline). Both a
strict (5 ms) and a relaxed (50 ms offline batch) threshold are reported.

Evidence-only: no acceptance/verdict is computed; the artifact feeds
review. Deterministic (fixed seeds per N). Status: audit-only (profiler).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    _pair_barriers,
    solve_cbf_qp,
)
from apedp_dyn.robotics.multi_agent_transport import (
    AgentRole,
    multi_agent_step,
)

OUT_JSON = Path("benchmarks/cbf_scaling_profile_results.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
SEED = 20261022
DT = 0.05          # module's control period: 20 Hz
V_MAX = 1.0
R_SAFE = 0.5
SPACING = 2.0      # grid spacing; min inter-agent distance >= SPACING - 2*JITTER > r_safe
JITTER = 0.4
BUDGET_STRICT_MS = 5.0     # 20 Hz real-time control, 90% pipeline headroom
BUDGET_RELAXED_MS = 50.0   # offline batch

CERT_SWEEP = (8, 16, 32, 64, 128, 256, 512)
HEUR_SWEEP = (8, 16, 32, 64, 128, 256, 512)
HEUR_REPS = {512: 1, 256: 2}   # tiny N → many reps; huge N → few
CERT_REPS = {512: 3, 256: 5}


def _fixture(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Seeded deterministic multi-agent fixture with h(0) > 0 for all pairs (A3)."""
    rng = np.random.default_rng(SEED)
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)[:n].astype(float) * SPACING
    pts += rng.uniform(-JITTER, JITTER, size=pts.shape)
    poses = np.concatenate([pts, np.zeros((n, 1))], axis=1)   # [x, y, theta]
    targets = pts + rng.uniform(-SPACING, SPACING, size=pts.shape)
    roles = [AgentRole.ACTIVE_STRIKER if i % 3 == 0
             else AgentRole.SUPPORT_RECOVERY for i in range(n)]
    velocities = rng.uniform(-V_MAX, V_MAX, size=(n, 2))
    # Fixture guard: A3 requires h(0) > 0 for EVERY pair.
    if n > 1:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        d[np.diag_indices(n)] = np.inf
        min_gap = float(d.min())
        if not min_gap > R_SAFE:
            raise RuntimeError(
                f"fixture violates A3 (min gap {min_gap:.3f} <= r_safe {R_SAFE})")
    return poses, targets, roles, velocities


def _median_ms(fn, args: tuple = (), kwargs: dict | None = None, reps: int = 3) -> float:
    """Median wall time of repeated calls (args bound explicitly, no late capture)."""
    times = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        fn(*args, **(kwargs or {}))
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def _indexed_build_ms(positions, assumptions, reps: int = 5) -> dict:
    """Measured KD-tree INDEX floor: cKDTree build + candidate query + sparse
    barrier components, versus the dense ``_pair_barriers`` build. This is
    the direct answer to the dense-floor limitation: the index must beat
    the [N,N,2] matrix without ever missing a constrained pair (soundness
    pinned in tests/test_dyn_proximity_margin_pins.py)."""
    from apedp_dyn.robotics.cbf_certificate import _pair_barriers
    from apedp_dyn.robotics.proximity_margin import indexed_barrier_components

    dense = _median_ms(_pair_barriers, (positions, assumptions.r_safe), reps=reps)
    indexed = _median_ms(indexed_barrier_components, (positions, assumptions),
                         reps=reps)
    _, _unit, _h, diag = indexed_barrier_components(positions, assumptions)
    return {
        "dense_build_ms": round(dense, 4),
        "indexed_build_ms": round(indexed, 4),
        "speedup": round(dense / indexed, 2),
        "candidate_pairs": diag["candidate_pairs"],
        "dense_pair_count": diag["dense_pair_count"],
        "pair_reduction": diag["pair_reduction"],
        "soundness": diag["soundness"],
    }


def _fit_exponent(nss: list[int], ts: list[float]) -> float | None:
    if len(nss) < 3:
        return None
    x = np.log(np.asarray(nss, dtype=float))
    y = np.log(np.asarray(ts, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return round(float(slope), 3)


def _gate_crossing(table: dict[int, dict], budget_ms: float) -> int | float | None:
    """First swept N (ascending) whose measured ms exceeds budget; else None."""
    for n in sorted(table):
        if table[n]["qp_ms"] > budget_ms:
            return n
    return None


def run() -> dict:
    assumptions_by_n = {
        n: CBFQPCertificateAssumptions(
            v_max=V_MAX, r_safe=R_SAFE, gamma=0.5, dt=DT, max_agents=n)
        for n in CERT_SWEEP
    }
    certified: dict[int, dict] = {}
    for n in CERT_SWEEP:
        poses, targets, roles, vel = _fixture(n)
        a = assumptions_by_n[n]

        barriers_ms = _median_ms(_pair_barriers, (poses[:, :2], R_SAFE),
                                 reps=CERT_REPS.get(n, 10))
        qp_ms = _median_ms(solve_cbf_qp, (poses[:, :2], vel, a),
                           reps=CERT_REPS.get(n, 10))
        pairs = n * (n - 1) // 2
        entry = {
            "qp_ms": round(qp_ms, 4),
            "barriers_build_ms": round(barriers_ms, 4),
            "pairs": pairs,
            # milliseconds per 1000 pair-constraints (throughput density)
            "us_per_pair": round(qp_ms * 1000.0 / pairs, 4),
        }
        if n >= 128:
            # the dense-floor limitation's direct counter-measure, measured
            entry["indexed_build"] = _indexed_build_ms(poses[:, :2], a)
            # the COMPOSED restricted path: index + restricted projections +
            # full all-pairs fail-closed re-check (end-to-end vs dense QP)
            from apedp_dyn.robotics.proximity_margin import solve_cbf_qp_proximity
            t_prox = _median_ms(solve_cbf_qp_proximity,
                                (poses[:, :2], vel, a), reps=CERT_REPS.get(n, 10))
            out = solve_cbf_qp_proximity(poses[:, :2], vel, a)
            entry["restricted_composed"] = {
                "total_ms": round(t_prox, 4),
                "vs_dense_ms": round(qp_ms / max(t_prox, 1e-9), 2),
                "constrained_pairs": out.constrained_pairs,
                "constraint_ratio": out.constraint_ratio,
                "note": ("index + restricted alternating projections + FULL "
                         "all-pairs fail-closed re-check; the re-check keeps "
                         "one O(N²) einsum pass by design — certified honesty "
                         "is not negotiable"),
            }
        # k-NN computational counterfactual (prototype estimate; NOT a
        # soundness claim — see module docstring).
        if n >= 16:
            bar = _pair_barriers(poses[:, :2], R_SAFE)
            dist = bar.h + R_SAFE
            for k in (4, 8):
                kk = min(k, n - 1)
                idx = np.argpartition(dist, kk, axis=1)[:, :kk]
                rows = np.repeat(np.arange(n), kk)
                directed = np.stack([rows, idx.ravel()], axis=1)
                und = np.sort(directed, axis=1)
                uniq = np.unique(und, axis=0)
                p_red = len(uniq)
                t0 = time.perf_counter()
                a_ij = bar.unit[uniq[:, 0], uniq[:, 1]]
                b_ij = -0.5 * DT * 2.0 * bar.h[uniq[:, 0], uniq[:, 1]]  # -gamma*h
                if not (np.isfinite(a_ij).all() and np.isfinite(b_ij).all()):
                    raise RuntimeError("k-NN restricted pair assembly produced non-finite values")
                setup_ms = (time.perf_counter() - t0) * 1e3
                entry[f"knn{k}"] = {
                    "pairs_reduced": p_red,
                    "pair_ratio_of_dense": round(p_red / pairs, 4),
                    "restricted_setup_ms": round(setup_ms, 4),
                    # estimated per-call ms if the QP-iteration work scales
                    # with the pair count (STRAIGHT-LINE estimate at the
                    # measured dense cost density; iterations are not
                    # guaranteed to shrink — the estimate is one-sided).
                    "estimated_qp_ms": round(
                        barriers_ms + (qp_ms - barriers_ms)
                        * (p_red / pairs), 4),
                }
        certified[n] = entry

    heuristic: dict[int, dict] = {}
    for n in HEUR_SWEEP:
        poses, targets, roles, _vel = _fixture(n)
        ms = _median_ms(
            multi_agent_step,
            (poses, targets, roles), {"dt": DT, "v_max": V_MAX, "k_rep": 2.0},
            reps=HEUR_REPS.get(n, 3))
        pair_calls = n * (n - 1)          # full loop, no halving
        heuristic[n] = {
            "step_ms": round(ms, 4),
            "pair_invocations": pair_calls,
            "us_per_pair_call": round(ms * 1000.0 / pair_calls, 4),
        }

    cert_ms = {n: v["qp_ms"] for n, v in certified.items()}
    heur_ms = {n: v["step_ms"] for n, v in heuristic.items()}
    out = {
        "seed": SEED,
        "protocol": (
            "deterministic grid+jitter fixture (A3 guarded: min gap > r_safe); "
            "median of repeated wall-clock calls; certified path = full "
            "all-pairs QP; heuristic = full N² Python pair loop; k-NN rows "
            "are COMPUTATIONAL counterfactuals (pair-reduction ratios and "
            "setup costs), explicitly NOT soundness claims"),
        "budget": {
            "strict_ms": BUDGET_STRICT_MS,
            "relaxed_ms": BUDGET_RELAXED_MS,
            "basis": f"dt={DT} control (20 Hz); 90% pipeline headroom",
        },
        "certified_path": {
            "per_n": certified,
            "exponent_fit": _fit_exponent(sorted(cert_ms), list(map(cert_ms.get, sorted(cert_ms)))),
            "gate_crossing_first_N_strict": _gate_crossing(certified, BUDGET_STRICT_MS),
            "gate_crossing_first_N_relaxed": _gate_crossing(certified, BUDGET_RELAXED_MS),
        },
        "heuristic_path": {
            "per_n": heuristic,
            "exponent_fit": _fit_exponent(sorted(heur_ms), list(map(heur_ms.get, sorted(heur_ms)))),
            "gate_crossing_first_N_strict": {
                b: next((n for n in sorted(heuristic)
                         if heuristic[n]["step_ms"] > b), None)
                for b in (BUDGET_STRICT_MS, BUDGET_RELAXED_MS)},
        },
        "decision_note": (
            "backlog ACTIVATES for a path when its measured per-call wall "
            "time exceeds the budget above; a proximity graph also requires "
            "the soundness re-verification of barrier certificates at "
            "reduced connectivity before ANY production use"),
    }
    out["certified_path"]["gate_crossing_first_N_strict"] = _gate_crossing(
        certified, BUDGET_STRICT_MS)
    return out


def main() -> None:
    res = run()
    OUT_JSON.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"\ncertified QP exponent fit: {res['certified_path']['exponent_fit']}")
    print(f"heuristic step exponent fit: {res['heuristic_path']['exponent_fit']}")
    print(f"certified strict-gate first N: {res['certified_path']['gate_crossing_first_N_strict']}"
          f" | relaxed: {res['certified_path']['gate_crossing_first_N_relaxed']}")
    print("heuristic gates:", res["heuristic_path"]["gate_crossing_first_N_strict"])
    for n in sorted(res["certified_path"]["per_n"]):
        v = res["certified_path"]["per_n"][n]
        extra = ""
        if "knn4" in v:
            e = v["knn4"]
            extra = f" | knn4 ratio {e['pair_ratio_of_dense']}, est {e['estimated_qp_ms']} ms"
        print(f"  N={n:4d} qp={v['qp_ms']:9.3f} ms (barriers {v['barriers_build_ms']:.3f}, "
              f"{v['pairs']} pairs, {v['us_per_pair']} us/pair){extra}")
    for n in sorted(res["heuristic_path"]["per_n"]):
        v = res["heuristic_path"]["per_n"][n]
        print(f"  N={n:4d} heuristic={v['step_ms']:10.3f} ms "
              f"({v['pair_invocations']} pair calls, {v['us_per_pair_call']} us/call)")


if __name__ == "__main__":
    main()
