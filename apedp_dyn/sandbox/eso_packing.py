"""ESO-1: Epistemically Safe Optimization via sphere packing.

CANONICAL ARCHITECTURE DIRECTIVE:
--------------------------------
`apedp_dyn.sandbox.sphere_packing_sandbox` is designated as the CANONICAL discrete
geometry and packing referee engine for forward APEDP-X research.

This module (`eso_packing.py`) preserves its full original implementation untouched
to ensure zero regression for existing cross-reference and instance-level tests.
It should be treated as a legacy reference implementation; all comprehensive Cohn-Elkies
LP bounds, isostatic rigidity Hessian audits, and adversarial traps are developed in
`sphere_packing_sandbox.py`.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')

GROUND_TRUTH: dict[tuple[str, int], dict] = {
    ("interval", 5): {"r_star": 1.0 / 10.0, "phi_star": 1.0,
                      "note": "n equal segments tile [0,1] exactly"},
    ("square", 1): {"r_star": 0.5, "phi_star": np.pi / 4.0,
                    "note": "single inscribed circle"},
    ("square", 2): {"r_star": 0.25, "phi_star": np.pi / 8.0,
                    "note": "side-by-side, bound r<=0.25 tight"},
    ("square", 4): {"r_star": 0.25, "phi_star": np.pi / 4.0,
                    "note": "2x2 grid, optimal"},
    ("square", 9): {"r_star": 1.0 / 6.0, "phi_star": np.pi / 4.0,
                    "note": "3x3 grid, optimal; hard for random starts"},
}


def _feasible_interval(r: float, n: int) -> bool:
    """Equal segments of radius r fit in [0,1] iff 2*n*r <= 1."""
    return bool(2.0 * n * r <= 1.0 + 1e-12)


def _try_place_square(r: float, n: int, trials: int, seed: int) -> bool:
    """One feasibility probe: random placements + short repulsion."""
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        c = rng.uniform(r, 1.0 - r, size=(n, 2))
        # Local repulsion passes (push overlapping pairs apart)
        for _ in range(30):
            d = c[:, None, :] - c[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1)) + np.eye(n)
            iu = np.triu_indices(n, 1)
            overlap = 2.0 * r - dist[iu]
            if bool((overlap <= 1e-9).all()):
                break
            for a, b in zip(*iu, strict=True):
                if dist[a, b] < 2.0 * r:
                    push = (2.0 * r - dist[a, b]) / 2.0 + 1e-9
                    direction = d[a, b] / (dist[a, b] + 1e-12)
                    c[a] = np.clip(c[a] + push * direction, r, 1.0 - r)
                    c[b] = np.clip(c[b] - push * direction, r, 1.0 - r)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(n, 1)
        if bool((d[iu] >= 2.0 * r - 1e-9).all()):
            return True
    return False


def _best_radius(family: str, n: int, budget: int, seed: int) -> float:
    """Bisection on r under a fixed probe budget. Deterministic per seed."""
    if family == "interval":
        # Exact bound is analytic; optimizer earns it only via bisection.
        lo, hi = 0.0, 0.5
        for _i in range(24):
            mid = (lo + hi) / 2.0
            if _feasible_interval(mid, n):
                lo = mid
            else:
                hi = mid
        return lo
    lo, hi = 0.0, 0.5
    for i in range(16):
        mid = (lo + hi) / 2.0
        per = max(20, budget // 16)
        if _try_place_square(mid, n, per, seed * 1000 + i):
            lo = mid
        else:
            hi = mid
    return lo


def _density(family: str, n: int, r: float) -> float:
    if family == "interval":
        return float(min(1.0, 2.0 * n * r))
    if family == "square":
        return float(n * np.pi * r ** 2)
    raise ValueError(f"unknown family {family!r}")


def run_eso_case(family: str, n: int, budget: int = 2000,
                 tau: float = 0.01, seeds: tuple[int, ...] = (0, 1),
                 perturb_trials: int = 200) -> dict:
    """Run one ground-truth case; return verdict + audit trail."""
    key = (family, n)
    if key not in GROUND_TRUTH:
        raise ValueError(f"no ground truth for {key}; known: {sorted(GROUND_TRUTH)}")
    truth = GROUND_TRUTH[key]
    r_star = truth["r_star"]

    per_seed = [_best_radius(family, n, budget, s) for s in seeds]
    best_r = max(per_seed) if per_seed else 0.0
    if best_r <= 0.0:
        return {"verdict": "INFEASIBLE", "best_r": 0.0, "gap": None,
                "truth": truth, "per_seed": per_seed}

    gap = abs(best_r - r_star)
    agree = (max(per_seed) - min(per_seed)) <= tau if len(per_seed) > 1 else True
    # Perturbation stability: slightly smaller radius must stay feasible.
    stable = _best_radius(family, n, perturb_trials, seeds[0] + 999) >= best_r - tau
    if gap <= tau and agree and stable:
        verdict = "GLOBALLY_OPTIMAL"
    else:
        verdict = "BEST_OBSERVED"
    return {
        "verdict": verdict,
        "best_r": float(best_r),
        "phi": _density(family, n, best_r),
        "phi_star": float(truth["phi_star"]),
        "gap": float(gap),
        "agree": bool(agree),
        "stable": bool(stable),
        "tau": float(tau),
        "truth": truth,
        "per_seed": [float(x) for x in per_seed],
    }


# ── CLI demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("ESO-1: best-observed vs globally-optimal (known ground truth)")
    print("=" * 72)
    print("\n[interval n=5: exact optimum r*=0.1]")
    r = run_eso_case("interval", 5, budget=500)
    print(f"  verdict={r['verdict']} best_r={r['best_r']:.4f} gap={r['gap']:.5f}")
    print("\n[square n=4: optimum r*=0.25]")
    for B in (200, 2000):
        r = run_eso_case("square", 4, budget=B)
        print(f"  budget={B:>6}: best_r={r['best_r']:.4f} phi={r['phi']:.4f} "
              f"gap={r['gap']:.4f} agree={r['agree']} stable={r['stable']} -> {r['verdict']}")
    print("\n[square n=9: optimum r*=0.1667, hard for random starts]")
    for B in (200, 2000):
        r = run_eso_case("square", 9, budget=B)
        print(f"  budget={B:>6}: best_r={r['best_r']:.4f} phi={r['phi']:.4f} "
              f"gap={r['gap']:.4f} agree={r['agree']} stable={r['stable']} -> {r['verdict']}")
    print("=" * 72)
