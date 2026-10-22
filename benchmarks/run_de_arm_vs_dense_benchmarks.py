"""DE-arm frozen benchmark: screened library vs dense flat library.

Measures the Level-1 dependency-screening hypothesis (external-review
correction A) on deterministic coupled-ODE ground truths:

  S1  Lotka-Volterra    : ẋ = 1.5x − xy,      ẏ = −3y + xy
  S2  FitzHugh–Nagumo   : ẋ = x − x³/3 − y,  ẏ = 0.5x − 0.2y
  S3  Lorenz            : ẋ = 10(y − x),     ẏ = 28x − y − xz,
                          ż = xy − (8/3)z

One RK4-integrated, noise-free trajectory per system (fixed seeds), with
Lorenz burned for 30 time units so the kept window covers both attractor
lobes (a narrow slow-manifold window makes polynomial columns nearly
collinear: with noise-free exact derivatives, MANY column combinations
achieve identically zero residual, and BOTH library kinds then fit exact
but non-physical structures — fixture under-excitation is itself the
reviewed artifact class, so coverage is part of the protocol).

For each system the SAME frame goes to BOTH library kinds:

  dense   — kernel ``discover_coupled_system`` (flat polynomial library,
            stridge over ALL columns) — the route coupled discovery uses
            today, and the arm's own escape hatch.
  screened— the arm's reduced-library mirror fit, at TWO screen depths:
            top_k=2 (keeps the ranking's two best parents) and top_k=1
            (an aggressive screen that MUST drop a true parent in the
            3-state system — the engineered false-exclusion setting that
            demonstrates the escape hatch is load-bearing).

Reported per system/state: library column count (MEASURED, never an
assumed reduction fraction), active terms vs ground-truth RHS structure,
screen false exclusions, train R², and wall time. Term names are
canonicalized before comparison (``y*x`` ≡ ``x*y``); the mirror's
intercept column ``1`` is retained by construction and is reported
separately, not scored as structure. Screen exclusions are REPORTED
verbatim, never hidden. Evidence-only: no acceptance/verdict is computed
here; the artifact feeds referee review.

Status: audit-only (benchmark harness).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from apedp_dyn.de_arm.dependency_screen import discover_ode_system

OUT_JSON = Path("benchmarks/de_arm_vs_dense_benchmark_results.json")
SEED = 20261022

# Ground-truth RHS parent set and polynomial structure per state.
GROUND_TRUTH: dict[str, dict] = {
    "lotka_volterra": {
        "parents": {"x": ("x", "y"), "y": ("x", "y")},
        "true_terms": {"x": {"x", "x*y"}, "y": {"y", "x*y"}},
        "degree": 2,
    },
    "fitzhugh_nagumo": {
        "parents": {"x": ("x", "y"), "y": ("x", "y")},
        "true_terms": {"x": {"x", "x^3", "y"}, "y": {"y", "x"}},
        "degree": 3,
    },
    "lorenz": {
        "parents": {"x": ("x", "y"), "y": ("x", "y", "z"), "z": ("x", "y", "z")},
        "true_terms": {
            "x": {"x", "y"},
            "y": {"x", "y", "x*z"},
            "z": {"z", "x*y"},
        },
        "degree": 2,
    },
}


def _rk4(f, ic, dt, n_steps, keep_last):
    def step(s):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    s = np.asarray(ic, dtype=float)
    for _ in range(n_steps - keep_last):
        s = step(s)
    traj = np.empty((keep_last, len(s)), dtype=float)
    for i in range(keep_last):
        traj[i] = s
        s = step(s)
    return traj


def _lorenz_rhs(s):
    x, y, z = s
    return np.array([10.0 * (y - x), 28.0 * x - y - x * z,
                     x * y - (8.0 / 3.0) * z])


def _lv_rhs(s):
    x, y = s
    return np.array([1.5 * x - 1.0 * x * y, -3.0 * y + 1.0 * x * y])


def _fhn_rhs(s):
    x, y = s
    return np.array([x - x**3 / 3.0 - y, 0.5 * x - 0.2 * y])


def _trajectories() -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    ic2 = rng.uniform(-1.0, 1.0, size=2)
    ic3 = rng.uniform(-1.0, 1.0, size=3)
    lv = _rk4(_lv_rhs, ic2 * 0.1 + 0.2, 0.01, 1500, 800)
    fhn = _rk4(_fhn_rhs, ic2 + 0.5, 0.01, 2000, 800)
    # 44-time-unit burn, 16-time-unit kept window: BOTH the full window AND
    # its first-70% train partition (what the screen and mirror actually
    # see) must span both lobes — a slow-manifold-heavy train partition
    # makes a state's variance small enough for the ridge prune to drop a
    # true linear term, which is FIXTURE bias, not engine failure.
    lor = _rk4(_lorenz_rhs, ic3, 0.002, 40200, 8000)
    out = {
        "lotka_volterra": {"states": ("x", "y"), "dt": 0.01, "traj": lv},
        "fitzhugh_nagumo": {"states": ("x", "y"), "dt": 0.01, "traj": fhn},
        "lorenz": {"states": ("x", "y", "z"), "dt": 0.002, "traj": lor},
    }
    # Fixture coverage guard: assert each state swept a WIDE range on the
    # attractor (the under-excitation artifact class is excluded by design).
    spans = {"lotka_volterra": (0.2, 3.0), "fitzhugh_nagumo": (0.3, 1.5),
             "lorenz": (8.0, 48.0)}
    for name, spec in out.items():
        n = spec["traj"].shape[0]
        n_train = int((n - 4) * 0.7) + 2   # interior-row train partition
        for k, col in zip(spec["states"], spec["traj"].T, strict=True):
            for part, sweep in (("full", float(col.max() - col.min())),
                                ("train", float(col[:n_train].max()
                                                - col[:n_train].min()))):
                lo, _hi = spans[name]
                if not sweep > lo:
                    raise RuntimeError(
                        f"fixture under-excitation for {name}.{k} "
                        f"({part} sweep {sweep:.2f}); refusing to benchmark "
                        "a degenerate window")
    return out


def _norm_term(t: str) -> str:
    """Canonical term name: cross terms are commutative (``y*x`` ≡ ``x*y``)."""
    if "*" in t:
        parts = sorted(t.split("*"))
        return "*".join(p for p in parts if p and p != "1")
    return t


def _norm_set(terms) -> set[str]:
    return {_norm_term(t) for t in terms}


def run() -> dict:
    results: dict = {
        "seed": SEED,
        "protocol": ("deterministic RK4 fixtures, noise-free; term names "
                     "canonicalized (y*x ≡ x*y); mirror intercept '1' "
                     "retained by construction and excluded from the "
                     "structure metrics"),
        "systems": {},
    }
    for sys_name, spec in _trajectories().items():
        states, dt, traj = spec["states"], spec["dt"], spec["traj"]
        series = {k: traj[:, i] for i, k in enumerate(states)}
        cfg = GROUND_TRUTH[sys_name]
        truth_parents, true_terms = cfg["parents"], cfg["true_terms"]

        t0 = time.perf_counter()
        outcome = discover_ode_system(
            series, dt, top_k=2, max_polynomial_degree=cfg["degree"])
        t_arm = time.perf_counter() - t0
        t0 = time.perf_counter()
        outcome_k1 = discover_ode_system(
            series, dt, top_k=1, max_polynomial_degree=cfg["degree"])
        t_arm_k1 = time.perf_counter() - t0

        dense = outcome.dense_result
        d = len(states)
        dense_cols = 1 + d
        if cfg["degree"] >= 2:
            dense_cols += d * (d + 1) // 2
        if cfg["degree"] >= 3:
            dense_cols += d

        per_state: dict[str, dict] = {}
        for state in sorted(states):
            eq = next(e for e in dense.equations if e.state_variable == state)
            truth = _norm_set(true_terms[state])
            truth_p = set(truth_parents[state])
            dense_terms = _norm_set(eq.active_terms)

            def screen_block(fit, rank, truth, truth_p):
                kept = sorted(set(rank.kept))
                false_excl = sorted(set(rank.excluded) & truth_p)
                scr = _norm_set(fit.active_terms) - {"1"}
                return {
                    "kept_parents": kept,
                    "false_exclusions": false_excl,
                    "library_columns": len(fit.library),
                    "active_terms_no_intercept": sorted(scr),
                    "missing_truth_terms": sorted(truth - scr),
                    "overcomplete_terms": sorted(scr - truth),
                    "structure_exact": (truth <= scr) and not (scr - truth),
                    "truth_recovered": truth <= scr,
                    "train_r2": round(fit.train_r2, 6),
                    "train_rmse": round(fit.train_rmse, 8),
                }

            per_state[state] = {
                "truth_parents": sorted(truth_p),
                "true_rhs_terms": sorted(truth),
                "dense": {
                    "library_columns": dense_cols,
                    "active_terms": sorted(dense_terms),
                    "truth_recovered": truth <= dense_terms,
                    "structure_exact": truth <= dense_terms
                    and not (dense_terms - truth),
                    "r2": round(eq.r2, 6),
                },
                "screened_k2": screen_block(outcome.screened[state],
                                            outcome.ranking[state], truth, truth_p),
                "screened_k1": screen_block(outcome_k1.screened[state],
                                            outcome_k1.ranking[state], truth, truth_p),
            }

        scr_lib = {s: p["screened_k2"]["library_columns"]
                   for s, p in per_state.items()}
        results["systems"][sys_name] = {
            "dt": dt,
            "n_rows": int(traj.shape[0]),
            "dense_columns": dense_cols,
            "screened_library_columns_k2": scr_lib,
            "library_reduction_measured_k2": {
                s: round(1.0 - v / dense_cols, 4) for s, v in scr_lib.items()},
            "dense_mean_system_r2": round(dense.mean_system_r2, 6),
            "dense_worst_state_r2": round(dense.worst_state_r2, 6),
            "dense_structure_exact_all_states": all(
                p["dense"]["structure_exact"] for p in per_state.values()),
            "dense_truth_recovered_all_states": all(
                p["dense"]["truth_recovered"] for p in per_state.values()),
            "screened_k2_structure_exact_all_states": all(
                p["screened_k2"]["structure_exact"] for p in per_state.values()),
            "screened_k2_false_exclusions_total": sum(
                len(p["screened_k2"]["false_exclusions"])
                for p in per_state.values()),
            "screened_k1_false_exclusions_total": sum(
                len(p["screened_k1"]["false_exclusions"])
                for p in per_state.values()),
            # Parents the aggressive screen suppressed (kept-pool omissions,
            # excluding the state itself — a state is never its own screen
            # candidate; the dense result in the same outcome carries these).
            "screened_k1_suppressed_parents": {
                s: sorted(
                    set(truth_parents[s])
                    - set(per_state[s]["screened_k1"]["kept_parents"])
                    - {s})
                for s in sorted(states)
            },
            "runtime_seconds": {
                "ode_arm_outcome_k2": round(t_arm, 4),
                "ode_arm_outcome_k1": round(t_arm_k1, 4),
            },
            "referee_note": (
                "screen exclusions reported verbatim; the dense result in the "
                "same outcome carries any structure the screen dropped "
                "(mandatory escape hatch); no acceptance computed here"),
            "per_state": per_state,
        }
    return results


def main() -> None:
    res = run()
    OUT_JSON.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    for name, s in res["systems"].items():
        print(f"\n=== {name} (dt={s['dt']}, n={s['n_rows']}) ===")
        print(f"  dense: mean R2 {s['dense_mean_system_r2']}, "
              f"truth recovered {s['dense_truth_recovered_all_states']}, "
              f"structure exact {s['dense_structure_exact_all_states']} "
              f"({s['dense_columns']} cols)")
        print(f"  screened k2: exact "
              f"{s['screened_k2_structure_exact_all_states']}, "
              f"cols {s['screened_library_columns_k2']}, "
              f"false exclusions {s['screened_k2_false_exclusions_total']}")
        print(f"  screened k1 (aggressive): false exclusions "
              f"{s['screened_k1_false_exclusions_total']}")


if __name__ == "__main__":
    main()
