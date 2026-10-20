"""Lorenz public benchmark — frozen confirmation harness (PREREG-LORENZ-PUB-V1).

Status: audit-only (benchmarks harness; the prereg IS the spec:
docs/PREREGISTRATION_LORENZ_PUBLIC_BENCHMARK_V1.md)

Frozen grid: sigma=10, rho=28, beta=8/3 world; seeds {101, 202, 303} x noise
{0.5%, 1%, 2%, 3%} = 12 registered runs per path; SG(21,3)+np.gradient
estimator; 70/30 dev/confirmation split; pairwise `adjudicate` with block
bootstrap n_boot=1000, seed=7, data_origin/benchmark_level "unknown".

Registered paths:
  PATH-R  referee-gated forward selection (primary; a term is kept only when
          pairwise adjudicate(new, incumbent) on the SEALED holdout returns
          unique_support promoting the new model)
  PATH-B  greedy forward TRAINING-BIC over the degree-<=3 library
          (analytic naive negative control)
  PATH-B' the external transcript's curated 4-candidate ranking (control A')
  PATH-C  chimera: ground-truth constants baked in + hardcoded -y + the
          transcript's multipliers, validated in the transcript by explicit
          Euler and a 3D attractor plot (control B; the plot is inadmissible
          here — H6 scores PATH-C on the registered instruments only)

Registered instruments (PREREG-LORENZ-PUB-V1):
  structure_clean, max_coefficient_error_se, winner_true_holdout_gap,
  referee_status(+referee_promoted), divergence_horizon_ratio
  (same-integrator control on both sides).

Smoke mode (--smoke): single wiring-check run (seed 101, 1%), referee
n_boot=100 — output is NOT a confirmation artifact.

Usage:  python3 benchmarks/run_lorenz_public_benchmark.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "lorenz_public_benchmark_results.json"
LEDGER = REPO_ROOT / "benchmarks" / "lorenz_public_benchmark_ledger.jsonl"

# ---- frozen constants (PREREG-LORENZ-PUB-V1) --------------------------------
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
TRUE_TERMS = {"x": RHO, "y": -1.0, "xz": -1.0}
SEEDS = (101, 202, 303)
NOISE_SWEEP = (0.005, 0.01, 0.02, 0.03)
SG_WINDOW, SG_ORDER = 21, 3
HOLDOUT_FRACTION = 0.30
REF_BOOT, REF_SEED = 1000, 7
T_SPAN, DT_SAMPLE = (0.0, 10.0), 0.005
HOLDOUT_X0 = (2.0, 2.0, 25.0)  # catalog holdout initial condition
SPURIOUS_TOL = 1e-3
LIB = ("1", "x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz",
       "x3", "y3", "z3", "xyz", "x2y", "x2z", "xy2", "xz2", "y2z", "yz2")
LIB = tuple(dict.fromkeys(LIB))
FROZEN = {
    "prereg_id": "PREREG-LORENZ-PUB-V1-2026-10-20",
    "world": {"sigma": SIGMA, "rho": RHO, "beta": BETA, "t_span": list(T_SPAN),
              "dt_sample": DT_SAMPLE, "ic": [1.0, 1.0, 1.0],
              "holdout_x0": list(HOLDOUT_X0)},
    "seeds": list(SEEDS), "noise_sweep": list(NOISE_SWEEP),
    "estimator": {"savgol": [SG_WINDOW, SG_ORDER], "derivative": "np.gradient"},
    "split": {"dev_fraction": 1.0 - HOLDOUT_FRACTION, "holdout_fraction": HOLDOUT_FRACTION},
    "referee": {"n_boot": REF_BOOT, "seed": REF_SEED,
                "data_origin": "unknown", "benchmark_level": "unknown"},
    "spurious_tolerance": SPURIOUS_TOL,
}
# PATH-C frozen numbers: the transcript's own operational model as written
# (ground-truth sigma/rho/beta baked in, -y hardcoded, transcript multipliers).
PATH_C_TERMS = {"x": 0.973879 * RHO, "y": -1.0, "xz": -0.100261 * SIGMA}


def _record_ledger(candidate_id: str, outcome: str) -> None:
    """Burn-record the sealed confirmation-window consumption (I3 discipline)."""
    from apedp_dyn.epistemic.experiment_ledger import (
        CandidateLifecycleState,
        DatasetRole,
        FullLineageExperimentLedger,
    )

    ledger = FullLineageExperimentLedger(persistence_path=LEDGER)
    ledger.register_candidate(candidate_id=candidate_id,
                              proposer_id="lorenz_public_benchmark",
                              formula=f"run::{candidate_id}")
    ledger.record_data_exposure(candidate_id=candidate_id,
                                dataset_id="lorenz_public_benchmark",
                                slice_identifier="confirmation_window",
                                role=DatasetRole.CONFIRMATION_HOLDOUT,
                                outcome_status=outcome)
    if outcome != "PASSED":
        ledger.update_lifecycle_state(candidate_id,
                                      CandidateLifecycleState.CONFIRMATION_REJECTED,
                                      rejection_reason=f"prereg hypothesis outcome: {outcome}")


def simulate(seed: int, noise_frac: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    def rhs(t, s):
        return [SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]]
    sol = solve_ivp(rhs, T_SPAN, [1.0, 1.0, 1.0],
                    t_eval=np.arange(T_SPAN[0], T_SPAN[1] + 1e-12, DT_SAMPLE),
                    rtol=1e-10, atol=1e-10)
    t, S = sol.t, sol.y
    S_noisy = S + rng.normal(0.0, 1.0, S.shape) * (noise_frac * S.std(axis=1, keepdims=True))
    S_sm = savgol_filter(S_noisy, SG_WINDOW, SG_ORDER, axis=1)
    dS = np.gradient(S_sm, DT_SAMPLE, axis=1)
    x, y, z = S_sm
    F = {"1": np.ones(t.size), "x": x, "y": y, "z": z,
         "x2": x * x, "y2": y * y, "z2": z * z, "xy": x * y, "xz": x * z, "yz": y * z,
         "x3": x ** 3, "y3": y ** 3, "z3": z ** 3, "xyz": x * y * z,
         "x2y": x * x * y, "x2z": x * x * z, "xy2": x * y * y,
         "xz2": x * z * z, "y2z": y * y * z, "yz2": y * z * z}
    return t, dS[1], F


def _fit(F, dy, idn, terms, hold=False):
    A = np.column_stack([F[tm] for tm in terms])
    sl = slice(idn, None) if hold else slice(0, idn)
    w, *_ = np.linalg.lstsq(A[sl], dy[sl], rcond=None)
    return w, dy[sl] - A[sl] @ w


def _scored(cid, terms, rh, dy_hold, rd):
    g4 = {"passed": True, "holdout_obs_rms": float(np.sqrt(np.mean(dy_hold ** 2))),
          "holdout_noise_sigma": None, "holdout_n_samples": int(rh.size),
          "holdout_residuals": rh.tolist()}
    cand = CandidateDynamics(lhs="dydt",
                             terms=tuple(Term(tm, f"b{i}") for i, tm in enumerate(terms)),
                             candidate_id=cid, metadata={})
    return ScoredCandidate(cand, {"gate4_status": g4},
                           train_loss=float(np.mean(rd ** 2)),
                           holdout_rmse=float(np.sqrt(np.mean(rh ** 2))))


def path_b(F, dy, idn) -> tuple[str, ...]:
    n = idn
    chosen: list[str] = []
    best = n * float(np.log(np.var(dy[:idn])))
    improved = True
    while improved and len(chosen) < 10:
        improved = False
        bestc = None
        for tm in LIB:
            if tm in chosen:
                continue
            _, r = _fit(F, dy, idn, tuple(sorted(chosen + [tm], key=LIB.index)))
            bic = n * float(np.log(np.mean(r ** 2))) + (len(chosen) + 1) * float(np.log(n))
            if bic < best - 1e-9 and (bestc is None or bic < bestc[0]):
                bestc = (bic, tm)
        if bestc:
            best, improved = bestc[0], True
            chosen.append(bestc[1])
    return tuple(sorted(chosen, key=LIB.index))


def path_r(F, dy, idn, n_boot: int):
    """Referee-gated forward selection exactly as frozen:
    'candidates are added from the library in dev-BIC order, and a term is
    kept only if pairwise adjudicate(new, incumbent) on the SEALED holdout
    returns unique_support promoting the new model; otherwise declined and
    the next candidate is tried.' Mechanism detail: the queue is re-ranked
    by marginal dev-BIC gain after each promotion; a term declined in a
    round stays declined for that run (deterministic, single pass per round)."""
    incumbent: tuple[str, ...] = ("1",)
    notes: list[dict] = []
    declined_perm = set()
    while True:
        remaining = [tm for tm in LIB if tm != "1" and tm not in incumbent
                     and tm not in declined_perm]
        if not remaining:
            break
        _, rd_i = _fit(F, dy, idn, incumbent)
        base_bic = idn * float(np.log(np.mean(rd_i ** 2))) + len(incumbent) * float(np.log(idn))
        queue = []
        for tm in remaining:
            cand_terms = tuple(sorted(incumbent + (tm,), key=LIB.index))
            _, rd_c = _fit(F, dy, idn, cand_terms)
            bic_c = idn * float(np.log(np.mean(rd_c ** 2))) + len(cand_terms) * float(np.log(idn))
            queue.append((base_bic - bic_c, tm, cand_terms))
        queue.sort(key=lambda g: -g[0])
        promoted_any = False
        for _, tm, cand_terms in queue:
            _, rd_c = _fit(F, dy, idn, cand_terms)
            rh_c = _fit(F, dy, idn, cand_terms, hold=True)[1]
            _, rd_i = _fit(F, dy, idn, incumbent)
            rh_i = _fit(F, dy, idn, incumbent, hold=True)[1]
            adj = adjudicate([_scored(f"cand_{tm}", cand_terms, rh_c, dy[idn:], rd_c),
                              _scored("incumbent", incumbent, rh_i, dy[idn:], rd_i)],
                             holdout_scale=None, resid_best=rh_c.tolist(),
                             resid_rival=rh_i.tolist(), bootstrap=(n_boot > 0),
                             bootstrap_n_boot=n_boot, bootstrap_seed=REF_SEED,
                             data_origin="unknown", benchmark_level="unknown")
            promoted = (adj.promoted is not None
                        and adj.promoted.candidate.candidate_id == f"cand_{tm}")
            if promoted:
                incumbent = cand_terms
                declined_perm.discard(tm) # formally: promoted
                if tm in declined_perm:
                    declined_perm.remove(tm)
                promoted_any = True
                break
            declined_perm.add(tm)
            notes.append({"term": tm, "status": adj.status.value,
                          "reason": str(adj.reason)[:200]})
        if not promoted_any:
            break
    return incumbent, notes


def _instruments(F, dy, idn, terms: tuple[str, ...], n_boot: int) -> dict:
    w_win, r_dev = _fit(F, dy, idn, terms)
    r_hold = _fit(F, dy, idn, terms, hold=True)[1]
    w_true, r_true_dev = _fit(F, dy, idn, ("x", "y", "xz"))
    r_true_hold = _fit(F, dy, idn, ("x", "y", "xz"), hold=True)[1]
    se_scale = float(np.sqrt(np.mean(r_true_dev ** 2)))
    max_err = 0.0
    for tm, c in zip(("x", "y", "xz"), w_true):
        se_j = se_scale / float(np.sqrt(np.mean(F[tm][:idn] ** 2)))
        max_err = max(max_err, abs(float(c) - TRUE_TERMS[tm]) / se_j)
    adj = adjudicate([_scored("winner", terms, r_hold, dy[idn:], r_dev),
                      _scored("true", ("x", "y", "xz"), r_true_hold, dy[idn:], r_true_dev)],
                     holdout_scale=None, resid_best=r_hold.tolist(),
                     resid_rival=r_true_hold.tolist(), bootstrap=(n_boot > 0),
                     bootstrap_n_boot=n_boot, bootstrap_seed=REF_SEED,
                     data_origin="unknown", benchmark_level="unknown")
    gap = float(abs(float(np.sqrt(np.mean(r_hold ** 2))) - float(np.sqrt(np.mean(r_true_hold ** 2)))))
    clean = not any(tm not in ("x", "y", "xz") and abs(float(c)) > SPURIOUS_TOL
                    for tm, c in zip(terms, w_win))
    refs = dict(zip(terms, (float(c) for c in w_win)))
    horizon = divergence_horizon({"x": refs.get("x", 0.0), "y": refs.get("y", 0.0),
                                  "xz": refs.get("xz", 0.0)}) if set(terms) - {"1"} else 0.0
    oracle = divergence_horizon(None)
    return {"terms": list(terms), "coefficients": [float(c) for c in w_win],
            "structure_clean": clean,
            "all_coeffs_within_2p5se": max_err <= 2.5,
            "max_coefficient_error_se": max_err,
            "winner_true_holdout_gap": gap,
            "referee_status": adj.status.value,
            "referee_promoted": (adj.promoted.candidate.candidate_id if adj.promoted else None),
            "referee_reason": str(adj.reason)[:250],
            "divergence_horizon_model": horizon,
            "divergence_horizon_oracle": oracle,
            "divergence_horizon_ratio": (horizon / oracle if oracle > 0 else None)}


def divergence_horizon(model_w: dict | None, chimera: bool = False) -> float:
    """Same-integrator control (both sides RK45 rtol/atol 1e-9, catalog x0):
    horizon t* is where ||x_model - x_true|| > 1; model_w None => oracle truth."""
    t_eval = np.arange(0.0, 10.0 + 1e-12, DT_SAMPLE)
    def truth_rhs(t, s):
        return [SIGMA * (s[1] - s[0]), RHO * s[0] - s[0] * s[2] - s[1], s[0] * s[1] - BETA * s[2]]
    def model_rhs(t, s):
        x, y, z = s
        if chimera or model_w is None:
            wx = PATH_C_TERMS["x"] if chimera else RHO
            wy = PATH_C_TERMS["y"] if chimera else -1.0
            wxz = PATH_C_TERMS["xz"] if chimera else -1.0
            return [SIGMA * (y - x), wx * x + wy * y + wxz * x * z, x * y - BETA * z]
        return [SIGMA * (y - x), model_w["x"] * x + model_w["y"] * y + model_w["xz"] * x * z,
                x * y - BETA * z]
    truth = solve_ivp(truth_rhs, (0, 10), HOLDOUT_X0, t_eval=t_eval, rtol=1e-9, atol=1e-9).y.T
    model = solve_ivp(model_rhs, (0, 10), HOLDOUT_X0, t_eval=t_eval, rtol=1e-9, atol=1e-9).y.T
    diffs = np.abs(model - truth).max(axis=1)
    idx = np.nonzero(diffs > 1.0)[0]
    if chimera or model_w is None:
        pass  # chimera trail is its own model trajectory vs truth (H6 instrument)
    return float(t_eval[int(idx[0])]) if idx.size else 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    noises = NOISE_SWEEP[2:3] if args.smoke else NOISE_SWEEP
    n_boot = 100 if args.smoke else REF_BOOT

    rows = []
    for seed in seeds:
        for nf in noises:
            dy, F = simulate(seed, nf)[1:]
            idn = int((1.0 - HOLDOUT_FRACTION) * dy.size)
            head = {"seed": seed, "noise_frac": nf}
            terms_b = path_b(F, dy, idn)
            rows.append(dict(head, path="B", **_instruments(F, dy, idn, terms_b, 0)))
            curated = {"true": ("x", "y", "xz"),
                       "spur1_plus_x2y": ("x", "y", "xz", "x2y"),
                       "spur2_plus_z3": ("x", "y", "xz", "z3"),
                       "spur3_plus_xyz": ("x", "y", "xz", "xyz")}
            bic_ranks = []
            for name, terms in curated.items():
                _, rd = _fit(F, dy, idn, terms)
                bic_ranks.append((idn * float(np.log(np.mean(rd ** 2)))
                                  + len(terms) * float(np.log(idn)), name, terms))
            _, sel_name, sel_terms = min(bic_ranks)
            rows.append(dict(head, path="B-prime", selected=sel_name,
                             **_instruments(F, dy, idn, sel_terms, 0)))
            terms_r, declined = path_r(F, dy, idn, n_boot)
            rows.append(dict(head, path="R", declined=declined[:3],
                             **_instruments(F, dy, idn, terms_r, n_boot)))
            chim_horizon = divergence_horizon({"x": PATH_C_TERMS["x"], "y": PATH_C_TERMS["y"],
                                               "xz": PATH_C_TERMS["xz"]}, chimera=True)
            oracle_horizon = divergence_horizon(None)
    # PATH-C instruments (H6): referee-certification absent by construction;
    # scored on the divergence-ratio and coefficient instruments.
            rows.append(dict(head, path="C", terms=["x", "y", "xz"],
                             coefficients=[PATH_C_TERMS["x"], PATH_C_TERMS["y"],
                                           PATH_C_TERMS["xz"]],
                             structure_clean=False,
                             referee_status="not-adjudicated-chimera",
                             referee_promoted=None,
                             divergence_horizon_model=chim_horizon,
                             divergence_horizon_oracle=oracle_horizon,
                             divergence_horizon_ratio=(chim_horizon / oracle_horizon
                                                       if oracle_horizon > 0 else None),
                             max_coefficient_error_se=None,
                             all_coeffs_within_2p5se=None,
                             winner_true_holdout_gap=None))
    head_h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=str(REPO_ROOT)).stdout.strip()
    outcome = "PASSED" if all(
        r.get("referee_status") in {"multi_hypothesis", "insufficient_evidence"}
        for r in rows if r["path"] == "B") else "FAILED"
    _record_ledger(f"lorenz_public_benchmark::{FROZEN['prereg_id']}", outcome)
    payload = {"prereg_id": FROZEN["prereg_id"], "frozen": FROZEN, "git_head": head_h,
               "mode": "smoke" if args.smoke else "full",
               "recorded_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"harness ok: mode={'smoke' if args.smoke else 'full'}, {len(rows)} rows -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
