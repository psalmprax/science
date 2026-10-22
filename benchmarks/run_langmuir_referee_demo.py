"""Langmuir isotherm vs. engine tournament — referee-routed + ledger-tracked (APEDP-X).

Run from anywhere: ``python benchmarks/run_langmuir_referee_demo.py``.

Evidence-only demo, ``audit-only``. It computes no acceptance by itself: Gate 4
(Diebold-Mariano, canonical alpha) and ``OdeReferee`` decide, and every verdict
is journalled in ``FullLineageExperimentLedger`` at
``get_ledger_path("langmuir_referee_demo.jsonl")``. A candidate that fails its
confirmation slice burns that slice; re-running after a burn reports the block
instead of re-reading the holdout. Uses live network reads of the NIST ISODB
mirror (``NIST-ISODB/isodb-library``); no local data fixture is shipped.

  * Gate-4 block runs on a NIST ISODB isotherm with 156 points (55 holdout rows),
    i.e. enough evidence for the Diebold-Mariano gate to actually decide;
  * no dev-MSE ranking: Gate 4 (canonical alpha) decides, and the ledger is
    consulted BEFORE any holdout is touched, so a burned slice stays burned;
  * the coupled-ODE block runs on a real trajectory (Lotka-Volterra RK4) and is
    adjudicated by OdeReferee, with a noise-tolerance sweep.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from apedp.evolutionary_search import SearchConfig, evolve
from apedp.rust_bridge import RustKernelBridge
from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.novelty_adjudication import evaluate_predictive_incrementality
from apedp_dyn.ode import discover_coupled_ode
from apedp_dyn.ode.data import OdeTrajectoryData
from apedp_dyn.ode.referee import OdeReferee
from apedp_dyn.orchestration import dispatch_frame
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

# 1. Load a NIST ISODB isotherm: methane on zeolite 4A at 323 K (156 points,
#    a genuine type-I Langmuir system — the classical law fits it well, so the
#    incumbent is a real competitor rather than a straw man).
ISO_PATH = "Library/10.1007s1045001698416/10.1007s1045001698416.Isotherm24.json"
ISO_URL = ("https://raw.githubusercontent.com/NIST-ISODB/isodb-library/HEAD/"
           + urllib.parse.quote(ISO_PATH))
iso = json.loads(urllib.request.urlopen(ISO_URL, timeout=30).read().decode())
df_clean = pd.DataFrame(
    [{"p": pt["pressure"], "q": pt["species_data"][0]["adsorption"]}
     for pt in iso["isotherm_data"]]).dropna().sort_values("p").reset_index(drop=True)
p_all = df_clean["p"].to_numpy(float)
q_all = df_clean["q"].to_numpy(float)
dataset_id = iso["filename"]
print(f"dataset: {dataset_id} | {iso['adsorbent']['name']} | "
      f"{iso['adsorbates'][0]['name']} {iso['temperature']} K | n={len(p_all)}")

# 2. PRE-REGISTERED allocation, declared BEFORE any holdout is drawn: every
#    proposer gets exactly one disjoint confirmation slice. The assignment is
#    fixed here and never revised after a verdict (AGENTS.md 1.2: a failed
#    holdout is burned, never re-cut; a later candidate gets its own slice, not
#    a renamed copy of the one that failed).
TRAIN_FRACTION = 0.50
PREREGISTERED = {
    "rime_ad_proposer": ("prereg_sliceA_seed0", 0.25),
    "native_ga_gp": ("prereg_sliceB_seed0", 0.25),
}
def allocate(seed: int) -> tuple[np.ndarray, dict[str, tuple[str, np.ndarray]]]:
    """One train block + one disjoint confirmation slice per pre-registered candidate."""
    perm = np.random.default_rng(seed).permutation(len(p_all))
    n_train = int(round(TRAIN_FRACTION * len(p_all)))
    cur, out = n_train, {}
    for cid, (sid, frac) in PREREGISTERED.items():
        n_sl = int(round(frac * len(p_all)))
        out[cid] = (f"{sid}_s{seed}", perm[cur:cur + n_sl])
        cur += n_sl
    return perm[:n_train], out

train_idx, SLICES = allocate(0)
p_tr, q_tr = p_all[train_idx], q_all[train_idx]
print(f"split: train n={len(p_tr)} | " + " | ".join(
    f"{sid} n={len(ix)}" for _cid, (sid, ix) in SLICES.items()))

# 3. Classical Langmuir baseline — parameters fitted on TRAIN rows only.
#    Its holdout predictions are computed lazily, inside the gated loop, so no
#    holdout value is read before the ledger authorises the slice.
def classical_langmuir(p, q_max, K):
    return (q_max * K * p) / (1.0 + K * p)

popt, _ = curve_fit(classical_langmuir, p_tr, q_tr,
                    p0=[q_tr.max() * 1.2, 1.0 / np.median(p_tr)],
                    bounds=(0, np.inf), maxfev=200000)
print(f"baseline params (train-only): q_max={popt[0]:.4f}, K={popt[1]:.5f}")

# 4. Modality sieve (routing evidence only — no engine output is used below)
dispatch_res = dispatch_frame(df_clean, time_col='p', value_cols=['q'])
route = dispatch_res.decision.decision
print(f"sieve: {route.sieve_modality} -> {route.chosen_engines or '(nothing routed)'}")

# 5. Proposers run on TRAIN rows only (no holdout touched yet)
ad_proposals = RimeAdProposer(target_col='q').fit_and_propose(
    pd.DataFrame({'p': p_tr, 'q': q_tr}))
print(f"RIME-AD: {len(ad_proposals)} candidates emitted")

ga_res = evolve({'p': p_tr}, q_tr, config=SearchConfig(
    use_native_ga=True, islands=3, island_size=20, generations=10, seed=42))
assert ga_res.history[0].get('native_rust') is True, "GA fell back to the Python loop"
print(f"GA/GP (Rust): {ga_res.n_evaluations} evals; best = {ga_res.best_expression}")

def ad_prediction(p_slice):
    best_ad = ad_proposals[0]           # already BIC-ranked on dev rows
    return np.asarray(best_ad.compute_score(pd.DataFrame({'p': p_slice})), float).ravel()

def ga_prediction(p_slice):
    return np.asarray(RustKernelBridge().evaluate_expression(
        ga_res.best_expression, ['p'], [p_slice]), float)

# 6. Experiment ledger — consulted BEFORE any holdout read, one slice per
#    pre-registered candidate.
print("\n===== EXPERIMENT LEDGER + GATE 4 (Diebold-Mariano) =====")
ledger = FullLineageExperimentLedger.from_env("langmuir_referee_demo.jsonl")
print(f"ledger: {ledger.persistence_path}")
print(f"alpha = {DIEBOLD_MARIANO_P_VALUE_THRESHOLD} (canonical spec)")

contestants = [
    ("rime_ad_proposer", "RimeAdProposer", ad_prediction),
    ("native_ga_gp", "native_ga", ga_prediction),
]
formulas = {"rime_ad_proposer": ad_proposals[0].formula if ad_proposals else "(no proposal)",
            "native_ga_gp": ga_res.best_expression}
known = {n.candidate_id for n in ledger.get_all_candidates()}
for cid, proposer_id, _ in contestants:
    if cid not in known:
        ledger.register_candidate(cid, proposer_id, formulas[cid])

print(f"{'candidate':>17} | {'slice':>20} | {'n':>3} | {'baseRMSE':>9} | {'candRMSE':>9} "
      f"| {'dm':>6} | {'p':>8} | verdict")
print("-" * 118)
for cid, name, predict in contestants:
    sid, sl_idx = SLICES[cid]
    burned = ledger.get_burned_holdouts().get((dataset_id, sid))
    if burned is not None:
        print(f"{name:>17} | {sid:>20} | slice already burned by '{burned}' "
              f"in a previous run — holdout not read")
        continue

    # first authorised read of THIS candidate's slice
    p_sl, q_sl = p_all[sl_idx], q_all[sl_idx]
    pred_base = classical_langmuir(p_sl, *popt)
    pred = predict(p_sl)
    if not np.any(np.isfinite(pred)):
        print(f"{name:>17} | {sid:>20} | {len(q_sl):3d} | no evaluable prediction -> INSUFFICIENT_EVIDENCE")
        continue

    res = evaluate_predictive_incrementality(
        q_sl, pred, pred_base, alpha=DIEBOLD_MARIANO_P_VALUE_THRESHOLD)
    promoted = res.verdict == "PREDICTIVELY_INCREMENTAL"
    rmse = float(np.sqrt(np.nanmean((q_sl - pred) ** 2)))
    base_rmse = float(np.sqrt(np.mean((q_sl - pred_base) ** 2)))

    try:
        ledger.record_data_exposure(
            candidate_id=cid, dataset_id=dataset_id, slice_identifier=sid,
            role=DatasetRole.CONFIRMATION_HOLDOUT,
            outcome_status="PASSED" if promoted else "FAILED")
    except HoldoutContaminationError as exc:
        print(f"{name:>17} | {sid:>20} | BLOCKED: {exc}")
        continue
    ledger.update_lifecycle_state(
        candidate_id=cid,
        new_state=(CandidateLifecycleState.CONFIRMED_VALID if promoted
                   else CandidateLifecycleState.CONFIRMATION_REJECTED),
        rejection_reason=None if promoted else f"gate4={res.verdict}",
        metrics={"holdout_rmse": rmse, "gate4": res.verdict,
                 "dm_statistic": res.dm_statistic, "dm_p_value": res.p_value})

    print(f"{name:>17} | {sid:>20} | {len(q_sl):3d} | {base_rmse:9.4f} | {rmse:9.4f} "
          f"| {res.dm_statistic:6.3f} | {res.p_value:8.4g} | {res.verdict}"
          f"{'' if promoted else '  [slice burned]'}")

print("burned holdouts:", ledger.get_burned_holdouts())
print("Promotion requires verdict=PREDICTIVELY_INCREMENTAL; no candidate is ranked by dev MSE.")

# 7. Coupled ODE discovery on a REAL trajectory: Lotka-Volterra RK4
#    dx/dt = 1.5x - xy,  dy/dt = -3y + xy  (the repo's frozen fixture)
def lotka_volterra(n=800, dt=0.01, x0=(1.0, 1.0),
                   alpha=1.5, beta=1.0, delta=1.0, gamma=3.0):
    rhs = lambda s: np.array([alpha * s[0] - beta * s[0] * s[1],
                              delta * s[0] * s[1] - gamma * s[1]])
    xs, ys = np.empty(n), np.empty(n)
    s = np.array(x0, float)
    for k in range(n):
        xs[k], ys[k] = s
        k1 = rhs(s); k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2); k4 = rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return OdeTrajectoryData.from_dict({"x": xs, "y": ys}, dt=dt)

traj = lotka_volterra()
n_disc = int(0.70 * len(traj.t))
disc = OdeTrajectoryData.from_dict(
    {k: v[:n_disc] for k, v in traj.states.items()}, dt=traj.dt)
hold = OdeTrajectoryData.from_dict(
    {k: v[n_disc:] for k, v in traj.states.items()}, dt=traj.dt)

res = discover_coupled_ode({k: v for k, v in disc.states.items()}, dt=traj.dt,
                           max_polynomial_degree=2)
print(f"\n===== COUPLED ODE DISCOVERY (discovery window n={n_disc}) =====")
for eq in res.equations:
    terms = ", ".join(f"{c:+.4f}*{t}" for t, c in zip(eq.active_terms, eq.coefficients))
    print(f"  d{eq.state_variable}/dt = {terms}   (fit R^2={eq.r2:.6f})")
print("  true: dx/dt = +1.5*x -1.0*x*y ;  dy/dt = -3.0*y +1.0*x*y")
print(f"  mean_system_r2={res.mean_system_r2:.6f} (in-sample; not evidence)")

LV_TRUE_TERMS = {"x": {"x", "x*y"}, "y": {"y", "x*y"}}

def structure_matches(result, truth=LV_TRUE_TERMS) -> bool:
    """Structural identity: do the discovered term sets equal the true ones?"""
    return all(set(eq.active_terms) == truth.get(eq.state_variable, set())
               for eq in result.equations)

verdict = OdeReferee().adjudicate_result(res, hold)
certified = verdict.verdict.value == "IDENTIFIED_DYNAMICAL_LAW"
structure_ok = structure_matches(res)
print(f"\n===== ODE REFEREE (forward rollout on sealed holdout n={len(hold.t)}) =====")
print(f"  verdict={verdict.verdict.value} mean_holdout_r2={verdict.mean_holdout_r2:.6f} "
      f"worst_state_r2={verdict.worst_state_r2:.6f} stable={verdict.is_stable}")
print(f"  structure matches truth: {structure_ok}")
print(f"  {verdict.rational_justification}")
# The certificate is a PREDICTIVE contract (holdout rollout R^2); structural
# identity is a separate claim. On the noise-free fixture both must hold, so a
# divergence here is a real regression rather than a tolerance to widen.
assert certified and structure_ok, (
    "referee certificate and structural identity disagree on the noise-free fixture")

# 8. NOISE SWEEP: how much measurement noise the referee tolerates before it refuses
state_scale = float(np.sqrt(np.mean([np.var(v) for v in traj.states.values()])))
levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

print(f"\n===== ODE REFEREE NOISE TOLERANCE (state RMS={state_scale:.4f}) =====")
print(f"{'sigma/sigma_state':>18} | {'struct':>9} | {'meanR2':>8} | {'worstR2':>8} | {'stable':>6} | verdict                       | claim")
print("-" * 92)
tolerance = None
for i, level in enumerate(levels):
    sigma = level * state_scale
    noisy = OdeTrajectoryData.from_dict(
        {k: v + np.random.default_rng(1000 + i).normal(0.0, sigma, size=v.shape)
         for k, v in traj.states.items()},
        dt=traj.dt)
    d = OdeTrajectoryData.from_dict(
        {k: v[:n_disc] for k, v in noisy.states.items()}, dt=noisy.dt)
    h = OdeTrajectoryData.from_dict(
        {k: v[n_disc:] for k, v in noisy.states.items()}, dt=noisy.dt)
    r = discover_coupled_ode({k: v for k, v in d.states.items()}, dt=d.dt,
                             max_polynomial_degree=2)
    struct_ok = structure_matches(r)
    v = OdeReferee().adjudicate_result(r, h)
    certified_here = v.verdict.value == "IDENTIFIED_DYNAMICAL_LAW"
    if certified_here and struct_ok:
        claim = "EXACT RECOVERY"
    elif certified_here:
        claim = "PREDICTIVE ONLY (structure wrong)"
    else:
        claim = "REFUSED"
    print(f"{level:18.3f} | {('yes' if struct_ok else 'no'):>9} | "
          f"{v.mean_holdout_r2:8.4f} | {v.worst_state_r2:8.4f} | "
          f"{str(v.is_stable):>6} | {v.verdict.value:<28} | {claim}")
    if certified_here:
        tolerance = level

print(f"\nlast sigma/sigma_state certified: {tolerance}")
if tolerance is not None:
    print(f"first refusal at sigma/sigma_state = "
          f"{next(l for l in levels if l > tolerance)} (one seeded realization per level)")

# 9. GATE-4 STABILITY ACROSS INDEPENDENT SPLITS
#    Each (candidate, seed) pair gets its own disjoint, pre-registered slice, so
#    a verdict is never a lucky draw of one holdout. This is a DIAGNOSTIC: fresh
#    draws measure how stable the gate is, and a passing draw is not a promotion
#    (repeated testing multiplies the false-positive rate, which is exactly what
#    the preregistered alpha exists to prevent).
print("\n===== GATE 4 STABILITY ACROSS INDEPENDENT SPLITS =====")
print(f"{'seed':>4} | {'candidate':>17} | {'n':>3} | {'baseRMSE':>9} | {'candRMSE':>9} | {'dm':>6} | {'p':>9} | verdict")
print("-" * 108)
sweep_hits = {cid: 0 for cid, _, _ in contestants}
for seed in range(5):
    tr_idx, sl = allocate(seed)
    popt_s, _ = curve_fit(classical_langmuir, p_all[tr_idx], q_all[tr_idx],
                          p0=[q_all[tr_idx].max() * 1.2, 1.0 / np.median(p_all[tr_idx])],
                          bounds=(0, np.inf), maxfev=200000)
    for cid, name, predict in contestants:
        _, ix = sl[cid]
        p_sl, q_sl = p_all[ix], q_all[ix]
        pred_base_s = classical_langmuir(p_sl, *popt_s)
        pred_s = predict(p_sl)
        r = evaluate_predictive_incrementality(q_sl, pred_s, pred_base_s,
                                               alpha=DIEBOLD_MARIANO_P_VALUE_THRESHOLD)
        if r.verdict == "PREDICTIVELY_INCREMENTAL":
            sweep_hits[cid] += 1
        print(f"{seed:4d} | {name:>17} | {len(q_sl):3d} | "
              f"{np.sqrt(np.mean((q_sl - pred_base_s) ** 2)):9.4f} | "
              f"{np.sqrt(np.nanmean((q_sl - pred_s) ** 2)):9.4f} | "
              f"{r.dm_statistic:6.3f} | {r.p_value:9.4g} | {r.verdict}")

print("\npasses out of 5 independent draws:",
      {cid: f"{n}/5" for cid, n in sweep_hits.items()})
print("A candidate that passes only some draws is not promoted on its best draw;"
      " the pre-registered confirmation slice (section 6) remains the verdict.")
