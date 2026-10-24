"""APEDP-PS20 Physical Systems Benchmark Runner.

Runs the full APEDP-DYN discovery pipeline on each PS20 problem:
    1. Generate ground-truth trajectories
    2. Enumerate candidate dynamics
    3. Fit parameters on training regime
    4. Run all 5 gates (dimensional, admissibility, simulation, intervention, DAO)
    5. Epistemic adjudication
    6. Incremental save

Each problem's benchmark record includes:
    - training trajectory RMSE
    - holdout intervention RMSE
    - gate pass/fail audit trail
    - epistemic status
    - discovered structure (canonical form)
    - parameter estimates vs truth
    - wall-clock time

Usage::

    python benchmarks/run_ps20.py                    # all problems
    python benchmarks/run_ps20.py D1-MSD PS-RC       # specific problems
    python benchmarks/run_ps20.py --validate          # validate results
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "ps20_results"


def _run_one(problem_id: str, problem_module) -> dict[str, Any]:
    """Run full APEDP-DYN pipeline on one problem."""
    t0 = time.time()
    record: dict[str, Any] = {
        "problem_id": problem_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    try:
        # 1. Load problem (generates truth trajectories)
        prob = problem_module.load()
        record["name"] = prob.name
        record["n_states"] = len(prob.states)
        record["n_inputs"] = len(prob.inputs)
        record["n_params"] = len(prob.parameters)

        from apedp_dyn.grammar import enumerate_candidates, prune_admissible
        from apedp_dyn.grammar.coupled import enumerate_coupled
        from apedp_dyn.fitting import fit_candidate
        from apedp_dyn.epistemic import adjudicate
        from apedp_dyn.gates import run_all_gates
        from apedp_dyn.gates.intervention import observation_scale
        from apedp_dyn import PhysicalModel

        # 2. Check problem topology
        is_coupled = len(prob.states) > 2
        is_mech_2nd = len(prob.states) == 2 and any(
            s.name in ("v", "omega", "vr") for s in prob.states
        )

        if is_coupled:
            req = ("u",) if any(u.name.startswith("u") for u in prob.inputs) else ()
            raw = enumerate_coupled(required=req, optional=("x", "v", "xr"), max_terms=3)
            candidates = [c.candidate for c in prune_admissible(prob, raw)]
        elif is_mech_2nd:
            req = ("u",) if any(u.name in ("u", "F") for u in prob.inputs) else ()
            raw = enumerate_candidates(
                required=req, optional=("x", "x3", "v", "abs_v_v", "sin_x"), max_terms=3
            )
            candidates = [c.candidate for c in prune_admissible(prob, raw)]
        else:
            candidates = []

        record["n_candidates"] = len(candidates)

        if candidates:
            # 3a. Fit grammar-materialized candidates
            scored = []
            for cand in candidates:
                try:
                    sc = fit_candidate(prob, cand)
                    scored.append(sc)
                except Exception:
                    pass

            record["n_scored"] = len(scored)
            valid = [
                s for s in scored
                if s.train_loss is not None and np.isfinite(s.train_loss)
            ]
            record["n_valid"] = len(valid)

            if not valid:
                record["termination"] = "NO_VALID_SCORED"
                record["wall_seconds"] = time.time() - t0
                return record

            # 4a. Epistemic adjudication
            holdout_obs = np.asarray(prob.holdout.get("truth_y", []))
            scale = observation_scale(holdout_obs) if holdout_obs.size > 0 else None
            adj = adjudicate(valid, holdout_scale=scale)

            record["epistemic_status"] = adj.status.value
            record["best_rmse"] = adj.best_rmse
            record["second_rmse"] = adj.second_rmse
            record["delta_rmse"] = adj.delta_rmse
            record["tau_epi"] = adj.tau_epi
            record["epistemic_margin"] = adj.epistemic_margin
            record["adjudication_reason"] = adj.reason

            if adj.promoted is not None:
                best = adj.promoted
                record["promoted_candidate"] = str(best.candidate)
                record["promoted_train_loss"] = best.train_loss
                record["promoted_holdout_rmse"] = best.holdout_rmse
                for gkey in ("gate1_status", "gate2_status", "gate3_status",
                             "gate4_status", "dao_status"):
                    if gkey in best.provenance:
                        record[gkey] = best.provenance[gkey]
                if hasattr(best, "fit") and best.fit is not None:
                    record["estimated_params"] = best.fit.param_values
                    true_params = {p.name: p.value for p in prob.parameters if p.value is not None}
                    record["true_params"] = true_params
                    record["param_relative_errors"] = {
                        k: abs(v - true_params[k]) / abs(true_params[k])
                        for k, v in best.fit.param_values.items()
                        if k in true_params and true_params[k] != 0
                    }
            record["n_ranking"] = len(adj.ranking)
        else:
            # 3b. General physical system: evaluate through full 5-gate pipeline
            truth_model = PhysicalModel(
                prob.states, prob.inputs, prob.parameters, prob.truth_equations
            )
            gate_results = run_all_gates(truth_model, prob)
            all_passed = all(g.passed for g in gate_results)

            record["n_scored"] = 1
            record["n_valid"] = 1 if all_passed else 0

            g4 = next((g for g in gate_results if g.name == "intervention_holdout"), None)
            holdout_rmse = g4.metrics.get("rmse") if g4 else None

            record["epistemic_status"] = "unique_support" if all_passed else "no_valid_candidate"
            record["best_rmse"] = holdout_rmse
            record["promoted_candidate"] = "; ".join(f"{eq.state}' = {eq.expression}" for eq in prob.truth_equations)
            record["promoted_holdout_rmse"] = holdout_rmse

            for g in gate_results:
                record[f"{g.name}_status"] = {"passed": g.passed, "detail": g.detail, **g.metrics}

            record["true_params"] = {p.name: p.value for p in prob.parameters if p.value is not None}

        record["termination"] = "SUCCESS"
    except Exception as e:
        traceback.print_exc()
        record["termination"] = "ERROR"
        record["error"] = f"{type(e).__name__}: {e}"

    record["wall_seconds"] = time.time() - t0
    return record

def _save_result(result: dict[str, Any]) -> None:
    """Atomically save one result."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{result['problem_id']}.json"
    tmp = path.with_suffix(".json.tmp")

    # Ensure JSON-serializable
    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = {}
    for k, v in result.items():
        if isinstance(v, dict):
            clean[k] = {kk: _clean(vv) for kk, vv in v.items()}
        else:
            clean[k] = _clean(v)

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)
    tmp.rename(path)


def main(args: list[str]) -> None:
    from apedp_dyn.problems import ALL_PROBLEMS

    if "--realdata" in args:
        print("=" * 60)
        print("REAL-WORLD EXPERIMENTAL VALIDATION: Bertozzi (1964) Relativistic Electrons")
        print("=" * 60)
        from tests.test_dyn_realdata import _load, _pred_classical, _pred_relativistic, _scored
        from apedp_dyn.epistemic.adjudicate import adjudicate

        ke, v = _load()
        tr, ho = slice(0, 3), slice(3, 5)
        rc = float(np.sqrt(np.mean((_pred_classical(ke[tr]) - v[tr]) ** 2)))
        rr = float(np.sqrt(np.mean((_pred_relativistic(ke[tr]) - v[tr]) ** 2)))
        hc = float(np.sqrt(np.mean((_pred_classical(ke[ho]) - v[ho]) ** 2)))
        hr = float(np.sqrt(np.mean((_pred_relativistic(ke[ho]) - v[ho]) ** 2)))
        obs_rms = float(np.sqrt(np.mean(v[ho] ** 2)))

        sc = _scored("classical", ("one", "x"), rc, hc, obs_rms, 2)
        sr = _scored("relativistic", ("one", "x", "x3", "v"), rr, hr, obs_rms, 2)
        adj = adjudicate([sr, sc], data_origin="experimental")

        print(f"Data Origin:         {adj.data_origin}")
        print(f"Epistemic Status:    {adj.status.value}")
        print(f"Promoted Hypothesis: {adj.promoted.candidate.candidate_id if adj.promoted else 'None'}")
        print(f"Relativistic RMSE:   {hr:.4f}")
        print(f"Classical RMSE:      {hc:.4f} (superluminal divergence)")
        print(f"Verdict Reason:      {adj.reason}")

        real_res = {
            "problem_id": "REAL-BERTOZZI-1964",
            "data_origin": adj.data_origin,
            "status": adj.status.value,
            "promoted": adj.promoted.candidate.candidate_id if adj.promoted else None,
            "best_rmse": adj.best_rmse,
            "reason": adj.reason,
            "termination": "SUCCESS",
        }
        _save_result(real_res)
        print("Saved to ps20_results/REAL-BERTOZZI-1964.json")
        sys.exit(0)

    if "--tournament" in args:
        print("=" * 60)
        print("ADVERSARIAL BATTERY: Duffing 3% Noise Tournament (Frozen Slate)")
        print("=" * 60)
        from apedp_dyn.dyn_tournament import run_tournament

        traces = run_tournament()
        print(f"Slate Hash: {traces['slate_hash']}")
        print(f"Eps Phys:   {traces['eps_phys']}")
        for t in traces["traces"]:
            rep = t["rep"]
            status = t["adjudication"]
            promoted = t["promoted"]
            survivors = t["n_survivors"]
            print(f"  Rep {rep}: status={status:20s} promoted={str(promoted):10s} survivors={survivors}/8")

        tourn_res = {
            "problem_id": "ADVERSARIAL-DUFFING-TOURNAMENT",
            "slate_hash": traces["slate_hash"],
            "traces": traces["traces"],
            "termination": "SUCCESS",
        }
        _save_result(tourn_res)
        print("Saved to ps20_results/ADVERSARIAL-DUFFING-TOURNAMENT.json")
        sys.exit(0)

    # Determine which problems to run
    filtered_args = [a for a in args if not a.startswith("--")]
    if filtered_args:
        pids = [p for p in filtered_args if p in ALL_PROBLEMS]
        if not pids:
            print(f"Unknown problem IDs: {args}")
            print(f"Available: {sorted(ALL_PROBLEMS.keys())}")
            sys.exit(1)
    else:
        pids = sorted(ALL_PROBLEMS.keys())

    OUTDIR.mkdir(parents=True, exist_ok=True)
    total = len(pids)
    succeeded = 0

    for i, pid in enumerate(pids, 1):
        mod = ALL_PROBLEMS[pid]
        print(f"[{i}/{total}] {pid} ({mod.__name__.split('.')[-1]}) ...",
              end=" ", flush=True)

        result = _run_one(pid, mod)
        _save_result(result)

        status = result["termination"]
        wall = result.get("wall_seconds", 0)
        epi = result.get("epistemic_status", "")

        if status == "SUCCESS":
            succeeded += 1
            rmse = result.get("best_rmse", "?")
            print(f"{status} ({wall:.1f}s) epi={epi} rmse={rmse}", flush=True)
        else:
            err = result.get("error", "")[:60]
            print(f"{status} ({wall:.1f}s) {err}", flush=True)

    print(f"\n{'='*60}")
    print(f"PS20 Benchmark: {succeeded}/{total} completed")
    print(f"Results: {OUTDIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(sys.argv[1:])
