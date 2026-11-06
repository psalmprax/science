"""Fit a two-DOF saturation model end to end, and record the run.

The grammar now binds a saturation law's reference level as its own fitted
parameter (`applied` per mass as `s0_langmuir`, `s1_langmuir`, ...), so the chain

    enumerate -> prune (gates 1-2) -> materialize -> multistart fit (gates 3-4)
        -> adjudicate against rival laws

can be walked on a two-DOF problem. This script walks it once, on a planted law,
and writes every number it uses to
``benchmarks/two_dof_saturation_fit_results.json``: the slate it enumerated, the
gates each candidate passed, the fitted parameters against the planted ones, the
adjudicated status, and a noise sweep.

Why a planted law rather than the repo's D2-COUPLED: D2-COUPLED is linear, so a
saturation law is a wrong hypothesis there by construction. That comparison is
still made -- a saturating rival that nests the linear truth must not beat it on
confirmation -- but it cannot show that the reference level is *identified*.
Only a plant that has a knee can do that.

Observation noise: `DynProblem.train["truth_y"]` is the series the fitter
consumes, so the sweep perturbs that key and leaves the holdout truth exact. The
noise is therefore in the fit's data only, which is what a measurement noise
level means; gate 4 still scores against uncorrupted confirmation truth.

Usage::

    python3 scripts/measure_two_dof_saturation_fit.py
    python3 scripts/measure_two_dof_saturation_fit.py --no-write   # print only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from apedp_dyn import (  # noqa: E402
    adjudicate,
    enumerate_coupled,
    fit_candidate,
    prune_admissible,
)
from apedp_dyn.fitting.parameter_fit import (  # noqa: E402
    fit_candidate_multistart,
)
from apedp_dyn.model import (  # noqa: E402
    DynamicalEquation,
    InputVariable,
    Parameter,
    StateVariable,
)
from apedp_dyn.problems.base import DynProblem, finalize, make_signal  # noqa: E402

LAWS = ("langmuir_bind", "hill2_bind", "hill3_bind")
RECORD = ROOT / "benchmarks" / "two_dof_saturation_fit_results.json"

# Planted: both masses carry Langmuir (Michaelis-Menten) drag, each with its own
# half-saturation level, so a single shared scale parameter would be wrong.
PLANT = {"c": (0.20, 0.15), "a": (2.0, 3.0), "s": (0.30, 0.45)}


def planted_two_dof() -> DynProblem:
    """Two masses, each with saturating drag, driven at distinct frequencies."""
    c1, c2 = PLANT["c"]
    a1, a2 = PLANT["a"]
    s1, s2 = PLANT["s"]
    states = (StateVariable("x1", "meter", 0.35),
              StateVariable("v1", "meter/second", 0.0),
              StateVariable("x2", "meter", -0.28),
              StateVariable("v2", "meter/second", 0.0))
    inputs = (InputVariable("u1", "kilogram*meter/second**2"),
              InputVariable("u2", "kilogram*meter/second**2"))
    parameters = (Parameter("c1", "1/second", c1),
                  Parameter("a1", "meter/second**2", a1),
                  Parameter("s1", "meter", s1),
                  Parameter("c2", "1/second", c2),
                  Parameter("a2", "meter/second**2", a2),
                  Parameter("s2", "meter", s2))
    truth = (DynamicalEquation("x1", "v1"),
             DynamicalEquation("v1", "-c1*v1 - a1*x1/(s1 + Abs(x1)) + u1"),
             DynamicalEquation("x2", "v2"),
             DynamicalEquation("v2", "-c2*v2 - a2*x2/(s2 + Abs(x2)) + u2"))
    train = {"t_span": (0.0, 8.0),
             "u_of_t": make_signal(({"kind": "sine", "amplitude": 1.0,
                                     "frequency": 1.0},
                                    {"kind": "sine", "amplitude": 0.8,
                                     "frequency": 0.6})),
             "x0": (0.35, 0.0, -0.28, 0.0)}
    # Confirmation moves both drives and both ICs, past the knee of the law.
    holdout = {"t_span": (0.0, 8.0),
               "u_of_t": make_signal(({"kind": "sine", "amplitude": 1.4,
                                       "frequency": 1.7},
                                      {"kind": "sine", "amplitude": 1.1,
                                       "frequency": 0.9})),
               "x0": (0.55, 0.0, -0.40, 0.05),
               "tolerance": "auto"}
    return finalize(DynProblem("PLANT-2DOF-LANGMUIR", "two-DOF Langmuir drag",
                               states, inputs, parameters, truth, train,
                               holdout))


def with_observation_noise(problem: DynProblem, relative: float,
                           seed: int = 0) -> DynProblem:
    """Perturb the series the fitter consumes; leave confirmation truth exact."""
    if relative <= 0.0:
        return problem
    rng = np.random.default_rng(seed)
    observed = np.asarray(problem.train["truth_y"], dtype=float)
    scale = np.sqrt(np.mean(observed ** 2, axis=0, keepdims=True))
    noise = rng.normal(0.0, 1.0, size=observed.shape) * relative * scale
    train = dict(problem.train)
    train["truth_y"] = observed + noise
    return dataclasses.replace(problem, train=train)


def role_map(system) -> dict[str, str]:
    """Parameter name per structural role, so reporting survives q-naming."""
    out: dict[str, str] = {}
    for dof, equation in enumerate(system.equations):
        for term in equation.terms:
            if term.name == "u":
                out[f"dof{dof}.input"] = term.parameter
            elif term.name == "v":
                out[f"dof{dof}.damping"] = term.parameter
            elif term.name in LAWS:
                out[f"dof{dof}.saturation"] = term.parameter
            if term.scale_parameter:
                out[f"dof{dof}.scale"] = term.scale_parameter
    return out


def plant_seeded_vector(problem, system) -> list[float]:
    """The planted parameters, in the materialised model's parameter order.

    Diagnostic only. It uses knowledge the pipeline does not have, so it measures
    the objective's floor -- what the data and the structure permit -- and not
    what the search can find. The two are recorded apart on purpose.
    """
    from apedp_dyn.grammar.coupled import materialize_system

    model = materialize_system(problem, system)
    roles = role_map(system)
    values: dict[str, float] = {}
    for dof in (0, 1):
        values[roles[f"dof{dof}.damping"]] = -PLANT["c"][dof]
        values[roles[f"dof{dof}.saturation"]] = -PLANT["a"][dof]
        values[roles[f"dof{dof}.scale"]] = PLANT["s"][dof]
        values[roles[f"dof{dof}.input"]] = 1.0
    return [values[p.name] for p in model.parameters]


def pick_law_slate(systems, law: str):
    for system in systems:
        if all(set(eq.term_set()) == {"u", "v", law} for eq in system.equations):
            return system
    return None


def pick_linear_slate(systems):
    for system in systems:
        if all(set(eq.term_set()) == {"u", "v"} for eq in system.equations):
            return system
    return None


def gates_of(scored) -> dict[str, object]:
    provenance = scored.provenance
    out = {}
    for key in ("gate1_status", "gate2_status", "dao_status",
                "gate3_status", "gate4_status"):
        status = provenance.get(key)
        if status is None:
            continue
        out[key] = {"passed": bool(status.get("passed")),
                    "detail": status.get("detail")}
    return out


def summarise(scored, roles: dict[str, str]) -> dict[str, object]:
    fitted = (scored.fit.param_values if scored.fit is not None else {})
    return {
        "candidate_id": scored.candidate.candidate_id,
        "term_sets": [list(terms) for terms in scored.candidate.term_sets()],
        "train_loss": scored.train_loss,
        "holdout_rmse": scored.holdout_rmse,
        "nfev": getattr(scored.fit, "nfev", None),
        "parameters_by_role": {
            role: {"name": name,
                   "value": (round(float(fitted[name]), 6)
                             if name in fitted else None)}
            for role, name in roles.items()},
        "gates": gates_of(scored),
        "rejection_reason": scored.provenance.get("rejection_reason"),
    }


def run(starts: int = 1, max_nfev: int = 100) -> dict[str, object]:
    started = time.time()
    problem = planted_two_dof()
    slate = enumerate_coupled(optional=("v", *LAWS), max_terms=4)
    kept = prune_admissible(problem, slate)

    fits: dict[str, object] = {}
    recovery: dict[str, object] = {}
    objective_floor: dict[str, object] = {}
    scored_by_law = {}
    for law in LAWS:
        system = pick_law_slate(slate, law)
        if system is None:
            fits[law] = {"error": "no candidate with this law in both equations"}
            continue
        # Multistart only for the planted law, whose recovery is reported. The
        # rivals exist to be adjudicated against, so a single deterministic
        # start each is enough and keeps the run bounded.
        if law == "langmuir_bind":
            scored, info = fit_candidate_multistart(
                problem, system, n_starts=starts, max_nfev=max_nfev)
        else:
            scored = fit_candidate(problem, system, max_nfev=max_nfev)
            info = {"n_starts": 1, "starts": ["single"]}
        print(f"  fitted {law}: train={scored.train_loss} "
              f"holdout={scored.holdout_rmse}", flush=True)
        scored_by_law[law] = scored
        roles = role_map(system)
        fits[law] = {**summarise(scored, roles),
                     "multistart": {"n_starts": info.get("n_starts"),
                                    "starts": info.get("starts")}}
        if law == "langmuir_bind" and scored.fit is not None:
            fitted = scored.fit.param_values
            for dof in (0, 1):
                for role, planted in (
                        ("damping", -PLANT["c"][dof]),
                        ("saturation", -PLANT["a"][dof]),
                        ("scale", PLANT["s"][dof])):
                    name = roles.get(f"dof{dof}.{role}")
                    if name is None or name not in fitted:
                        continue
                    value = float(fitted[name])
                    reference = planted if role == "scale" else abs(planted)
                    measured = value if role == "scale" else abs(value)
                    recovery[f"dof{dof}.{role}"] = {
                        "parameter": name,
                        "planted": reference,
                        "fitted": round(measured, 6),
                        "signed_fitted": round(value, 6),
                        "relative_error": round(
                            abs(measured - reference) / reference, 4),
                    }

            # What the objective permits, seeded from the plant. If this lands
            # far below the self-started fit, the recorded recovery is a warm
            # start problem and not a limit of the data.
            seeded = fit_candidate(
                problem, system, x0=plant_seeded_vector(problem, system),
                max_nfev=max_nfev)
            floor_fit = (seeded.fit.param_values if seeded.fit is not None else {})
            floor_recovery: dict[str, object] = {}
            for dof in (0, 1):
                for role, planted in (("damping", -PLANT["c"][dof]),
                                      ("saturation", -PLANT["a"][dof]),
                                      ("scale", PLANT["s"][dof])):
                    name = roles.get(f"dof{dof}.{role}")
                    if name is None or name not in floor_fit:
                        continue
                    value = float(floor_fit[name])
                    reference = planted if role == "scale" else abs(planted)
                    measured = value if role == "scale" else abs(value)
                    floor_recovery[f"dof{dof}.{role}"] = {
                        "planted": reference,
                        "fitted": round(measured, 6),
                        "relative_error": round(
                            abs(measured - reference) / reference, 4),
                    }
            objective_floor = {
                "note": ("plant-seeded fit: the objective's floor, not what the "
                         "search can find"),
                "train_loss": seeded.train_loss,
                "holdout_rmse": seeded.holdout_rmse,
                "nfev": getattr(seeded.fit, "nfev", None),
                "warm_start_train_loss": scored.train_loss,
                "warm_start_over_floor": (
                    round(float(scored.train_loss) / float(seeded.train_loss), 1)
                    if seeded.train_loss else None),
                "recovery": floor_recovery,
            }
            print(f"  objective floor: plant-seeded loss={seeded.train_loss} "
                  f"vs warm start {scored.train_loss}", flush=True)

    linear = pick_linear_slate(slate)
    scored_linear = None
    if linear is not None:
        scored_linear = fit_candidate(problem, linear, max_nfev=max_nfev)
        fits["linear_control"] = summarise(scored_linear, role_map(linear))

    family = [scored for scored in scored_by_law.values() if scored.fit is not None]
    adjudication: dict[str, object] = {}
    if len(family) >= 2:
        adjudged = adjudicate(family)
        adjudication = {
            "status": str(getattr(adjudged.status, "value", adjudged.status)),
            "candidates": [s.candidate.candidate_id for s in family],
            "detail": getattr(adjudged, "reason", None),
        }

    sweep = []
    langmuir = pick_law_slate(slate, "langmuir_bind")
    if langmuir is not None:
        roles = role_map(langmuir)
        for relative in (0.0, 0.01, 0.03):
            noisy = with_observation_noise(problem, relative, seed=7)
            scored = fit_candidate(noisy, langmuir, max_nfev=max_nfev)
            fitted = scored.fit.param_values if scored.fit is not None else {}
            entry: dict[str, object] = {
                "relative_noise": relative,
                "train_loss": scored.train_loss,
                "holdout_rmse": scored.holdout_rmse,
                "scale_recovery": {},
            }
            for dof in (0, 1):
                name = roles[f"dof{dof}.scale"]
                if name in fitted:
                    reference = PLANT["s"][dof]
                    entry["scale_recovery"][f"dof{dof}.scale"] = {
                        "planted": reference,
                        "fitted": round(float(fitted[name]), 6),
                        "relative_error": round(
                            abs(float(fitted[name]) - reference) / reference, 4),
                    }
            sweep.append(entry)

    return {
        "schema": "two_dof_saturation_fit/v1",
        "generated_by": "scripts/measure_two_dof_saturation_fit.py",
        "plant": {
            "problem_id": problem.problem_id,
            "laws": list(LAWS),
            "parameters": {
                "c": list(PLANT["c"]), "a": list(PLANT["a"]),
                "s": list(PLANT["s"]),
            },
            "note": ("per mass: -c*v - a*x/(s+|x|) + u, both masses; the two "
                     "reference levels differ so a shared scale would be wrong"),
        },
        "slate": {
            "raw_systems": len(slate),
            "kept_after_gates_1_2": len(kept),
            "laws_enumerated": list(LAWS),
        },
        "fits": fits,
        "recovery": recovery,
        "adjudication": adjudication,
        "noise_sweep": sweep,
        "objective_floor": objective_floor,
        "settings": {"starts": starts, "max_nfev": max_nfev,
                     "noise_seed": 7},
        "wall_seconds": round(time.time() - started, 2),
        "limitations": (
            "One planted law, one excitation design, one IC pair. The plant's "
            "structure is in the enumerated slate, so this measures the "
            "fitter and the binding, not the grammar's search power."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECORD))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--starts", type=int, default=1,
                        help="deterministic multistarts for the planted law "
                             "(raise it to separate optimizer luck from a "
                             "structural failure; the committed record uses 1)")
    parser.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args()

    record = run(starts=args.starts, max_nfev=args.max_nfev)
    print(f"slate   raw={record['slate']['raw_systems']} "
          f"kept={record['slate']['kept_after_gates_1_2']}")
    for key, entry in record["recovery"].items():
        print(f"  {key:18s} planted {entry['planted']:<8} "
              f"fitted {entry['fitted']:<9} rel.err {entry['relative_error']}")
    for law in LAWS:
        fit = record["fits"].get(law, {})
        print(f"  {law:14s} train {fit.get('train_loss')} "
              f"holdout {fit.get('holdout_rmse')} "
              f"g4 {fit.get('gates', {}).get('gate4_status', {}).get('passed')}")
    print(f"adjudication: {record['adjudication'].get('status')}")
    floor = record.get("objective_floor") or {}
    if floor:
        print(f"objective floor: {floor['train_loss']} vs warm start "
              f"{floor['warm_start_train_loss']} "
              f"({floor['warm_start_over_floor']}x worse)")
    for entry in record["noise_sweep"]:
        errors = {k: v["relative_error"] for k, v in entry["scale_recovery"].items()}
        print(f"  noise {entry['relative_noise']:.0%}: scale rel.err {errors}")
    print(f"wall {record['wall_seconds']}s")

    if not args.no_write:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
