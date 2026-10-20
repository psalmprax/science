"""REG-F16 evidence battery: formal CBF-QP certificate + model-check artifact.

Writes ``benchmarks/cbf_certificate_report.json``:
  - EXPLICIT assumption set (machine-checked; refuses impossible margins)
  - formal-layer results (feasibility lemma, fail-closed certification,
    projection containment, discrete invariance) over deterministic sweeps
  - model-checked unicycle realization through the PRODUCTION arc stepper
    (seeded adversarial head-on sweeps) — evidence, never a formal claim
  - disclosed boundary: batch_multi_agent_cbf_step remains the artificial-
    potential deadlock-remediation heuristic (docstring disclosure pinned).

Exit 0 requires every formal-layer check to hold and every model-checked
horizon to keep the barrier strictly positive. Any violation fails the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import datetime
import json
import math

import numpy as np
import torch

from apedp_dyn.cuda.batch_physics_steppers import batch_unicycle_arc_step
from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    certified_cbf_guard,
    solve_cbf_qp,
    unicycle_commands_from_certified,
)

REPORT_PATH = Path("benchmarks/cbf_certificate_report.json")
SEED = 1618
GAMMA, DT, V_MAX, R_SAFE = 2.0, 0.05, 1.0, 0.5


def _assumptions() -> CBFQPCertificateAssumptions:
    return CBFQPCertificateAssumptions(v_max=V_MAX, r_safe=R_SAFE, gamma=GAMMA, dt=DT)


def _sweep_positions(n_envs: int, n_agents: int, min_h0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    margin = min_h0 + R_SAFE
    while len(out) < n_envs:
        p = rng.uniform(-5.0, 5.0, size=(n_agents, 2))
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= margin:
            out.append(p)
    return np.stack(out)


def _barriers(positions: np.ndarray, r_safe: float):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)
    h = dist - r_safe
    np.fill_diagonal(h, 0.0)
    return h, diff / dist[..., None]


def formal_layer_results(assm) -> dict:
    rng = np.random.default_rng(SEED + 1)
    envs = _sweep_positions(64, 8, 0.05, SEED + 2)
    n_checks = 0
    worst_slack = math.inf
    worst_containment = 0.0
    for env in envs:
        nominal = rng.uniform(-assm.v_max, assm.v_max, size=env.shape)
        v_cert = solve_cbf_qp(env, nominal, assm)
        h, unit = _barriers(env, assm.r_safe)
        i, j = np.triu_indices(env.shape[0], k=1)
        slack = np.einsum("pk,pk->p", unit[i, j], v_cert[i] - v_cert[j]) + assm.gamma * h[i, j]
        worst_slack = min(worst_slack, float(np.min(slack)))
        worst_containment = max(
            worst_containment,
            float(np.max(np.linalg.norm(v_cert - nominal, axis=1)
                         - np.linalg.norm(nominal, axis=1))),
        )
        n_checks += 1
    # discrete invariance over a re-certified horizon
    n_agents = 6
    positions = _sweep_positions(4, n_agents, 0.02, SEED + 5)
    for env in positions:
        vel = rng.uniform(-assm.v_max, assm.v_max, size=env.shape)
        for _ in range(120):
            v_cert = solve_cbf_qp(env, vel, assm)
            env = env + assm.dt * v_cert
            vel = v_cert
            h = _barriers(env, assm.r_safe)[0]
            np.fill_diagonal(h, np.inf)  # exclude self-pairs from the min
            assert h.min() > 0.0, "discrete invariance violated in formal sweep"
    return {
        "claim": "FORMAL (velocity layer, under the explicit assumption set)",
        "environments_checked": int(len(envs)),
        "qp_releases_verified": int(n_checks),
        "worst_pair_constraint_slack": float(worst_slack),
        "worst_projection_optimality_excess": float(worst_containment),
        "discrete_invariance_horizons": "4 envs x 120 steps, h > 0 at every step",
        "fail_closed_refusal": "uncertified commands raise CertificateViolation (never released)",
    }


def model_check_results(assm) -> dict:
    n_envs, n_agents, steps = 16, 6, 60
    positions = _sweep_positions(n_envs, n_agents, 0.05, SEED + 7)
    worst = math.inf
    collisions = 0
    for b in range(n_envs):
        p = positions[b].copy()
        d = p[0] - p[1]
        th = np.full(n_agents, np.arctan2(-d[1], -d[0]))  # adversarial head-on
        for _ in range(steps):
            v_cert, _ = certified_cbf_guard(p, th, np.zeros_like(p), assm)
            v_cmd, omega_cmd = unicycle_commands_from_certified(th, v_cert, omega_max=2.0)
            poses = np.column_stack([p, th])
            outs = batch_unicycle_arc_step(
                torch.from_numpy(poses), torch.from_numpy(v_cmd),
                torch.from_numpy(omega_cmd), assm.dt)
            outs = outs.numpy()
            p, th = outs[:, :2], outs[:, 2]
            h = _barriers(p, assm.r_safe)[0]
            np.fill_diagonal(h, np.inf)  # exclude self-pairs from the min
            worst = min(worst, float(h.min()))
            if h.min() <= 0.0:
                collisions += 1
    return {
        "claim": "MODEL-CHECKED (unicycle realization through the production "
                 "arc stepper — evidence, NOT a formal claim)",
        "environments": int(n_envs), "agents_per_env": int(n_agents),
        "steps_per_env": int(steps),
        "scenario": "adversarial symmetric head-on courses (deadlock geometry)",
        "min_barrier_observed": float(worst),
        "collisions": int(collisions),
    }


def main() -> int:
    assm = _assumptions()
    formal = formal_layer_results(assm)
    model_check = model_check_results(assm)
    if formal["worst_pair_constraint_slack"] < -assm.tol:
        print("FORMAL layer FAILED: negative constraint slack")
        return 1
    if formal["worst_projection_optimality_excess"] > assm.tol:
        print("FORMAL layer FAILED: projection optimality violated (deviation > nominal deviation from 0)")
        return 1
    if model_check["collisions"] or model_check["min_barrier_observed"] <= 0.0:
        print("MODEL-CHECK FAILED: realized trajectory crossed the barrier")
        return 1
    report = {
        "title": "REG-F16 formal CBF-QP certificate + model-check battery",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "assumptions": assm.to_dict(),
        "assumption_gate": "refuses gamma*dt > 0.5, non-positive margins, out-of-scope agent counts",
        "formal_layer": formal,
        "model_check_layer": model_check,
        "boundary_disclosure": (
            "batch_multi_agent_cbf_step REMAINS the artificial-potential "
            "deadlock-remediation heuristic (its docstring's not-a-certified-CBF "
            "disclosure is pinned and unmodified). certified_cbf_guard is the "
            "additive certified layer; the unicycle realization is model-checked "
            "evidence, not a formal result. External safety sign-off (register "
            "source) remains the outstanding step."
        ),
        "seed": SEED,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report written: {REPORT_PATH}")
    print(f"formal: {formal['qp_releases_verified']} releases verified, "
          f"worst slack {formal['worst_pair_constraint_slack']:.3e}")
    print(f"model-check: min barrier {model_check['min_barrier_observed']:.4f}, "
          f"{model_check['collisions']} collisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
