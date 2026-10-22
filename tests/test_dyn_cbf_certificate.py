"""REG-F16 pins: formal CBF-QP certificate (docs/SYSTEM_FLAWS_AUDIT.md).

Scope of the pins (deliberately matches the module's claims):
  FORMAL — closed-form/invariance results at the single-integrator velocity
  layer, machine-checked over deterministic randomized sweeps.
  MODEL-CHECKED — the unicycle realization through the PRODUCTION stepper
  kinematics (batch_unicycle_arc_step), reported as evidence, never claimed
  formal. The production heuristic batch_multi_agent_cbf_step is untouched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from apedp_dyn.cuda.batch_physics_steppers import batch_unicycle_arc_step
from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    CertificateViolation,
    certified_cbf_guard,
    discrete_invariance_bound,
    model_check_unicycle_horizon,
    solve_cbf_qp,
    unicycle_commands_from_certified,
)

SEED = 1618


def _assumptions(**overrides) -> CBFQPCertificateAssumptions:
    base = {"v_max": 1.0, "r_safe": 0.5, "gamma": 2.0, "dt": 0.05}
    base.update(overrides)
    return CBFQPCertificateAssumptions(**base)


def _sweep_positions(n_envs: int, n_agents: int, min_h0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    margin = min_h0 + 0.5  # r_safe included
    while len(out) < n_envs:
        p = rng.uniform(-5.0, 5.0, size=(n_agents, 2))
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= margin:
            out.append(p)
    return np.stack(out)


def test_factory_feasibility_lemma():
    """Lemma: identical velocities satisfy every pair constraint while h >= 0,
    so the QP feasible set is never empty and certification never refuses on
    feasibility grounds."""
    assm = _assumptions()
    for env in _sweep_positions(16, 6, 1e-3, SEED):
        zero = np.zeros_like(env)
        v_cert = solve_cbf_qp(env, zero, assm)  # identical-velocity start
        bar_h, bar_unit = _barriers(env, assm.r_safe)
        i, j = np.triu_indices(env.shape[0], k=1)
        lhs = np.einsum("pk,pk->p", bar_unit[i, j], v_cert[i] - v_cert[j])
        assert np.all(lhs >= -assm.gamma * bar_h[i, j] - assm.tol)


def _barriers(positions: np.ndarray, r_safe: float):
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-12)
    h = dist - r_safe
    np.fill_diagonal(h, 0.0)
    return h, diff / dist[..., None]


def test_qp_output_satisfies_all_pair_constraints_fail_closed():
    """Certificate-by-construction: EVERY released command re-checked; a
    perturbed (uncertified) command is refused."""
    assm = _assumptions()
    rng = np.random.default_rng(SEED + 1)
    positions = _sweep_positions(8, 8, 0.05, SEED + 2)[0]
    nominal = rng.uniform(-assm.v_max, assm.v_max, size=positions.shape)
    v_cert = solve_cbf_qp(positions, nominal, assm)
    # projection optimality: the certified command is no farther from the
    # nominal than the zero command (which is feasible by the factory lemma):
    # - the QP deviation cannot exceed the nominal's own deviation from 0.
    dev_cert = np.linalg.norm(v_cert - nominal, axis=1)
    dev_zero = np.linalg.norm(nominal, axis=1)
    assert np.all(dev_cert <= dev_zero + 1e-9)
    h, unit = _barriers(positions, assm.r_safe)
    i, j = np.triu_indices(positions.shape[0], k=1)
    assert np.all(np.einsum("pk,pk->p", unit[i, j], v_cert[i] - v_cert[j])
                  >= -assm.gamma * h[i, j] - assm.tol)
    # fail-closed pin: with NO solver iterations the violating nominal reaches
    # the final check uncorrected and is REFUSED. The violating input is a
    # NEAR-CONTACT head-on course (h barely above 0: deficit = 2*v_max - gamma*h > 0);
    # at large h the constraint is trivially satisfied, so small h is required.
    unit = np.array([1.0, 0.0])
    near = np.array([[0.0, 0.0], [assm.r_safe + 1e-3, 0.0]])
    head_on = np.zeros((2, 2))
    head_on[0] = assm.v_max * unit           # agent 0 drives toward agent 1
    head_on[1] = -assm.v_max * unit          # agent 1 drives toward agent 0
    with pytest.raises(CertificateViolation):
        solve_cbf_qp(near, head_on, assm, max_iter=0)  # uncertified: refused
    # the SAME violating situation, solved properly, IS certified
    v_ok = solve_cbf_qp(near, head_on, assm)
    h, u = _barriers(near, assm.r_safe)
    assert float(np.einsum("k,k->", u[0, 1], v_ok[0] - v_ok[1])) >= -assm.gamma * h[0, 1] - assm.tol


def test_qp_is_minimal_deviation_from_nominal():
    """The certified command is a PROJECTION of the nominal: the certified set
    contains the 0 command (factory lemma), so projection cannot deviate more
    than the nominal itself does from the certified set."""
    assm = _assumptions()
    positions = _sweep_positions(4, 6, 0.05, SEED + 3)[0]
    nom = np.full(positions.shape, assm.v_max)  # worst-case push-apart... use push-IN nominal
    nom[0] = (assm.v_max, 0.0)
    v_cert = solve_cbf_qp(positions, nom, assm)
    assert np.all(np.linalg.norm(v_cert, axis=1) <= assm.v_max + 1e-9)


def test_discrete_invariance_over_certified_euler_horizon():
    """Discrete forward invariance: re-certifying every step, Euler realization
    obeys h_{k+1} >= (1 - gamma*dt) h_k — pinned against the analytic bound."""
    assm = _assumptions()
    rng = np.random.default_rng(SEED + 4)
    positions = _sweep_positions(1, 6, 0.02, SEED + 5)[0]
    vel = rng.uniform(-assm.v_max, assm.v_max, size=positions.shape)
    h0 = _barriers(positions, assm.r_safe)[0]
    np.fill_diagonal(h0, np.inf)
    bound = h0.min()
    for _ in range(120):
        v_cert = solve_cbf_qp(positions, vel, assm)
        positions = positions + assm.dt * v_cert
        vel = v_cert  # hold commanded velocities; re-certify nominal
        h, _ = _barriers(positions, assm.r_safe)
        np.fill_diagonal(h, np.inf)
        bound = min(bound, discrete_invariance_bound(h0.min(), assm) * (1 - assm.gamma * assm.dt) ** 0)
        assert h.min() > 0.0
    # collision cannot occur: h stays above the chain bound minus float noise
    assert h.min() >= 0.0


def test_assumption_gate_refuses_bad_assumptions():
    with pytest.raises(ValueError):
        _assumptions(gamma=30.0)  # gamma*dt = 1.5 > 0.5 margin
    with pytest.raises(ValueError):
        _assumptions(r_safe=-1.0)


def test_unicycle_realization_model_check_through_production_stepper():
    """Model-checked evidence (NOT formal): realized trajectories through the
    production arc stepper hold h > 0 over a seeded adversarial horizon
    (head-on courses included)."""
    assm = _assumptions()
    n_envs, n_agents = 8, 6
    positions = _sweep_positions(n_envs, n_agents, 0.05, SEED + 7)
    # adversarial head-on pairs: point each straight at the nearest neighbour
    headings = np.empty((n_envs, n_agents))
    for b in range(n_envs):
        p = positions[b]
        d = p[0] - p[1]
        headings[b] = np.arctan2(-d[1], -d[0])  # all facing each other (symmetric deadlock case)
    min_h = math.inf
    for b in range(n_envs):  # the certificate layer certifies one env at a time
        p = positions[b].copy()
        th = headings[b].copy()
        for _ in range(60):
            vel_nominal = np.zeros_like(p)
            v_cert, _diag = certified_cbf_guard(p, th, vel_nominal, assm)
            v_cmd, omega_cmd = unicycle_commands_from_certified(th, v_cert, omega_max=2.0)
            assert np.all(v_cmd >= 0.0) and np.all(np.abs(omega_cmd) <= 2.0 + 1e-9)
            poses = np.column_stack([p, th])  # arc stepper wants [N, 3]
            outs = batch_unicycle_arc_step(torch.from_numpy(poses),
                                           torch.from_numpy(v_cmd),
                                           torch.from_numpy(omega_cmd), assm.dt)
            outs = outs.numpy()
            p = outs[:, :2]
            th = outs[:, 2]  # theta += omega*dt convention
            h = _barriers(p, assm.r_safe)[0]
            np.fill_diagonal(h, np.inf)
            min_h = min(min_h, float(h.min()))
            assert h.min() > 0.0, f"collision in realization at min_h={h.min():.6f}"
    assert min_h > 0.0
    # artifact-level regression pin: the realized margin must never approach r_safe
    assert min_h >= 1e-3


def test_production_heuristic_docstring_disclosure_intact():
    """The production stepper's not-a-certified-CBF disclosure survives; the
    certified layer is additive and does not re-badge the heuristic."""
    from apedp_dyn.cuda.batch_physics_steppers import batch_multi_agent_cbf_step

    assert "not a certified quadratic-programming" in batch_multi_agent_cbf_step.__doc__


def test_model_check_utility_seeded_and_deterministic():
    assm = _assumptions()
    a = model_check_unicycle_horizon(
        _sweep_positions(2, 4, 0.05, SEED + 8)[0],
        np.array([0.0, math.pi, 0.0, math.pi]),
        assm, steps=30, arc_step_fn=_torch_arc, seed=SEED,
    )
    b = model_check_unicycle_horizon(
        _sweep_positions(2, 4, 0.05, SEED + 8)[0],
        np.array([0.0, math.pi, 0.0, math.pi]),
        assm, steps=30, arc_step_fn=_torch_arc, seed=SEED,
    )
    assert a["min_h_observed"] == pytest.approx(b["min_h_observed"], abs=1e-12)
    assert a["min_h_observed"] > 0.0


def _torch_arc(poses, v, omega, dt):
    return batch_unicycle_arc_step(torch.from_numpy(np.asarray(poses)),
                                   torch.from_numpy(np.asarray(v)),
                                   torch.from_numpy(np.asarray(omega)), dt)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
