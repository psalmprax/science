"""Non-neighbor margin argument for a proximity-restricted certified CBF set.

THE PITCH (velocity layer, same contract as ``cbf_certificate``)
================================================================
The certified CBF contract is ALL-PAIRS by definition: every pair (i, j)
must satisfy hdot_ij = (v_i − v_j)·n_ij ≥ −γ·h_ij, re-checked fail-closed.
Imposing it explicitly on every pair is what makes the certified QP
O(N²) per sweep. The margin argument shows a SUBSET of pairs needs no
constraint at all:

  For any pair and any commands inside the assumption box A1
  (|v_i| ≤ v_max as a magnitude):

      (v_i − v_j)·n_ij ≥ −‖v_i − v_j‖ ≥ −(‖v_i‖ + ‖v_j‖) ≥ −2·v_max.

  Hence WHENEVER h_ij ≥ h* := 2·v_max / γ  (units: length; γ in 1/time
  via A2's γ·dt ≤ 0.5), the pair condition holds CONSTRUCTIVELY:

      −2·v_max ≥ −γ·h_ij      necessarily,

  with equality at h_ij = h* — the worst-case anti-aligned commands
  exactly saturate the barrier law, never violate it. So a pair whose
  barrier sits at or beyond h* needs NO explicit constraint element:
  no command inside the box can violate its condition.

THE PROXIMITY LOCK (discrete realization, assumptions A1/A2/A5)
===============================================================
Solve the alternating-projection QP over the RESTRICTED pair set
P_restricted = {(i, j) : h_ij < h*}, then re-check fail-closed against
ALL pairs. Per step, every pair is then one of:

  * CONSTRAINED (h_ij < h*): certified by the restricted solve's
    all-pairs re-check — the pair is in the solve, so its deficit was
    repaired (or the solve refused, fail-closed);
  * FREE (h_ij ≥ h*): its condition holds by the margin algebra above.

Discrete forward invariance therefore extends to every pair: a
constrained pair obeys h_{k+1} ≥ (1 − γ·dt)·h_k by the certified
argument; a free pair obeys h_{k+1} ≥ h_k − 2·v_max·dt (relative closing
bounded by 2·v_max·dt through A5) ≥ (1 − γ·dt)·h_k exactly because
h_k ≥ 2·v_max/γ. One induction step per pair, per step, over any horizon.
A pair may only CROSS from free to constrained between steps, and it is
re-decided EVERY step from the h matrix the barriers build already
computes — the gate is a threshold filter over data that exists anyway;
no extra pair-distance work is added.

HONEST SCOPE (why this stays ``experimental``)
==============================================
* The margin argument is a sufficient-condition argument under A1/A2/A5 —
  the same assumption box as the certified module, nothing stronger.
* This module is a PITCH + machine-check harness, not a production path:
  it is wired to no pipeline, its restricted solver mirrors the certified
  projection schedule over the reduced set (provenance names both), and
  its every release passes the FULL all-pairs fail-closed re-check.
* Gate context (``docs/cbf_proximity_graph_backlog_gate.md``): the dense
  certified path holds real-time through N ≈ 128; this machinery is the
  prerequisite for N ≥ 256 real-time certified control. The [N,N,2]
  barriers matrix remains the dense floor — a production proximity
  structure would additionally need a spatial index (the counted numbers
  below quantify the CONSTRAINT reduction, not a wall-clock claim).

Status: experimental (certified-adjacent pitch; never self-adjudicating).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    CertificateViolation,
    _pair_barriers,
)

__all__ = [
    "ProximityOutcome",
    "free_margin",
    "solve_cbf_qp_proximity",
]


def free_margin(assumptions: CBFQPCertificateAssumptions) -> float:
    """h* = 2·v_max / γ: pairs at or beyond this barrier height are FREE.

    The threshold under which the worst-case box-bounded relative
    velocity (−2·v_max) can no longer violate the barrier law
    (−γ·h_ij): at h = h* the worst case EXACTLY saturates it.
    """
    if not (assumptions.gamma > 0 and assumptions.v_max > 0):
        raise ValueError("margin needs positive gamma and v_max")
    return 2.0 * assumptions.v_max / assumptions.gamma


@dataclass
class ProximityOutcome:
    """Evidence-only outcome of a proximity-restricted certified solve."""

    commands: np.ndarray            # [N, 2] certified commands (all-pairs re-checked)
    constrained_pairs: int          # pairs explicitly constrained in the solve
    free_pairs: int                 # pairs carrying NO constraint (margin-held)
    all_pairs: int                  # total pair count (the reduction bookkeeping)
    h_star: float
    dense_floor_last_deficit: float  # max all-pairs deficit of the released command
    meta: dict[str, Any]

    @property
    def constraint_ratio(self) -> float:
        """Fraction of pairs explicitly constrained (1.0 = dense QP)."""
        return self.constrained_pairs / max(1, self.all_pairs)


def solve_cbf_qp_proximity(
    positions: np.ndarray,
    nominal_velocities: np.ndarray,
    assumptions: CBFQPCertificateAssumptions,
    max_iter: int = 256,
) -> ProximityOutcome:
    """Alternating-projection QP over the RESTRICTED pair set (h < h*),
    with the SAME fail-closed all-pairs re-check as :func:`solve_cbf_qp`.

    The projection schedule mirrors the certified solver over the reduced
    pair set (provenance: python mirror OF solve_cbf_qp; kernel-parity is
    not claimed and not relevant here). Equality cases: identical-agent
    factory commands make every reduced deficit zero in one sweep.
    Fail-closed: ANY all-pairs defect beyond tol raises (max_iter=0 asks
    for an unsolved command and reaches the re-check unguarded — refused
    honestly, never silently repaired).
    """
    p = np.asarray(positions, dtype=float)
    v = np.clip(np.asarray(nominal_velocities, dtype=float).copy(),
                -assumptions.v_max, assumptions.v_max)
    n = p.shape[0]
    if n > assumptions.max_agents:
        raise CertificateViolation(
            f"assumption scope: {n} agents > max_agents {assumptions.max_agents}")
    if n < 2:
        return ProximityOutcome(
            commands=v, constrained_pairs=0, free_pairs=0, all_pairs=0,
            h_star=free_margin(assumptions), dense_floor_last_deficit=0.0,
            meta={"n": int(n), "note": "single agent: no pairs, no constraints"})

    bar = _pair_barriers(p, assumptions.r_safe)
    h_star = free_margin(assumptions)
    idx_i, idx_j = np.triu_indices(n, k=1)
    h_ij = bar.h[idx_i, idx_j]
    keep = h_ij < h_star                       # FREE pairs: h ≥ h* (margin-held)
    a_ij = bar.unit[idx_i, idx_j][keep]        # [P_r, 2]
    b_ij = -assumptions.gamma * h_ij[keep]     # constraint RHS: −γ·h_ij

    if a_ij.shape[0] == 0:
        # Every pair is free: the margin argument certifies the NOMINAL
        # (box-projected) commands with no projection loop at all.
        box = assumptions.v_max
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(
            1.0,
            np.divide(box, norms, out=np.full_like(norms, box), where=norms > 1e-12))
        v *= scale
    box = assumptions.v_max
    for _ in range(max_iter):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(
            1.0,
            np.divide(box, norms, out=np.full_like(norms, box),
                      where=norms > 1e-12))
        v *= scale
        lhs = np.einsum("pk,pk->p", a_ij, v[idx_i[keep]] - v[idx_j[keep]])
        deficit = b_ij - lhs
        bad = deficit > assumptions.tol
        if not bad.any():
            break
        kappa = deficit[bad] / 2.0
        corr = a_ij[bad] * kappa[:, None]
        np.add.at(v, idx_i[keep][bad], corr)
        np.add.at(v, idx_j[keep][bad], -corr)

    # FULL all-pairs fail-closed re-check (the certified contract is
    # all-pairs; the margin argument is re-verified NUMERICALLY here, in
    # addition to holding algebraically).
    idxi, idxj = np.triu_indices(n, k=1)
    a_all = bar.unit[idxi, idxj]
    h_all = bar.h[idxi, idxj]
    lhs_all = np.einsum("pk,pk->p", a_all, v[idxi] - v[idxj])
    rhs_all = -assumptions.gamma * h_all
    free_mask = ~keep
    worst_free = float(np.max(rhs_all[free_mask] - lhs_all[free_mask])) if free_mask.any() else float("-inf")
    worst_constrained = float(np.max(rhs_all[keep] - lhs_all[keep])) if keep.any() else float("-inf")
    worst = max(worst_free, worst_constrained)
    norms = np.linalg.norm(v, axis=1)
    if float(np.max(norms)) > box + assumptions.tol:
        raise CertificateViolation("CBF-QP(proximity) failed the norm-ball constraint (A1)")
    if float(np.max(np.abs(v))) > box + assumptions.tol:
        raise CertificateViolation("CBF-QP(proximity) failed the box constraint (A1)")
    if worst > assumptions.tol:
        raise CertificateViolation(
            f"CBF-QP(proximity) failed to certify: max all-pairs deficit {worst:.3e} "
            f"(free-pair worst {worst_free:.3e}; margin covers it algebraically)")

    meta = {
        "n": int(n),
        "max_iter_allowed": int(max_iter),
        "gamma": float(assumptions.gamma),
        "v_max": float(assumptions.v_max),
        "note": ("restricted-set solve + all-pairs fail-closed re-check; "
                 "free pairs hold by the margin algebra h >= 2*v_max/gamma, "
                 "re-verified numerically at release"),
    }
    return ProximityOutcome(
        commands=v,
        constrained_pairs=int(keep.sum()),
        free_pairs=int((~keep).sum()),
        all_pairs=len(h_ij),
        h_star=float(h_star),
        dense_floor_last_deficit=worst if worst > float("-inf") else 0.0,
        meta=meta,
    )
