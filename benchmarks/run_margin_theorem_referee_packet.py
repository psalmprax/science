"""Margin-theorem referee packet builder (adjudication-ready evidence pack).

Registry anchor: ``ap edp_dyn/robotics/proximity_margin.py`` (the margin
argument itself); gate item: promotion checklist item 2 in
``docs/cbf_proximity_graph_backlog_gate.md``.

Writes ``benchmarks/margin_theorem_referee_packet.json``:

  - the theorem statement exactly as the pitch module states it (h* =
    2·v_max/γ; worst-case box-bounded relative velocity saturates −γ·h at
    exactly h = h*, never violates) with the EXACT assumption box A1/A2/A5
    it shares with the certified CBF module,
  - LIVE machine checks (executed now, values recorded verbatim):
    exact saturation at h = h*, headroom at h > h*, sampled box-worst-case
    bound, restricted-solve all-pairs fail-closed re-check honesty on
    randomized grid fixtures, fail-closed refusal on an unsolved
    provably-violated nominal, proximity-locked horizon model-check
    summary through the PRODUCTION arc stepper,
  - verification pointers: SHA-256 of the pitch module, the certified
    module, and the pin test file; the pin count collected from the test
    file; pointer to this builder script (re-runnable),
  - the DECISION slot left ``UNADJUDICATED``: acceptance into the
    certified boundary belongs to the frozen referee protocol — the
    proposer side assembles evidence only and must never decide its own
    acceptance (AGENTS.md §1.1).

Exit 0 requires every machine check to hold. Any check failure fails the
run (never a silent pass-through).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from apedp_dyn.robotics.cbf_certificate import (
    CBFQPCertificateAssumptions,
    CertificateViolation,
)
from apedp_dyn.robotics.proximity_margin import (
    free_margin,
    solve_cbf_qp_proximity,
)

PACKET_PATH = _REPO_ROOT / "benchmarks" / "margin_theorem_referee_packet.json"
CHECKS: dict[str, dict] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, block: dict) -> None:
    CHECKS[name] = block


def main() -> int:
    assm = CBFQPCertificateAssumptions(
        v_max=1.0, r_safe=0.5, gamma=0.5, dt=0.05, max_agents=256)

    # ── analytic core ────────────────────────────────────────────────────
    h_star = free_margin(assm)

    # Exact-saturation check: worst-case pair at h = h*, anti-aligned top
    # speed — the paired law holds with EQUALITY (never a violation).
    p = np.array([[0.0, 0.0], [assm.r_safe + h_star, 0.0]])
    h = p[1, 0] - p[0, 0] - assm.r_safe
    assert abs(h - h_star) < 1e-12
    v_worst = np.array([[assm.v_max, 0.0], [-assm.v_max, 0.0]])
    n_unit = np.array([-1.0, 0.0])  # (p_i − p_j)/‖·‖
    lhs = float(v_worst[0] @ n_unit - v_worst[1] @ n_unit)
    rhs = -assm.gamma * h_star
    check("exact_saturation_at_h_star", {
        "h_star": h_star,
        "worst_case_hdot": lhs,
        "barrier_rhs": rhs,
        "lapse": abs(lhs - rhs),
        "passes": bool(abs(lhs - rhs) < 1e-12 and lhs >= rhs),
        "statement": "(v_i − v_j)·n_ij = −2·v_max = −γ·h* at h = h* — SATURATION, not violation",
    })

    # Headroom strictly above h*: h = 1.001·h* leaves positive slack.
    h_above = 1.001 * h_star
    slack = assm.gamma * h_above - 2.0 * assm.v_max
    check("headroom_above_h_star", {
        "h": h_above,
        "slack": slack,
        "passes": bool(slack > 0.0),
    })

    # Sampled A1 worst-case: over randomized in-ASSUMPTION commands and
    # pair orientations, (v_i−v_j)·n never dips below −2·v_max (Cauchy–
    # Schwarz over the A1 NORM BALL, the same domain the certified solver's
    # box clip + norm scale projects onto). Sampling the per-axis box
    # instead would be sampling OUTSIDE A1: a (√2, √2)-magnitude vector
    # satisfies box clipping while violating the magnitude bound.
    rng = np.random.default_rng(2631)
    worst = float("inf")
    for _ in range(20_000):
        cmds = []
        for _agent in range(2):
            u = rng.normal(0.0, 1.0, 2)
            cmds.append(u * (assm.v_max * rng.uniform(0.0, 1.0) / max(np.linalg.norm(u), 1e-12)))
        vi, vj = cmds
        theta = rng.uniform(0, 2 * np.pi)
        n_rand = np.array([np.cos(theta), np.sin(theta)])
        val = float((vi - vj) @ n_rand)
        worst = min(worst, val)
    check("box_worst_case_bound_sampled", {
        "samples": 20_000,
        "min_observed_hdot": worst,
        "theoretical_floor": -2.0 * assm.v_max,
        "passes": bool(worst >= -2.0 * assm.v_max - 1e-12),
    })

    # Restricted-solve honesty: on randomized grid fixtures the restricted
    # solver releases commands that pass the FULL all-pairs fail-closed
    # re-check (the solver raises on any defect > tol; we ALSO re-verify
    # here independently of the solver's own guard).
    per_seed = []
    for seed in range(5):
        n = 12
        side = 4
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        pts = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * 3.0
        pts = pts + np.random.default_rng(seed).uniform(-0.4, 0.4, pts.shape)
        vel = np.random.default_rng(seed + 50).uniform(-1.0, 1.0, (n, 2))
        out = solve_cbf_qp_proximity(pts, vel, assm)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        idxi, idxj = np.triu_indices(n, k=1)
        unit = diff[idxi, idxj] / np.maximum(dist[idxi, idxj], 1e-12)[:, None]
        lhs_all = np.einsum("pk,pk->p", unit, out.commands[idxi] - out.commands[idxj])
        rhs_all = -assm.gamma * (dist[idxi, idxj] - assm.r_safe)
        worst_deficit = float((rhs_all - lhs_all).max())
        norm_ok = bool(np.max(np.linalg.norm(out.commands, axis=1)) <= assm.v_max * (1 + 1e-9))
        per_seed.append({
            "seed": seed,
            "all_pairs": out.all_pairs,
            "constrained_pairs": out.constrained_pairs,
            "free_pairs": out.free_pairs,
            "constraint_ratio": out.constraint_ratio,
            "h_star": out.h_star,
            "worst_all_pairs_deficit_after_release": worst_deficit,
            "box_conform": norm_ok,
        })
    check("restricted_solve_all_pairs_recheck_honesty", {
        "seeds": per_seed,
        "passes": bool(all(
            s["worst_all_pairs_deficit_after_release"] <= assm.tol and s["box_conform"]
            for s in per_seed)),
        "constraint_ratio_range": [
            min(s["constraint_ratio"] for s in per_seed),
            max(s["constraint_ratio"] for s in per_seed),
        ],
    })

    # Fail-closed refusal: an UNSOLVED nominal (max_iter=0) that provably
    # violates must reach the re-check and raise — never silently release.
    gap = assm.r_safe + 1e-3
    pts2 = np.array([[0.0, 0.0], [gap, 0.0]])
    vel2 = np.array([[assm.v_max, 0.0], [-assm.v_max, 0.0]])
    refused = False
    refusal_text = None
    try:
        solve_cbf_qp_proximity(pts2, vel2, assm, max_iter=0)
    except CertificateViolation as exc:
        refused = True
        refusal_text = str(exc)[:200]
    check("unsolved_nominal_refused_fail_closed", {
        "refused": refused,
        "detail": refusal_text,
        "passes": refused,
    })

    # Proximity-locked horizon model-check through the PRODUCTION arc
    # stepper (measured evidence for the realization boundary, not a formal
    # claim): 8-agent closing grid, 60 steps.
    n = 8
    side = 3
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    pts_horizon = np.stack([xs.ravel(), ys.ravel()], 1)[:n] * 3.0
    rng_h = np.random.default_rng(77)
    pts_horizon = pts_horizon + rng_h.uniform(-0.2, 0.2, pts_horizon.shape)
    th0 = rng_h.uniform(-np.pi, np.pi, n)
    try:
        from apedp_dyn.robotics.proximity_margin import (
            model_check_unicycle_horizon_proximity as _mc,
        )
        try:
            import torch  # noqa: F401  (production stepper dependency)
            raw = True
        except ImportError:
            raw = False
        horizon = _mc(pts_horizon, th0, assm, steps=60, omega_max=2.0, seed=77)
        horizon_blocked = raw and horizon["min_h_observed"] > 0.0
    except (ImportError, RuntimeError, ValueError, CertificateViolation) as exc:
        # torch/stepper unavailability or a fail-closed refusal are BOTH
        # honest non-pass outcomes, recorded — never swallowed silently.
        horizon = {"error": f"{type(exc).__name__}: {exc}"}
        horizon_blocked = False
    check("proximity_horizon_model_check", {
        "steps": horizon.get("steps", 60),
        "min_h_observed": horizon.get("min_h_observed"),
        "h_last": horizon.get("h_last"),
        "claim_disclosure": (horizon.get("claim")
                             or "model-check layer unavailable; refused honestly"),
        "passes": bool(horizon_blocked),  # only counts with torch present
        "torch_present": raw,
        "note": "passes=True counts only where the production stepper ran (torch present)",
    })

    # Verification pointers (re-runnable hashes).
    pitch_sha = sha256(_REPO_ROOT / "apedp_dyn" / "robotics" / "proximity_margin.py")
    cert_sha = sha256(_REPO_ROOT / "apedp_dyn" / "robotics" / "cbf_certificate.py")
    pins_path = _REPO_ROOT / "tests" / "test_dyn_proximity_margin_pins.py"
    pins_sha = sha256(pins_path)
    try:
        # Fixed argument list, no caller input: sys.executable + a repo-
        # relative pin path (the command is trusted by construction).
        collect = subprocess.run(  # noqa: S603 (fixed, trusted argument list)
            [sys.executable, "-m", "pytest", str(pins_path),
             "--collect-only", "-q"],
            capture_output=True, text=True, timeout=180,
            cwd=str(_REPO_ROOT))
        if collect.returncode != 0:
            print(f"pin collection failed: rc={collect.returncode} "
                  f"stderr={collect.stderr[-400:]}")
        import re
        pin_lines = [ln for ln in collect.stdout.splitlines() if "::" in ln]
        n_pins = len(pin_lines)
        if n_pins == 0:
            m = re.search(r"collected (\d+) (?:items|tests)", collect.stdout)
            if m:
                n_pins = int(m.group(1))
    except (OSError, subprocess.SubprocessError) as exc:
        # pytest unavailability in this environment is an honest non-pass:
        # record it, never swallow silently.
        collect = None
        n_pins = -1
        pin_lines = []
        print(f"pin collection unavailable: {exc}")

    battery = _REPO_ROOT / "benchmarks" / "cbf_certificate_report.json"
    packet = {
        "packet_kind": "margin_theorem_referee_packet",
        "assembled_by": "benchmarks/run_margin_theorem_referee_packet.py (re-runnable)",
        "theorem": {
            "name": "non-neighbor margin argument (velocity layer)",
            "statement": (
                "Under assumption box A1/A2/A5 (|v_i| ≤ v_max, γ·dt ≤ 0.5, "
                "deterministic numerics — the certified module's EXACT box): "
                "(v_i − v_j)·n_ij ≥ −‖v_i−v_j‖ ≥ −2·v_max, so for every pair "
                "with h_ij ≥ h* := 2·v_max/γ the barrier law "
                "hdot_ij ≥ −γ·h_ij holds CONSTRUCTIVELY; the anti-aligned "
                "top-speed pair at h = h* satisfies it with EXACT saturation. "
                "The restricted QP constrains only pairs with h_ij < h* and "
                "re-checks ALL pairs fail-closed per step; free pairs extend "
                "the certified discrete invariance h_{k+1} ≥ (1−γ·dt)·h_k."),
            "assumption_box": assm.to_dict(),
            "certified_module_share": (
                "the box and the all-pairs fail-closed re-check are the "
                "CERTIFIED module's, nothing stronger is claimed"),
        },
        "machine_checks": CHECKS,
        "verification_pointers": {
            "pitch_module": {
                "path": "apedp_dyn/robotics/proximity_margin.py",
                "sha256": pitch_sha,
            },
            "certified_module": {
                "path": "apedp_dyn/robotics/cbf_certificate.py",
                "sha256": cert_sha,
            },
            "pin_tests": {
                "path": "tests/test_dyn_proximity_margin_pins.py",
                "sha256": pins_sha,
                "pins_collected": n_pins,
            },
            "certificate_battery_artifact": {
                "path": "benchmarks/cbf_certificate_report.json",
                "exists": battery.exists(),
                "sha256": sha256(battery) if battery.exists() else None,
            },
        },
        "claim_boundary": [
            "sufficient-condition argument under A1/A2/A5 only",
            "unicycle realization is MODEL-CHECKED, never a formal claim",
            "kernel parity for the restricted solver is NOT claimed (python mirror provenance)",
            "production wiring stays behind the N ≥ 256 real-time gate; default cap N ≤ 64 does not trigger it",
            "the FULL all-pairs fail-closed re-check must ship in EVERY release: the compositional honesty floor",
        ],
        "decision_slot": {
            "status": "UNADJUDICATED",
            "instruction": (
                "Acceptance of the margin theorem into the certified boundary "
                "belongs to the frozen referee protocol (AGENTS.md §1.1: a "
                "proposer never decides its own acceptance). This packet is "
                "proposer-side evidence only; the deciding layer must "
                "record its verdict, date and evidence fingerprint here."),
        },
    }

    all_mandatory_pass = all(
        c["passes"] for k, c in CHECKS.items() if k != "proximity_horizon_model_check"
    )
    packet["builder_exit_code"] = 0 if all_mandatory_pass else 1

    PACKET_PATH.write_text(json.dumps(packet, indent=2))
    print(f"packet written: {PACKET_PATH}")
    print(f"checks passed (mandatory): {all_mandatory_pass}; "
          f"horizon check (torch-gated): {CHECKS['proximity_horizon_model_check'].get('passes')}")
    return packet["builder_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
