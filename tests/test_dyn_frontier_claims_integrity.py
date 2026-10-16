"""Claims-integrity pins for the Frontier 9/10 architectural record.

Regenerates every headline number in
``docs/frontier_9_of_10_architectural_resolution.md`` and asserts both that
the measured value matches the documented value within a tight tolerance and
that the artifact on disk actually states that value. A method change that
invalidates a documented claim must therefore fail here before it can ship
unnoticed (see AGENTS.md section 1.4, documentary integrity).

Numbers documented (2026-10-16 regeneration):
  - Planck Riccati: traj R2 = 0.999965, jet RMSE = 0.004906 (approximate fit)
  - Harmonic oscillator: traj R2 = 0.9999948646, jet RMSE = 6.11e-14 (exact)
  - Heat: dev R2 = 0.996306, holdout-window R2 = 0.999997, coef = 0.49719
  - Burgers: holdout-window R2 = 0.999909, adv = -0.996212, diff = 0.111180
  - TV composite: in-sample R2 = 0.999965
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import apedp_kernel as ak
from apedp_dyn.epistemic.pde_discovery import discover_pde
from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "frontier_9_of_10_architectural_resolution.md"
)


def _read_artifact() -> str:
    assert ARTIFACT.is_file(), f"Missing claim artifact: {ARTIFACT}"
    return ARTIFACT.read_text(encoding="utf-8")


def test_artifact_exists_and_declares_provenance():
    text = _read_artifact()
    assert "experimental" in text
    assert "2026-10-16" in text
    assert "test_dyn_frontier_claims_integrity" in text


def test_planck_riccati_claim_matches_record():
    v = np.linspace(0.3, 5.0, 80)
    u = (v ** 3) / (np.exp(v) - 1.0)
    cfg = {
        "max_x_degree": 3,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.02,
        "rcond_threshold": 1e-12,
        "include_riccati": True,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(v.tolist()), json.dumps(u.tolist()), json.dumps(cfg)
        )
    )
    assert abs(res["integrated_trajectory_r2"] - 0.999999999759) < 1e-6, res
    assert abs(res["jet_rmse"] - 8.597e-07) < 1e-7, res
    assert res.get("snapped_terms") is True, res
    assert "y^2" in res["formula"]
    text = _read_artifact()
    assert "0.999999999759" in text
    assert "8.597e-07" in text


def test_harmonic_oscillator_claim_matches_record():
    t = np.linspace(0.1, 3.0, 70)
    y = np.sin(2.0 * t)
    cfg = {
        "max_x_degree": 0,
        "max_diff_order": 2,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
        "include_riccati": False,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(t.tolist()), json.dumps(y.tolist()), json.dumps(cfg)
        )
    )
    assert abs(res["integrated_trajectory_r2"] - 0.9999948646) < 1e-6, res
    assert res["jet_rmse"] < 1e-12, res
    assert res["is_exact_recovery"]
    text = _read_artifact()
    assert "0.9999948646" in text


def _simulate_heat() -> tuple[np.ndarray, float, float]:
    nx, nt = 50, 40
    dx, dt = 0.1, 0.005
    u = np.zeros((nt, nx))
    x = np.linspace(0, (nx - 1) * dx, nx)
    u[0, :] = np.sin(x) + 0.5 * np.sin(2.0 * x)
    for i in range(nt - 1):
        for j in range(1, nx - 1):
            u[i + 1, j] = u[i, j] + dt * 0.5 * (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dx ** 2)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]
    return u, dt, dx


def _simulate_burgers() -> tuple[np.ndarray, float, float]:
    nx, nt = 60, 50
    dx, dt = 0.1, 0.002
    u = np.zeros((nt, nx))
    x = np.linspace(0, (nx - 1) * dx, nx)
    u[0, :] = np.sin(x)
    nu = 0.1
    for i in range(nt - 1):
        for j in range(1, nx - 1):
            adv = -u[i, j] * (u[i, j + 1] - u[i, j - 1]) / (2.0 * dx)
            diff = nu * (u[i, j + 1] - 2.0 * u[i, j] + u[i, j - 1]) / (dx ** 2)
            u[i + 1, j] = u[i, j] + dt * (adv + diff)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]
    return u, dt, dx


def test_heat_claim_matches_record():
    u, dt, dx = _simulate_heat()
    res = discover_pde(u, dt, dx, holdout_ratio=0.25)
    assert abs(res.r2_dev - 0.996306) < 2e-3, res
    assert abs(res.holdout_r2 - 0.999997) < 1e-4, res
    assert res.active_terms == ["u_xx"]
    assert abs(res.coefficients[0] - 0.49719) < 2e-3, res
    text = _read_artifact()
    assert "0.49719" in text
    assert "0.999997" in text


def test_burgers_claim_matches_record():
    u, dt, dx = _simulate_burgers()
    res = discover_pde(u, dt, dx, holdout_ratio=0.25)
    assert abs(res.holdout_r2 - 0.999909) < 1e-3, res
    assert "u*u_x" in res.active_terms
    assert "u_xx" in res.active_terms
    adv = res.coefficients[res.active_terms.index("u*u_x")]
    diff = res.coefficients[res.active_terms.index("u_xx")]
    assert abs(adv - (-0.996212)) < 5e-3, res
    assert abs(diff - 0.111180) < 5e-3, res
    assert "u" in res.active_terms, "spurious u-term documented as selected"
    text = _read_artifact()
    assert "0.999909" in text
    assert "0.11118" in text
    assert "11.2%" in text


def test_tv_composite_claim_matches_record():
    x = np.linspace(0.2, 8.0, 80)
    y = 2.0 * np.floor(x / 1.5) + 0.5 * x + 0.08 * (x ** 2)
    df = pd.DataFrame({"x": x, "target": y})
    proposer = RimeAdProposer(target_col="target")
    candidates = proposer.fit_and_propose(df)
    comp = [c for c in candidates if c.candidate_id == "tv_smooth_composite"]
    assert comp, "Rep 7 candidate must be emitted"
    best = comp[0]
    assert abs(best.metadata["r2"] - 0.999965) < 1e-4, best.metadata
    text = _read_artifact()
    assert "1.27%" in text
    assert "1.4810" in text
    assert "overstated" in text
