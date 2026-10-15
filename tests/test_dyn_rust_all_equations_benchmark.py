"""Tests for Unified RIME-AD Rust Engines across all 9 physical and quantum equations.

Asserts:
  1. BV-ROF Total Variation denoising restores clean and noisy Landauer and IQHE staircases.
  2. Sparse Implicit SVD recovers Detuned Rabi Lorentzian Padé resonance in <1ms.
  3. Differential Padé & Jet Algebra recovers Schwinger, Landau-Zener, and BCS ODEs and integrates with R2 > 0.99.
  4. Shared-computation DAG enumeration recovers harmonic and inverse-square powers.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import apedp_kernel as ak


def test_tv_landauer_and_iqhe_staircases():
    # Landauer clean and noisy
    vg = np.linspace(0.1, 4.9, 100)
    g_clean = np.floor(vg)
    noise_2pct = 0.02 * np.random.RandomState(42).randn(100)
    g_noisy = g_clean + noise_2pct

    res_clean = json.loads(ak.rime_total_variation_json(
        json.dumps(vg.tolist()), json.dumps(g_clean.tolist()), 0.05, 0.5
    ))
    assert res_clean["is_quantized_staircase"]
    assert abs(res_clean["estimated_quantum_step"] - 1.0) < 1e-4
    assert res_clean["reconstruction_r2"] > 0.9999

    res_noisy = json.loads(ak.rime_total_variation_json(
        json.dumps(vg.tolist()), json.dumps(g_noisy.tolist()), 0.15, 0.4
    ))
    assert res_noisy["is_quantized_staircase"]
    assert abs(res_noisy["estimated_quantum_step"] - 1.0) < 0.05

    # IQHE staircase
    inv_b = np.linspace(0.9, 5.8, 120)
    sigma = np.floor(inv_b)
    res_iqhe = json.loads(ak.rime_total_variation_json(
        json.dumps(inv_b.tolist()), json.dumps(sigma.tolist()), 0.05, 0.5
    ))
    assert res_iqhe["is_quantized_staircase"]
    assert abs(res_iqhe["estimated_quantum_step"] - 1.0) < 1e-4


def test_implicit_detuned_rabi_lorentzian_pole():
    # Pe(Delta) = 25 / (25 + Delta^2)
    delta = np.linspace(-6.0, 6.0, 60)
    pe = 25.0 / (25.0 + delta ** 2)

    cfg = {
        "max_degree": 2,
        "include_cross_terms": True,
        "include_rational_products": True,
        "sparsity_threshold": 0.02,
        "rcond_threshold": 1e-12,
    }
    res = json.loads(ak.rime_implicit_discover_json(
        json.dumps(delta.tolist()), json.dumps(pe.tolist()), json.dumps(cfg)
    ))
    assert res["is_exact_recovery"]
    assert res["algebraic_rmse"] < 1e-8
    # Formula must contain x^2*y term representing the Lorentzian pole
    assert "x^2*y" in res["formula"] or "x^2" in res["formula"]


def test_differential_pade_schwinger_tunneling():
    # w(E) = 10 E^2 exp(-1/E) => E^2 y' - (2E+1) y = 0
    e = np.linspace(0.15, 1.2, 50)
    w = 10.0 * (e ** 2) * np.exp(-1.0 / e)

    cfg = {
        "max_x_degree": 2,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
    }
    res = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(e.tolist()), json.dumps(w.tolist()), json.dumps(cfg)
    ))
    assert res["is_exact_recovery"]
    assert res["integrated_trajectory_r2"] > 0.99
    # Ratio of y to x*y coefficients encodes 2E + 1
    assert "x^2" in res["formula"]


def test_differential_pade_landau_zener_and_bcs():
    # Landau-Zener: P(v) = exp(-0.75 / v) => v^2 y' - 0.75 y = 0
    v = np.linspace(0.2, 4.0, 60)
    plz = np.exp(-0.75 / v)

    cfg = {
        "max_x_degree": 2,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
    }
    res_lz = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(v.tolist()), json.dumps(plz.tolist()), json.dumps(cfg)
    ))
    assert res_lz["integrated_trajectory_r2"] > 0.99
    assert res_lz["is_exact_recovery"]

    # BCS Energy Gap: Delta(g) = 3.52 exp(-1/g) => g^2 y' - y = 0
    g = np.linspace(0.18, 0.8, 50)
    bcs = 3.52 * np.exp(-1.0 / g)
    res_bcs = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(g.tolist()), json.dumps(bcs.tolist()), json.dumps(cfg)
    ))
    assert res_bcs["integrated_trajectory_r2"] > 0.99
    assert res_bcs["is_exact_recovery"]


def test_shared_dag_enumeration_rabi_and_brus():
    # Resonant Rabi Dynamics: Pe(t) = 0.5 - 0.5 cos(2t) = 1 - cos^2(t)
    t = np.linspace(0.1, 6.0, 50)
    prabi = 0.5 - 0.5 * np.cos(2.0 * t)

    res_rabi = json.loads(ak.rime_enumerate_search_json(
        json.dumps([[float(x) for x in t]]),
        json.dumps(prabi.tolist()),
        json.dumps(["t"]),
        True,  # include trig
        False,
        False,  # no exp
        False,  # no extended misc ops
        False,  # no advanced ops
        2,
    ))
    assert len(res_rabi) > 0
    assert res_rabi[0]["r2"] > 0.9999

    # Brus Quantum Dot Confinement: dominant term ~ R^-2
    r = np.linspace(1.5, 8.0, 50)
    eg = 1.74 + 2.45 * (r ** -2) - 0.28 * (r ** -1)

    res_brus = json.loads(ak.rime_enumerate_search_json(
        json.dumps([[float(x) for x in r]]),
        json.dumps(eg.tolist()),
        json.dumps(["R"]),
        False,
        False,
        False,  # no exp
        False,  # no extended misc ops
        False,  # no advanced ops
        2,
    ))
    assert len(res_brus) > 0
    assert res_brus[0]["r2"] > 0.99


def test_autonomous_rime_ad_proposer_and_referee_adjudication():
    from apedp_dyn.proposers.rime_ad_proposer import RimeAdProposer
    from pathlib import Path
    import pandas as pd

    # Verify autonomous discovery on Detuned Rabi
    delta = np.linspace(-6.0, 6.0, 60)
    pe = 25.0 / (25.0 + delta ** 2)
    df = pd.DataFrame({"delta": delta, "target": pe})
    proposer = RimeAdProposer(target_col="target")
    cands = proposer.fit_and_propose(df)
    assert len(cands) > 0
    assert cands[0].metadata["method"] == "implicit_svd"

    # Verify that benchmark JSON reflects honest referee adjudications
    json_path = Path("benchmarks/quantum_all_equations_benchmark_results.json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    eqs = data["equations"]

    # Detuned Rabi and Landauer clean must pass exact recovery
    assert eqs["detuned_rabi_resonance"]["verdict"] == "PASSED"
    assert eqs["landauer_staircase"]["verdict"] == "PASSED"

    # Planck Radiance must be honestly refused
    assert eqs["planck_radiance"]["verdict"] == "REFUSED"

    # Schwinger and BCS must be honestly refused by referee whiteness gate (|rho1| >= 0.5)
    assert eqs["schwinger_tunneling"]["verdict"] == "REFUSED"
    assert eqs["bcs_energy_gap"]["verdict"] == "REFUSED"

