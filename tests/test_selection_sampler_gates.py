"""Fast deterministic tests for the flaw-fix batch (selector, sampler, gates).

Covers: BIC-forward exact recovery, reciprocal powers, cross-family
overfit guard, extrapolation audit, HMC core + frozen-chain diagnostics,
MH validation errors, adequacy gate. All seeded; no data files needed.
"""
from __future__ import annotations

import numpy as np
import pytest


def _rng():
    return np.random.default_rng(0)


def test_bic_forward_recovers_poly_exactly():
    pytest.importorskip("sklearn")
    from apedp.discovery import sparse_select_terms
    rng = _rng()
    x = rng.uniform(-3, 3, 60)
    y = x**2 - 2 * x + 1 + rng.normal(0, 0.4, 60)
    out = sparse_select_terms({"x": x}, y)
    assert set(out["terms"]) == {"x**1", "x**2"}, out
    assert out["adequacy"]["verdict"] == "OK", out["adequacy"]


def test_bic_forward_brus_from_raw_R():
    pytest.importorskip("sklearn")
    from apedp.discovery import sparse_select_terms
    rng = _rng()
    R = np.linspace(1.0, 6.0, 40)
    y = (3.728 / R**2 - 0.254 / R) * (1 + rng.normal(0, 0.02, len(R)))
    out = sparse_select_terms({"R": R}, y)
    assert set(out["terms"]) == {"1/R**2", "1/R"}, out


def test_reciprocal_powers_in_library():
    pytest.importorskip("sklearn")
    from apedp.discovery import sparse_select_terms
    rng = _rng()
    x = rng.uniform(1.0, 5.0, 40)
    y = 2.0 / x**2 + rng.normal(0, 0.005, 40)
    out = sparse_select_terms({"x": x}, y)
    assert "1/x**2" in out["terms"], out


def test_overfit_guard_rejects_9param_winner():
    from apedp.discovery import cross_family_select
    lam = np.array([480, 500, 520, 550, 580, 600, 610], float)
    D = 1.6122e-9 * lam**4 - 2.6575e-6 * lam**3 + 1.6242e-3 * lam**2 - 0.4277 * lam + 41.57
    R, dE = D / 2.0, 1240.0 / lam - 1.74
    out = cross_family_select({"R": R}, dE)
    assert out.get("best") is not None
    assert out["best"]["num_params"] <= 4, out["best"]
    assert any("rejected" in n for n in out.get("notes", [])), out


def test_extrapolation_audit():
    from apedp.scientific_layers import extrapolation_audit
    bad = extrapolation_audit("exp(1.98/R)", {"R": (1.0, 2.5)}, 0.84)
    assert bad["status"] == "FAIL", bad
    good = extrapolation_audit("R**(-2)", {"R": (1.0, 2.5)}, 0.84)
    assert good["status"] == "PASS", good
    unparseable = extrapolation_audit("1/0", {"R": (1.0, 2.5)}, 0.84)
    assert unparseable["status"] in ("FAIL", "CONDITIONAL"), unparseable


def test_hmc_1d_normal():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from apedp.priors import hmc_sample
    draws, acc = hmc_sample(lambda th: -0.5 * jnp.sum(th**2),
                            np.array([0.0]), n_iter=100, step_size=0.5,
                            n_leapfrog=10, mass=np.ones(1), seed=0)
    assert acc > 0.5, acc
    assert abs(float(draws.mean())) < 0.3, float(draws.mean())
    assert 0.7 < float(draws.std()) < 1.3, float(draws.std())


def test_hmc_jitter_breaks_resonance():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from apedp.priors import hmc_sample
    CV = np.array([[4.0, 0.9], [0.9, 0.25]])
    Prec = np.linalg.inv(CV)

    def lp(th):
        return -0.5 * (th @ (jnp.asarray(Prec) @ th))

    def lag1(draws):
        x = np.asarray(draws[100:, 0], dtype=float)
        x = x - x.mean()
        return float(x[:-1] @ x[1:] / (x @ x))

    d_fix, _ = hmc_sample(lp, np.array([5.0, -3.0]), n_iter=400,
                          step_size=0.3, n_leapfrog=10,
                          mass=Prec, seed=0, jitter=False)
    d_jit, acc = hmc_sample(lp, np.array([5.0, -3.0]), n_iter=400,
                            step_size=0.3, n_leapfrog=10,
                            mass=Prec, seed=0, jitter=True)
    assert abs(lag1(d_jit)) < abs(lag1(d_fix)), (lag1(d_fix), lag1(d_jit))
    assert acc > 0.9, acc


def test_frozen_chain_diagnostics():
    pytest.importorskip("jax")
    from apedp.priors import posterior_diagnostics
    const = np.ones((2, 50))
    out = posterior_diagnostics({"a": const})
    assert out["a"]["ess"] == 0.0, out
    assert out["a"]["r_hat"] == float("inf"), out


def _dummy_mh(n_iter=60, **kw):
    from apedp.priors import build_semf_posterior
    Z = np.array([6.0, 8.0, 26.0])
    N = np.array([6.0, 8.0, 30.0])
    A = Z + N
    BE = np.array([7.6, 8.0, 8.8])

    def like(coeffs, Z, N, A, BE):
        tot = sum((float(coeffs[k]) - t) ** 2
                  for k, t in (("a_v", 15.0), ("a_s", 18.0),
                               ("a_c", 0.7), ("a_a", 23.0)))
        return -tot

    def sampler(key, n_samples=0):
        rng = np.random.default_rng(1)
        n = max(n_samples, 1)
        return {"a_v": rng.normal(15.0, 1.0, n),
                "a_s": rng.normal(18.0, 1.0, n),
                "a_c": rng.normal(0.7, 0.1, n),
                "a_a": rng.normal(23.0, 1.0, n),
                "Delta": 1.6}

    return build_semf_posterior(like, sampler, Z, N, A, BE,
                                n_chain=2, n_iter=n_iter,
                                proposal_scale=0.05, seed=0, **kw)


def test_mh_runs_and_reports():
    pytest.importorskip("jax")
    out = _dummy_mh()
    assert set(out["samples"]) == {"a_v", "a_s", "a_c", "a_a"}
    assert all(0.0 <= v <= 1.0 for v in out["acceptance_rates"].values())
    assert "diagnostics" in out and "final_state" in out


def test_mh_rejects_bad_proposal_cov():
    pytest.importorskip("jax")
    with pytest.raises(ValueError):
        _dummy_mh(proposal_cov=np.eye(3))
    with pytest.raises(ValueError):
        _dummy_mh(init_state={"a_v": np.array([1.0])})


def test_two_stage_smoke():
    pytest.importorskip("jax")

    from apedp.priors import two_stage_posterior
    Z = np.array([6.0, 8.0, 26.0, 28.0])
    N = np.array([6.0, 8.0, 30.0, 30.0])
    A = Z + N
    BE = np.array([7.6, 8.0, 8.8, 8.79])

    def like(coeffs, Z, N, A, BE):
        tot = sum((float(coeffs[k]) - t) ** 2
                  for k, t in (("a_v", 15.0), ("a_s", 18.0),
                               ("a_c", 0.7), ("a_a", 23.0)))
        return -tot

    def sampler(key, n_samples=0):
        rng = np.random.default_rng(1)
        n = max(n_samples, 1)
        return {"a_v": rng.normal(15.0, 1.0, n),
                "a_s": rng.normal(18.0, 1.0, n),
                "a_c": rng.normal(0.7, 0.1, n),
                "a_a": rng.normal(23.0, 1.0, n),
                "Delta": 1.6}

    out = two_stage_posterior(like, sampler, Z, N, A, BE, n_chain=2,
                              n_warmup=120, n_main=200,
                              proposal_scale=0.05, seed=0, burn=50)
    assert out["method"] == "two_stage_legacy_plus_fixed_mvn", out["method"]
    assert out["samples"]["a_v"].shape == (2, 150), out["samples"]["a_v"].shape


def test_mlp_finds_narrow_range_variable():
    pytest.importorskip("sklearn")
    from apedp.discovery import neural_relevance
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 1000, 100)
    x2 = rng.uniform(0, 1, 100)
    y = 0.02 * x1 + 15.0 * x2 + rng.normal(0, 0.5, 100)
    out = neural_relevance({"x1": x1, "x2": x2}, y)
    assert set(out["relevant"]) == {"x1", "x2"}, out
    assert out["surrogate_r2"] > 0.9, out


def test_mlp_reports_convergence():
    pytest.importorskip("sklearn")
    from apedp.discovery import neural_relevance
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 5.0, 80)
    out = neural_relevance({"x": x}, 2.0 * x + rng.normal(0, 0.1, 80))
    assert isinstance(out["converged"], bool), out
    assert out["n_iter"] >= 1, out


def test_mlp_gate_refuses_noise():
    pytest.importorskip("sklearn")
    from apedp.discovery import neural_relevance
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 40)
    out = neural_relevance({"x": x}, rng.normal(0, 1, 40))
    assert out["relevant"] == [], out


def test_distill_grid_stays_honest():
    pytest.importorskip("sklearn")
    from apedp.discovery import distill_via_surrogate
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 60)
    y = rng.uniform(0.5, 5.0, 60)
    out = distill_via_surrogate({"x": x, "y": y}, x * y)
    assert "x*y" in out["terms"], out
    assert out["surrogate_r2"] > 0.9, out
    assert "converged" in out, out


def test_ess_geyer_handles_antithetic():
    pytest.importorskip("jax")
    from apedp.priors import posterior_diagnostics
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2, 200))
    anti = np.empty_like(base)
    anti[:, ::2] = base[:, ::2]
    anti[:, 1::2] = -base[:, ::2]  # perfect alternation
    out = posterior_diagnostics({"z": anti})
    assert np.isfinite(out["z"]["ess"]), out
    assert 0.0 < out["z"]["ess"] <= 400.0, out


def test_rediscover_reports_strict_and_tolerant():
    import os

    from apedp.semf_validation import design_columns, parse_mas20, rediscover_magic
    path = os.path.join(os.path.dirname(__file__), "..", "data", "mass_1.mas20.txt")
    assert os.path.isfile(path), "vendored AME2020 missing"
    data = parse_mas20(path)
    assert len(data) > 2000
    mask = data.A >= 16
    X = design_columns(data.Z[mask], data.N[mask], data.A[mask])
    coef, *_ = np.linalg.lstsq(X, data.BE_per_nucleon[mask] * data.A[mask], rcond=None)
    resid = data.BE_per_nucleon[mask] * data.A[mask] - X @ coef
    from types import SimpleNamespace
    sub = SimpleNamespace(Z=data.Z[mask], N=data.N[mask], A=data.A[mask],
                          BE_per_nucleon=data.BE_per_nucleon[mask])
    out = rediscover_magic(sub, resid, by="N")
    assert "recall_strict" in out and "recall" in out, out
    assert set(out["hits_strict"]) <= set(out["hits"]), out


def test_hybrid_shuffle_note():
    from apedp.hybrid import hybrid_discover
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 5.0, 30)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 30)
    out = hybrid_discover({"x": x}, y, ["x"], generations=2, pop_size=8,
                          seed=0, shuffle=True)
    assert any("shuffled" in n for n in out.get("probes", [])), out.get("probes")
    out2 = hybrid_discover({"x": x}, y, ["x"], generations=2, pop_size=8,
                           seed=0, shuffle=False)
    assert any("sequential" in n for n in out2.get("probes", [])), out2.get("probes")


def test_adequacy_gate_flags_thin():
    from apedp.discovery import check_data_adequacy
    R = np.linspace(1.0, 1.6, 6)
    rep = check_data_adequacy({"R": R}, R ** -2)
    assert rep["verdict"] == "THIN", rep
    assert any("max/min" in r for r in rep["reasons"]), rep
    Rok = np.linspace(1.0, 6.0, 40)
    rep_ok = check_data_adequacy({"R": Rok}, Rok ** -2)
    assert rep_ok["verdict"] == "OK", rep_ok
