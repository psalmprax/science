"""Focused regression tests for the remaining scientific capabilities."""

import numpy as np
import pytest


def test_jax_heat_validation_and_neumann_constant():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from apedp.jax_pde_solver import heat_1d_explicit_jit, validate_explicit_grid

    with pytest.raises(ValueError, match="unstable heat"):
        validate_explicit_grid(1.0, 0.1, 0.01, 2)
    out = heat_1d_explicit_jit(
        jnp.ones(16), 0.1, 1.0 / 15.0, 0.0001, 10, bctype="neumann"
    )
    assert np.allclose(np.asarray(out), 1.0)


def test_hierarchical_prior_and_posterior_diagnostics():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from apedp.priors import (
        build_semf_posterior,
        hierarchical_semf_prior,
        posterior_diagnostics,
        semf_prior_samples,
    )

    prior = hierarchical_semf_prior(jax.random.PRNGKey(3), n_domains=2, n_samples_per_domain=4)
    assert prior["a_v"].shape == (2, 4)
    assert "hyper_means" in prior

    def likelihood(coeffs, _z, _n, _a, _observed):
        return -0.5 * (coeffs["a_v"] - 15.8) ** 2

    result = build_semf_posterior(
        likelihood, semf_prior_samples, jnp.array([1]), jnp.array([1]),
        jnp.array([2]), jnp.array([1.0]), n_chain=2, n_iter=12, seed=4,
    )
    assert result["samples"]["a_v"].shape == (2, 12)
    assert set(result["diagnostics"]["a_v"]) == {"r_hat", "ess", "mean", "std"}
    assert posterior_diagnostics(result["samples"])["a_v"]["ess"] > 0


def test_uncertainty_decomposition_and_batch_experiment_design():
    from apedp.epistemic import UncertaintyDecomposition
    from apedp.scientific_layers import design_experiments

    budget = UncertaintyDecomposition.from_components(
        measurement=1.0, parameter=2.0, structural=2.0, extrapolation=1.0
    )
    assert budget.aleatoric_total == 1.0
    assert np.isclose(budget.epistemic_total, 3.0)
    assert np.isclose(budget.combined_expanded_uncertainty, 2.0 * np.sqrt(10.0))

    design = design_experiments(
        [lambda x: x, lambda x: 2.0 * x, lambda x: 3.0 * x],
        np.arange(6.0),
        uncertainty_fn=lambda x: np.ones_like(x),
        feasible_fn=lambda x: x >= 1.0,
        n_points=2,
        min_distance=2.0,
    )
    assert len(design.selected_points) == 2
    assert all(point >= 1.0 for point in design.selected_points)
    assert abs(design.selected_points[0] - design.selected_points[1]) >= 2.0


def test_jax_nd_heat_and_wave_constant_fields():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from apedp.jax_pde_solver import (
        heat_nd_grad_D,
        solve_heat_nd,
        solve_wave_nd,
        validate_nd_grid,
    )

    with pytest.raises(ValueError, match="n-D heat"):
        validate_nd_grid(1.0, (0.1, 0.1), 0.1, 2)

    initial = jnp.ones((8, 7))
    heat = solve_heat_nd(initial, 0.1, (1.0 / 7.0, 1.0 / 6.0), 0.001, 5, bctype="neumann")
    assert np.allclose(np.asarray(heat), 1.0)

    wave = solve_wave_nd(initial, jnp.zeros_like(initial), 1.0,
                         (1.0 / 7.0, 1.0 / 6.0), 0.001, 5, bctype="neumann")
    assert np.allclose(np.asarray(wave), 1.0)

    varying = jnp.sin(jnp.pi * jnp.linspace(0.0, 1.0, 8))[:, None] * jnp.ones((1, 7))
    gradient = heat_nd_grad_D(varying, 0.1, (1.0 / 7.0, 1.0 / 6.0), 0.0001, 5)
    assert np.isfinite(gradient)


def test_posterior_predictive_calibration_reports_coverage():
    from apedp.calibration import posterior_predictive_calibration

    observed = np.array([0.0, 1.0, 2.0, 3.0])
    predictive = np.vstack([observed - 0.2, observed, observed + 0.2, observed + 0.1])
    report = posterior_predictive_calibration(predictive, observed, coverage=0.95)
    assert report["empirical_coverage"] == 1.0
    assert report["coverage_error"] > 0.0



def test_model_discovery_ranks_equations_and_designs_experiment():
    from apedp.model_discovery import ModelDiscoveryEngine

    x = np.linspace(1.0, 5.0, 20)
    y = 2.0 * x + 1.0
    train = {"x": x[:14]}
    holdout = {"x": x[14:]}
    engine = ModelDiscoveryEngine(["x"])
    result = engine.discover(
        {"linear": "x + 1", "quadratic": "x**2"},
        train,
        y[:14],
        x_holdout=holdout,
        y_holdout=y[14:],
        design_grid={"x": np.linspace(1.0, 5.0, 30)},
    )
    assert result.selected is not None
    assert result.selected.train_r_squared > 0.99
    assert result.selected.holdout_r_squared > 0.99
    assert result.experiment_design is not None


def test_model_discovery_exposes_joint_structure_posterior():
    from apedp.model_discovery import ModelDiscoveryEngine

    x = np.linspace(1.0, 5.0, 16)
    y = 2.0 * x + 1.0
    result = ModelDiscoveryEngine(["x"]).discover_bayesian(
        {"linear": "x + 1", "quadratic": "x**2"},
        {"x": x}, y, bootstrap_samples=3, seed=7,
    )
    assert result.structure_posterior is not None
    assert np.isclose(sum(result.structure_posterior.values()), 1.0)
    assert result.parameter_bootstrap is not None
    assert result.posterior_method is not None


def test_model_discovery_exact_finite_candidate_evidence():
    from apedp.model_discovery import ModelDiscoveryEngine

    x = np.linspace(1.0, 5.0, 16)
    y = 2.0 * x + 1.0
    result = ModelDiscoveryEngine(["x"]).discover_exact_bayesian(
        {"linear": "x + 1", "quadratic": "x**2"},
        {"x": x}, y, noise_std=0.1, prior_scale=10.0,
    )
    assert result.exact_log_evidence is not None
    assert result.parameter_posterior is not None
    assert np.isclose(sum(result.structure_posterior.values()), 1.0)
    assert result.posterior_method.startswith("exact finite-candidate")


def test_model_discovery_supports_multi_output_and_regimes():
    from apedp.model_discovery import ModelDiscoveryEngine

    x = np.linspace(0.0, 10.0, 24)
    engine = ModelDiscoveryEngine(["x"])
    outputs = engine.discover_multi_output(
        {"linear": "x + 1"}, {"x": x},
        {"first": 2.0 * x + 1.0, "second": 3.0 * x + 2.0},
    )
    assert set(outputs.outputs) == {"first", "second"}
    assert all(item.selected is not None for item in outputs.outputs.values())
    regimes = engine.discover_regimes(
        {"linear": "x + 1"}, {"x": x}, 2.0 * x + 1.0, "x", min_segment_size=8
    )
    assert "combined_bic" in regimes


def test_pipeline_job_manager_recovers_stale_running_job(tmp_path, monkeypatch):
    import json
    import time
    from datetime import datetime, timedelta, timezone

    from apedp.orchestration import PipelineJobManager

    job_id = "stale-job"
    old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    state_dir = tmp_path / "jobs"
    state_dir.mkdir()
    (state_dir / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "status": "RUNNING", "progress": "working",
        "parameters": {"csv_path": "data.csv", "target_column": "y"},
        "attempt": 1, "max_attempts": 1, "submitted_at": old,
        "updated_at": old, "heartbeat_at": old, "result": None,
        "error": None, "cancel_requested": False,
    }))

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            return {"status": "PERSISTED", "model_id": "recovered-model"}

    monkeypatch.setattr("apedp.orchestration.DiscoveryPipeline", FakePipeline)
    manager = PipelineJobManager(str(state_dir), max_workers=1)
    try:
        deadline = time.time() + 2.0
        current = manager.get(job_id)
        while current["status"] not in {"SUCCEEDED", "FAILED", "REFUSED", "CANCELLED"}:
            assert time.time() < deadline
            time.sleep(0.01)
            current = manager.get(job_id)
        assert current["status"] == "SUCCEEDED"
        assert current["attempt"] == 1
    finally:
        manager.shutdown()


def test_pipeline_job_manager_persists_and_completes(tmp_path, monkeypatch):
    import time

    from apedp.orchestration import PipelineJobManager

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            return {"status": "PERSISTED", "model_id": "test-model"}

    monkeypatch.setattr("apedp.orchestration.DiscoveryPipeline", FakePipeline)
    manager = PipelineJobManager(str(tmp_path / "jobs"), max_workers=1)
    try:
        submitted = manager.submit({"csv_path": "data.csv", "target_column": "y"})
        deadline = time.time() + 2.0
        current = manager.get(submitted["job_id"])
        while current["status"] not in {"SUCCEEDED", "FAILED", "REFUSED", "CANCELLED"}:
            assert time.time() < deadline
            time.sleep(0.01)
            current = manager.get(submitted["job_id"])
        assert current["status"] == "SUCCEEDED"
        assert (tmp_path / "jobs" / f"{submitted['job_id']}.json").exists()
    finally:
        manager.shutdown()
