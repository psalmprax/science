"""Regression tests for the noise-vs-rediscovery benchmark harness.

Scope: harness mechanics only (deterministic design generation, preregistered
verdict gates, report schema, world-table integrity). The full 18-world ×
5-sigma × 3-rep sweep is a separate artifact run, not a test job; tests pin
the smallest cells that still exercise every adapter path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_feynman_noise_rediscovery import (  # noqa: E402
    ALLOWED_VERDICTS,
    DEFAULT_SIGMAS,
    PREDICTIVE_R2_MIN,
    STRUCTURE_R2_MIN,
    _design_seed,
    _noise_seed,
    _feynman_worlds,
    adapter_synthesis,
    all_worlds,
    build_design,
    probe_design,
    r_squared,
    run_benchmark,
    score_trial,
)


def _run_minimal():
    return run_benchmark(
        engines=["synthesis"],
        sigmas=[0.05],
        reps=1,
        n_train=60,
        n_val=20,
        case_substr="quadratic_constant",
        output_path=None,
    )


# ---------------------------------------------------------------------------
# Scoring gates
# ---------------------------------------------------------------------------


def test_score_trial_exact_prediction_recovers():
    """Exact predictions of a high-variance law pass both gates."""
    x_val = np.linspace(0.0, 4.0, 30)
    truth_val = 2.0 * x_val**1.5
    s = score_trial(truth_val, truth_val, truth_val, truth_val)
    assert s["verdict"] == "recovered_structure"


def test_score_trial_flat_truth_is_never_a_pass():
    """ss_tot=0 makes R² undefined; an undefined gate is not a pass."""
    truth = np.full(30, 1.0)
    s = score_trial(truth, truth, truth, truth)
    assert s["verdict"] == "failed"


def test_score_trial_predictive_only_is_indistinguishable_alias():
    """Constant fit to a non-constant law passes neither gate."""
    truth = np.linspace(0.0, 5.0, 64)
    bad = np.full(64, np.mean(truth))
    s = score_trial(bad, bad, truth, truth)
    assert s["verdict"] == "failed"


def test_score_trial_structure_gate_rejects_train_range_alias():
    """Predictive in-range but wrong shape on the probe grid -> predictive_only."""
    rng = np.random.default_rng(7)
    # Truth is quadratic on both val and probe; alias is linear through 2 pts
    x_val = np.linspace(0.0, 4.0, 30)
    true_val = 3.0 * x_val**2 - 2.0 * x_val + 1.0
    # pretend the engine emitted the quadratic in-range but linear probe.
    noise = rng.normal(0, 1e-9, size=30)
    x_probe = np.linspace(-1.0, 5.0, 64)
    true_probe = 3.0 * x_probe**2 - 2.0 * x_probe + 1.0
    linear_probe = 7.0 * x_probe + 9.0
    s = score_trial(true_val + noise, linear_probe, true_val, true_probe)
    assert s["verdict"] == "recovered_predictive_only"


def test_score_trial_nonfinite_majority_fails():
    truth = np.ones(30)
    pred = np.full(30, np.nan)
    s = score_trial(pred, pred, truth, truth)
    assert s["verdict"] == "failed"


def test_r_squared_bounds_and_degenerate_total():
    y = np.array([1.0, 3.0, 5.0])
    assert r_squared(y, y) == 1.0
    assert r_squared(y, -y) < -1.0  # anti-correlated worse than the mean
    assert r_squared(np.ones(5), np.zeros(5)) == 0.0  # flat truth guards


# ---------------------------------------------------------------------------
# Deterministic design generation
# ---------------------------------------------------------------------------


def test_seed_separation_design_from_noise():
    """Design seed varies with rep only; noise seed also varies with sigma."""
    assert _design_seed("w", 0) == _design_seed("w", 0)
    assert _design_seed("w", 0) != _design_seed("w", 1)
    assert _design_seed("w", 0) == _design_seed("w", 0)  # stable under re-call
    assert _noise_seed("w", 0.01, 0) != _noise_seed("w", 0.02, 0)
    assert _noise_seed("w", 0.01, 0) != _noise_seed("w", 0.01, 1)
    assert _noise_seed("w", 0.01, 0) == _noise_seed("w", 0.01, 0)


def test_build_design_is_reproducible_and_noisy_only_in_target():
    world = all_worlds()[0]
    a = build_design(world, 0.02, 50, 15, rep=2)
    b = build_design(world, 0.02, 50, 15, rep=2)
    assert np.array_equal(a["x_train"], b["x_train"])
    assert np.array_equal(a["y_train"], b["y_train"])
    assert np.array_equal(a["y_val_clean"], b["y_val_clean"])
    clean = build_design(world, 0.0, 50, 15, rep=2)
    # Fixed-design convention: the same (world, rep) draws identical design
    # points at every sigma; only the target noise phase changes.
    assert np.array_equal(a["x_train"], clean["x_train"])
    assert np.array_equal(clean["y_train"], world.eval(clean["x_train"]))
    # Additive noise adds variance, not standard deviation:
    # var(noisy) = var(clean) + (sigma_rel * std(clean))**2
    var_clean = float(np.var(clean["y_train"]))
    sigma_abs = 0.02 * float(np.std(clean["y_train"]))
    assert abs(float(np.var(a["y_train"])) - (var_clean + sigma_abs**2)) < 0.1 * var_clean


def test_val_design_has_no_noise_draw():
    world = all_worlds()[0]
    d = build_design(world, 0.10, 80, 25, rep=0)
    clean = build_design(world, 0.0, 80, 25, rep=0)
    # Validation design points are seed-separated from train; the scored
    # quantity y_val_clean must be identical between noisy and clean runs
    # (noise only enters the training target).
    v_other_rep = build_design(world, 0.10, 80, 25, rep=1)
    assert not np.array_equal(d["x_val"], v_other_rep["x_val"])
    assert np.array_equal(d["y_val_clean"], world.eval(d["x_val"]))
    # The scored quantity is the noise-free truth, invariant to sigma config:
    v_noisy = build_design(world, 0.10, 80, 25, rep=0)
    assert np.array_equal(d["x_val"], v_noisy["x_val"])
    assert np.array_equal(d["y_val_clean"], v_noisy["y_val_clean"])


# ---------------------------------------------------------------------------
# World table integrity
# ---------------------------------------------------------------------------


def test_world_table_unique_ids_and_feynman_import():
    worlds = all_worlds()
    ids = [w.equation_id for w in worlds]
    assert len(ids) == len(set(ids))
    imported = {w.equation_id for w in _feynman_worlds()}
    assert len(imported) == 5
    assert all(any(equation_id == i for i in ids) for equation_id in imported)


def test_all_worlds_eval_finite_on_probe_grid():
    """Every evaluation lambda stays finite across its declared probe range."""
    for world in all_worlds():
        probe = probe_design(world, n_points=64)
        y = world.eval(probe)
        assert np.all(np.isfinite(y)), world.equation_id


def test_default_protocol_constants_are_preregistered():
    assert PREDICTIVE_R2_MIN == 0.95
    assert STRUCTURE_R2_MIN == 0.90
    assert DEFAULT_SIGMAS == (0.0, 0.01, 0.02, 0.05, 0.10)
    assert "skipped_not_applicable" in ALLOWED_VERDICTS


# ---------------------------------------------------------------------------
# End-to-end harness runs
# ---------------------------------------------------------------------------


def test_minimal_run_report_schema_and_determinism():
    report_a = _run_minimal()
    report_b = _run_minimal()
    def without_wall(report: dict) -> list[dict]:
        return [{k: v for k, v in t.items() if k != "wall_seconds"} for t in report["trials"]]

    # Same config replays to the same seeds, designs, predictions and
    # verdicts; wall-clock latency is instrumentation, not protocol.
    assert without_wall(report_a) == without_wall(report_b)
    assert [
        (t["world"], t["engine"], t["rep"], t["verdict"]) for t in report_a["trials"]
    ] == [(t["world"], t["engine"], t["rep"], t["verdict"]) for t in report_b["trials"]]
    assert report_a["harness_version"]
    assert report_a["protocol"]["abstention_counts_toward_recovery"] is False
    for trial in report_a["trials"]:
        assert trial["verdict"] in ALLOWED_VERDICTS
        assert trial["engine"] == "synthesis"
        assert trial["world"] == "quadratic_constant"
        assert trial["sigma_rel"] == 0.05
    summary = report_a["overall"]["synthesis"]
    total = sum(summary["counts"].values())
    assert total == len(report_a["trials"])
    assert abs(
        summary["structure_recovery_rate"]
        + summary["predictive_only_rate"]
        + summary["failed_rate"]
        + summary["abstain_rate"]
        - 1.0
    ) < 1e-9


def test_resume_state_skips_completed_cells_and_rejects_protocol_conflict(tmp_path):
    """Kill-resume semantics: same state replays zero cells; different config fails closed."""
    state = tmp_path / "state.json"
    first = run_benchmark(
        engines=["synthesis"], sigmas=[0.05, 0.1], reps=1, n_train=40, n_val=15,
        case_substr="quadratic_constant", output_path=None, state_path=state,
    )
    assert state.exists()
    resumed = run_benchmark(
        engines=["synthesis"], sigmas=[0.05, 0.1], reps=1, n_train=40, n_val=15,
        case_substr="quadratic_constant", output_path=None, state_path=state,
    )
    assert resumed["cells_executed_this_invocation"] == 0
    assert resumed["cells_scoped"] == 2
    def without_wall(report: dict) -> list[dict]:
        return [{k: v for k, v in t.items() if k != "wall_seconds"} for t in report["trials"]]
    assert without_wall(resumed) == without_wall(first)
    assert resumed["overall"]["synthesis"]["counts"].values()
    with pytest.raises(RuntimeError, match="refusing to mix protocols"):
        run_benchmark(
            engines=["synthesis"], sigmas=[0.05], reps=1, n_train=40, n_val=15,
            case_substr="quadratic_constant", output_path=None, state_path=state,
        )


def test_synthesis_adapter_multivariate_world_is_marked_not_applicable():
    from benchmarks.run_feynman_noise_rediscovery import adapter_synthesis as synth

    world = next(w for w in all_worlds() if w.equation_id == "product_two_var")
    design = build_design(world, 0.02, 40, 10, rep=0)
    out = synth(world, design, "target")
    assert out.abstained
    assert out.abstain_reason == "not_applicable_multivariate"


def _kernel_ready() -> bool:
    try:
        import apedp_kernel  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _kernel_ready(), reason="native apedp_kernel wheel absent")
def test_rime_ad_adapter_recovers_visible_monomial_world():
    """Blind RIME-AD on the easiest in-grammar world (sigma=0, monomial)."""
    from benchmarks.run_feynman_noise_rediscovery import adapter_rime_ad

    world = next(w for w in all_worlds() if w.equation_id == "mono_power_15")
    design = build_design(world, 0.0, 80, 20, rep=0)
    out = adapter_rime_ad(world, design, "target")
    assert out is not None
    if out.abstained:
        assert out.abstain_reason == "apedp_kernel_unavailable"
        pytest.skip("kernel present but proposer returned none")
    probe_true = world.eval(probe_design(world))
    s = score_trial(out.y_val_hat, out.y_probe_hat, design["y_val_clean"], probe_true)
    assert s["verdict"] in ("recovered_structure", "recovered_predictive_only", "failed")
    assert isinstance(out.detail.get("top_formula"), str)
