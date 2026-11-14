"""Pins for the 2026-11-15 engineering and statistics round.

  * A missing native build is not an abstention. ``RimeProposer`` returned ``[]``
    without the kernel -- the same answer it gives for a world with nothing in
    it -- so every test that pins "no mechanism found" passed on a machine where
    no search had run.
  * The suite reports a missing build as a skip, and as a failure where the
    builds are required (CI).
  * Tests do not write into the repository.
  * The promotion gates' false-pass rates and power are measured, and stage 4
    measures serial dependence instead of trusting a declared horizon.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.conf import canonical_spec
from apedp_dyn.epistemic.promotion_lifecycle import (
    effective_sample_size,
    information_coefficient,
    one_sided_ic_p_value,
)
from apedp_dyn.kernel_shim import NativeKernelUnavailableError, missing_kernel
from apedp_dyn.proposers import rime_proposer
from tests import conftest as suite_hooks

REPO = Path(__file__).resolve().parent.parent
CALIBRATION = REPO / "benchmarks" / "promotion_gate_calibration.json"
HARNESS = REPO / "benchmarks" / "run_promotion_gate_calibration.py"


# ---------------------------------------------------------------------------
# No kernel is an error, not an abstention
# ---------------------------------------------------------------------------


def _planted_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    return pd.DataFrame({"x": x, "z": rng.normal(size=300), "target": 3.0 * x})


def test_rime_without_the_kernel_raises_instead_of_abstaining(monkeypatch):
    """Before: an obvious planted law came back as "no mechanism found"."""
    monkeypatch.setattr(rime_proposer, "_RIME_AVAILABLE", False)
    proposer = rime_proposer.RimeProposer(feature_cols=["x", "z"], target_col="target")
    with pytest.raises(NativeKernelUnavailableError, match="not an abstention"):
        proposer.fit_and_propose(_planted_frame())


def test_the_shim_raises_the_same_named_error():
    with pytest.raises(NativeKernelUnavailableError):
        missing_kernel.rime_search_json  # noqa: B018 - attribute access is the call under test
    # Still a RuntimeError, so handlers written for the old error keep working.
    assert issubclass(NativeKernelUnavailableError, RuntimeError)
    assert not hasattr(missing_kernel, "__file__")


# ---------------------------------------------------------------------------
# The suite's handling of a missing build
# ---------------------------------------------------------------------------


def test_missing_build_is_recognised_through_a_wrapped_error():
    try:
        try:
            raise NativeKernelUnavailableError("apedp_kernel.x is unavailable")
        except RuntimeError as inner:
            raise ValueError("discovery failed") from inner
    except ValueError as wrapped:
        assert "native kernel not built" in (suite_hooks.missing_native_reason(wrapped) or "")
    missing_torch = ModuleNotFoundError("No module named 'torch'", name="torch.nn")
    assert "torch" in (suite_hooks.missing_native_reason(missing_torch) or "")


@pytest.mark.parametrize(
    "exc",
    [
        AssertionError("0 > 0"),
        RuntimeError("something else broke"),
        ModuleNotFoundError("No module named 'apedp_dyn.typo'", name="apedp_dyn.typo"),
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
    ],
)
def test_ordinary_failures_are_not_mistaken_for_a_missing_build(exc):
    """Only the kernel and torch are optional; a typo or a broken core import is a failure."""
    assert suite_hooks.missing_native_reason(exc) is None


def _run_probe(tmp_path: Path, strict: bool) -> str:
    probe = tmp_path / "test_probe_native.py"
    probe.write_text(
        "from apedp_dyn.kernel_shim import NativeKernelUnavailableError\n"
        "def test_needs_kernel():\n"
        "    raise NativeKernelUnavailableError('apedp_kernel.rime_search_json is unavailable')\n"
        "def test_real_bug():\n"
        "    assert 1 == 2\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        (REPO / "tests" / "conftest.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    env.pop("APEDP_REQUIRE_NATIVE", None)
    if strict:
        env["APEDP_REQUIRE_NATIVE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider",
         "-o", "addopts=", "--no-header", "--rootdir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, check=False, env=env, cwd=tmp_path,
    )
    return proc.stdout.strip().splitlines()[-1]


def test_a_missing_build_is_a_skip_on_a_bare_machine_and_a_failure_in_ci(tmp_path):
    relaxed = _run_probe(tmp_path, strict=False)
    assert "1 failed" in relaxed and "1 skipped" in relaxed  # the real bug still fails
    strict = _run_probe(tmp_path, strict=True)
    assert "2 failed" in strict and "skipped" not in strict


def test_ci_requires_the_native_builds():
    workflow = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert 'APEDP_REQUIRE_NATIVE: "1"' in workflow


# ---------------------------------------------------------------------------
# Tests do not write into the repository
# ---------------------------------------------------------------------------


def test_api_persistence_is_redirected_out_of_the_tracked_directory():
    configured = Path(os.environ["APEDP_PERSISTENCE_DIR"]).resolve()
    assert configured != (REPO / "apedp_persistence").resolve()
    assert REPO.resolve() not in configured.parents


def test_lorenz_harness_can_write_outside_the_registered_files(tmp_path):
    harness = (REPO / "benchmarks" / "run_lorenz_public_benchmark.py").read_text(encoding="utf-8")
    assert "--out-dir" in harness
    wiring_test = (REPO / "tests" / "test_dyn_lorenz_public_benchmark.py").read_text(encoding="utf-8")
    assert '"--out-dir"' in wiring_test


# ---------------------------------------------------------------------------
# Serial dependence in stage 4
# ---------------------------------------------------------------------------


def _overlapping(rng: np.random.Generator, n: int = 400, horizon: int = 21) -> np.ndarray:
    return np.convolve(rng.normal(size=n + horizon), np.ones(horizon), mode="valid")[:n]


def test_independent_series_keep_almost_all_their_observations():
    rng = np.random.default_rng(1)
    sizes = [effective_sample_size(rng.normal(size=400), rng.normal(size=400)) for _ in range(100)]
    assert min(sizes) > 340 and max(sizes) <= 400


def test_overlapping_series_are_recognised_without_being_declared():
    rng = np.random.default_rng(2)
    sizes = [effective_sample_size(_overlapping(rng), _overlapping(rng)) for _ in range(100)]
    assert float(np.median(sizes)) < 80  # 400 bars of 21-bar overlap carry a few dozen


def test_measured_dependence_restores_the_nominal_error_rate():
    """Unrelated overlapping series: about a third "significant" at n, about 5% at n_eff."""
    rng = np.random.default_rng(3)
    naive = corrected = 0
    trials = 600
    for _ in range(trials):
        signal, targets = _overlapping(rng), _overlapping(rng)
        ic, n = information_coefficient(signal, targets)
        naive += one_sided_ic_p_value(ic, n) < 0.05
        corrected += one_sided_ic_p_value(ic, effective_sample_size(signal, targets)) < 0.05
    assert naive / trials > 0.25
    assert corrected / trials < 0.10


def test_one_slow_series_alone_costs_nothing():
    """Dependence in only one of the two series does not inflate the correlation's variance."""
    rng = np.random.default_rng(4)
    sizes = [effective_sample_size(_overlapping(rng), rng.normal(size=400)) for _ in range(100)]
    assert float(np.median(sizes)) > 330


# ---------------------------------------------------------------------------
# Gate calibration
# ---------------------------------------------------------------------------


def test_calibration_harness_runs_end_to_end(tmp_path):
    out = tmp_path / "calibration.json"
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--smoke", "--json", str(out)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(out.read_text())
    assert payload["mode"] == "smoke"
    assert {row["n"] for row in payload["stage_2"]} == {250, 600, 2000}
    assert {row["world"] for row in payload["pipeline_stages_2_to_5"]} == {"null", "planted", "planted_strong"}
    assert len(payload["stage_5"]) == 12


@pytest.fixture(scope="module")
def calibration() -> dict:
    payload = json.loads(CALIBRATION.read_text())
    assert payload["mode"] == "full"
    return payload


def test_recorded_calibration_is_for_the_current_specification(calibration):
    """The measured rates describe these thresholds; a new spec needs a new run."""
    assert calibration["canonical_spec_sha256"] == canonical_spec.PINNED_SPEC_SHA256
    assert calibration["thresholds"]["stage2_min_ic"] == canonical_spec.STAGE2_MIN_IC
    assert calibration["thresholds"]["dsr_threshold"] == canonical_spec.DSR_THRESHOLD
    assert calibration["simulations_per_cell"] >= 1000


def test_the_bare_stage_2_threshold_is_not_a_test_and_the_fold_rule_is(calibration):
    """IC >= 0.02 alone passes a large share of pure noise at every sample size."""
    for row in calibration["stage_2"]:
        if row["world"] != "null":
            continue
        bare = row["bare_threshold_only"]
        gate = row["gate_threshold_and_five_positive_folds"]
        assert bare["wilson95"][0] > 0.12, row
        assert gate["wilson95"][1] < 0.06, row


def test_stage_2_power_is_recorded_not_assumed(calibration):
    by_cell = {(row["n"], row["true_ic"]): row["gate_threshold_and_five_positive_folds"]["rate"]
               for row in calibration["stage_2"]}
    # More data and more signal both help, and a weak signal is usually missed.
    assert by_cell[(2000, 0.10)] > by_cell[(600, 0.10)] > by_cell[(250, 0.10)]
    assert by_cell[(2000, 0.10)] > 0.75
    assert by_cell[(600, 0.05)] < 0.35


def test_stage_4_holds_its_level_on_overlapping_targets_without_a_declaration(calibration):
    stage4 = calibration["stage_4"]
    assert stage4["null_candidate_independent_targets"]["wilson95"][1] < 0.10
    assert stage4["null_candidate_overlapping_targets_horizon_not_declared"]["wilson95"][1] < 0.12
    assert stage4["null_candidate_overlapping_targets_horizon_declared"]["wilson95"][1] < 0.08


def test_stage_5_does_not_pass_the_best_of_a_zero_edge_search(calibration):
    for row in calibration["stage_5"]:
        cell = row["best_of_search_passes_stage_5"]
        if row["world"] == "null":
            assert cell["wilson95"][1] < 0.08, row
        if row["true_annual_sharpe"] >= 3.0:
            assert cell["rate"] > 0.95, row


def test_nothing_from_a_null_world_reaches_stage_5(calibration):
    rows = {row["world"]: row for row in calibration["pipeline_stages_2_to_5"]}
    assert rows["null"]["reached_stage_5"]["passed"] == 0
    assert rows["null"]["reached_stage_3"]["wilson95"][1] < 0.02
    assert rows["planted_strong"]["reached_stage_5"]["rate"] > 0.85
