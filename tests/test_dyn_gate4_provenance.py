"""Gate-4 provenance producer + CandidateProposal bridge (core-side, no sandbox).

Pins the contract that lets ``adjudicate`` run from ordinary proposer output:
error metrics are measured (never invented), the gate stays closed when
incrementality was not tested, and the emitted provenance is what the referee
actually reads.
"""

import numpy as np
import pandas as pd
import pytest

from apedp_dyn.conf.canonical_spec import DIEBOLD_MARIANO_P_VALUE_THRESHOLD
from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.epistemic.gate4_provenance import (
    ProposalNotEvaluableError,
    gate4_status,
    measure_proposal,
    noise_sigma_from_residuals,
    scored_candidate,
)
from apedp_dyn.grammar.proposer_interface import CandidateProposal, StrategyArchetype


def _proposal(name: str, fn) -> CandidateProposal:
    return CandidateProposal(
        proposer_id="test_proposer",
        candidate_id=name,
        formula=f"{name}(x)",
        strategy_archetype=StrategyArchetype.PHYSICAL_INVARIANT,
        declared_scale_invariant=True,
        declared_nominal_dependencies=("x",),
        compute_score=fn,
    )


def _data(n: int = 40, offset: float = 0.0):
    x = np.linspace(-2.0, 2.0, n)
    y = 2.0 * x + 3.0 + offset * np.sin(2.0 * x)
    frame = pd.DataFrame({"x": x})
    return frame, y


# ---------------------------------------------------------------- error metrics

def test_measure_proposal_reports_real_metrics():
    frame, y = _data()
    exact = _proposal("exact", lambda f: 2.0 * f["x"].to_numpy(float) + 3.0)
    m = measure_proposal(exact, frame, y)
    assert m.n == len(y)
    assert m.mse == pytest.approx(0.0, abs=1e-18)
    assert m.rmse == pytest.approx(0.0, abs=1e-12)
    assert m.r2 == pytest.approx(1.0)
    assert m.residuals.shape == y.shape

    biased = _proposal("biased", lambda f: 2.0 * f["x"].to_numpy(float) + 2.0)
    mb = measure_proposal(biased, frame, y)
    assert mb.rmse == pytest.approx(1.0)
    assert mb.mae == pytest.approx(1.0)
    assert mb.r2 < 0.99


def test_measure_proposal_fails_closed():
    frame, y = _data(10)
    short = _proposal("short", lambda f: np.zeros(3))
    with pytest.raises(ProposalNotEvaluableError):
        measure_proposal(short, frame, y)

    all_nan = _proposal("nan", lambda f: np.full(len(f), np.nan))
    with pytest.raises(ProposalNotEvaluableError):
        measure_proposal(all_nan, frame, y)

    raising = _proposal("boom", lambda f: (_ for _ in ()).throw(RuntimeError("kernel down")))
    with pytest.raises(ProposalNotEvaluableError):
        measure_proposal(raising, frame, y)


def test_noise_sigma_estimator_is_not_applied_implicitly():
    frame, y = _data()
    cand = _proposal("cand", lambda f: 2.0 * f["x"].to_numpy(float) + 2.9)
    status = gate4_status(
        cand, train_frame=frame, y_train=y, holdout_frame=frame, y_holdout=y)
    assert status["holdout_noise_sigma"] is None
    assert noise_sigma_from_residuals(np.array([0.1, -0.1, 0.1, -0.1])) > 0.0


# ---------------------------------------------------------------- gate 4 status

def test_gate_is_closed_when_incrementality_was_not_tested():
    frame, y = _data()
    cand = _proposal("cand", lambda f: 2.0 * f["x"].to_numpy(float) + 3.0)
    status = gate4_status(
        cand, train_frame=frame, y_train=y, holdout_frame=frame, y_holdout=y,
        noise_sigma=1e-3)
    assert status["passed"] is False
    assert "not tested" in status["detail"]
    # the evidence the referee needs is still recorded
    assert status["holdout_n_samples"] == len(y)
    assert len(status["holdout_residuals"]) == len(y)
    assert status["holdout_obs_rms"] > 0.0


def test_gate_passes_only_against_a_measurably_worse_incumbent():
    frame, y = _data()
    cand = _proposal("cand", lambda f: 2.0 * f["x"].to_numpy(float) + 3.0)
    weak_incumbent = np.full_like(y, np.mean(y))          # constant mean predictor

    passed = gate4_status(
        cand, train_frame=frame, y_train=y, holdout_frame=frame, y_holdout=y,
        incumbent_pred=weak_incumbent, incumbent_id="sample_mean", noise_sigma=1e-3)
    assert passed["passed"] is True
    assert passed["dm_statistic"] > 0
    assert passed["dm_p_value"] < DIEBOLD_MARIANO_P_VALUE_THRESHOLD

    # Against the TRUE law (same structure), the candidate cannot be incremental.
    equal = gate4_status(
        cand, train_frame=frame, y_train=y, holdout_frame=frame, y_holdout=y,
        incumbent_pred=np.asarray(y, float), incumbent_id="true_law")
    assert equal["passed"] is False
    assert "INCREMENTALITY_UNPROVEN" in equal["detail"]


# ------------------------------------------------------------------ the bridge

def test_scored_candidate_feeds_adjudicate_without_sandbox_harness():
    frame, y = _data()
    cand = _proposal("cand", lambda f: 2.0 * f["x"].to_numpy(float) + 3.05)
    incumbent = np.full_like(y, np.mean(y))

    scored = scored_candidate(
        cand, train_frame=frame, y_train=y, holdout_frame=frame, y_holdout=y,
        incumbent_pred=incumbent, incumbent_id="sample_mean", noise_sigma=1e-3)
    assert scored.train_loss == pytest.approx(
        scored.provenance["gate4_status"]["proposal_metrics"]["train"]["mse"])
    assert scored.holdout_rmse == pytest.approx(
        scored.provenance["gate4_status"]["rmse"])

    verdict = adjudicate([scored])
    # The referee consumed the provenance we produced (not a hand-built dict):
    # without it the resolution floor would be unassessable and it would abstain.
    assert verdict.holdout_scale_source == "gate4_provenance"
    assert verdict.best_rmse == pytest.approx(scored.holdout_rmse)
    assert "no scale" not in verdict.reason
