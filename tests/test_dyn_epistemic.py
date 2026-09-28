"""Epistemic adjudication tests: ranking is not promotion.

- Indistinguishable rivals (Duffing linear vs cubic under a small-amplitude
  intervention) → MULTI_HYPOTHESIS, nothing promoted.
- Decisive intervention (large amplitude) → UNIQUE_SUPPORT with margin record.
- Best fails gate 4 → INSUFFICIENT_EVIDENCE (no least-bad promotion).
- Nothing scored → NO_VALID_CANDIDATE.
"""
import dataclasses

import numpy as np
import pytest

from apedp_dyn import (
    REL_FLOOR,
    EpistemicStatus,
    adjudicate,
    enumerate_candidates,
    enumerate_coupled,
    fit_candidate,
    make_signal,
    prune_admissible,
    resolution_floor,
)
from apedp_dyn.problems import coupled, duffing


def _duffing_pair_for(prob):
    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    kept = {tuple(sorted(s.candidate.term_set())): s for s in prune_admissible(prob, cands)}
    return kept[("u", "v", "x", "x3")], kept[("u", "v", "x")]


def _duffing_pair():
    prob = duffing.load()
    truth_s, linear_s = _duffing_pair_for(prob)
    return prob, truth_s, linear_s


def _expected_tau(prob) -> float:
    """Resolution floor reproduced independently from the raw holdout record."""
    obs = np.asarray(prob.holdout["truth_y"])[:, 0]
    rms = float(np.sqrt(np.mean(obs ** 2)))
    sig = float(1.4826 * np.median(np.abs(np.diff(obs, n=3))) / np.sqrt(6.0))
    return max(REL_FLOOR * rms, sig / np.sqrt(len(obs)))


def _small_holdout_duffing():
    base = duffing.spec()
    holdout = dict(base.holdout)
    holdout["u_of_t"] = make_signal({"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
    from apedp_dyn.problems.base import finalize

    return finalize(dataclasses.replace(base, holdout=holdout))


def test_duffing_ambiguity_small_holdout_is_multi():
    """Linear model wins train but cubic wins holdout → MULTI_HYPOTHESIS."""
    prob = _small_holdout_duffing()
    _, truth_s, linear_s = _duffing_pair()
    rt = fit_candidate(prob, truth_s.candidate)
    rl = fit_candidate(prob, linear_s.candidate)
    adj = adjudicate([rt, rl])
    assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS, adj
    assert adj.promoted is None
    assert adj.best_rmse is not None and adj.second_rmse is not None
    assert adj.delta_rmse is not None and adj.tau_epi is not None
    assert adj.epistemic_margin is not None and adj.epistemic_margin <= 0


def test_duffing_decisive_holdout_is_unique():
    """Cubic model wins both train and holdout → UNIQUE_SUPPORT."""
    prob = duffing.load()
    _, truth_s, linear_s = _duffing_pair()
    rt = fit_candidate(prob, truth_s.candidate)
    rl = fit_candidate(prob, linear_s.candidate)
    adj = adjudicate([rt, rl])
    assert adj.status is EpistemicStatus.UNIQUE_SUPPORT, adj
    assert adj.promoted is not None
    assert adj.epistemic_margin is not None and adj.epistemic_margin > 0


def test_insufficient_evidence_when_best_fails_holdout():
    """Best candidate fails gate 4 → INSUFFICIENT_EVIDENCE, no promotion."""
    prob = coupled.load()
    kept = prune_admissible(prob, enumerate_coupled(
        required=("u",), optional=("x", "v", "xr", "dxr"), max_terms=4))
    # Uncoupled candidate (no xr/dxr terms) - guaranteed to survive pruning
    unc = [s for s in kept if [set(e) for e in s.candidate.term_sets()]
           == [{"u", "x", "v"}, {"u", "x", "v"}]][0]
    ru = fit_candidate(prob, unc.candidate)
    adj = adjudicate([ru])
    assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
    assert adj.promoted is None


def test_no_valid_candidate_on_empty():
    adj = adjudicate([])
    assert adj.status is EpistemicStatus.NO_VALID_CANDIDATE
    assert adj.promoted is None


def test_resolution_floor_is_scale_free_and_none_when_unavailable():
    """No scale -> None (an abstention), never a silent 0.0."""
    assert resolution_floor(None, None, None) is None
    assert resolution_floor(0.0, None, None) is None
    # Amplitude floor binds on a large, quiet record.
    assert resolution_floor(1.0, 0.001, 400) == pytest.approx(REL_FLOOR)
    # Noise floor binds when the record is small and noisy.
    assert resolution_floor(1.0, 1.0, 4) == pytest.approx(0.5)
    # Scale-free: the floor tracks the record, not the problem's units.
    assert resolution_floor(2.0, None, None) == pytest.approx(
        2.0 * resolution_floor(1.0, None, None))


def test_tau_epi_tracks_holdout_scale_not_gate4_tolerance():
    """Two records, one hardcoded acceptance tolerance, different resolution."""
    big = duffing.load()
    small = _small_holdout_duffing()
    # Identical gate-4 acceptance tolerance in both problems.
    assert big.holdout["tolerance"] == small.holdout["tolerance"]

    tau = {}
    for label, prob in (("big", big), ("small", small)):
        truth_s, linear_s = _duffing_pair_for(prob)
        adj = adjudicate([fit_candidate(prob, truth_s.candidate),
                          fit_candidate(prob, linear_s.candidate)])
        assert adj.tau_epi_source == "holdout-scale"
        assert adj.tau_epi == pytest.approx(_expected_tau(prob))
        tau[label] = adj.tau_epi

    # The record with the larger excited amplitude earns the coarser floor.
    assert tau["big"] > tau["small"]
    # Decoupled from the acceptance threshold it used to borrow.
    assert tau["big"] != big.holdout["tolerance"]


def test_explicit_tau_epi_overrides_the_derived_floor():
    prob = duffing.load()
    truth_s, linear_s = _duffing_pair_for(prob)
    rt = fit_candidate(prob, truth_s.candidate)
    rl = fit_candidate(prob, linear_s.candidate)
    derived = adjudicate([rt, rl]).tau_epi
    adj = adjudicate([rt, rl], tau_epi=derived + 1.0)
    assert adj.tau_epi_source == "explicit"
    assert adj.tau_epi == pytest.approx(derived + 1.0)
    assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS


def test_absent_holdout_scale_abstains_instead_of_promoting():
    """A missing scale must not read as "zero tolerance" and promote rank-1."""
    prob = duffing.load()
    truth_s, linear_s = _duffing_pair_for(prob)
    rt = fit_candidate(prob, truth_s.candidate)
    g4 = dict(rt.provenance["gate4_status"])
    assert g4["passed"]
    for key in ("holdout_obs_rms", "holdout_noise_sigma", "holdout_n_samples"):
        g4.pop(key, None)
    stripped = dataclasses.replace(
        rt, provenance={**rt.provenance, "gate4_status": g4})
    adj = adjudicate([stripped, fit_candidate(prob, linear_s.candidate)])
    assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
    assert adj.promoted is None
    assert "no scale" in adj.reason


def test_gate4_records_observation_scale():
    """Gate 4 must publish the scale adjudication needs, alongside acceptance."""
    prob = duffing.load()
    truth_s, _ = _duffing_pair_for(prob)
    rt = fit_candidate(prob, truth_s.candidate)
    g4 = rt.provenance["gate4_status"]
    for key in ("holdout_obs_rms", "holdout_noise_sigma", "holdout_n_samples"):
        assert key in g4, (key, sorted(g4))
    assert g4["holdout_n_samples"] == len(prob.holdout["truth_t"])