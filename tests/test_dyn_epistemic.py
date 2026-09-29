"""Epistemic adjudication tests: ranking is not promotion.

- Indistinguishable rivals (Duffing linear vs cubic under a small-amplitude
  intervention) → MULTI_HYPOTHESIS, nothing promoted.
- Decisive intervention (large amplitude) → UNIQUE_SUPPORT with margin record.
- Best fails gate 4 → INSUFFICIENT_EVIDENCE (no least-bad promotion).
- Nothing scored → NO_VALID_CANDIDATE.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn import (
    REL_FLOOR,
    EpistemicStatus,
    ScoredCandidate,
    adjudicate,
    enumerate_candidates,
    enumerate_coupled,
    fit_candidate,
    make_signal,
    prune_admissible,
    resolution_floor,
    scale_from_observations,
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
    """Resolution floor reproduced independently from the raw holdout record.

    All channels, matching gate 4's all-channel RMSE: the floor and the
    statistic it resolves must be measured on the same record.
    """
    obs = np.asarray(prob.holdout["truth_y"], dtype=float)
    rms = float(np.sqrt(np.mean(obs ** 2)))
    per_channel = (1.4826 * np.median(np.abs(np.diff(obs, n=3, axis=0)), axis=0)
                   / np.sqrt(6.0))
    sig = float(np.sqrt(np.mean(per_channel ** 2)))
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
    unc = next(s for s in kept if [set(e) for e in s.candidate.term_sets()]
               == [{"u", "x", "v"}, {"u", "x", "v"}])
    ru = fit_candidate(prob, unc.candidate)
    adj = adjudicate([ru])
    assert adj.status is EpistemicStatus.INSUFFICIENT_EVIDENCE, adj
    assert adj.promoted is None


def test_no_valid_candidate_on_empty():
    adj = adjudicate([])
    assert adj.status is EpistemicStatus.NO_VALID_CANDIDATE
    assert adj.promoted is None


_D2A_REPORT = Path(__file__).resolve().parent.parent / "outputs" / "d2a_ab_report.json"


@pytest.mark.skipif(not _D2A_REPORT.exists(),
                    reason="D2a A/B artifact not recorded in this checkout")
def test_rel_floor_clears_the_d2a_estimator_spread():
    """The floor must exceed the RMSE gap between two fits of the SAME structure.

    Otherwise the adjudicator reads estimator variability as a resolved
    structural difference. This is the recorded lower edge of the window
    documented on REL_FLOOR, re-derived from the artifact.
    """
    from apedp_dyn.problems import mass_spring_damper as msd
    from apedp_dyn.problems import pendulum

    report = json.loads(_D2A_REPORT.read_text())
    obs_rms = {}
    for pid, mod in (("D1-MSD", msd), ("D1-PEND", pendulum)):
        y = np.asarray(mod.load().holdout["truth_y"], dtype=float)
        obs_rms[pid] = float(np.sqrt(np.mean(y ** 2)))

    gaps = []
    for pid, info in report["problems"].items():
        for arms in info["levels"].values():
            refined = [arms[a].get("refined") for a in arms]
            if not all(isinstance(v, dict) for v in refined):
                continue  # this arm did not refine at this noise level
            gaps.append(abs(refined[0]["holdout_rmse"] - refined[1]["holdout_rmse"])
                        / obs_rms[pid])
    assert gaps, "no refined arms recorded in the D2a report"
    assert REL_FLOOR >= max(gaps), (REL_FLOOR, max(gaps))


def test_rel_floor_sits_inside_the_structural_window():
    """Re-derive both edges from the live problems: one floor, two verdicts.

    The small-amplitude pair must stay unresolved and the decisive pair must be
    resolved, so REL_FLOOR is bracketed by their relative RMSE gaps. Pinning
    the window (not just the two verdicts) is what stops the constant from
    being moved out of calibration unnoticed.
    """
    ratios = {}
    for label, prob in (("decisive", duffing.load()),
                        ("small", _small_holdout_duffing())):
        truth_s, linear_s = _duffing_pair_for(prob)
        rt = fit_candidate(prob, truth_s.candidate)
        rl = fit_candidate(prob, linear_s.candidate)
        y = np.asarray(prob.holdout["truth_y"], dtype=float)
        obs_rms = float(np.sqrt(np.mean(y ** 2)))
        ratios[label] = (rl.holdout_rmse - rt.holdout_rmse) / obs_rms

    # Rivals closer than REL_FLOOR of the record stay unresolved; one far
    # beyond it must be resolved.
    assert ratios["small"] < REL_FLOOR < ratios["decisive"], ratios
    # Not merely inside the window: clear of the binding edge by a factor.
    assert REL_FLOOR >= 1.5 * ratios["small"], (REL_FLOOR, ratios["small"])
    assert ratios["decisive"] >= 1.5 * REL_FLOOR, (ratios["decisive"], REL_FLOOR)


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
    """One acceptance RULE, two derived thresholds; resolution is not that bar."""
    big = duffing.load()
    small = _small_holdout_duffing()
    # Both problems use the noise-floor rule, but no longer share a constant:
    # the number is derived per problem rather than hand-set once.
    assert big.holdout["tolerance"] == "auto"
    assert small.holdout["tolerance"] == "auto"

    tau: dict[str, float] = {}
    tol: dict[str, float] = {}
    for label, prob in (("big", big), ("small", small)):
        truth_s, linear_s = _duffing_pair_for(prob)
        rt = fit_candidate(prob, truth_s.candidate)
        adj = adjudicate([rt, fit_candidate(prob, linear_s.candidate)])
        assert adj.tau_epi_source == "holdout-scale"
        assert adj.tau_epi == pytest.approx(_expected_tau(prob))
        tau[label] = adj.tau_epi
        tol[label] = rt.provenance["gate4_status"]["holdout_tolerance"]

    # The record with the larger excited amplitude earns the coarser floor.
    assert tau["big"] > tau["small"]
    # The acceptance thresholds are derived too, and therefore differ.
    assert tol["big"] > tol["small"]
    # Resolution stays decoupled from the acceptance threshold it used to
    # borrow: sharing one constant made abstention track gate-4 leniency.
    assert tau["big"] != tol["big"]


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


def test_holdout_scale_can_be_supplied_without_gate4_provenance():
    """The resolution floor's AVAILABILITY must not depend on the scoring path.

    Mining the scale out of gate4_status coupled it to gate 4 having run, so a
    candidate scored any other way could never earn UNIQUE_SUPPORT however
    decisive its evidence. Supplying the problem's own holdout record unblocks
    it, and which route was used is recorded.
    """
    prob = _small_holdout_duffing()
    truth_s, linear_s = _duffing_pair_for(prob)
    rt = fit_candidate(prob, truth_s.candidate)
    rl = fit_candidate(prob, linear_s.candidate)

    g4 = dict(rt.provenance["gate4_status"])
    assert g4["passed"]
    for key in ("holdout_obs_rms", "holdout_noise_sigma", "holdout_n_samples"):
        g4.pop(key, None)
    stripped = dataclasses.replace(
        rt, provenance={**rt.provenance, "gate4_status": g4})

    # With nothing to go on, the stripped record abstains ...
    assert adjudicate([stripped, rl]).status is EpistemicStatus.INSUFFICIENT_EVIDENCE
    # ... and the same candidate adjudicates once the scale is handed in.
    adj = adjudicate(
        [stripped, rl],
        holdout_scale=scale_from_observations(
            np.asarray(prob.holdout["truth_y"], dtype=float)))
    assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS, adj
    assert adj.holdout_scale_source == "argument"
    assert adj.tau_epi == pytest.approx(_expected_tau(prob))
    # The provenance route still works, and says so.
    assert adjudicate([rt, rl]).holdout_scale_source == "gate4_provenance"


def test_identity_distinguishes_parameter_sharing_not_just_term_sets():
    """Reciprocal kc vs independent k12/k21 are distinct hypotheses.

    Both have the SAME term sets, so the canonical form cannot separate them;
    the parameter-binding signature is what must.
    """
    from apedp_dyn.epistemic.adjudicate import _identity
    from apedp_dyn.grammar import Term, canonical_system

    prob = coupled.load()
    wanted = [{"u", "x", "v", "xr"}, {"u", "x", "v", "xr"}]
    base = next(s for s in prune_admissible(prob, enumerate_coupled(
        required=("u",), optional=("x", "v", "xr"), max_terms=4))
        if [set(e) for e in s.candidate.term_sets()] == wanted)

    # Rewrite the second DOF's coupling term to share the first's parameter.
    e0, e1 = base.candidate.equations
    remote0 = next(t for t in e0.terms if t.name == "xr")
    e1_shared = dataclasses.replace(
        e1,
        terms=tuple(Term(t.name,
                         remote0.parameter if t.name == "xr" else t.parameter)
                    for t in e1.terms),
        parameter_bounds={**e0.parameter_bounds, **e1.parameter_bounds})
    reciprocal = dataclasses.replace(base.candidate, equations=(e0, e1_shared))

    assert base.candidate.term_sets() == reciprocal.term_sets()
    assert canonical_system(base.candidate) == canonical_system(reciprocal)
    assert _identity(base) != _identity(ScoredCandidate(candidate=reciprocal))


def test_identity_dispatches_on_ir_type_and_rejects_unknown():
    from apedp_dyn.epistemic.adjudicate import _identity
    from apedp_dyn.grammar import CandidateDynamics, SystemCandidate

    single = _duffing_pair_for(duffing.load())[0]
    assert isinstance(single.candidate, CandidateDynamics)
    assert _identity(single)  # resolves without attribute probing

    cprob = coupled.load()
    system = prune_admissible(cprob, enumerate_coupled(
        required=("u",), optional=("x", "v"), max_terms=3))[0]
    assert isinstance(system.candidate, SystemCandidate)
    assert _identity(system)

    unknown = ScoredCandidate(candidate=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unknown candidate IR type"):
        _identity(unknown)


def test_gate4_records_observation_scale():
    """Gate 4 must publish the scale adjudication needs, alongside acceptance."""
    prob = duffing.load()
    truth_s, _ = _duffing_pair_for(prob)
    rt = fit_candidate(prob, truth_s.candidate)
    g4 = rt.provenance["gate4_status"]
    for key in ("holdout_obs_rms", "holdout_noise_sigma", "holdout_n_samples"):
        assert key in g4, (key, sorted(g4))
    assert g4["holdout_n_samples"] == len(prob.holdout["truth_t"])
    # All-channel, so the scale is commensurate with the all-channel RMSE.
    assert g4["holdout_n_channels"] == np.asarray(prob.holdout["truth_y"]).shape[1]


def test_gate4_records_margin_not_only_verdict():
    """A bare `passed` saturates; the margin says how much headroom was left."""
    prob = duffing.load()
    _, linear_s = _duffing_pair_for(prob)
    bad = fit_candidate(prob, linear_s.candidate)
    g4 = bad.provenance["gate4_status"]
    assert not g4["passed"]
    # Margin ratio is rmse/tol: >1 is a failure, and by how much is recorded.
    assert g4["holdout_margin_ratio"] == pytest.approx(
        g4["holdout_rmse"] / g4["holdout_tolerance"])
    assert g4["holdout_margin_ratio"] > 1.0
    assert g4["holdout_margin"] < 0.0