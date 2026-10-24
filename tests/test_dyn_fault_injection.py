"""Fault-injection pins: the referee must fail closed, never promote,
when numerical primitives are corrupted, gates fail, or artifacts alter.

Deterministic (no randomness evaluated here).
"""

from apedp_dyn.epistemic.adjudicate import adjudicate
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term


def _cand(cid, train_loss, holdout_rmse, passed=True, n_samples=57,
          obs_rms=0.5, noise_sigma=0.02, omit_gate=False, term="one"):
    prov = {} if omit_gate else {"data_origin": "synthetic", "gate4_status": {
        "passed": passed, "holdout_obs_rms": obs_rms,
        "holdout_noise_sigma": noise_sigma, "holdout_n_samples": n_samples,
        "rmse": holdout_rmse, "r2": 0.9, "rho1": 0.1, "holdout_residuals": [0.01] * 10}}
    return ScoredCandidate(
        CandidateDynamics(lhs="N", terms=(Term(term, "theta_0"),),
                          candidate_id=cid),
        provenance=prov, train_loss=train_loss, holdout_rmse=holdout_rmse)


def test_failed_gate_never_promotes():
    adj = adjudicate([_cand("A", 0.01, 0.02, passed=False)])
    assert adj.status.value == "insufficient_evidence"
    assert adj.promoted is None


def test_nan_losses_yield_no_valid_candidate():
    adj = adjudicate([_cand("A", float("nan"), 0.02),
                      _cand("B", float("nan"), 0.03)])
    assert adj.status.value == "no_valid_candidate"
    assert adj.promoted is None


def test_missing_gate_scale_abstains():
    adj = adjudicate([_cand("A", 0.01, 0.02, omit_gate=True)])
    assert adj.status.value == "insufficient_evidence"
    assert adj.promoted is None


def test_corrupt_holdout_is_invalid_evidence():
    # NaN holdout RMSE is metric-integrity failure, not ambiguity:
    # insufficient_evidence, never multi_hypothesis, never promoted.
    adj = adjudicate([_cand("A", 0.01, float("nan")),
                      _cand("B", 0.02, 0.05)])
    assert adj.status.value == "insufficient_evidence"
    assert adj.promoted is None


def test_indistinguishable_rivals_abstain():
    # MULTI_HYPOTHESIS explicitly prohibits unique promotion: a rival and
    # no promoted candidate must both hold.
    adj = adjudicate([_cand("A", 0.0100, 0.0200, term="x"),
                      _cand("B", 0.0101, 0.0201, term="v")])
    assert adj.status.value == "multi_hypothesis"
    assert adj.promoted is None
    assert adj.rival is not None


def test_duplicate_structures_canonicalized():
    # Identical structures are deduplicated before uniqueness is assessed:
    # canonical ranking holds one entry; duplicates can neither promote
    # nor serve as rivals.
    adj = adjudicate([_cand("A", 0.0100, 0.0200, term="x"),
                      _cand("A2", 0.0099, 0.0199, term="x")])
    assert len(adj.ranking) == 1
    assert adj.ranking[0].candidate.candidate_id == "A2"
    assert adj.status.value != "unique_support"


def _C(cid, terms, tl, ho):
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term
    return ScoredCandidate(
        CandidateDynamics(lhs="N",
                          terms=tuple(Term(n, p) for n, p in terms),
                          candidate_id=cid),
        provenance={"data_origin": "synthetic", "gate4_status": {
            "passed": True, "holdout_obs_rms": 0.5, "holdout_noise_sigma": 0.02,
            "holdout_n_samples": 57, "rmse": ho, "r2": 0.9, "rho1": 0.1,
            "holdout_residuals": [0.01] * 10}},
        train_loss=tl, holdout_rmse=ho)


def test_reordered_terms_are_duplicates():
    adj = adjudicate([_C("A", [("x", "t0"), ("v", "t1")], 0.01, 0.02),
                      _C("A2", [("v", "t1"), ("x", "t0")], 0.009, 0.019)])
    assert [c.candidate.candidate_id for c in adj.ranking] == ["A2"]
    assert adj.promoted is None or adj.promoted.candidate.candidate_id == "A2"


def test_parameter_sharing_stays_distinct():
    # Reciprocal-kc sharing vs independent k12/k21: same term names,
    # different bindings -> distinct hypotheses, both survive.
    adj = adjudicate([_C("S", [("x", "kc"), ("x2", "kc")], 0.01, 0.02),
                      _C("I", [("x", "k12"), ("x2", "k21")], 0.0101, 0.0201)])
    assert sorted(c.candidate.candidate_id for c in adj.ranking) == ["I", "S"]
    assert adj.status.value == "multi_hypothesis"
    assert adj.promoted is None


def test_determinism_repeated_runs_agree():
    fitted = [_C("A", [("x", "t0")], 0.01, 0.02),
              _C("A2", [("x", "t0")], 0.009, 0.019),
              _C("B", [("v", "t1")], 0.0102, 0.0202)]
    outs = [adjudicate(fitted) for _ in range(3)]
    assert all(o.status.value == outs[0].status.value for o in outs)
    assert all([c.candidate.candidate_id for c in o.ranking] ==
               [c.candidate.candidate_id for c in outs[0].ranking] for o in outs)


def test_dropped_duplicates_reconstructible():
    # Dropped IDs = input IDs minus canonical ranking IDs (auditable
    # on every path, including abstentions).
    fitted = [_C("A", [("x", "t0")], 0.01, 0.02),
              _C("A2", [("x", "t0")], 0.009, 0.019),
              _C("B", [("v", "t1")], 0.0102, 0.0202)]
    adj = adjudicate(fitted)
    dropped = {c.candidate.candidate_id for c in fitted} - \
              {c.candidate.candidate_id for c in adj.ranking}
    assert dropped == {"A"}


def test_interaction_dupes_corrupt_metric_rivals():
    # Duplicates + NaN holdout + two near-equivalent distinct rivals:
    # invalid evidence must not become promotion via canonicalization.
    adj = adjudicate([_C("A", [("x", "t0")], 0.01, 0.02),
                      _C("A2", [("x", "t0")], 0.009, 0.019),
                      _C("N", [("u", "t9")], 0.05, float("nan")),
                      _C("B", [("v", "t1")], 0.0102, 0.0202),
                      _C("D", [("v2", "t2")], 0.0103, 0.0203)])
    assert adj.promoted is None
    assert adj.status.value == "multi_hypothesis"
    assert "A" not in [c.candidate.candidate_id for c in adj.ranking]
