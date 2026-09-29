"""D2b tests: nonlinear restoring (Duffing) + coupling topology (2-DOF)."""
import inspect
from itertools import combinations

from apedp_dyn import (
    Term,
    canonical_system,
    enumerate_candidates,
    enumerate_coupled,
    fit_candidate,
    prune_admissible,
    select_by_train_loss,
)
from apedp_dyn.problems import coupled, duffing


def _pmap(scored):
    """Term-name -> fitted value, for single-DOF candidates ({} otherwise)."""
    from apedp_dyn.grammar import CandidateDynamics

    cand = scored.candidate
    if not isinstance(cand, CandidateDynamics):
        return {}
    return {t.name: scored.fit.param_values[t.parameter] for t in cand.terms}


def test_duffing_chain_truth_wins():
    prob = duffing.load()
    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    kept = prune_admissible(prob, cands)
    assert len(cands) == 8 and len(kept) == 8
    fitted = [fit_candidate(prob, s.candidate) for s in kept]
    by_terms = {tuple(sorted(s.candidate.term_set())): s for s in fitted}
    truth, linear = by_terms[("u", "v", "x", "x3")], by_terms[("u", "v", "x")]
    assert truth.train_loss <= linear.train_loss  # nested: no holdout peeking
    assert truth.provenance["gate4_status"]["passed"]
    assert not linear.provenance["gate4_status"]["passed"]
    assert truth.holdout_rmse < 0.01 * linear.holdout_rmse
    assert select_by_train_loss(fitted).candidate is truth.candidate
    p = _pmap(truth)
    assert abs(p["u"] - 1.0) < 1e-2, p
    assert abs(p["x"] - (-4.0)) < 1e-2, p
    assert abs(p["x3"] - (-2.0)) < 5e-2, p
    assert abs(p["v"] - (-0.2)) < 1e-2, p


def test_duffing_linear_trap_is_real():
    # Small-amplitude train: the linear rival fits to 2.8e-05 — the trap is
    # genuine. Only the large-amplitude holdout exposes the missing cubic.
    prob = duffing.load()
    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    lin = next(s for s in prune_admissible(prob, cands)
               if set(s.candidate.term_set()) == {"u", "v", "x"})
    res = fit_candidate(prob, lin.candidate)
    assert res.train_loss < 1e-3, res.train_loss
    assert res.holdout_rmse > 0.1, res.holdout_rmse
    assert not res.provenance["gate4_status"]["passed"]


def test_xr_is_coupling_family_and_rejected_single_dof():
    assert Term("xr", "q0").family == "coupling"
    from apedp_dyn.problems import mass_spring_damper as msd

    prob = msd.load()
    cands = enumerate_candidates(required=("xr",), optional=("x",), max_terms=2)
    kept = prune_admissible(prob, cands)
    # Remote position is meaningless with one mass: nothing survives to fitting.
    assert len(cands) == 2 and len(kept) == 0


def test_coupled_enumeration_and_prune():
    prob = coupled.load()
    allc = enumerate_coupled()
    # Derive the raw count from the DECLARED optional terms rather than
    # hardcoding it. A literal rots whenever the grammar grows: this read 225
    # until c1/c2 were added for D3, when it became 1764 with no real change in
    # behaviour. Per-DOF subsets of size <= (max_terms - |required|), squared
    # over the two DOFs.
    sig = inspect.signature(enumerate_coupled)
    optional = sig.parameters["optional"].default
    required = sig.parameters["required"].default
    max_terms = sig.parameters["max_terms"].default
    per_dof = sum(len(list(combinations(optional, k)))
                  for k in range(max_terms - len(required) + 1))
    assert len(allc) == per_dof ** 2

    kept = prune_admissible(prob, allc)
    # Measured 2026-09-28 against the D3-widened grammar (optional gained
    # c1/c2): 676 of the 1764 survive dimensional + admissibility pruning.
    assert len(kept) == 676
    keys = [canonical_system(s.candidate) for s in kept]
    assert len(set(keys)) == len(keys)  # pruning keeps distinct hypotheses only
    # DOF-labeled canonicals: a swapped pair are distinct hypotheses, both kept.
    assert canonical_system(kept[0].candidate) != canonical_system(kept[-1].candidate)
    truth = [s for s in kept if [set(e) for e in s.candidate.term_sets()]
             == [{"u", "x", "v", "xr"}, {"u", "x", "v", "xr"}]]
    assert len(truth) == 1


def test_coupled_chain_topology_wins():
    prob = coupled.load()
    kept = prune_admissible(prob, enumerate_coupled(
        required=("u",), optional=("x", "v", "xr"), max_terms=4))
    def get(sets):
        return next(s for s in kept if [set(e) for e in s.candidate.term_sets()] == sets)

    truth = get([{"u", "x", "v", "xr"}, {"u", "x", "v", "xr"}])
    unc = get([{"u", "x", "v"}, {"u", "x", "v"}])
    rt = fit_candidate(prob, truth.candidate)
    ru = fit_candidate(prob, unc.candidate)
    assert rt.train_loss <= ru.train_loss
    assert rt.provenance["gate4_status"]["passed"], rt.provenance
    assert not ru.provenance["gate4_status"]["passed"], ru.provenance
    assert rt.holdout_rmse < 0.01 * ru.holdout_rmse
    assert select_by_train_loss([rt, ru]).candidate is truth.candidate
    # Coupling coefficients recover +kc/m on both masses; local stiffness
    # absorbs the coupling shift (-(k+kc)): the shifted, not bare, values.
    m0, m1 = truth.candidate.equations
    p0 = {t.name: rt.fit.param_values[t.parameter] for t in m0.terms}
    p1 = {t.name: rt.fit.param_values[t.parameter] for t in m1.terms}
    assert abs(p0["xr"] - 1.5) < 0.1, p0
    assert abs(p1["xr"] - 1.5) < 0.1, p1
    assert abs(p0["x"] - (-5.5)) < 0.1, p0
    assert abs(p1["x"] - (-7.5)) < 0.1, p1
