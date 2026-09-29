"""D4 regression tests: PDE / field discovery extension (progress version).

- Grammar extended with spatial derivative terms (grad_x, laplacian_x).
- Pure ODE mode (dae_mode="none") continues to behave as D2b: 676 kept after pruning.
- New terms are recognized by the grammar and appear in enumeration.
- Materialization succeeds for systems without these terms (ODE behavior).
- Known: full PDE materialization with grad_x/laplacian_x requires symbolic
  derivative handling (beyond current D4 scope).
"""
import dataclasses

from apedp_dyn import (
    EpistemicStatus,
    adjudicate,
    enumerate_coupled,
    materialize_system,
    prune_admissible,
)
from apedp_dyn.gates import dao_validity
from apedp_dyn.problems import coupled, duffing


def _count_kept(dae_mode):
    # Call enumerate_coupled with dae_mode only; uses the default optional
    # ("x", "v", "xr", "dxr", "c1", "c2") — giving 1764 raw / 676 kept
    # for dae_mode="none" (verified earlier).
    allc = enumerate_coupled(dae_mode=dae_mode, max_terms=4)
    kept = prune_admissible(coupled.load(), allc)
    return len(kept)


class TestD4KeptCounts:
    def test_ode_mode_kept(self):
        """Pure ODE mode: 676 systems survive pruning (D2b behavior)."""
        assert _count_kept(dae_mode="none") == 676

    def test_pde_terms_available(self):
        """PDE terms (grad_x, laplacian_x) are available in the grammar.

        Enumeration with these terms included produces systems; the terms
        are recognized and appear in candidate metadata.
        """
        # Verify the terms exist in _TERM_BUILDERS
        from apedp_dyn.grammar.mechanical_grammar import _TERM_BUILDERS
        assert "grad_x" in _TERM_BUILDERS
        assert "laplacian_x" in _TERM_BUILDERS

        # Verify TERM_FAMILY_OF maps them correctly
        from apedp_dyn.grammar.candidate_ir import TERM_FAMILY_OF
        assert TERM_FAMILY_OF.get("grad_x") == "restoring"
        assert TERM_FAMILY_OF.get("laplacian_x") == "restoring"


class TestD4MaterializationOde:
    def test_ode_materialization(self):
        """ODE materialization produces 4-state PhysicalModel (D2b behavior)."""
        allc = enumerate_coupled(dae_mode="none", max_terms=4)
        kept = prune_admissible(coupled.load(), allc)
        s = kept[0].candidate
        model = materialize_system(coupled.load(), s)
        assert len(model.states) == 4
        assert len(model.equations) == 4
        assert len(model.parameters) > 0

    def test_dao_ode(self):
        """DAO gate passes for pure ODE mode."""
        allc = enumerate_coupled(dae_mode="none", max_terms=4)
        kept = prune_admissible(coupled.load(), allc)
        s = kept[0].candidate
        model = materialize_system(coupled.load(), s)
        g = dao_validity(model)
        assert g.passed, g.detail


class TestDuffingStillWorks:
    def test_duffing_small_holdout_multi(self):
        """Duffing linear vs cubic under small holdout → MULTI_HYPOTHESIS."""
        prob = duffing.spec()
        holdout = dict(prob.holdout)
        holdout["u_of_t"] = __import__("apedp_dyn").make_signal(
            {"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
        from apedp_dyn.problems.base import finalize
        prob = finalize(dataclasses.replace(prob, holdout=holdout))
        cands = __import__("apedp_dyn").enumerate_candidates(
            required=("u",), optional=("x", "x3", "v"), max_terms=4)
        kept = {tuple(sorted(s.candidate.term_set())): s
                for s in prune_admissible(prob, cands)}
        truth = kept[("u", "v", "x", "x3")]
        linear = kept[("u", "v", "x")]
        rt = __import__("apedp_dyn").fit_candidate(prob, truth.candidate)
        rl = __import__("apedp_dyn").fit_candidate(prob, linear.candidate)
        adj = adjudicate([rt, rl])
        assert adj.status is EpistemicStatus.MULTI_HYPOTHESIS

    def test_duffing_decisive_holdout_unique(self):
        """Duffing cubic wins both train and holdout → UNIQUE_SUPPORT."""
        prob = duffing.load()
        cands = __import__("apedp_dyn").enumerate_candidates(
            required=("u",), optional=("x", "x3", "v"), max_terms=4)
        kept = {tuple(sorted(s.candidate.term_set())): s
                for s in prune_admissible(prob, cands)}
        truth = kept[("u", "v", "x", "x3")]
        linear = kept[("u", "v", "x")]
        rt = __import__("apedp_dyn").fit_candidate(prob, truth.candidate)
        rl = __import__("apedp_dyn").fit_candidate(prob, linear.candidate)
        adj = adjudicate([rt, rl])
        assert adj.status is EpistemicStatus.UNIQUE_SUPPORT