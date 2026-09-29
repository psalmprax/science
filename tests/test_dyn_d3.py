"""D3 regression tests: DAE and multi-timescale extensions.

- Constrained DAE mode (c1, c2 terms) → REFUSED as unimplemented. The c1/c2
  basis is the constant 1 and no index-1 reduction is performed, so every
  candidate that carried a constraint was rejected at gate 1 and the ones that
  survived carried none. The gate's contract for a constrained model is stated
  directly in test_dao_constrained_gate_specification, which is the acceptance
  test for implementing it.
- Multi-timescale mode (fast/slow τ_fast/τ_slow) → survives pruning, materializes
  with tau parameters; fast/slow variable labeling consistent with metadata.
  Note: multi-timescale kept count depends on gate configuration; current pruning
  yields 0 survivors until dimensional scaling is resolved (see known issue).
- Pure ODE mode (dae_mode="none") → behaves as D2b: 676 kept after pruning.
- Dao gate (dao_validity) → passes for all applicable modes.
"""

import pytest

from apedp_dyn import (
    EpistemicStatus,
    adjudicate,
    enumerate_coupled,
    materialize_system,
    prune_admissible,
)
from apedp_dyn.gates import dao_validity
from apedp_dyn.model import Parameter, PhysicalModel
from apedp_dyn.problems import coupled, duffing, mass_spring_damper


def _count_kept(dae_mode):
    # Call enumerate_coupled with dae_mode only; it uses the default optional
    # which is ("x", "v", "xr", "dxr", "c1", "c2") — giving 1764 raw / 676 kept
    # for dae_mode="none" (verified earlier).
    allc = enumerate_coupled(dae_mode=dae_mode, max_terms=4)
    kept = prune_admissible(coupled.load(), allc)
    return len(kept)


class TestD3KeptCounts:
    def test_ode_mode_kept(self):
        """Pure ODE mode: 676 systems survive pruning (D2b behavior)."""
        assert _count_kept(dae_mode="none") == 676

    def test_constrained_mode_is_refused(self):
        """Constrained DAE mode is not implemented, and says so.

        This test used to assert `n > 0` survivors and passed, which was the
        tell: the survivors were the 225 candidates carrying no constraint at
        all, and the 1539 that carried one were all rejected at gate 1 for
        mixing a dimensionless constraint into an acceleration. A mode that
        can only produce unconstrained models is not a mode.
        """
        with pytest.raises(NotImplementedError, match="not implemented"):
            _count_kept(dae_mode="constrained")

    def test_multi_timescale_kept(self):
        """Multi-timescale mode: fast/slow candidates survive pruning.

        This used to assert `len(kept) == 0`, documenting a known issue: the
        materializer emits `(q / tau_slow) * term` but declared q with the
        pure-ODE unit, target/term. The extra 1/time made every term
        [length]/[time]**3 against a [length]/[time]**2 state rate, so gate 1
        rejected all of them and the mode produced nothing at all.

        The coefficient now carries the time factor, and all 225 enumerated
        candidates survive.
        """
        allc = enumerate_coupled(dae_mode="multi_timescale",
                                 optional=("x_f", "v_f", "x_s", "v_s"),
                                 timescales=("fast", "slow"),
                                 max_terms=4)
        kept = prune_admissible(coupled.load(), allc)
        assert len(kept) == len(allc), (
            f"{len(kept)} of {len(allc)} survived; the tau-scaled parameter "
            "units are probably wrong again")
        # And the models really do carry the timescale provenance.
        model = materialize_system(coupled.load(), kept[0].candidate)
        assert model.dae_mode == "multi_timescale"
        assert model.dae_timescales == ("fast", "slow")
        assert {"tau_fast", "tau_slow"} <= {p.name for p in model.parameters}


class TestD3Materialization:
    def test_ode_materialization(self):
        """ODE materialization produces 4-state PhysicalModel."""
        allc = enumerate_coupled(dae_mode="none")
        kept = prune_admissible(coupled.load(), allc)
        s = kept[0].candidate
        model = materialize_system(coupled.load(), s)
        assert len(model.states) == 4
        assert len(model.equations) == 4
        assert len(model.parameters) > 0

    def test_constrained_materialization_is_refused(self):
        """A constrained candidate cannot be built, and must not pretend to be.

        Rejects at the enumeration boundary rather than materialising a model
        whose constraint is silently dropped. See the raise site in
        enumerate_coupled for what a real implementation needs.
        """
        with pytest.raises(NotImplementedError, match="not implemented"):
            enumerate_coupled(dae_mode="constrained", max_terms=4)


class TestDaoGate:
    def test_dao_pure_ode(self):
        """DAO gate passes for pure ODE mode."""
        allc = enumerate_coupled(dae_mode="none")
        kept = prune_admissible(coupled.load(), allc)
        s = kept[0].candidate
        model = materialize_system(coupled.load(), s)
        # A pure-ODE model carries no DAE provenance, so the gate must say
        # so. Asserting the verdict is the point: this call was previously
        # wrapped in try/except/pass, which passed for a gate that returned
        # garbage -- the reason dao_validity was dead code for so long.
        g = dao_validity(model)
        assert g.passed, g.detail
        assert "pure ODE" in g.detail

    def test_dao_constrained_gate_specification(self):
        """The gate's contract for a constrained model, stated directly.

        enumerate_coupled refuses dae_mode="constrained" (the c1/c2 basis is
        the constant 1 and no index-1 reduction exists), so there is no
        constrained candidate to run the gate on. The gate is still specified
        and still enforced, so this builds the model a correct materializer
        would produce and checks both directions:

          - a model that claims the mode but carries no constraint is refused,
            because "constrained" with nothing constrained is a mislabel;
          - a model carrying a dimensionless constraint parameter passes,
            because the c-basis is X**0 and its coefficient must be too.

        This is the acceptance test for the work described at the raise site
        in enumerate_coupled. It should keep passing unchanged once the
        reduction lands.
        """
        p = mass_spring_damper.spec()

        unbacked = PhysicalModel(p.states, p.inputs, p.parameters,
                                 p.truth_equations, dae_mode="constrained")
        g_unbacked = dao_validity(unbacked)
        assert not g_unbacked.passed
        assert "no constraint parameters" in g_unbacked.detail

        con = (Parameter("con", "dimensionless", 0.0, (-1e6, 1e6)),)
        backed = PhysicalModel(p.states, p.inputs, con, p.truth_equations,
                               dae_mode="constrained",
                               dae_constraint_params=("con",))
        g_backed = dao_validity(backed)
        assert g_backed.passed, g_backed.detail

        dimensionful = (Parameter("con", "newton", 0.0, (-1e6, 1e6)),)
        bad = PhysicalModel(p.states, p.inputs, dimensionful, p.truth_equations,
                            dae_mode="constrained", dae_constraint_params=("con",))
        g_bad = dao_validity(bad)
        assert not g_bad.passed
        assert "dimensionally inhomogeneous" in g_bad.detail


class TestDuffingStillWorks:
    def test_duffing_small_holdout_multi(self):
        """Duffing linear vs cubic under small holdout → MULTI_HYPOTHESIS."""
        prob = duffing.spec()
        holdout = dict(prob.holdout)
        holdout["u_of_t"] = __import__("apedp_dyn").make_signal(
            {"kind": "sine", "amplitude": 0.4, "frequency": 1.0})
        from apedp_dyn.problems.base import finalize
        prob = finalize(__import__("dataclasses").replace(prob, holdout=holdout))
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