"""Grammar-vocabulary scope (regression boundary).

Finding: the flagship 7-term template is dimensionally stillborn on
meter-state problems (sin_x over meters) yet survives intact on
radian-state problems. Vocabulary validity is PROBLEM-RELATIVE, not a
property of the term list. Pinned so the grammar cannot silently drift:
transcendental terms are scoped to dimensionless states by Gate 1, and the
template degrades gracefully (sin-free subset survives) instead of vanishing.
"""
from apedp_dyn import full_topology_candidate, prune_admissible
from apedp_dyn.grammar.candidate_ir import CandidateDynamics, Term
from apedp_dyn.problems import duffing, pendulum


def _without(full, drop):
    names = [t.name for t in full.terms if t.name not in drop]
    return CandidateDynamics(
        lhs="xddot",
        terms=tuple(Term(n, f"a{i}") for i, n in enumerate(names)),
        candidate_id="scope-probe")


class TestVocabularyScope:
    def test_full_template_dead_on_meter_states(self):
        kept = prune_admissible(duffing.load(), [full_topology_candidate()])
        assert kept == []

    def test_sin_free_subset_survives_meter_states(self):
        sub = _without(full_topology_candidate(), {"sin_x"})
        kept = prune_admissible(duffing.load(), [sub])
        assert len(kept) == 1
        assert "sin_x" not in kept[0].candidate.term_set()

    def test_full_template_alive_on_radian_states(self):
        kept = prune_admissible(pendulum.load(), [full_topology_candidate()])
        assert len(kept) == 1
        assert "sin_x" in kept[0].candidate.term_set()


class TestProposerSeam:
    def test_grammar_proposer_reproduces_enumeration(self):
        from apedp_dyn import enumerate_candidates
        from apedp_dyn.grammar.proposer import (
            PROPOSER_VERSION,
            CandidateProposer,
            GrammarProposer,
        )
        from apedp_dyn.problems import duffing

        assert PROPOSER_VERSION == "1.0.0-ir-seam"
        prop = GrammarProposer(required=("u",), optional=("x", "x3", "v"), max_terms=4)
        assert isinstance(prop, CandidateProposer)
        assert prop.proposer_id == "grammar-v1"
        prob = duffing.load()
        via_seam = prop.propose(prob)
        direct = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
        assert [c.candidate_id for c in via_seam] == [c.candidate_id for c in direct]
        assert [c.term_set() for c in via_seam] == [c.term_set() for c in direct]


class TestEnumerationScaling:
    """Search-scaling bounds (informational pins): enumeration+pruning is
    cheap; FITTING survivors is the bottleneck (each fit ~seconds), which is
    why sweeps run hand-picked slates instead of full grids. A 676-system
    COUP grid at ~2s/fit would cost ~25 min."""

    def test_duff_extended_counts(self):
        from apedp_dyn import enumerate_candidates, prune_admissible
        from apedp_dyn.problems import duffing

        prob = duffing.load()
        cands = enumerate_candidates(required=("u",),
                                     optional=("x", "x2", "x3", "v", "abs_v_v", "x_v"),
                                     max_terms=4)
        assert len(cands) == 42
        assert len(prune_admissible(prob, cands)) == 42

    def test_coup_restricted_counts(self):
        from apedp_dyn import enumerate_coupled, prune_admissible
        from apedp_dyn.problems import coupled

        prob = coupled.load()
        allc = enumerate_coupled(required=("u",), optional=("x", "v", "xr", "dxr"),
                                 max_terms=4)
        assert len(allc) == 225
        assert len(prune_admissible(prob, allc)) == 64


class TestLibraryProposer:
    def test_stridge_recovers_duffing_truth(self):
        from apedp_dyn.grammar.proposer import (
            STRIDGE_ITERS,
            STRIDGE_REL_TOL,
            CandidateProposer,
            LibraryProposer,
        )
        from apedp_dyn.problems import duffing

        assert (STRIDGE_REL_TOL, STRIDGE_ITERS) == (0.05, 10)
        prop = LibraryProposer()
        assert isinstance(prop, CandidateProposer)
        props = prop.propose(duffing.load())
        assert len(props) == 1
        assert set(props[0].term_set()) == {"u", "x", "x3", "v"}
