"""The shared corpus: a celled sample, reproducible from seeds, and not favourable.

A corpus that can be quietly regenerated to a different draw is not citable, and a
corpus of targets one engine was written for is not shared. These tests pin both
properties on the module rather than on the record:

* **a sample, not a set** -- every graph family is drawn at every declared width with
  the declared number of replicates, so a family's behaviour cannot be read off one
  hand-picked instance;
* **reproducible** -- rebuilding from the seeds alone yields the same specifications,
  hash for hash, and every seed is the one the documented rule produces, so a record's
  citation of a target is a citation of a cell;
* **as specified** -- a "3-regular" target really is simple, connected and 3-regular,
  checked by reading the degrees back out of the stabilizers rather than trusting the
  generator's note;
* **shared in the honest sense** -- the graph states are buildable by the grammar, and
  the random stabilizer states are not: the corpus contains the abstention instead of
  only the favourable case.
"""
from __future__ import annotations

from collections import Counter

import pytest

from apedp_dyn.vqc.corpus import (
    CORPUS,
    GRAPH_FAMILIES,
    SAMPLE_REPLICATES,
    SAMPLE_WIDTHS,
    SEED_BASE,
    STABILIZER_WIDTHS,
    CorpusTarget,
    random_regular_graph_edges,
    random_stabilizer_target,
    sampled_graph_corpus,
    seed_for,
    shared_target_corpus,
)
from apedp_dyn.vqc.synthesis import (
    SynthesisError,
    Z,
    default_families,
    synthesize,
)


def _adjacency(entry: CorpusTarget) -> list[set[int]]:
    """The edge set, read out of the target's own stabilizers."""
    neighbours: list[set[int]] = [set() for _ in range(entry.target.num_qubits)]
    for vertex, (labels, _phase) in enumerate(entry.target.stabilizers):
        for other, label in enumerate(labels):
            if label == Z:
                neighbours[vertex].add(other)
    return neighbours


def _connected(adjacency: list[set[int]]) -> bool:
    reached, frontier = {0}, [0]
    while frontier:
        for neighbour in adjacency[frontier.pop()]:
            if neighbour not in reached:
                reached.add(neighbour)
                frontier.append(neighbour)
    return len(reached) == len(adjacency)


def test_the_corpus_is_reproducible_from_its_seeds_alone():
    first = [(entry.label, entry.kind, entry.seed, entry.target.sha256())
             for entry in shared_target_corpus()]
    second = [(entry.label, entry.kind, entry.seed, entry.target.sha256())
              for entry in shared_target_corpus()]
    assert first == second
    assert first == [(entry.label, entry.kind, entry.seed, entry.target.sha256())
                     for entry in CORPUS]
    assert len({hash_ for *_rest, hash_ in first}) == len(first)


def test_every_cell_is_drawn_and_the_sample_covers_its_design():
    """Three families x five widths x five replicates, plus the abstention anchors."""
    expected = (len(GRAPH_FAMILIES) * len(SAMPLE_WIDTHS) * SAMPLE_REPLICATES
                + len(STABILIZER_WIDTHS))
    assert len(CORPUS) == len({entry.label for entry in CORPUS}) == expected

    cells = Counter((entry.family, entry.width) for entry in CORPUS
                    if entry.family != "random_stabilizer")
    for family, _degree in GRAPH_FAMILIES:
        for width in SAMPLE_WIDTHS:
            assert cells[(family, width)] == SAMPLE_REPLICATES, (family, width)
        assert {entry.replicate for entry in CORPUS if entry.family == family} == set(
            range(SAMPLE_REPLICATES))

    anchors = [entry for entry in CORPUS if entry.family == "random_stabilizer"]
    assert [entry.width for entry in anchors] == list(STABILIZER_WIDTHS)


def test_every_seed_is_the_one_the_documented_rule_produces():
    """The seed is arithmetic, so a cell can be rebuilt without the corpus around it."""
    for entry in CORPUS:
        assert entry.seed == seed_for(entry.family, entry.width, entry.replicate)
        assert entry.seed // 10_000 * 10_000 == SEED_BASE[entry.family]
    with pytest.raises(SynthesisError, match="replicate"):
        seed_for("regular3", 12, 10)      # a cell's offset is one digit wide
    with pytest.raises(SynthesisError, match="unknown corpus family"):
        seed_for("lattice", 12, 0)


def test_a_label_names_the_cell_that_produced_it():
    for entry in CORPUS:
        assert entry.label == f"{entry.family}_{entry.width}q_{entry.replicate}"


def test_the_corpus_holds_matched_targets_and_an_abstention():
    kinds = {entry.kind for entry in CORPUS}
    assert kinds == {"regular_graph", "random_graph", "random_stabilizer"}

    # One replicate per cell is enough for the buildability claim: whether a draw is
    # constructible is structural, and the record carries every instance's verdict.
    representatives = [entry for entry in CORPUS if entry.replicate == 0]
    for entry in representatives:
        if entry.kind == "random_stabilizer":
            # No construction family applies, and asking for one is a refusal with a
            # reason rather than an empty success.
            assert default_families(entry.target) == ()
            with pytest.raises(SynthesisError, match="no construction family"):
                synthesize(entry.target)
        else:
            outcome = synthesize(entry.target)
            assert outcome.promoted is not None, outcome.refusal
            assert outcome.ranked


def test_a_regular_draw_is_simple_connected_and_actually_regular():
    for entry in CORPUS:
        if entry.kind != "regular_graph":
            continue
        adjacency = _adjacency(entry)
        degrees = [len(row) for row in adjacency]
        degree = degrees[0]
        assert degree in (3, 4), entry.label
        assert degrees == [degree] * entry.target.num_qubits, entry.label
        assert degree == entry.degree
        assert all(vertex not in row for vertex, row in enumerate(adjacency))
        assert _connected(adjacency)
        assert sum(degrees) // 2 == entry.as_dict()["edges"] == (
            entry.target.num_qubits * degree) // 2
        assert entry.as_dict()["min_degree"] == entry.as_dict()["max_degree"] == degree


def test_the_corpus_graphs_are_not_rings_or_lattices():
    """A ring is 2-regular; a square lattice is at most degree 4 but has boundary
    vertices of degree 2, and the corpus's regular draws have no degree-2 vertex."""
    for entry in CORPUS:
        if entry.kind == "random_stabilizer":
            continue
        degrees = {len(row) for row in _adjacency(entry)}
        assert degrees != {2}, entry.label
        if entry.kind == "regular_graph":
            assert 2 not in degrees, entry.label
        else:
            assert len(degrees) > 1, entry.label    # irregular by construction


def test_the_irregular_family_is_still_irregular_at_every_width():
    for entry in CORPUS:
        if entry.kind != "random_graph":
            continue
        degrees = [len(row) for row in _adjacency(entry)]
        assert min(degrees) < max(degrees), entry.label
        assert entry.as_dict()["mean_degree"] == pytest.approx(
            sum(degrees) / len(degrees))


def test_a_smaller_sample_is_the_same_sample_restricted():
    """The design is a function of its two parameters, not a lookup table."""
    smaller = sampled_graph_corpus(widths=(12, 20), replicates=2)
    assert len(smaller) == len(GRAPH_FAMILIES) * 2 * 2 + len(STABILIZER_WIDTHS)
    by_label = {entry.label: entry for entry in CORPUS}
    for entry in smaller:
        if entry.family == "random_stabilizer":
            assert by_label[entry.label].target.sha256() == entry.target.sha256()
            continue
        assert entry.label in by_label
        assert by_label[entry.label].target.sha256() == entry.target.sha256()


def test_an_impossible_regular_draw_is_refused_by_arithmetic():
    with pytest.raises(SynthesisError, match="even"):
        random_regular_graph_edges(5, 3, seed=1)      # 5 * 3 is odd
    with pytest.raises(SynthesisError, match="degree"):
        random_regular_graph_edges(8, 8, seed=1)
    with pytest.raises(SynthesisError, match="four vertices"):
        random_regular_graph_edges(3, 1, seed=1)


def test_a_random_stabilizer_target_is_a_pure_state_with_no_graph_structure():
    target = random_stabilizer_target(8, seed=7)
    assert target.name == "random_stabilizer"
    assert len(target.stabilizers) == 8
    target.stabilizer_state().validate()       # raises if it is not a pure state
    # It is the same state twice, and a different state at a different seed.
    assert target.sha256() == random_stabilizer_target(8, seed=7).sha256()
    assert target.sha256() != random_stabilizer_target(8, seed=8).sha256()
