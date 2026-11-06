"""Pins for the two native integer-relation searchers and their comparison record.

`find_integer_relation` is a branch-and-bound over a box (`n > 8` refused, no
certificate). `find_integer_relation_lll` reduces `[ I | round(x * 10^digits) ]`
and verifies every candidate by exact substitution on the `f64` inputs. The tests
below decide which of those claims the build actually supports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "integer_relation_methods.json"

kernel = pytest.importorskip("apedp_kernel")


def _has_lll() -> bool:
    return hasattr(kernel, "rime_number_theory_lll_relation_json")


def _lll(x, **kwargs):
    # No coefficient bound: the reduction's reach is the thing under test, and a
    # positive `max_coeff` would filter exactly the candidates it reaches for.
    defaults = {
        "max_coeff": 0,
        "eps": 1e-10,
        "labels": None,
        "start_digits": 17,
        "max_digits": 100,
        "max_iterations": 20_000,
    }
    defaults.update(kwargs)
    return json.loads(kernel.rime_number_theory_lll_relation_json(x=x, **defaults))


def _bnb(x, max_coeff=10, eps=1e-10, labels=None):
    return json.loads(
        kernel.rime_number_theory_integer_relation_json(x, max_coeff, eps, labels)
    )


def test_the_lll_binding_exists_and_names_its_method():
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    found = _lll([math.sqrt(2.0), math.sqrt(3.0), math.sqrt(2.0) + math.sqrt(3.0)])
    assert found is not None
    assert found["method"] == "lll_integer_relation_pslq"
    assert found["coefficients"] == [1, 1, -1]


def test_a_candidate_is_only_returned_after_exact_substitution():
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    found = _lll([math.sqrt(2.0), math.sqrt(3.0), math.sqrt(2.0) + math.sqrt(3.0)])
    assert found["verified"] is True
    # The residual is exact arithmetic on the f64 inputs, so a real relation holds
    # to the inputs' own precision and says how far that is.
    assert found["relative_residual"] <= 1e-10
    assert found["digits_held"] >= 10.0
    assert any("exact rational arithmetic" in s for s in found["statements"])
    assert found["working_digits"] >= 17
    # The decisive guard: the vector must be far shorter than what a random
    # embedding of the same size produces, or it is arithmetic rather than law.
    assert found["max_coefficient"] * 10.0 <= found["generic_coefficient_bound"]


def test_branch_and_bound_refuses_ten_inputs_and_lll_does_not():
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    bases = [math.sqrt(p) for p in primes]
    x = [*bases, sum((i + 1) * v for i, v in enumerate(bases))]
    assert _bnb(x, max_coeff=20, eps=1e-10) is None, "the box search caps at n = 8"
    found = _lll(x, eps=1e-10)
    assert found is not None
    # Several exact relations exist among the first nine primes (2 + 3 = 5 is the
    # shortest); the claim is that a ten-input relation is searched at all, so any
    # relation whose residual is exactly zero answers it.
    assert found["verified"] is True
    assert found["residual"] == 0.0


def test_a_relation_outside_the_box_is_found_by_reduction_and_not_by_search():
    """Coefficients beyond `max_coeff`, with the relation exact in the inputs.

    The values are dyadic, so 137 and -89 hold with zero residual. The same
    relation built on sqrt(2)/sqrt(3) would hold only to f64 rounding and would
    need about 21 working digits — `n*log10(137) + 16` — which an f64 cannot
    supply, so it is not detectable by either searcher and would test the float
    format rather than the box limit.
    """
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    a = 1_234_567 / 2**20
    b = 7_654_321 / 2**20
    x = [a, b, 137.0 * a - 89.0 * b]
    assert _bnb(x, max_coeff=20, eps=1e-12) is None, "137 and -89 do not fit in the box"
    found = _lll(x, eps=1e-12)
    assert found is not None
    assert found["coefficients"] == [137, -89, -1]


def test_a_tolerance_below_the_exact_residual_is_refused():
    """At the input's own precision, no inexact relation can clear 1e-300."""
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    x = [math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)]
    assert _lll(x, eps=1e-300, max_digits=17) is None


def test_escalation_finds_a_relation_among_the_rounded_values_not_the_constants():
    """An f64 is a rational, so three of them always have an exact relation.

    Allowed past the input precision, the reduction returns one with coefficients
    around 1e8 and a residual of exactly zero — an integer relation among the three
    52-bit rationals, not among sqrt(2), sqrt(3), sqrt(5). The searcher is honest
    about which it found: the result is `verified` against the inputs it was given,
    and `working_digits` says the ladder had to leave the input's precision.
    """
    if not _has_lll():
        pytest.skip("this build of the extension predates the LLL binding")
    x = [math.sqrt(2.0), math.sqrt(3.0), math.sqrt(5.0)]
    found = _lll(x, eps=1e-300, max_digits=40)
    assert found is not None
    assert found["verified"] is True
    # The kernel's payload carries the absolute residual as `residual`; the
    # comparison record renames it `absolute_residual` beside the relative one.
    assert found["residual"] == 0.0
    assert found["working_digits"] > 17, "this relation only exists above f64 precision"
    assert max(abs(c) for c in found["coefficients"]) > 10**6


def test_the_record_is_recomputable_from_its_own_rows():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    rows = record["rows"]
    summary = record["summary"]
    assert summary["cases"] == len(rows)
    assert summary["agree"] == sum(1 for r in rows if r["agreement"])
    for method in ("branch_and_bound", "lll_pslq"):
        for key in ("correct", "refused_out_of_scope", "wrong"):
            assert summary[method][key] == sum(
                1 for r in rows if r[f"{method}_assessment"] == key
            ), (method, key)
    # Refusing outside a documented scope is behaviour, not error; answering
    # wrongly is the only failure either searcher is allowed to avoid.
    assert summary["branch_and_bound"]["wrong"] == 0
    assert summary["lll_pslq"]["wrong"] == 0
    assert summary["lll_pslq"]["refused_out_of_scope"] == 0
    assert summary["branch_and_bound"]["refused_out_of_scope"] == sum(
        1
        for r in rows
        if r["branch_and_bound_scope"] == "out_of_scope"
        and not r["branch_and_bound"]["found"]
    )
    # A `specific` case that either method answers must answer with the relation
    # its construction fixed; a different vector is a wrong answer, not a variant.
    for row in rows:
        if row["criterion"] != "specific":
            continue
        for method in ("branch_and_bound", "lll_pslq"):
            if row[method]["found"]:
                assert row[method]["coefficients"] == row["expected_coefficients"], row
    assert record["methods"]["lll_pslq"]["constant"] == "lll_integer_relation_pslq"
    assert record["design"]["not_a_distribution"] is True


def test_the_randomized_block_is_seeded_and_recomputable_from_its_rows():
    """The random families are a distribution only over the families they define.

    Every rate in the record has to be recomputable from the rows, the seed and the
    generator parameters have to be present (a seed alone reproduces nothing), and
    the three families have to show the behaviour their construction fixes: the two
    refusal-free families always carry a relation, and the independent family never
    does.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    randomized = record["summary"]["randomized"]
    assert randomized["seed"] == 20261102
    assert randomized["per_family"] > 0
    assert "random.Random(seed)" in record["design"]["randomized"]

    rows = [r for r in record["rows"] if r["family"] != "designed"]
    designed = [r for r in record["rows"] if r["family"] == "designed"]
    assert len(designed) == 8, "the hand-built cases must stay listed separately"
    assert len(rows) == randomized["per_family"] * len(randomized["families"])

    for family, entry in randomized["families"].items():
        subset = [r for r in rows if r["family"] == family]
        assert entry["cases"] == len(subset), family
        for method in ("branch_and_bound", "lll_pslq"):
            for key in ("correct", "refused_out_of_scope", "wrong"):
                assert entry[method][key] == sum(
                    1 for r in subset if r[f"{method}_assessment"] == key
                ), (family, method, key)
            assert entry[method]["pass_rate"] == pytest.approx(
                entry[method]["correct"] / len(subset)
            )
        assert entry["planted_vector_recovered_by_lll_pslq"] == sum(
            1
            for r in subset
            if r["expected_coefficients"] is not None
            and r["lll_pslq"]["coefficients"] == r["expected_coefficients"]
        )

    independent = randomized["families"]["random_independent"]
    assert independent["branch_and_bound"]["correct"] == independent["cases"], (
        "no relation exists at input precision, so refusing is the right answer"
    )
    assert independent["lll_pslq"]["correct"] == independent["cases"]

    for family in ("random_planted_in_box", "random_planted_out_of_box"):
        entry = randomized["families"][family]
        assert entry["lll_pslq"]["wrong"] == 0, family
        assert entry["lll_pslq"]["correct"] == entry["cases"], (
            "every planted case must be answered by the reduction"
        )

    # The box search is out of scope for the out-of-box family by construction, and
    # refusing there is recorded as scope, not as error.
    out_of_box = randomized["families"]["random_planted_out_of_box"]
    assert out_of_box["branch_and_bound"]["refused_out_of_scope"] == out_of_box["cases"]
    assert out_of_box["branch_and_bound"]["wrong"] == 0
