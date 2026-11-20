"""MMLM Track S learning from the trial ledger by domain (apedp_mmlm/ledger_learning.py)."""
from __future__ import annotations

import pytest
import sympy as sp

from apedp_mmlm import equation_language as el
from apedp_mmlm import ledger_learning as ll
from apedp_referee.trial_ledger import CandidateTrialLedger

a, b, c = sp.symbols("a b c")
SEED = [el.to_tree(e) for e in (a * b, a / b, a * b**2, sp.sqrt(a / b), a * b / c)]


def _record(ledger, cid, formula, *, domain="optics", status="confirmed", **meta):
    ledger.record_trial(candidate_id=cid, proposer_id="x", formula=formula, strategy_archetype="equation_discovery",
                        in_sample_sharpe=0.0, status=status, metadata={"domain": domain, "resolved_by": "contrast", **meta})


def test_only_confirmations_are_learned_and_only_by_their_domain():
    ledger = CandidateTrialLedger()
    _record(ledger, "c1", "exp(u/w)")
    _record(ledger, "c2", "exp(u/w) + v", status="widen_search")                      # a search step, not a confirmation
    _record(ledger, "c3", "sin(u)", status="unique_support", stage="proposer_validation", resolved_by="loop")
    _record(ledger, "c4", "exp(p/q)", status="unique_support", stage="confirmation", resolved_by="loop")
    learner = ll.DomainLearner(SEED)
    before = learner.model("optics").bits(el.to_tree(sp.exp(a / b)))
    untouched = learner.model("finance").state()
    assert learner.sync(ledger) == {"learned": 2, "retracted": 0, "unparseable": 0}
    assert learner.model("optics").bits(el.to_tree(sp.exp(a / b))) < before
    assert learner.model("finance").state() == untouched
    assert learner.sync(ledger) == {"learned": 0, "retracted": 0, "unparseable": 0}   # idempotent


def test_a_retraction_in_the_ledger_unlearns_the_law():
    ledger = CandidateTrialLedger()
    learner = ll.DomainLearner(SEED)
    clean = learner.model("optics").state()
    _record(ledger, "c1", "sin(cos(u)) - exp(w)")
    learner.sync(ledger)
    assert learner.model("optics").state() != clean
    _record(ledger, "c1-retraction", "", status="retracted", retracts="c1")
    assert learner.sync(ledger)["retracted"] == 1
    assert learner.model("optics").state() == clean


@pytest.mark.parametrize("text", ["__import__('os').system('x')", "u.__class__", "open(f)[0]", "u; w", "'a'", "", "3.5"])
def test_ledger_text_that_is_not_a_plain_expression_is_never_parsed(text):
    assert ll.parse_law(text) is None


def test_the_ledgers_own_spellings_parse():
    # Sorted: the order of a sum's terms follows sympy's ordering of the variable names (a known
    # limitation of the tokeniser: commutative operands are not put in a name-independent order).
    assert sorted(el.tokens(ll.parse_law("((u)^(2) + w)"))) == sorted(el.tokens(el.to_tree(a**2 + b)))
    assert sorted(el.tokens(ll.parse_law("u**2 + 2*w"))) == sorted(el.tokens(el.to_tree(a**2 + b)))
    assert el.tokens(ll.parse_law("(a * m)")) == ["*", "v0", "v1"]
    assert ll.parse_law("1.5e3*x0/x1") is not None


def test_an_unparseable_confirmation_is_counted_not_learned():
    ledger = CandidateTrialLedger()
    _record(ledger, "c1", "d x0/dt = 1.0 * (x1)")
    assert ll.DomainLearner(SEED).sync(ledger) == {"learned": 0, "retracted": 0, "unparseable": 1}
