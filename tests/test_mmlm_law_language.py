"""MMLM-L: canonical tokens, proper probabilities, domain back-off, learning and unlearning."""
from __future__ import annotations

import random

import numpy as np
import pytest
import sympy as sp

from apedp_mmlm import equation_language as el
from apedp_mmlm import law_corpora
from apedp_mmlm import law_language as ll

LAWS = ["x0*x1/x2", "x0*x1**2", "sqrt(x0/x1)", "x0/(x0 + x1)", "exp(-x0/x1)", "x0*x1*x2", "x0**2*x1/x2",
        "x0*(x1 - x2)", "x0/x1**2", "log(x0/x1)", "x0*sin(x1)", "x0*x1/(x2*x3)", "sqrt(x0**2 + x1**2)",
        "x0*x1 + x2*x3", "x0/(x1*x2)", "x0*exp(x1*x2)", "x0*x1**3", "(x0 - x1)/x2", "x0*cos(x1)", "x0**2/x1"]


def _trees(texts=LAWS):
    return [el.to_tree(sp.sympify(t)) for t in texts]


def _model():
    return ll.LawLanguageModel(folds=5).fit(_trees(), ["mechanics"] * 10 + ["fields"] * 10)


def test_tokens_do_not_depend_on_variable_names():
    rng = random.Random(3)
    for text in [*LAWS, "x0*x1 + x0*x2", "x0*x1*(x2 + x3) + x4**2", "x0**2*x1**2*x2**2*(1 + x2*x3/x4) + x5**2"]:
        expr = sp.sympify(text)
        for _ in range(4):
            renamed = expr.xreplace({s: sp.Symbol(f"q{rng.randrange(10**6)}") for s in expr.free_symbols})
            assert ll.law_tokens(el.to_tree(expr)) == ll.law_tokens(el.to_tree(renamed)), text


def test_every_next_token_distribution_is_a_distribution_over_the_allowed_tokens():
    model = _model()
    for tree in _trees(["x0*x1/(x2 + x0)", "sqrt(x0)*x1"]):
        for _written, context in ll._walk(ll.canonical(tree)):
            allowed = ll._allowed(context["_nvars"][0])
            dist = ll._mix(model._blend("mechanics")[context["_group"][0]], model.counts.log_probabilities(context, "mechanics"), allowed)
            assert np.exp(dist[allowed]).sum() == pytest.approx(1.0)
            assert np.all(np.isneginf(dist[~allowed]))
    # before any variable has appeared, the only variable token allowed is v0
    assert ll._allowed(0)[ll.ID["v0"]] and not ll._allowed(0)[ll.ID["v1"]]


def test_a_domain_with_no_laws_is_scored_by_the_pooled_model():
    model = _model()
    tree = _trees(["x0*x1/x2"])[0]
    assert model.bits(tree, "never_seen") == pytest.approx(model.bits(tree, None))
    assert model.bits(tree, "mechanics") != pytest.approx(model.bits(tree, None))


def test_observing_a_law_makes_it_cheaper_and_retracting_restores_the_model():
    model = _model()
    law = _trees(["tanh(x0)*x1/x2"])[0]
    before_bits, before_state = model.bits(law, "mechanics"), model.state()
    cost = model.observe(law, "mechanics")
    assert cost == pytest.approx(before_bits)
    assert model.bits(law, "mechanics") < before_bits
    assert model.state()["observed_laws"] == 1
    assert model.retract(law, "mechanics") is True
    assert model.bits(law, "mechanics") == pytest.approx(before_bits)
    assert model.state() == before_state
    assert model.retract(law, "mechanics") is False


def test_familiar_shapes_cost_fewer_bits_than_unfamiliar_ones():
    model = _model()
    familiar, strange = _trees(["x0*x1/x2", "tanh(acos(x0)) - asin(x1)"])
    assert model.bits(familiar) < model.bits(strange)


def test_search_prior_prices_a_search_node():
    prior = ll.search_prior(_model())
    simple = ("/", ("x", 0), ("x", 1))
    nested = ("sin", ("exp", ("cos", ("x", 0))))
    assert 0 < prior(simple) < prior(nested)


def test_expression_parser_accepts_arithmetic_and_refuses_everything_else():
    expr = law_corpora.parse_expression("c_0*x_0^2*(1 - x_0/c_1) + sin(x_1)**2", {"c_0": 0.5, "c_1": 2.0})
    x0, x1 = sp.Symbol("x_0"), sp.Symbol("x_1")
    assert sp.simplify(expr - (0.5 * x0**2 * (1 - x0 / 2) + sp.sin(x1) ** 2)) == 0
    for hostile in ("__import__('os').system('true')", "x.__class__", "[x for x in y]", "open('f')", "lambda: 0"):
        with pytest.raises((ValueError, SyntaxError)):
            law_corpora.parse_expression(hostile)


def test_odebench_file_is_parsed_without_being_executed():
    source = '''
raise SystemExit("this file must never be executed")
equations = [
 {'id': 1, 'eq': 'c_0 * x_0 | -x_1 + c_1*x_0^2', 'dim': 2, 'consts': [[1/3, 2.5]], 'init': [[1., 2.]]},
]
'''
    entries = law_corpora.odebench_entries(source)
    assert entries == [{"id": 1, "dim": 2, "components": ["c_0 * x_0", "-x_1 + c_1*x_0^2"], "consts": [pytest.approx(1 / 3), 2.5]}]


def test_ledger_learner_learns_confirmations_once_and_applies_retractions():
    from types import SimpleNamespace

    from apedp_mmlm.law_ledger import LawLedgerLearner

    def record(candidate, stamp, status, formula="", **meta):
        return SimpleNamespace(candidate_id=candidate, timestamp=stamp, status=status, formula=formula, metadata=meta)

    records = [
        record("c1", 1, "confirmed", "tanh(x0)*x1/x2", resolved_by="contrast", domain="mechanics"),
        record("c2", 2, "refuted", "x0 + x1"),
        record("c3", 3, "unique_support", "x0*exp(x1)", stage="confirmation"),
    ]
    ledger = SimpleNamespace(get_records=lambda: list(records))
    model = _model()
    law = el.to_tree(sp.sympify("tanh(x0)*x1/x2"))
    before = model.bits(law, "mechanics")
    learner = LawLedgerLearner(model)
    assert learner.sync(ledger) == {"learned": 2, "retracted": 0, "unparseable": 0}
    assert model.bits(law, "mechanics") < before
    assert learner.sync(ledger) == {"learned": 0, "retracted": 0, "unparseable": 0}
    records.append(record("r1", 4, "retracted", retracts="c1"))
    assert learner.sync(ledger)["retracted"] == 1
    assert model.state()["observed_laws"] == 1 and learner.state()["laws_learned_from_ledger"] == 1
