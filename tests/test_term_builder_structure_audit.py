"""The term-builder structure audit: every emission site binds its coefficient.

The defect under audit is string assembly. A term is emitted as
``(coefficient * <built expression>)``; drop the inner parentheses and a builder
that returns a sum has its coefficient applied to the first addend only.
``dxr`` (``XR - X``) emitted ``q2*x2 - x1`` for two tranches' worth of runs, so
every coupled candidate died at gate 1 for a formatting reason.

`scripts/audit_term_builder_structure.py` asks the question numerically of every
builder and every site that formats a coefficient onto an operand:
``rhs(p=2) - 2*rhs(p=1) == 0``. These tests pin the recorded answer and the two
behaviours it corrected, so a new builder or a new mutation template has to move
the record rather than slip past it.

They also pin the second face of "not a single product": eight builders add a
literal dimensionless ``1`` to a dimensional state, so gate 1 refuses them on
every dimensional problem. That is documented vocabulary
(`candidate_ir.TERM_FAMILIES`, `test_dyn_typed_saturation_grammar.py`), not a
defect to silence, and the record has to keep saying so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "term_builder_structure_audit.json"

# The eight builders whose expression adds the literal 1 to a dimensional state.
DIMENSIONLESS_ONLY = (
    "sat_x", "sat_v", "sat_u", "inv_x", "inv_x2",
    "log_abs_x", "sqrt_abs_x", "ret_x",
)


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD.exists():
        pytest.skip(f"{RECORD.name} has not been generated")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _rows(record: dict) -> dict[str, dict]:
    return {row["term"]: row for row in record["builders"]}


def test_the_record_asks_a_question_it_can_answer(record):
    assert record["schema"] == "term_builder_structure_audit/v3"
    assert record["generated_by"] == "scripts/audit_term_builder_structure.py"
    assert "rhs(p=2) - 2*rhs(p=1) == 0" in record["question"]
    assert "first addend" in record["defect_class"]


def test_every_builder_in_the_vocabulary_is_classified(record):
    from apedp_dyn.grammar.mechanical_grammar import _TERM_BUILDERS

    rows = _rows(record)
    assert set(rows) == set(_TERM_BUILDERS)
    for name, row in rows.items():
        assert row["structure"], name
        assert row["structure"] in {
            "monomial", "product", "function_of_symbol", "sum",
            "quotient_with_sum", "builder_refuses"}, name


def test_no_emitted_term_binds_its_coefficient_to_part_of_an_expression(record):
    """The whole point: nothing in the vocabulary mis-binds after the fix."""
    assert record["summary"]["coefficient_binding_failures"] == []
    for name, row in _rows(record).items():
        if "emitted_rhs" not in row:
            continue
        assert row["coefficient_binds_whole_term"] is True, (name, row)
        assert abs(row["residual"]) < 1e-9, (name, row["residual"])


def test_the_two_sums_are_terms_and_are_emitted_whole(record):
    """A sum is a legitimate term; only the formatting made it wrong."""
    rows = _rows(record)
    assert rows["u_relax"]["structure"] == "sum"
    assert rows["u_relax"]["coefficient_binds_whole_term"] is True
    # It is buildable only where the input shares the velocity's units; the
    # Duffing plant's force input makes the sum incoherent, which is gate 1's
    # job rather than a formatting fault. The audit uses a velocity drive so
    # that the coefficient binding can be checked at all.
    assert rows["u_relax"]["plant"] == "velocity_driven_probe"
    assert rows["u_relax"]["gate1"] == "pass"
    assert rows["u_relax"]["adds_a_literal_constant"] is False
    assert rows["dxr"]["structure"] == "sum"
    # dxr cannot be built single-DOF at all; the coupled site covers it below.
    assert rows["dxr"]["needs_remote_position"] is True
    assert rows["dxr"]["gate1"] == "not_reached"


def test_the_coupled_sum_is_emitted_whole(record):
    """The defect that started this: `dxr` in the two-DOF acceleration."""
    site = next(s for s in record["emission_sites"]
                if s["site"] == "coupled._materialize_ode")
    assert site["coefficient_binds_whole_operand"] is True
    assert abs(site["residual"]) < 1e-9, site["residual"]
    assert site["emitted_rhs"] == "q1*(-x1 + x2)", site["emitted_rhs"]


def test_each_emission_site_binds_its_operand_and_records_the_old_failure(record):
    sites = {site["site"]: site for site in record["emission_sites"]}
    assert set(sites) == {
        "coupled._materialize_ode",
        "coupled._materialize_multiscale",
        "proposers.sindy_proposer.SINDyProposer.fit_and_propose",
        "advanced_search.HierarchicalSearch._expansions",
    }
    assert record["summary"]["emission_sites_failing"] == []
    for name, site in sites.items():
        assert site["coefficient_binds_whole_operand"] is True, name
        # A passing audit that cannot say what it would have caught is not
        # evidence: a site either carries the unparenthesised form it replaced,
        # or says why it never had one.
        if "counterexample_before_fix" in site:
            assert "without" in site["counterexample_before_fix"]["form"], name
            assert site["counterexample_before_fix"]["evaluates_as"], name
        else:
            assert site["defect_not_present"], name


def test_the_multiscale_site_binds_both_the_coefficient_and_the_time_factor(record):
    site = next(s for s in record["emission_sites"]
                if s["site"] == "coupled._materialize_multiscale")
    assert site["outer_factor_binds"] is True
    assert abs(site["outer_factor_residual"]) < 1e-9
    # The fast equation must use the fast timescale. It used to emit
    # `tau_slow` because the labels were read from the system metadata, where
    # `enumerate_coupled` never puts them: `tau_fast` was declared and unused.
    assert "tau_fast" in site["emitted_rhs"], site["emitted_rhs"]


def test_the_literal_one_family_is_recorded_as_dimensionless_only(record):
    """Gate 1 is the right outcome here, and the record has to say why."""
    recorded = set(record["summary"]["dimensionless_only_vocabulary"])
    assert set(DIMENSIONLESS_ONLY) <= recorded
    rows = _rows(record)
    for name in DIMENSIONLESS_ONLY:
        # `materialize` derives the parameter unit with gate 1's own checker, so
        # the refusal can surface there instead of at the gate. Either is the
        # same verdict on the same expression.
        assert rows[name]["gate1"] in {"fail", "materialisation_raises"}, name
        assert "Add" in rows[name]["gate1_detail"], name
        # Not emitted at all, so there is no coefficient to mis-bind.
        assert rows[name]["coefficient_binds_whole_term"] is None, name


def test_the_dimensionless_argument_family_is_recorded_separately(record):
    """sin/cos/tanh of a dimensional state: a different refusal, same gate."""
    assert set(record["summary"]["dimensional_argument_only"]) == {
        "sin_x", "cos_x", "tanh_x"}


def test_expansions_apply_the_template_to_the_seed_as_a_unit():
    """The live behaviour behind the record: a sum seed is not split apart."""
    from apedp import advanced_search as module

    search = object.__new__(module.HierarchicalSearch)
    operand = "x0 + 1"
    var = "x1"
    parenthesised = f"({operand})"
    search.template_set = "classic"
    intents = [t.format(e=parenthesised) for t in
               ("{e}**2", "{e}*2", "2*{e}", "sin({e})", "{e}+1")]
    produced = list(search._expansions(operand, var))
    assert produced
    for text in produced:
        got = sp.sympify(text)
        assert any(sp.simplify(got - sp.sympify(intent)) == 0
                   for intent in intents), text

    search.template_set = "extended"
    intents = [t.format(e=parenthesised, v=var) for t in
               (*module._EXTENDED_EXPANSIONS, "{e}*{v}", "{e}/(1+{v})")]
    intents.append(operand)
    for text in search._expansions(operand, var):
        got = sp.sympify(text)
        assert any(sp.simplify(got - sp.sympify(intent)) == 0
                   for intent in intents), text


def test_each_sindy_candidate_scores_with_its_own_terms_and_coefficients():
    """A scorer that read the loop's variables would score two candidates alike.

    The factory used to close over `_lib_names` from the enclosing loop body.
    That was latent rather than active — the loop reassigned it to the same
    `names` list every iteration — but it made the scorer depend on when it was
    called rather than on what it was built from. Both terms and coefficients
    are now bound at build time, which this pins from both directions.
    """
    from apedp_dyn.proposers.sindy_proposer import SINDyProposer

    class _TwoVariableProposer(SINDyProposer):
        def build_library(self, X, poly_order=3):
            return (np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))]),
                    ["x0", "x1", "1"])

        def sparse_regression(self, Theta, dX):
            Xi = np.zeros((Theta.shape[1], dX.shape[1]))
            Xi[0, 0] = 2.0          # dx0/dt = 2*x0
            Xi[1, 1] = 3.0          # dx1/dt = 3*x1
            return Xi, [0, 1]

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    dX = np.column_stack([2.0 * X[:, 0], 3.0 * X[:, 1]])
    proposals = list(_TwoVariableProposer().fit_and_propose(X, dX))
    assert len(proposals) == 2

    panel = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "1": 1.0})
    first, second = proposals
    assert first.parameters["terms"] == ["x0"]
    assert second.parameters["terms"] == ["x1"]
    assert np.allclose(first.compute_score(panel), 2.0 * X[:, 0])
    assert np.allclose(second.compute_score(panel), 3.0 * X[:, 1])

    # Swapping the frame's columns must follow each scorer's own term: a shared
    # closure would return the same series for both.
    swapped = pd.DataFrame({"x0": X[:, 1], "x1": X[:, 0], "1": 1.0})
    assert np.allclose(first.compute_score(swapped), 2.0 * X[:, 1])
    assert np.allclose(second.compute_score(swapped), 3.0 * X[:, 0])


def test_sindy_formula_gives_a_sum_named_column_one_shared_coefficient():
    """A library column named for a sum must not split its coefficient."""
    from apedp_dyn.proposers.sindy_proposer import SINDyProposer

    class _SumColumnProposer(SINDyProposer):
        def build_library(self, X, poly_order=3):
            return (np.column_stack([X[:, 0] + X[:, 1], np.ones(len(X))]),
                    ["x0 + x1", "1"])

        def sparse_regression(self, Theta, dX):
            Xi = np.zeros((Theta.shape[1], dX.shape[1]))
            Xi[0, :] = 2.0
            return Xi, [0]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    dX = np.column_stack([2.0 * (X[:, 0] + X[:, 1])] * 2)
    proposal = next(iter(_SumColumnProposer().fit_and_propose(X, dX)))

    parsed = sp.sympify(proposal.formula)
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
    assert sp.simplify(parsed.coeff(x0) - parsed.coeff(x1)) == 0, proposal.formula
    assert parsed.coeff(x0) == 2


# --- the repo-wide coefficient-prefix sweep ---------------------------------
#
# The grammar's own sites were audited first. The same question was then put to
# every other f-string that formats a field, a `*` and another field, because
# that shape is the defect rather than the file it lives in. One more live site
# turned up (`ui/server.py`, the promoted theta-double-dot equation), four
# caller-named operands were converted, and the rest are recorded with the
# reason they are safe.


def test_every_coefficient_prefix_site_in_the_repo_is_adjudicated(record):
    """A new `{coef}*{operand}` template has to be adjudicated, not just merged."""
    assembly = record["string_assembly"]
    assert assembly["sites_found"], "the sweep found nothing; it is not sweeping"
    assert assembly["unrecorded_sites"] == [], [
        f"{site['file']}:{site['line']}: {site['source']}"
        for site in assembly["unrecorded_sites"]]
    assert assembly["stale_registry_entries"] == [], (
        "a registry entry no longer matches any site; either the site was "
        "removed (say so in the note) or its needle drifted",
        assembly["stale_registry_entries"])


def test_the_registry_uses_the_declared_vocabulary(record):
    assembly = record["string_assembly"]
    vocabulary = assembly["verdict_vocabulary"]
    assert set(vocabulary) == {
        "parenthesised", "atomic_operand", "numeric_product_label",
        "diagnostic_text", "guarantee_itself", "converted"}
    for row in assembly["registry"]:
        assert row["verdict"] in vocabulary, row
        assert row["note"], row["site"]
        if row["verdict"] == "converted":
            # A converted site may not still be formatting the literal shape;
            # the module has to be calling the formatter that guarantees it.
            assert row["sites"] == [], (row["site"], row["sites"])
            assert row["conversion_marker_present"] is True, row["site"]
        else:
            assert row["sites"], row["site"]


def test_no_coefficient_prefix_site_is_left_unremediated(record):
    assert record["summary"]["coefficient_prefix_sites_failing"] == []
    assert record["summary"]["coefficient_prefix_sites_unrecorded"] == []


# --- the sibling shapes: a field raised to a power, or divided by ----------
#
# The sweep above looked for `}*{`. `**` and `/` bind tighter than `*`, so the
# same defect reaches further: a name that is a sum loses everything after its
# first factor. `1/{name}` on `a + b` is `1/a + b` -- the reciprocal never saw
# the second addend, while the scorer divided the whole column.


def test_every_power_and_division_site_is_adjudicated(record):
    audit = record["power_and_division"]
    assert audit["in_scope"], "the sweep found no in-scope site; it is not sweeping"
    assert audit["unrecorded_sites"] == [], [
        f"{site['file']}:{site['line']} ({site['shape']}: {site['field']}) "
        f"{site['source']}" for site in audit["unrecorded_sites"]]
    assert audit["stale_registry_entries"] == [], (
        "a registry entry no longer matches a site; either the site was removed "
        "(say so in the note) or its needle drifted",
        audit["stale_registry_entries"])


def test_the_scope_rule_accounts_for_every_site_it_excluded(record):
    """An exclusion has to be reported, or the audit is silent about what it skipped."""
    audit = record["power_and_division"]
    by_shape = audit["by_shape"]
    assert by_shape["in_scope"] + by_shape["numeric_operand"] + \
        by_shape["display_only"] == by_shape["total"]
    assert by_shape["total"] == sum(by_shape[shape] for shape in
                                    ("power", "division", "call_argument",
                                     "left_factor"))
    accounted = {f"{site['file']}:{site['line']}" for site in audit["sites_found"]}
    asserted = {f"{site['file']}:{site['line']}"
                for site in audit["in_scope"] + audit["out_of_scope"]}
    assert accounted == asserted
    assert audit["out_of_scope"], "nothing was excluded; the rule is not filtering"
    for site in audit["out_of_scope"]:
        assert site["scope_reason"], site
        assert site["scope"] in {"numeric_operand", "display_only"}, site


def test_power_division_registry_uses_declared_vocabulary(record):
    audit = record["power_and_division"]
    vocabulary = audit["verdict_vocabulary"]
    assert set(vocabulary) == {
        "converted", "retired", "parenthesised", "formatted_operand",
        "atomic_operand", "numeric_fraction_text", "non_arithmetic_text",
        "argument_is_whole_call", "guarantee_itself"}
    assert audit["registry"], "the registry is empty"
    for row in audit["registry"]:
        assert row["verdict"] in vocabulary, row
        assert row["note"], row["site"]
        if row["verdict"] in ("converted", "retired"):
            assert row["sites"] == [], (row["site"], row["sites"])
            assert row["conversion_marker_present"] is True, row["site"]
        else:
            assert row["sites"], row["site"]


def test_powered_and_over_bind_sums():
    from apedp_dyn.grammar.term_format import over, powered

    assert powered("x", "2") == "x**2"
    assert powered("a + b", "2") == "(a + b)**2"
    assert over("1", "x") == "1/x"
    assert over("1", "a + b") == "1/(a + b)"
    assert over("XR - X", "tau") == "(XR - X)/tau"
    a, b = sp.Symbol("a"), sp.Symbol("b")
    assert sp.simplify(sp.sympify(powered("a + b", "2")) - (a + b) ** 2) == 0
    assert sp.simplify(sp.sympify(over("1", "a + b")) - 1 / (a + b)) == 0
    # A formatted number keeps its sign inside the parentheses: `-0.5**2` is
    # `-(0.25)`, which is a different value from `(-0.5)**2`.
    assert powered("-0.5", "2") == "(-0.5)**2"


def test_recombine_power_law_binds_a_sum_named_variable():
    """The live behaviour behind the record: a power on a sum-named variable."""
    from apedp.discovery._core import recombine_power_law

    a, b, v = sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("v")
    squared = sp.sympify(recombine_power_law(["a + b"], [2.0]))
    assert sp.simplify(squared - (a + b) ** 2) == 0, squared
    linear = sp.sympify(recombine_power_law(["a + b"], [1.0]))
    assert sp.simplify(linear - (a + b)) == 0, linear
    # The exponent-1 branch is the subtle one: the terms are joined with `*`,
    # so the name still has to arrive wrapped.
    product = sp.sympify(recombine_power_law(["v", "a + b"], [1.0, 1.0]))
    assert sp.simplify(product - v * (a + b)) == 0, product
    # A fractional exponent: assert on the base rather than on the value, since
    # sympy keeps 0.5 as a Float and will not reduce it to a Rational.
    fractional = sp.sympify(recombine_power_law(["a + b"], [0.5]))
    assert getattr(fractional, "base", None) == a + b, fractional


def test_the_formatter_leaves_atoms_alone_and_binds_sums():
    from apedp_dyn.grammar.term_format import atom, scaled

    for name in ("x", "x0", "sin(x)", "x**2", "u_x", "(a + b)", "a/(1+b)"):
        assert atom(name) == name, name
        assert scaled("2.0", name) == f"2.0*{name}", name
    for name in ("a + b", "XR - X", "x + 1", "-x"):
        assert atom(name) == f"({name})", name
        assert scaled("2.0", name) == f"2.0*({name})", name
    # A power on a sum-named operand has to land on the whole name, not the
    # last addend: `2.0*(a + b)^2`, never `2.0*(a + b^2)`.
    assert f"2.0*{atom('a + b')}^2" == "2.0*(a + b)^2"


def test_a_harmonic_probe_has_one_canonical_spelling():
    """`harmonic` is the only place a frequency-scaled call is spelled.

    The engine emits `sin(2.0*x0)` for an atomic seed and `sin(2.0*(x0 + 1))`
    for a sum. A test or a candidate id that spells either by hand can drift
    from the engine -- that is how the harmonic recovery test came to expect
    `sin(2.0*(x0))`, a law the search had stopped emitting.
    """
    from apedp_dyn.grammar.term_format import harmonic

    assert harmonic("sin", 2.0, "x0") == "sin(2.0*x0)"
    assert harmonic("cos", "0.5", "x0") == "cos(0.5*x0)"
    assert harmonic("sin", 3, "x0 + 1") == "sin(3.0*(x0 + 1))"
    # A symbolic frequency is already text, and is kept verbatim.
    assert harmonic("sin", "omega", "t") == "sin(omega*t)"
    # One law, one id: the caller's numeric type does not change the spelling.
    for frequency in (2, 2.0, "2", "2.0", "2e0"):
        assert harmonic("sin", frequency, "x0") == "sin(2.0*x0)", frequency
    # The argument is bound as a unit, so the frequency applies to the whole seed.
    assert sp.sympify(harmonic("sin", 2.0, "x0 + 1")) == sp.sin(2.0 * (sp.Symbol("x0") + 1))


def test_a_sum_named_library_term_keeps_one_shared_coefficient():
    """`TermLibrary.render` names caller-supplied terms; a term may be a sum."""
    from apedp_dyn.qed.library import TermLibrary

    library = TermLibrary.from_specs((
        ("a + b", lambda s: s[:, 0] + s[:, 1]),
        ("1", lambda s: np.ones(s.shape[0])),
    ))
    rendered = library.render((0,), (2.0,), lhs="dy/dt")
    parsed = sp.sympify(rendered.split("=", 1)[1])
    a, b = sp.Symbol("a"), sp.Symbol("b")
    assert sp.simplify(parsed.coeff(a) - parsed.coeff(b)) == 0, rendered
    assert parsed.coeff(a) == 2
