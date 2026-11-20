"""Audit every term builder for expressions that are not a single product.

The defect this audits is string assembly, not algebra. A term is emitted as
``(coefficient * <built expression>)``. Two materialiser sites used to emit
``(coefficient*<built expression>)`` without the inner parentheses, which is
harmless while every builder returns a single monomial and silently wrong the
moment one does not: ``q2*(XR - X)`` became ``q2*XR - X``, so the remote
position entered with coefficient 1 and gate 1 rejected every coupling
candidate as a length added to an acceleration.

The audit therefore asks one question of each builder and each emission site,
and answers it numerically rather than by inspection:

    does the coefficient multiply the whole expression?    For a **single-term** equation the emitted right-hand side must satisfy
``rhs(p=2) == 2 * rhs(p=1)`` at arbitrary state values, where ``p`` is that
term's own parameter. An expression that is a sum, a quotient of sums, or a
function of a sum is a legitimate term; a coefficient that reaches only part of
one is not.

    Single-term probes are load-bearing, and the first version of this script
got it wrong. With two terms on one right-hand side, doubling one coefficient
leaves the other term's contribution in place, so ``rhs(2) - 2*rhs(1)`` equals
minus that other term and *every* multi-term equation looks broken. The coupled
probe was duly reported as failing until it was reduced to one term per
equation; the residual it produced was exactly ``-q0*u1``. Each builder is also classified structurally and run through
materialisation and gate 1, because "not a single product" has a second face:
an expression that adds a dimensionless ``1`` to a dimensional state is refused
by gate 1 on *every* dimensional problem, and that is a property of the
vocabulary worth having in a record rather than in a reader's memory.

Emission sites audited:

- ``mechanical_grammar.materialize`` — single-DOF ODE
- ``coupled._materialize_ode`` — two-DOF ODE, where the ``dxr`` sum is emitted
- ``coupled._materialize_multiscale`` — the ``(p/tau)`` form
- ``proposers.sindy_proposer.SINDyProposer`` — ``f"{c}*{name}"`` formula strings
- ``advanced_search.HierarchicalSearch._expansions`` — ``f"{expr}*2"`` mutation

The last two are not grammar materialisers but assemble a coefficient onto an
operand by string formatting, so they belong to the same defect class and are
checked the same way: the emitted string must mean "the template applied to the
operand as a unit".

Output: ``benchmarks/term_builder_structure_audit.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from apedp_dyn.gates import dimensional_validity  # noqa: E402
from apedp_dyn.grammar.candidate_ir import (  # noqa: E402
    CandidateDynamics,
    SystemCandidate,
    Term,
)
from apedp_dyn.grammar.coupled import materialize_system  # noqa: E402
from apedp_dyn.grammar.mechanical_grammar import (  # noqa: E402
    _SCALED_TERMS,
    _TERM_BUILDERS,
    materialize,
)
from apedp_dyn.model import InputVariable, StateVariable  # noqa: E402

OUT = ROOT / "benchmarks" / "term_builder_structure_audit.json"

# Deliberately awkward, distinct, non-unit values: a coefficient that lands on
# the wrong addend cannot coincide with the correct value by accident.
SAMPLE_VALUES = (0.37, 0.71, 1.13, 2.29, 0.53, 1.71)


def classify(built: sp.Expr) -> str:
    """Structural class of a built expression.

    The classes exist because they fail differently: a ``sum`` is legitimate
    when its addends share units, while a ``quotient_with_sum`` manufactured
    from a literal ``1`` is refused by gate 1 for every dimensional state.
    """
    numerator, denominator = sp.fraction(sp.together(built))
    if denominator.atoms(sp.Add):
        return "quotient_with_sum"
    if numerator.atoms(sp.Add):
        return "sum"
    if built.is_Mul:
        return "product"
    if built.is_Pow or built.is_Symbol or built.is_Number:
        return "monomial"
    return "function_of_symbol"


class _VelocityDrivenPlant:
    """Minimal plant whose input carries velocity units.

    ``u_relax`` is ``U - V``. On the Duffing plant the input is a force, so the
    sum is dimensionally incoherent and the unit check refuses it for a reason
    that has nothing to do with formatting. A velocity drive makes the sum
    coherent, which is what lets the audit check that its coefficient covers
    *both* addends rather than the first.
    """

    states = (StateVariable("x", "meter", 0.0),
              StateVariable("v", "meter/second", 0.0))
    inputs = (InputVariable("u", "meter/second"),)


def binding_residual(rhs: str, parameter: str, *, value: float = 2.0,
                     expected: float | None = None) -> float | None:
    """``rhs(p=value) - expected*rhs(p=1)``; 0.0 means the binding is correct.

    Numeric rather than symbolic on purpose: the question is what the emitted
    text evaluates to, and a symbolic ``simplify`` can hide a shape it renamed.

    ``expected`` exists because a parameter is not always a numerator: the
    multiscale emission is ``(p/tau)*f``, so doubling ``tau`` must *halve* the
    value. Asserting a doubling there reports a correct expression as broken.
    """
    expr = sp.sympify(rhs)
    free = sorted(expr.free_symbols, key=str)
    p = sp.Symbol(parameter)
    if p not in free:
        return None
    values = {s: SAMPLE_VALUES[i % len(SAMPLE_VALUES)]
              for i, s in enumerate(free) if s != p}
    function = sp.lambdify(free, expr, modules=["numpy"])
    scaled = function(*[value if s == p else values[s] for s in free])
    single = function(*[1.0 if s == p else values[s] for s in free])
    factor = value if expected is None else expected
    return float(scaled) - factor * float(single)


def audit_builders() -> list[dict]:
    """Every builder: structure, materialisation, gate 1, coefficient binding."""
    from apedp_dyn.problems import duffing

    problem = duffing.load()
    X, V, U, XR, SX = (sp.Symbol(n) for n in ("gx", "gv", "gu", "gxr", "gL"))
    rows = []
    for name in _TERM_BUILDERS:
        row: dict = {"term": name}
        try:
            built = _TERM_BUILDERS[name](X, V, U, XR, SX)
        except ValueError as exc:  # UnavailableGrammarTerm is a ValueError
            rows.append({**row, "structure": "builder_refuses",
                         "refusal": str(exc), "gate1": "not_reached",
                         "coefficient_binds_whole_term": None})
            continue
        row["structure"] = classify(built)
        row["needs_remote_position"] = XR in built.atoms(sp.Symbol)
        row["expression"] = sp.sstr(built)
        # Did the builder make its dimensional coherence depend on a literal
        # constant? That is the difference between a legal sum (dxr, u_relax)
        # and a term that can only ever build for a dimensionless state.
        row["adds_a_literal_constant"] = any(
            arg.is_number for add in built.atoms(sp.Add) for arg in add.args)
        if row["needs_remote_position"]:
            # The single-DOF materialiser refuses remote terms, so its own
            # binding is exercised at the coupled site below.
            row["gate1"] = "not_reached"
            row["note"] = ("single-DOF problems have no remote position; the "
                           "coupled emission site is audited separately")
            rows.append(row)
            continue
        keyword = {"scale_parameter": "s0"} if name in _SCALED_TERMS else {}
        bounds = {"a0": (-1e6, 1e6)}
        if keyword:
            bounds["s0"] = (1e-9, 1e9)
        candidate = CandidateDynamics(
            lhs="xddot", terms=(Term(name, "a0", **keyword),),
            metadata={}, candidate_id=f"audit-{name}", parameter_bounds=bounds)
        # u_relax is a sum of the input and the velocity; only a plant whose
        # input shares the velocity's units can build it at all.
        plant = (_VelocityDrivenPlant() if name == "u_relax" else problem)
        row["plant"] = ("velocity_driven_probe" if name == "u_relax"
                        else "duffing")
        try:
            model = materialize(plant, candidate)
        except (TypeError, ValueError) as exc:
            # `materialize` derives the parameter unit with the same checker
            # gate 1 uses, so an incoherent Add is refused *before* the gate is
            # reached. The outcome is identical (the term cannot be built) and
            # the message is the same one gate 1 would give, so the bucket is
            # chosen from the message, not from where the exception surfaced.
            rows.append({**row, "gate1": "materialisation_raises",
                         "gate1_detail": str(exc),
                         "coefficient_binds_whole_term": None})
            continue
        rhs = sp.sstr(model.equations[1].sympy_expr)
        gate = dimensional_validity(model)
        row.update({
            "emitted_rhs": rhs,
            "parameter_units": model.parameters[0].units,
            "gate1": "pass" if gate.passed else "fail",
            "gate1_detail": gate.detail,
            "coefficient_binds_whole_term":
                _binds(binding_residual(rhs, "a0")),
            "residual": binding_residual(rhs, "a0"),
        })
        rows.append(row)
    return rows


def _binds(residual: float | None) -> bool | None:
    if residual is None:
        return None
    return math.isfinite(residual) and abs(residual) < 1e-9


def audit_coupled_site() -> dict:
    """The two-DOF ODE emission, where ``dxr`` is a sum."""
    from apedp_dyn.problems import coupled

    problem = coupled.load()
    # One term per equation. A second term would contribute a constant to the
    # doubling residual and make a correct emission look broken.
    equations = (
        CandidateDynamics(lhs="xddot", terms=(Term("dxr", "q1"),),
                          metadata={"family": "audit"}, candidate_id="do-0"),
        CandidateDynamics(lhs="xddot", terms=(Term("u", "q2"),),
                          metadata={"family": "audit"}, candidate_id="do-1"),
    )
    model = materialize_system(
        problem, SystemCandidate(equations=equations,
                                 metadata={"dae_mode": "none"},
                                 candidate_id="audit-ode"))
    rhs = sp.sstr(model.equations[1].sympy_expr)
    residual = binding_residual(rhs, "q1")
    return {
        "site": "coupled._materialize_ode",
        "operand_kind": "sum (dxr = XR - X)",
        "emitted_rhs": rhs,
        "coefficient_binds_whole_operand": _binds(residual),
        "residual": residual,
        "counterexample_before_fix": {
            "form": "f\"({p}*{sstr(built)})\" without the inner parentheses",
            "emitted": "q1*x2 - x1",
            "evaluates_as": "the remote position enters with coefficient 1",
        },
    }


def audit_multiscale_site() -> dict:
    """The ``(p/tau)`` emission: two coefficient-like factors on one operand."""
    from apedp_dyn.problems import coupled

    problem = coupled.load()
    equations = (
        CandidateDynamics(lhs="xddot",
                          terms=(Term("x_f", "q0"),),
                          metadata={"family": "audit",
                                    "fast_variables": ("x_f", "v_f")},
                          candidate_id="ms-0"),
        CandidateDynamics(lhs="xddot", terms=(Term("x_s", "q2"),),
                          metadata={"family": "audit",
                                    "slow_variables": ("x_s", "v_s")},
                          candidate_id="ms-1"),
    )
    model = materialize_system(
        problem,
        SystemCandidate(equations=equations,
                        metadata={"dae_mode": "multi_timescale",
                                  "timescales": ("fast", "slow")},
                        candidate_id="audit-ms"))
    rhs = sp.sstr(model.equations[1].sympy_expr)
    residual = binding_residual(rhs, "q0")
    # The factor is 1/tau_fast, so doubling tau must halve the value.
    tau_residual = binding_residual(rhs, "tau_fast", value=2.0, expected=0.5)
    return {
        "site": "coupled._materialize_multiscale",
        "operand_kind": "monomial under a (p/tau) factor",
        "emitted_rhs": rhs,
        "coefficient_binds_whole_operand": _binds(residual),
        "residual": residual,
        "outer_factor_binds": _binds(tau_residual),
        "outer_factor_residual": tau_residual,
        "defect_not_present": (
            "this site always parenthesised both the operand and the (p/tau) "
            "factor, so it never carried the unparenthesised form; it is "
            "audited because a coefficient-like factor divided by tau is the "
            "same binding question with a different arithmetic"),
    }


def audit_sindy_site() -> dict:
    """``f"{c}*{name}"`` — a library column whose name is a sum.

    Today every library column is a product, so this is latent rather than
    active. It is audited with a synthetic sum-valued column because the
    formatting is what decides the outcome: with the coefficient prefixed
    without parentheses the emitted formula is ``c*x0 + x1``, which claims the
    second variable has coefficient 1.
    """
    from apedp_dyn.proposers.sindy_proposer import SINDyProposer

    class _SumColumnProposer(SINDyProposer):
        """Library with one column named after a sum, coefficient pinned to 2."""

        def build_library(self, X: np.ndarray, poly_order: int = 3):
            return (np.column_stack([X[:, 0] + X[:, 1], np.ones(len(X))]),
                    ["x0 + x1", "1"])

        def sparse_regression(self, Theta: np.ndarray, dX: np.ndarray):
            Xi = np.zeros((Theta.shape[1], dX.shape[1]))
            Xi[0, :] = 2.0
            return Xi, [0]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    dX = np.column_stack([2.0 * (X[:, 0] + X[:, 1])] * 2)
    proposals = list(_SumColumnProposer().fit_and_propose(X, dX))
    formula = proposals[0].formula if proposals else ""
    parsed = sp.sympify(formula) if formula else None
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
    c0 = parsed.coeff(x0) if parsed is not None else None
    c1 = parsed.coeff(x1) if parsed is not None else None
    return {
        "site": "proposers.sindy_proposer.SINDyProposer.fit_and_propose",
        "operand_kind": "library column named for a sum ('x0 + x1')",
        "emitted_formula": formula,
        "coefficient_of_first_addend": str(c0),
        "coefficient_of_second_addend": str(c1),
        "coefficient_binds_whole_operand":
            parsed is not None and sp.simplify(c0 - c1) == 0,
        "counterexample_before_fix": {
            "form": "f\"{c:.6g}*{t}\" without parentheses",
            "emitted": "2*x0 + x1 for a coefficient of 2",
            "evaluates_as": "coefficient 2 on the first addend, 1 on the second",
        },
    }


def audit_expansion_site() -> dict:
    """``f"{expr}*2"`` and friends, where ``expr`` is a whole seed law.

    Seeds are frequently sums — the coarse templates emit ``v1/(1+v2)`` and the
    extension set wraps surviving forms — so an unparenthesised mutation changes
    which family is being searched, not merely how it prints.
    """
    from apedp import advanced_search as module

    search = object.__new__(module.HierarchicalSearch)
    operand = "x0 + 1"
    var = "x1"
    expected_symbol = sp.sympify(f"({operand})")
    checks = []
    for template_set in ("classic", "extended"):
        search.template_set = template_set
        produced = list(search._expansions(operand, var))
        checks.append({**audit_expansion_forms(
            produced, template_set, operand, var), "template_set": template_set})
    return {
        "site": "advanced_search.HierarchicalSearch._expansions",
        "operand_kind": "seed law, here a sum 'x0 + 1'",
        "operand_symbol": sp.sstr(expected_symbol),
        "template_sets": checks,
        "coefficient_binds_whole_operand": all(
            c["all_expansions_mean_the_template_on_the_operand"] for c in checks),
        "counterexample_before_fix": {
            "form": "f\"{expr}*2\" without parentheses",
            "emitted": "x0 + 1*2",
            "evaluates_as": "x0 + 2, i.e. the doubling never reached the seed",
        },
    }


def audit_expansion_forms(produced: list[str], template_set: str,
                          operand: str, var: str) -> dict:
    """Does each emitted string mean "the template applied to the operand"?"""
    parenthesised = f"({operand})"
    if template_set == "classic":
        templates = ("{e}**2", "{e}*2", "2*{e}", "sin({e})", "{e}+1")
        intents = [template.format(e=parenthesised) for template in templates]
    else:
        from apedp import advanced_search as module

        templates = (*module._EXTENDED_EXPANSIONS, "{e}*{v}", "{e}/(1+{v})")
        intents = [template.format(e=parenthesised, v=var)
                   for template in templates]
    intents.append(operand)          # the extended set keeps the seed itself
    found = []
    for text in produced:
        try:
            got = sp.sympify(text)
        except Exception as exc:  # noqa: BLE001 - a parse failure is a finding
            found.append({"expression": text, "matches_intent": False,
                          "error": str(exc)})
            continue
        matches = any(sp.simplify(got - sp.sympify(intent)) == 0
                      for intent in intents)
        found.append({"expression": text, "matches_intent": bool(matches)})
    return {
        "expansions": found,
        "all_expansions_mean_the_template_on_the_operand": all(
            f["matches_intent"] for f in found) and bool(found),
    }


#: Directories swept for coefficient-prefix string assembly. `sandbox/` holds a
#: vendored upstream tree and `scratch/` holds one-off analysis scripts; neither
#: emits an expression the repo records or adjudicates.
_SWEEP_DIRS = ("apedp", "apedp_dyn", "scripts", "benchmarks")
_SWEEP_SKIP_PARTS = ("sandbox", "_qualtran_src", "__pycache__", "scratch")

#: This file carries the pattern as data in its registry and docstrings, so it
#: would report itself. The tests that read the record are outside the sweep.
_SWEEP_SKIP_FILES = ("scripts/audit_term_builder_structure.py",)
_VOCABULARY = {
    "parenthesised": "the operand reaches the template already wrapped",
    "atomic_operand": "the operand is a name that cannot contain a top-level "
                      "operator, or the field itself calls atom()",
    "numeric_product_label": "the string names a product that is computed "
                             "numerically; no coefficient is formatted onto it",
    "diagnostic_text": "prose in a message, docstring or test assertion",
    "guarantee_itself": "the formatter that supplies the guarantee",
    "converted": "was formatted without parentheses and now routes through "
                 "apedp_dyn.grammar.term_format",
}

#: A format field, then ``*``, then another format field: the coefficient-prefix
#: shape. Whether the second field is wrapped in parentheses is the question.
_COEFFICIENT_PREFIX = re.compile(r"\}\s*\*\s*(?P<paren>\()?\{[A-Za-z_]")


#: One entry per site the sweep finds, and one per site that was fixed out of the
#: sweep's reach. The test fails on a site that is neither, so a new
#: coefficient-prefix format string has to be adjudicated before it is reviewed.
STRING_ASSEMBLY_SITES: tuple[dict, ...] = (
    # --- the grammar materialisers, all parenthesised by the original fix ---
    {"site": "mechanical_grammar.materialize", "file": "apedp_dyn/grammar/mechanical_grammar.py",
     "needle": 'parts.append(f"({t.parameter}*({sp.sstr(rexpr)}))")',
     "verdict": "parenthesised", "note": "single-DOF emission, fixed in the first pass"},
    {"site": "coupled._materialize_ode", "file": "apedp_dyn/grammar/coupled.py",
     "needle": 'parts.append(f"({t.parameter}*({sp.sstr(built)}))")',
     "verdict": "parenthesised", "note": "two-DOF ODE and DAE emissions; both carry dxr"},
    {"site": "coupled._materialize_constrained", "file": "apedp_dyn/grammar/coupled.py",
     "needle": 'constraint_parts.append(f"({t.parameter}*({sp.sstr(built)}))")',
     "verdict": "parenthesised", "note": "constraint row"},
    {"site": "term_format.scaled", "file": "apedp_dyn/grammar/term_format.py",
     "needle": 'return f"{coefficient}*{atom(operand)}"',
     "verdict": "guarantee_itself",
     "note": "the operand is passed through atom(), so the parentheses it "
             "renders are the guarantee rather than the hazard"},
    # --- converted: the literal format is gone, the module routes through scaled() ---
    {"site": "proposers.sindy_proposer.SINDyProposer.fit_and_propose",
     "file": "apedp_dyn/proposers/sindy_proposer.py",
     "needle": 'formula_parts.append(f"{c:.6g}*({t})")',
     "verdict": "parenthesised",
     "note": "converted in the first pass; the parens are still in the format"},
    {"site": "proposers.sparse_microstructure_proposer",
     "file": "apedp_dyn/proposers/sparse_microstructure_proposer.py",
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import scaled", "scaled("),
     "note": "ElasticNet column names are data-derived text"},
    {"site": "proposers.regime_switching_proposer",
     "file": "apedp_dyn/proposers/regime_switching_proposer.py",
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import scaled", "scaled("),
     "note": "two branches, same column set"},
    {"site": "proposers.rime_ad_proposer._make_tv_composite_candidate",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "verdict": "converted",
     "markers": ("slope0 = scaled(", "slope2 = scaled("),
     "note": "floor, linear and squared terms each bind the column name"},
    {"site": "proposers.distributed_lag_interaction_proposer",
     "file": "apedp_dyn/proposers/distributed_lag_interaction_proposer.py",
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import scaled", "scaled("),
     "note": "selected column names come from the caller's frame"},
    {"site": "qed.library.TermLibrary.render", "file": "apedp_dyn/qed/library.py",
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import scaled", "scaled("),
     "note": "term names are caller-supplied by TermLibrary.from_specs"},
    # --- still present, operand atomic or field calls atom() ---
    {"site": "proposers.invariant_transfer_proposer",
     "file": "apedp_dyn/proposers/invariant_transfer_proposer.py",
     "needle": 'f"{c:.4f}*{atom(feat)}^{p}"',
     "verdict": "atomic_operand",
     "note": "the field calls atom(), so the power applies to the whole name"},
    {"site": "proposers.stem_proposers.CompartmentalSirProposer",
     "file": "apedp_dyn/proposers/stem_proposers.py",
     "needle": "{atom(i_col)}",
     "verdict": "atomic_operand",
     "note": "compartment column names are caller-supplied; both uses call atom()"},
    {"site": "proposers.rime_ad_proposer._propose_multivariate",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needle": 'col_names.append(f"{c1}*{c2}")',
     "verdict": "numeric_product_label",
     "note": "names an interaction column whose values are multiplied numerically"},
    {"site": "proposers.rime_ad_proposer (rational pair / DAG)",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needle": "f\"{q_max:.6g} * ({feat_name} / ({c_best:.6g} + {feat_name})",
     "verdict": "parenthesised", "note": "Langmuir and Hill forms"},
    {"site": "proposers.rime_proposer", "file": "apedp_dyn/proposers/rime_proposer.py",
     "needle": "f\"{scale:+.6g} * ({p['equation']}) {intercept:+.6g}\"",
     "verdict": "parenthesised", "note": "whole discovered equation is the operand"},
    {"site": "proposers.rime_ad_proposer (Hill form)",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needle": 'f"{q_max:.6g} * ({squared} / ',
     "verdict": "parenthesised",
     "note": "cooperative Hill quotient: the whole quotient is the operand, "
             "and the squared column inside it is atom()-bound"},
    {"site": "proposers.rime_ad_proposer (multi-rational basis)",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needles": ("{coef[1]:.4f}*({reciprocal})",
                 "{coef[2]:.4f}*({reciprocal}^2)"),
     "verdict": "parenthesised",
     "note": "the coefficient multiplies the reciprocal as a unit; "
             "`reciprocal` is 1/atom(feat_name), so the denominator is "
             "bound too"},
    {"site": "pde.higher_order_engine", "file": "apedp_dyn/pde/higher_order_engine.py",
     "needle": 'f"{c:+.4f}*{name}"',
     "verdict": "atomic_operand",
     "note": "term names are built in the same module (u, u_sq, u*u_x, (u_x)^2, "
             "laplacian(u)); each is a symbol or a product"},
    # --- atomic operands built in-module, recorded rather than guessed at ---
    {"site": "epistemic.high_dim_pde.terms_str", "file": "apedp_dyn/epistemic/high_dim_pde.py",
     "needle": 'f"{c:+.4f} * {name}"',
     "verdict": "atomic_operand",
     "note": "three discovery paths share the line; term names are library "
             "labels (u, u_sq, u_cube, u_x, u_xx)"},
    {"site": "model_synthesis._s1_templates", "file": "apedp/model_synthesis.py",
     "needle": 'f"A * {_o} * {_e}"',
     "verdict": "atomic_operand",
     "note": "onsets and extinctions are atomic fragments (x**1.5, exp(-C * x)); "
             "the sum-valued saturations are always parenthesised where used"},
    {"site": "model_synthesis._s1_templates (onset x extinction)", "file": "apedp/model_synthesis.py",
     "needle": 'f"A * {o} * {e}"',
     "verdict": "atomic_operand",
     "note": "same template table; saturation is absent from this branch, and "
             "where it is present it is written `({s})`"},
    {"site": "run_langmuir_referee_demo", "file": "benchmarks/run_langmuir_referee_demo.py",
     "needle": 'f"{c:+.4f}*{t}"',
     "verdict": "atomic_operand",
     "note": "display only; coupled discovery terms are x and x*y"},
    {"site": "quantum_signal.rotation_angle", "file": "apedp_dyn/quantum_compilation/quantum_signal.py",
     "needle": 'f"ry({alpha}*{name}{beta:+}) q{index}"',
     "verdict": "atomic_operand",
     "note": "an affine rotation angle: alpha*theta + beta, emitted inside the "
             "gate's own parentheses; `name` is a rotation-parameter label"},
    # --- parenthesised emission sites outside the grammar ---
    {"site": "ui.server.promoted_equation (theta)", "file": "apedp_dyn/ui/server.py",
     "needle": 'f"ddot(theta) = {coef:.4g} * ({expr})',
     "verdict": "parenthesised",
     "note": "third live site of this class, found by the sweep and fixed: it "
             "emitted the coefficient onto the first addend of a canonical sum"},
    {"site": "ui.server.promoted_equation (state)", "file": "apedp_dyn/ui/server.py",
     "needle": 'f"d({last_state})/dt = {coef:.4g} * ({can_expr})"',
     "verdict": "parenthesised", "note": "already carried the parentheses"},
    {"site": "ui.server.energy_expression", "file": "apedp_dyn/ui/server.py",
     "needle": 'f"E = {coef:.4g} * ({expr})"',
     "verdict": "parenthesised", "note": "already carried the parentheses"},
    {"site": "discovery_l2.ode_engine", "file": "apedp_dyn/discovery_l2/ode_engine.py",
     "needle": 'f"d({target_state})/dt = {coef:.4g} * ({expr_str})"',
     "verdict": "parenthesised", "note": "whole canonical expression is the operand"},
    {"site": "discovery_l2.multiterm_engine (DAG node)",
     "file": "apedp_dyn/discovery_l2/multiterm_engine.py",
     "needle": 'f"theta_{j+1}*({node.to_str()})"',
     "verdict": "parenthesised", "note": "per-term parameter onto a whole node"},
    {"site": "discovery_l2.multiterm_engine (term)",
     "file": "apedp_dyn/discovery_l2/multiterm_engine.py",
     "needle": 'f"{val:+.4g} * ({t_obj.name})"',
     "verdict": "parenthesised", "note": "term name is the operand"},
    {"site": "epistemic.coordinate_transmuter",
     "file": "apedp_dyn/epistemic/coordinate_transmuter.py",
     "needle": "({atom(feature_name)} ** {k_power:.4g})",
     "verdict": "parenthesised",
     "note": "both sites (with and without the barrier factor); the "
             "exponent is applied through atom()"},
    {"site": "epistemic.rime_repair_engine (interaction)",
     "file": "apedp_dyn/epistemic/rime_repair_engine.py",
     "needle": 'f"({a}*{b})"',
     "verdict": "numeric_product_label",
     "note": "names an interaction column; the values are multiplied numerically"},
    {"site": "epistemic.rime_repair_engine (repair)",
     "file": "apedp_dyn/epistemic/rime_repair_engine.py",
     "needle": 'f"{alpha:+.6g}*({parent.formula}){beta:+.6g}"',
     "verdict": "parenthesised", "note": "the repaired parent formula is the operand"},
    {"site": "epistemic.rime_repair_engine (repair with term label)",
     "file": "apedp_dyn/epistemic/rime_repair_engine.py",
     "needle": 'f"{alpha:+.6g}*({parent.formula}){beta:+.6g}*({term_label})',
     "verdict": "parenthesised", "note": "every operand is parenthesised in the format"},
    {"site": "epistemic.rime_repair_engine (saturating repair)",
     "file": "apedp_dyn/epistemic/rime_repair_engine.py",
     "needle": 'f"({a_param:+.6g}*({parent.formula})) / (1 + {b_param:.6g}*',
     "verdict": "parenthesised", "note": "numerator and denominator both wrapped"},
    {"site": "advanced_search.ExpressionTree.to_string", "file": "apedp/advanced_search.py",
     "needle": 'f"({self.to_string(node.children[0])} * {self.to_string(node.children[1])})"',
     "verdict": "parenthesised", "note": "every child of the printer is parenthesised"},
    {"site": "advanced_search.HierarchicalSearch._expansions", "file": "apedp/advanced_search.py",
     "needle": 'f"{operand}*{var}"',
     "verdict": "parenthesised",
     "note": "`operand` is the pre-parenthesised seed; fixed in the first pass"},
    {"site": "discovery_benchmark", "file": "apedp/discovery_benchmark.py",
     "needle": 'f"{c:.6g}*({t})"',
     "verdict": "parenthesised", "note": "screened design column"},
    {"site": "discovery._core.cross_family_select (exp)", "file": "apedp/discovery/_core.py",
     "needle": 'f"exp({coef[0]:.4f}*({name}))"',
     "verdict": "parenthesised",
     "note": "fixed by the sweep: the exponential's argument is the whole "
             "operand, so a sum-named column cannot shift the exponent"},
    {"site": "family_detection.cross_family_select (exp)", "file": "apedp/family_detection.py",
     "needle": 'f"exp({coef[0]:.4f}*({name}))"',
     "verdict": "parenthesised", "note": "same fix, duplicated implementation"},
    # --- strings that name a numeric product or say nothing at all ---
    {"site": "discovery._core design labels", "file": "apedp/discovery/_core.py",
     "needle": 'feats[f"{a}*{b}"] = cols[a]',
     "verdict": "numeric_product_label",
     "note": "two sites, spelled `cols[a] * cols[b]` and `cols[a]*cols[b]`; the "
             "label names a column whose values are multiplied "
             "numerically, and no coefficient is formatted onto it"},
    {"site": "family_detection design labels", "file": "apedp/family_detection.py",
     "needle": 'feats[f"{a}*{b}"] = cols[a] * cols[b]',
     "verdict": "numeric_product_label", "note": "same construction"},
    {"site": "feature_selection design labels", "file": "apedp/feature_selection.py",
     "needle": 'feats[f"{a}*{b}"] = cols[a] * cols[b]',
     "verdict": "numeric_product_label", "note": "same construction"},
    {"site": "de_arm.dependency_screen library labels", "file": "apedp_dyn/de_arm/dependency_screen.py",
     "needle": 'lib[f"{state}*{parent}"]',
     "verdict": "numeric_product_label", "note": "reduced-library column name"},
    {"site": "de_arm.dependency_screen kept-pair labels",
     "file": "apedp_dyn/de_arm/dependency_screen.py",
     "needle": 'lib[f"{a}*{b}"] = kept_vals[a] * kept_vals[b]',
     "verdict": "numeric_product_label", "note": "reduced-library column name"},
    {"site": "core.DimensionArithmetic.batch_validate_power", "file": "apedp/core.py",
     "needle": "Dimension exponent overflow in power operation",
     "verdict": "diagnostic_text", "note": "an OverflowError message"},
    {"site": "quantum_compilation.referee", "file": "apedp_dyn/quantum_compilation/referee.py",
     "needle": "max_x * shots_per_x",
     "verdict": "diagnostic_text", "note": "a budget message"},
    {"site": "test_viable_optimization_path receipt check",
     "file": "scripts/test_viable_optimization_path.py",
     "needle": '{q0} * {e0} = {q0 * e0}',
     "verdict": "diagnostic_text", "note": "an assertion message"},
)


def scan_coefficient_prefix_sites() -> list[dict]:
    """Every ``...}*{operand}`` format site in library code, with its shape."""
    found: list[dict] = []
    for base in _SWEEP_DIRS:
        for path in sorted((ROOT / base).rglob("*.py")):
            if any(part in _SWEEP_SKIP_PARTS for part in path.parts):
                continue
            if path.relative_to(ROOT).as_posix() in _SWEEP_SKIP_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if 'f"' not in line and "f'" not in line:
                    continue
                for match in _COEFFICIENT_PREFIX.finditer(line):
                    found.append({
                        "file": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "operand_parenthesised": bool(match.group("paren")),
                        "source": line.strip(),
                    })
    return found


def audit_string_assembly() -> dict:
    """Adjudicate every coefficient-prefix site outside the grammar materialisers."""
    found = scan_coefficient_prefix_sites()
    matched: set[int] = set()
    rows: list[dict] = []
    for entry in STRING_ASSEMBLY_SITES:
        needles = entry.get("needles",
                            (entry["needle"],) if entry.get("needle") else ())
        hits = [index for index, site in enumerate(found)
                if site["file"] == entry["file"]
                and any(n and n in site["source"] for n in needles)]
        matched.update(hits)
        row = {**entry, "sites": [found[index] for index in hits]}
        if entry["verdict"] == "converted":
            path = ROOT / entry["file"]
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            row["conversion_marker_present"] = bool(entry["markers"]) and all(
                marker in text for marker in entry["markers"])
        rows.append(row)
    unrecorded = [site for index, site in enumerate(found) if index not in matched]
    stale = [row["site"] for row in rows
             if not row["sites"] and row["verdict"] != "converted"]
    return {
        "question": (
            "is a coefficient formatted onto an operand that is not a single "
            "syntactic unit, and if so does the operand get parenthesised? "
            "answered by the source shape and by the operand's provenance"),
        "verdict_vocabulary": _VOCABULARY,
        "sites_found": found,
        "registry": rows,
        "unrecorded_sites": unrecorded,
        "stale_registry_entries": stale,
        "parenthesised": [site for site in found
                          if site["operand_parenthesised"]],
        "unparenthesised": [site for site in found
                            if not site["operand_parenthesised"]],
    }


#: An operator that binds tighter than ``*``: a power, a quotient, a call or a
#: left factor. The coefficient sweep looked for ``}*{``; these are its
#: siblings, and they fail the same way -- the operator reaches the first factor
#: of an operand that is a sum, and the rest of the expression falls out.
_SHAPES = ("power", "division", "call_argument", "left_factor")

_POWER_ON_FIELD = re.compile(r"\{(?P<base>[^{}]*)\}\s*\*\*")
_DIVIDED_BY_FIELD = re.compile(r"(?:\b1|\})\s*/\s*\{(?P<den>[^{}]*)\}")
#: `sin({x})`: the call's own parentheses delimit the argument, so this is safe
#: when the field *is* the argument and hazardous only when an operator follows
#: it before the closing paren (`sin({x}*2)` on `a + b` is `sin(a + b*2)`).
_CALL_ARG_FIELD = re.compile(
    r"(?P<fn>sin|cos|tan|tanh|exp|log|sqrt|abs|arctan|sinh|cosh)\(\s*"
    r"(?P<sign>[-+])?\s*\{(?P<arg>[^{}]*)\}(?P<after>[^)]*)")
#: `{x}*2` / `{x}/2`: the mirror of the coefficient prefix. `**` is excluded
#: because the power shape already covers it.
_FIELD_AS_LEFT_FACTOR = re.compile(
    r"\{(?P<field>[^{}]*)\}\s*(?P<op>[*/])\s*(?P<rhs>[A-Za-z0-9_.]+)")

_POWER_DIVISION_VOCABULARY = {
    "converted": "the literal shape is gone and the module builds the operand "
                 "through term_format.atom/powered/over",
    "retired": "the string was rewritten (bracketed or renamed) and the literal "
               "shape no longer occurs; the entry records what was there",
    "parenthesised": "the field itself calls atom()/over(), so the operand "
                     "reaches the operator already wrapped",
    "formatted_operand": "the field holds a pre-formatted operand that was "
                         "already bracketed by over()",
    "atomic_operand": "the base or denominator is a literal or a name built in "
                      "the same module from a fixed table",
    "argument_is_whole_call": "the call's own parentheses delimit the field as "
                              "the entire argument, so no extra bracket is needed",
    "numeric_fraction_text": "two integers rendered as a fraction for display",
    "non_arithmetic_text": "a path, URL, object key, progress counter or "
                           "diagnostic message; the slash is not division",
    "guarantee_itself": "term_format.powered / term_format.over, which supply "
                        "the guarantee",
}


#: A field with a numeric format spec renders a number, so it can never carry a
#: top-level operator: `{eps:.4g}**2` is arithmetic, `{name}**2` is a hazard.
_NUMERIC_FIELD = re.compile(r":\s*[<>^=]?[+\- ]?\.?\d*[dfeEgG%]")

#: A right-hand side that is a number, so `{operand}*2` is the mirror of the
#: coefficient prefix. `{a}*{b}` is not: both sides are fields, and the
#: coefficient sweep already adjudicated those as numeric labels.
_NUMERIC_RHS = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

#: A field named like an operand. The hazard needs a field that can hold an
#: expression; a progress counter, a markdown label and a formatted number all
#: render something that cannot. The repo's operand-shaped fields are named for
#: what they are, which is what makes this decidable without parsing the file.
_OPERAND_WORDS = frozenset(
    ("name", "feat", "feature", "variable", "var", "column", "col", "term",
     "expr", "operand", "seed"))


def _names_an_operand(field: str) -> bool:
    """Whether the field is named like an operand rather than a counter.

    A substring test would read `s["seeds"]` as an operand; splitting every
    identifier on `_` keeps `feat_name` an operand and `seeds` a count.
    """
    text = field.strip()
    if any(marker in text
           for marker in ("tokens[", "reciprocal", "atom(", "to_string(")):
        return True
    words = {part for ident in re.findall(r"[A-Za-z_]+", text)
             for part in ident.split("_") if part}
    return bool(words & _OPERAND_WORDS)


def _field_is_formatted(field: str) -> bool:
    return field.strip().startswith(("atom(", "over(", "powered("))


def _scope(field: str, line: str, matched: str) -> tuple[str, str]:
    """Where a site sits: ``in_scope``, ``numeric_operand`` or ``display_only``."""
    if _NUMERIC_FIELD.search(field):
        return "numeric_operand", (
            "the field carries a numeric format spec, so it renders a number "
            "and cannot contain a top-level operator")
    if _names_an_operand(field):
        return "in_scope", ""
    return "display_only", (
        "the field is not named like an operand: a counter, a path, a markdown "
        "label or a shape tuple, so no expression is being built here")


def scan_power_and_division_sites() -> list[dict]:
    """Every f-string that raises a field to a power or divides by one."""
    found: list[dict] = []
    for base in _SWEEP_DIRS:
        for path in sorted((ROOT / base).rglob("*.py")):
            if any(part in _SWEEP_SKIP_PARTS for part in path.parts):
                continue
            if path.relative_to(ROOT).as_posix() in _SWEEP_SKIP_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if 'f"' not in line and "f'" not in line:
                    continue
                for match in _POWER_ON_FIELD.finditer(line):
                    scope, reason = _scope(match.group("base"), line, match.group(0))
                    found.append({
                        "file": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "shape": "power",
                        "field": match.group("base"),
                        "field_is_formatted": _field_is_formatted(match.group("base")),
                        "scope": scope,
                        "scope_reason": reason,
                        "source": line.strip(),
                    })
                for match in _DIVIDED_BY_FIELD.finditer(line):
                    scope, reason = _scope(match.group("den"), line, match.group(0))
                    found.append({
                        "file": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "shape": "division",
                        "field": match.group("den"),
                        "field_is_formatted": _field_is_formatted(match.group("den")),
                        "scope": scope,
                        "scope_reason": reason,
                        "source": line.strip(),
                    })
                for match in _CALL_ARG_FIELD.finditer(line):
                    after = match.group("after").strip()
                    sign = match.group("sign")
                    scope, reason = _scope(match.group("arg"), line, match.group(0))
                    # A field that is the entire argument is safe: the call's
                    # parentheses already bind it. Two cases are not: an
                    # operator before the closing paren (`sin({x}*2)` on
                    # `a + b` is `sin(a + b*2)`), and a leading sign
                    # (`exp(-{x})` is `exp(-a + b)`, the negation reaching only
                    # the first addend).
                    whole = after == "" and sign is None
                    if scope != "numeric_operand" and not whole and (
                            sign is not None or (after and after[0] in "*/")):
                        scope, reason = "in_scope", ""
                    found.append({
                        "file": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "shape": "call_argument",
                        "field": match.group("arg"),
                        "field_is_formatted": _field_is_formatted(match.group("arg")),
                        "argument_is_whole_call": whole,
                        "trailing": (sign or "") + after[:12],
                        "scope": scope,
                        "scope_reason": reason,
                        "source": line.strip(),
                    })
                for match in _FIELD_AS_LEFT_FACTOR.finditer(line):
                    scope, reason = _scope(match.group("field"), line, match.group(0))
                    if scope != "numeric_operand" and _NUMERIC_RHS.match(match.group("rhs")):
                        # `{x}*2` is arithmetic on the operand. `{a}*{b}` is
                        # not: both sides are fields and the coefficient
                        # sweep already adjudicated that as a numeric label.
                        scope, reason = "in_scope", ""
                    found.append({
                        "file": path.relative_to(ROOT).as_posix(),
                        "line": lineno,
                        "shape": "left_factor",
                        "field": match.group("field"),
                        "field_is_formatted": _field_is_formatted(match.group("field")),
                        "argument_is_whole_call": None,
                        "trailing": match.group("op") + match.group("rhs")[:10],
                        "scope": scope,
                        "scope_reason": reason,
                        "source": line.strip(),
                    })
    return found


#: One entry per site. ``needles`` matches several lines of the same file when
#: the reason is identical, so a file does not need an entry per f-string.
POWER_DIVISION_SITES: tuple[dict, ...] = (
    {"site": "term_format.powered / term_format.over",
     "file": "apedp_dyn/grammar/term_format.py",
     "needles": ('f"{atom(operand)}**{exponent}"',
                 'f"{atom(numerator)}/{atom(denominator)}"'),
     "verdict": "guarantee_itself",
     "note": "the two formatters that supply the guarantee: the base and the "
             "denominator are passed through atom(), so a sum arrives wrapped"},
    {"site": "advanced_search.ExpressionTree.to_string",
     "file": "apedp/advanced_search.py",
     "needles": ("to_string(node.children[0])",),
     "verdict": "parenthesised",
     "note": "the printer wraps every binary node it renders, so each child "
             "arrives as one unit before the / or ** is applied"},
    {"site": "advanced_search.HierarchicalSearch._expansions",
     "file": "apedp/advanced_search.py",
     "needles": ('f"{operand}**2"',),
     "verdict": "parenthesised",
     "note": "`operand = f\"({expr})\"` on the line above; the seed is wrapped "
             "once and every template is applied to that"},
    {"site": "discovery._core reciprocal power labels",
     "file": "apedp/discovery/_core.py",
     "needles": ('f"{reciprocal}**{q}"',),
     "verdict": "formatted_operand",
     "note": "`reciprocal` holds over('1', k), which already bracketed a "
             "non-atomic column name; the power then applies to that bracket"},
    {"site": "discovery._core power-law render",
     "file": "apedp/discovery/_core.py",
     "needles": ("powered(name, f\"({p})\")", "powered(name, str(round(p)))"),
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import atom, over, powered",
                 "powered("),
     "note": "recombine_power_law joins its terms with `*`, so the exponent "
             "and the join both need the name bound as a unit"},
    {"site": "feature_selection reciprocal power labels",
     "file": "apedp/feature_selection.py",
     "needles": ('f"{reciprocal}**{q}"',),
     "verdict": "formatted_operand",
     "note": "the base is over('1', k)'s output, already bracketed"},
    {"site": "feature_selection hypothesis atom set",
     "file": "apedp/feature_selection.py",
     "needles": ('over("1", variable)', "powered(variable,"),
     "shapes": ("power", "division"),
     "verdict": "converted",
     "markers": ("from apedp_dyn.grammar.term_format import atom, over, powered",
                 "powered("),
     "note": "the atom set is a list of expression strings built from a "
             "caller-supplied variable name; a reciprocal and a square were "
             "formatted straight onto it"},
    {"site": "epistemic.coordinate_transmuter power law",
     "file": "apedp_dyn/epistemic/coordinate_transmuter.py",
     "needles": ("({atom(feature_name)} ** ",),
     "verdict": "parenthesised",
     "note": "the exponent is applied through atom(); the surrounding "
             "parentheses alone would not have helped, because `(x ** 2)` with "
             "x a sum still raises only its last addend"},
    {"site": "epistemic.coordinate_transmuter tunneling exponential",
     "file": "apedp_dyn/epistemic/coordinate_transmuter.py",
     "needles": ("exp(-{c_barrier:.6g} / {atom(feature_name)})",),
     "verdict": "parenthesised",
     "note": "the barrier is a reciprocal of a caller-supplied feature name"},
    {"site": "proposers.rime_ad_proposer multi-rational basis",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needles": ('f"1/{atom(feat_name)}"',),
     "verdict": "parenthesised",
     "note": "the reciprocal's denominator goes through atom(), so the whole "
             "column is divided, not its first addend"},
    {"site": "proposers.rime_ad_proposer Hill form",
     "file": "apedp_dyn/proposers/rime_ad_proposer.py",
     "needles": ('squared = f"{atom(feat_name)}**2"',),
     "verdict": "parenthesised",
     "note": "the cooperative Hill term squares the column as a unit"},
    {"site": "record_epo_ops_fixture output path",
     "file": "scripts/record_epo_ops_fixture.py",
     "needles": ("{FIXTURES.relative_to(ROOT)}/{name}",),
     "verdict": "non_arithmetic_text",
     "note": "a filesystem path being printed; the slash separates directories"},
    {"site": "run_quantum_all_equations_benchmark markdown table",
     "file": "benchmarks/run_quantum_all_equations_benchmark.py",
     "needles": ("| **{i}** |", "| **{row['name']}** |"),
     "verdict": "non_arithmetic_text",
     "note": "`**` is markdown bold around a cell, not a power: the field is a "
             "row index or a name rendered into a table"},
    {"site": "trig_birth_fit_planck_benchmark counters",
     "file": "benchmarks/trig_birth_fit_planck_benchmark.py",
     "needles": ("{name}: pin exact", "small exact {sm['exact_counts']}"),
     "verdict": "non_arithmetic_text",
     "note": "counts over seeds in a printed line, not a quotient"},
    {"site": "grow_patent_corpus ratio_of_words",
     "file": "scripts/grow_patent_corpus.py",
     "needles": ('expression = f"{tokens[0]}/{tokens[1]}"',),
     "verdict": "atomic_operand",
     "note": "_strict_ratio_tokens admits only [A-Za-z][A-Za-z0-9_]{0,23}, so "
             "each operand is one identifier by construction"},
    # --- function arguments: safe when the field is the whole argument ---
    {"site": "advanced_search.ExpressionTree.to_string (calls)",
     "file": "apedp/advanced_search.py",
     "needles": ("sin({self.to_string(node.children[0])})",
                 "cos({self.to_string(node.children[0])})",
                 "exp({self.to_string(node.children[0])})",
                 "log({self.to_string(node.children[0])})",
                 "sqrt({self.to_string(node.children[0])})",
                 "abs({self.to_string(node.children[0])})"),
     "verdict": "argument_is_whole_call",
     "note": "the field is the entire argument and to_string already wraps every "
             "binary node, so the call's parentheses bind it twice over"},
    {"site": "advanced_search.HierarchicalSearch._expansions (sin)",
     "file": "apedp/advanced_search.py",
     "needle": 'f"sin({operand})"',
     "verdict": "argument_is_whole_call",
     "note": "`operand` is the seed pre-wrapped as `({expr})`"},
    {"site": "feature_selection hypothesis atom set (calls)",
     "file": "apedp/feature_selection.py",
     "needles": ("sin({variable})", "sqrt({variable})", "log({variable})"),
     "verdict": "argument_is_whole_call",
     "note": "the call's parentheses delimit the whole argument, so a "
             "sum-named variable needs nothing further"},
    {"site": "feature_selection hypothesis atom set (negated exp)",
     "file": "apedp/feature_selection.py",
     "needle": "exp(-{atom(variable)})",
     "verdict": "parenthesised",
     "note": "a leading sign is not a delimiter: `exp(-a + b)` is not "
             "`exp(-(a + b))`, so this one goes through atom()"},
    {"site": "model_synthesis damped/undamped oscillation",
     "file": "apedp/model_synthesis.py",
     "needle": "cos({w_str} * x)",
     "verdict": "atomic_operand",
     "note": "`w_str = f\"{float(w):.6g}\"` on the line above: the field is a "
             "stringified frequency, so the call's argument is a number times x"},
    {"site": "quantum_compilation.quantum_signal rotation argument",
     "file": "apedp_dyn/quantum_compilation/quantum_signal.py",
     "needles": ("cos({alphas[0]}*x_i", "cos({alpha}*x_i",
                 "cos({trial_alpha}*x_i"),
     "verdict": "atomic_operand",
     "note": "the leading factor is a fitted rotation angle (a float); the "
             "argument is a number times a feature label"},
    {"site": "generate_synthesis_50_corpus ground truth",
     "file": "benchmarks/generate_synthesis_50_corpus.py",
     "needle": "cos({omega} * x)",
     "verdict": "atomic_operand",
     "note": "a corpus generator writing a numeric frequency into a template"},
    # --- left factors: `{operand}*2`, the mirror of the coefficient prefix ---
    {"site": "advanced_search.HierarchicalSearch._expansions (doubling)",
     "file": "apedp/advanced_search.py",
     "needle": 'f"{operand}*2"',
     "verdict": "parenthesised",
     "note": "`operand = f\"({expr})\"` on the line above; without it the "
             "doubling reached the constant and the search explored another "
             "family than the one it named"},
    {"site": "advanced_search._expansions docstring",
     "file": "apedp/advanced_search.py",
     "needle": 'f"{expr}*2"',
     "verdict": "non_arithmetic_text",
     "note": "prose in a docstring recording the defect that was fixed"},
    {"site": "falsification.constraints derivative notation",
     "file": "apedp/falsification/constraints.py",
     "needles": ("d{eq.name}/dt", "d{ineq.name}/dt"),
     "verdict": "non_arithmetic_text",
     "note": "`d<state>/dt` is derivative notation inside a message; the slash "
             "is not a division of the state's name"},
    {"site": "run_langmuir_referee_demo labels and counts",
     "file": "benchmarks/run_langmuir_referee_demo.py",
     "needles": ("d{eq.state_variable}/dt", 'f"{n}/5"'),
     "verdict": "non_arithmetic_text",
     "note": "derivative notation and a count of sweep hits out of five"},
    {"site": "run_ps20 counters",
     "file": "benchmarks/run_ps20.py",
     "needle": "survivors}/8",
     "verdict": "non_arithmetic_text",
     "note": "a promoted-survivor count out of the eight problems"},
    {"site": "run_rime_microsample_adjudication fold counts",
     "file": "benchmarks/run_rime_microsample_adjudication.py",
     "needles": ("sat_family_top_rank_folds']}/11",
                 "origin_strict_satisfied_count']}/11"),
     "verdict": "non_arithmetic_text",
     "note": "folds out of eleven in a printed summary"},
    {"site": "run_scope_refusal_mv_validation cell count",
     "file": "benchmarks/run_scope_refusal_mv_validation.py",
     "needle": "{len(g2_defined)}/12",
     "verdict": "non_arithmetic_text",
     "note": "a count of defined G2 cells out of twelve"},
    {"site": "synthetic synthesis corpus ground truth",
     "file": "benchmarks/generate_synthesis_50_corpus.py",
     "needles": ("({A} * x**{alpha}", "({A_true} * x**{alpha_true}"),
     "verdict": "atomic_operand",
     "note": "a corpus generator formatting numeric constants into a ground-truth "
             "template: the field is a number, so `exp(-C * x**g)` negates a "
             "number rather than a sum"},
    {"site": "synthetic_synthesis_generator ground truth",
     "file": "benchmarks/synthetic_synthesis_generator.py",
     "needle": "({A_true} * x**{alpha_true}",
     "verdict": "atomic_operand",
     "note": "same template, numeric constants throughout"},
)


def audit_power_and_division() -> dict:
    """Adjudicate every power-on-a-field and division-by-a-field site."""
    found = scan_power_and_division_sites()
    matched: set[int] = set()
    rows: list[dict] = []
    for entry in POWER_DIVISION_SITES:
        needles = entry.get("needles",
                            (entry["needle"],) if entry.get("needle") else ())
        hits = [index for index, site in enumerate(found)
                if site["file"] == entry["file"]
                # An entry that records a converted shape names the shapes it
                # replaced: one line can hold both a reciprocal that was fixed
                # and a call that was always safe, and only the first belongs
                # to the conversion.
                and (not entry.get("shapes") or site["shape"] in entry["shapes"])
                and any(needle and needle in site["source"] for needle in needles)]
        matched.update(hits)
        row = {**entry, "sites": [found[index] for index in hits]}
        if entry["verdict"] in ("converted", "retired"):
            path = ROOT / entry["file"]
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            row["conversion_marker_present"] = bool(entry.get("markers")) and all(
                marker in text for marker in entry["markers"])
        rows.append(row)
    unrecorded = [site for index, site in enumerate(found)
                  if index not in matched and site["scope"] == "in_scope"]
    stale = [row["site"] for row in rows
             if not row["sites"] and row["verdict"] not in ("converted", "retired")]
    return {
        "question": (
            "is a field raised to a power, divided by, used as a function "
            "argument, or multiplied by a literal, without being bound as one "
            "syntactic unit? `**`, `/` and the call-opening paren all bind "
            "tighter than `+`, so an operand that is a sum loses everything "
            "after its first addend"),
        "scope_rule": (
            "a site is in scope when the operand field is named like an operand "
            "(name, feature, variable, term, expr, operand, seed, tokens, "
            "reciprocal, or a formatter call) and carries no numeric format "
            "spec; counters, paths, markdown labels, shape tuples and formatted "
            "numbers are reported with their reason instead of carrying an entry"),
        "in_scope": [site for site in found if site["scope"] == "in_scope"],
        "out_of_scope": [site for site in found if site["scope"] != "in_scope"],
        "verdict_vocabulary": _POWER_DIVISION_VOCABULARY,
        "sites_found": found,
        "registry": rows,
        "unrecorded_sites": unrecorded,
        "stale_registry_entries": stale,
        "by_shape": {
            **{shape: sum(1 for site in found if site["shape"] == shape)
               for shape in _SHAPES},
            "total": len(found),
            "in_scope": sum(1 for site in found if site["scope"] == "in_scope"),
            "numeric_operand": sum(1 for site in found
                                   if site["scope"] == "numeric_operand"),
            "display_only": sum(1 for site in found
                                if site["scope"] == "display_only"),
        },
    }


def build_record() -> dict:
    builders = audit_builders()
    sites = [audit_coupled_site(), audit_multiscale_site(),
             audit_sindy_site(), audit_expansion_site()]
    assembly = audit_string_assembly()
    power_division = audit_power_and_division()
    refused = ("fail", "materialisation_raises")
    dimensionless_only = [row["term"] for row in builders
                          if row.get("gate1") in refused
                          and row.get("adds_a_literal_constant")]
    unit_mismatch_sums = [row["term"] for row in builders
                          if row.get("gate1") in refused
                          and "Add" in str(row.get("gate1_detail", ""))
                          and not row.get("adds_a_literal_constant")]
    dimensional_argument_only = [
        row["term"] for row in builders
        if row.get("gate1") in refused
        and "dimensional argument" in str(row.get("gate1_detail", ""))]
    return {
        "schema": "term_builder_structure_audit/v3",
        "generated_by": "scripts/audit_term_builder_structure.py",
        "defect_class": (
            "a coefficient formatted onto an operand without parentheses binds "
            "to the first addend of a non-product expression; and, separately, "
            "an expression that is not a single product may be dimensionally "
            "incoherent by construction, in which case gate 1 is the right "
            "place for it to die"),
        "question": (
            "does the emitted coefficient multiply the whole built expression? "
            "checked numerically as rhs(p=2) - 2*rhs(p=1) == 0"),
        "builders": builders,
        "emission_sites": sites,
        "string_assembly": assembly,
        "power_and_division": power_division,
        "summary": {
            "builders_classified": len(builders),
            "builder_refusals": sum(1 for row in builders
                                    if row.get("structure") == "builder_refuses"),
            "unavailable_builders": [row["term"] for row in builders
                                     if row.get("structure") == "builder_refuses"],
            "remote_position_terms": [row["term"] for row in builders
                                      if row.get("needs_remote_position")],
            "not_a_single_product": [
                row["term"] for row in builders
                if row.get("structure") in ("sum", "quotient_with_sum")],
            "dimensionless_only_vocabulary": dimensionless_only,
            "unit_mismatch_sums": unit_mismatch_sums,
            "dimensional_argument_only": dimensional_argument_only,
            "gate1_passing_builders": [row["term"] for row in builders
                                       if row.get("gate1") == "pass"],
            "coefficient_binding_failures": [
                row["term"] for row in builders
                if row.get("coefficient_binds_whole_term") is False],
            "emission_sites_checked": len(sites),
            "emission_sites_failing": [
                site["site"] for site in sites
                if site.get("coefficient_binds_whole_operand") is False],
            "coefficient_prefix_sites_found": len(assembly["sites_found"]),
            "coefficient_prefix_sites_unrecorded": [
                f"{site['file']}:{site['line']}"
                for site in assembly["unrecorded_sites"]],
            "power_division_sites_found": len(power_division["sites_found"]),
            "power_division_sites_unrecorded": [
                f"{site['file']}:{site['line']}" + site["shape"]
                for site in power_division["unrecorded_sites"]],
            "power_division_registry_entries": len(POWER_DIVISION_SITES),
            "power_division_sites_failing": [
                row["site"] for row in power_division["registry"]
                if row.get("conversion_marker_present") is False],
            "coefficient_prefix_registry_entries": len(STRING_ASSEMBLY_SITES),
            "coefficient_prefix_sites_failing": [
                row["site"] for row in assembly["registry"]
                if row.get("conversion_marker_present") is False
                or row["verdict"] == "latent_open"],
        },
        "limitations": (
            "One problem per materialiser: the Duffing plant for single-DOF and "
            "the coupled 2-DOF plant for the system paths. A builder can only be "
            "refused by gate 1 on a problem whose units expose it, so "
            "'dimensionless_only' is a statement about dimensional problems, not "
            "about every problem. The expansion and SINDy sites are audited with "
            "a synthetic sum-valued operand because the libraries in use today "
            "are products; that makes their defect latent, not absent."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    record = build_record()
    print(f"{'term':14s} {'structure':20s} {'gate1':10s} binds")
    for row in record["builders"]:
        print(f"  {row['term'][:12]:12s} {row.get('structure')!s:20s} "
              f"{row.get('gate1')!s:10s} "
              f"{row.get('coefficient_binds_whole_term')}")
    print("\nemission sites")
    for site in record["emission_sites"]:
        print(f"  {site['site'][:52]:54s} "
              f"binds={site['coefficient_binds_whole_operand']}")
    summary = record["summary"]
    print(f"\nnot a single product: {summary['not_a_single_product']}")
    print(f"unavailable builders: {summary['unavailable_builders']}")
    print(f"dimensionless-only (unit check refuses no literal-1 family): "
          f"{summary['dimensionless_only_vocabulary']}")
    print(f"unit-mismatch sums: {summary['unit_mismatch_sums']}")
    print(f"dimensional-argument-only: {summary['dimensional_argument_only']}")
    print(f"coefficient-binding failures: {summary['coefficient_binding_failures']}")
    print(f"emission sites failing: {summary['emission_sites_failing']}")
    print("\ncoefficient-prefix assembly outside the materialisers")
    for site in record["string_assembly"]["sites_found"]:
        print(f"  {site['file']}:{site['line']:<5d} "
              f"parens={site['operand_parenthesised']!s:5s} {site['source'][:70]}")
    print(f"  unrecorded: {summary['coefficient_prefix_sites_unrecorded']}")
    print(f"  registry:   {summary['coefficient_prefix_registry_entries']} entries, "
          f"failing={summary['coefficient_prefix_sites_failing']}")
    print("\npower and division on a field")
    for site in record["power_and_division"]["in_scope"]:
        print(f"  {site['file']}:{site['line']:<5d} {site['shape']:8s} "
              f"wrapped={site['field_is_formatted']!s:5s} {site['source'][:64]}")
    scope = record["power_and_division"]["by_shape"]
    print(f"  in scope: {scope['in_scope']}, numeric operand: "
          f"{scope['numeric_operand']}, display only: {scope['display_only']}")
    print(f"  unrecorded: {summary['power_division_sites_unrecorded']}")
    print(f"  registry:   {summary['power_division_registry_entries']} entries, "
          f"failing={summary['power_division_sites_failing']}")
    if not args.no_write:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
