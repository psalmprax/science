"""Fixture-declaration hygiene for ``ModelEvidenceRecord`` literals.

The assurance gate can only measure what a record *declares*: an equation it cannot
parse, or one that references symbols the record never declares, yields ``UNVERIFIED``
or ``FAIL`` rather than a measurement. That is the right runtime behaviour, but it is
silent at authoring time -- the declarations live inline in tests, so nothing checked
them until this file did.

Two properties are enforced, both through the repository's own parser and the gate's
own measurement (:func:`apedp.gated_assurance._measure_dimensional_consistency`), never
through a copy of the rule:

1. Every record literal's ``equation_str`` is machine-parseable.
2. Every symbol it references is declared by the record (closure), so the equation
   describes the record rather than an unstated model.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from apedp.epistemic import (
    EvidenceMaturityLevel,
    ModelEvidenceRecord,
    ModelInterventionClass,
)
from apedp.gated_assurance import _measure_dimensional_consistency

TESTS_DIR = pathlib.Path(__file__).parent


def _string_keys(node: ast.AST | None) -> set[str]:
    """String keys of a dict literal, or string elements of a tuple/list literal."""
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    if isinstance(node, ast.Tuple | ast.List):
        return {e.value for e in node.elts if isinstance(e, ast.Constant)}
    return set()


def _record_literals() -> list[tuple[str, str, str, set[str], set[str], set[str]]]:
    """Every ``ModelEvidenceRecord(...)`` literal in this test suite, with a literal equation."""
    found = []
    for path in sorted(TESTS_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ModelEvidenceRecord"):
                continue
            keywords = {k.arg: k.value for k in node.keywords if k.arg}
            equation = keywords.get("equation_str")
            if not isinstance(equation, ast.Constant):
                continue  # contract-derived strings are covered by the contract test below
            model_id = keywords.get("model_id")
            found.append((
                f"{path.name}:{node.lineno}",
                model_id.value if isinstance(model_id, ast.Constant) else "(non-literal)",
                equation.value,
                _string_keys(keywords.get("variables")),
                _string_keys(keywords.get("parameters")),
                _string_keys(keywords.get("validated_domain")),
            ))
    return found


def _as_record(identifier: str, equation: str, variables: set[str], parameters: set[str],
               domains: set[str]) -> ModelEvidenceRecord:
    """Builds a record carrying the extracted declaration, with placeholder values elsewhere."""
    return ModelEvidenceRecord(
        model_id=f"DECL-{identifier}",
        model_version="1.0.0",
        parent_model_id=None,
        model_lineage=(f"DECL-{identifier}",),
        intervention_class=ModelInterventionClass.MATHEMATICAL_REGULARIZATION,
        evidence_maturity=EvidenceMaturityLevel.THEORETICAL_SPECIFICATION,
        domain="declaration hygiene",
        mechanism_family="declaration hygiene",
        equation_str=equation,
        variables=tuple(sorted(variables)),
        parameters={name: 1.0 for name in parameters},
        derivation_basis="declaration hygiene",
        mathematical_guarantees=(),
        numerical_guarantees=(),
        physical_constraints=(),
        assumptions=(),
        known_failure_modes=(),
        validated_domain={name: (1.0, 2.0) for name in domains},
        extrapolation_limits={name: (0.5, 4.0) for name in domains},
        falsification_target="",
        falsification_tests=(),
        failure_interpretation="",
    )


def test_every_record_literal_declares_a_parseable_closed_equation():
    """No fixture may declare an equation the gate cannot parse or cannot close over."""
    literals = _record_literals()
    assert len(literals) >= 8, f"the fixture scan found only {len(literals)} record literals"

    failures = []
    for identifier, model_id, equation, variables, parameters, domains in literals:
        record = _as_record(identifier, equation, variables, parameters, domains)
        status, _, evidence = _measure_dimensional_consistency(record)
        reason = str(evidence.get("reason", ""))
        if status == "FAIL" or "not parseable" in reason:
            failures.append(f"{identifier} ({model_id}): {equation!r} -> {reason}")

    assert not failures, "record declarations the gate cannot measure:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize(
    "factory_name",
    [
        "create_hall_petch_benchmark_contract",
        "create_casimir_benchmark_contract",
        "create_butler_volmer_benchmark_contract",
        "create_prandtl_glauert_benchmark_contract",
        "create_gibbs_thomson_benchmark_contract",
    ],
)
def test_canonical_contract_models_are_machine_parseable(factory_name):
    """Canonical model strings must survive the repository's own formula parser.

    Regression: every contract's ``refined_model`` used ``min(...)``, which the parser
    resolved to a plain Symbol (``'Symbol' object is not callable``), so any record
    deriving its ``equation_str`` from a contract was unmeasurable.
    """
    import apedp.gated_assurance as gate
    from apedp.safety import safe_parse_expr

    contract = getattr(gate, factory_name)()
    for field in ("baseline_model", "refined_model"):
        model_str = getattr(contract, field)
        # Equations read "<observable> = <expression>"; parse the expression.
        _, separator, expression = model_str.partition("=")
        parsed = safe_parse_expr(expression if separator else model_str)
        assert parsed.free_symbols, f"{factory_name}.{field} parsed to a constant: {model_str!r}"


def test_lowercase_min_and_max_are_allowlisted_functions():
    """``min``/``max`` are the natural spelling of a bounded physical law."""
    import sympy as sp

    from apedp.safety import safe_parse_expr

    assert isinstance(safe_parse_expr("min(a, b)"), sp.Min)
    assert isinstance(safe_parse_expr("max(a, b)"), sp.Max)
    # ...and an identifier that is not a function is still rejected rather than called.
    with pytest.raises(ValueError, match="Unparseable formula"):
        safe_parse_expr("totally_unknown_function(a)")
