"""Pins for Experiment E: the official score, and what the 5.09% is made of.

`docs/DQI_OFFICIAL_COMPATIBILITY.md` §§5-6 measured a phase-safe candidate at 66
Toffolis per controlled multiply against the reference's 75, and stopped at "not a
submission until it is run through the verifier's shot sample end to end". Experiment
E runs the competition's own `verify()` on both circuits at the competition's own
1,024-shot budget, decomposes the saving to the gate, and measures what the pinned
builder can express as a read. These tests pin the arithmetic, not the shot noise: the
executed Toffoli mean is a random variable because the decoder has classically
conditioned Toffolis, so the static stream is what the decomposition is checked on.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "dqi_submission_score.json"
SCRIPT = ROOT / "scripts" / "measure_dqi_submission_score.py"
DOC = ROOT / "docs" / "DQI_OFFICIAL_COMPATIBILITY.md"
DOCUMENTATION = ROOT / "docs" / "DOCUMENTATION.md"
VERIFIER = (ROOT / "apedp_dyn" / "sandbox" /
            "google-verifiable-quantum-circuit-compilation" / "verifiers" / "dqi" /
            "verify_submission_dqi.py")

#: How many Toffoli-like gates one controlled multiply costs on each side.
REFERENCE_PER_CALL = 75
CANDIDATE_PER_CALL = 66
#: The multiply the reference emits, counted at build time.
MULTIPLY_CALLS = 32_408
#: The reference's static Toffoli count, from `circuit.operation_statistics()`.
REFERENCE_STATIC = 5_725_461


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def repetition_overhead(eps: float) -> float:
    """`R(eps)`, recomputed here rather than read out of the verifier."""
    if eps == 0.0:
        return 1.0
    return math.log(2.0) / -math.log(0.5 + 2.0 * math.sqrt(eps))


def test_both_circuits_pass_the_official_verifier(record):
    verified = record["verified"]
    for name in ("reference", "candidate"):
        row = verified[name]
        assert row["status"] == "PASS"
        assert row["value_errors"] == 0
        assert row["phase_errors"] == 0
        assert row["effective_error_rate"] < 1 / 16
        assert row["verified_shots"] == record["instance"]["shots"]
    assert record["instance"]["shots"] == 400 * 256


def test_the_budget_is_the_competitions_own_and_is_the_smallest_that_passes(record):
    budget = record["budget"]
    assert budget["verified_shots"] == budget["official_num_shots"] == 102_400
    assert budget["fraction_of_official"] == 1.0
    # A zero-error run also passes at 1,536 shots and fails below it; that is why the
    # gate in `verify()` is not the only reason the record names its budget.
    assert budget["smallest_passing_budget"] == 1_536
    assert budget["zero_error_bound_at_this_budget"]["passes"] is True


def test_the_score_is_R_times_q_squared_times_t(record):
    for name, row in record["verified"].items():
        r = repetition_overhead(row["effective_error_rate"])
        assert row["repetition_factor"] == pytest.approx(r, rel=1e-12)
        assert row["score"] == pytest.approx(
            r * row["num_qubits"] ** 2 * row["toffoli_count"], rel=1e-12), name


def test_the_score_ratio_is_the_toffoli_ratio_at_equal_q(record):
    ratio = record["score_ratio"]
    assert ratio["qubits"] == 1.0
    assert record["verified"]["reference"]["num_qubits"] == 1893
    assert record["verified"]["candidate"]["num_qubits"] == 1893
    # Both runs have zero errors, so they share `eps_plus` and `R`: the score ratio is
    # the Toffoli ratio and carries no error-rate or qubit penalty either way.
    assert record["verified"]["reference"]["effective_error_rate"] == \
        record["verified"]["candidate"]["effective_error_rate"]
    assert ratio["score"] == pytest.approx(ratio["toffoli"], rel=1e-12)
    assert ratio["score"] == pytest.approx(0.94906, abs=1e-5)


def test_the_saving_is_exactly_the_multiplys_nine_toffolis_per_call(record):
    decomposition = record["decomposition"]
    assert decomposition["multiply_calls"] == MULTIPLY_CALLS
    assert decomposition["toffolis_per_call"] == {
        "reference": REFERENCE_PER_CALL, "candidate": CANDIDATE_PER_CALL, "delta": 9}
    assert decomposition["predicted_delta"] == 9 * MULTIPLY_CALLS
    # Exact on the static stream...
    assert decomposition["reference_static_toffoli_count"] == REFERENCE_STATIC
    assert decomposition["static_delta"] == decomposition["predicted_delta"]
    # ...and the executed mean lands on it to shot noise, because the decoder has
    # classically conditioned Toffolis.
    assert decomposition["executed_delta"] == pytest.approx(
        decomposition["predicted_delta"], rel=1e-4)
    # No Toffoli appears or disappears anywhere else: the CCX difference is the
    # executed Toffoli difference, per shot.
    shots = record["instance"]["shots"]
    assert decomposition["operation_count_deltas"]["CCX"] / shots == pytest.approx(
        decomposition["executed_delta"], rel=1e-9)
    assert decomposition["multiply_share_of_static"] == pytest.approx(
        REFERENCE_PER_CALL * MULTIPLY_CALLS / REFERENCE_STATIC, rel=1e-12)


def test_no_pinned_primitive_reads_a_qubit_into_a_bit(record):
    surface = record["lever_primitives"]["bit_surface"]
    assert record["build"]["has_mr"] is False
    writers = sorted(name for name, row in surface.items() if row["role"] == "output")
    assert writers == ["hmr"]
    probe = record["lever_primitives"]["probes"]["hmr"]
    # On |1> it wrote the value on neither all nor none of the shots: it is a
    # measurement, and its outcome is not the qubit's value.
    assert probe["reads_qubit"] is False
    assert 0 < probe["bit_ones"] < probe["shots"]
    assert probe["phase_flips"] == probe["bit_ones"]
    # Every other method that mentions a bit only consumes one.
    assert len(surface) == 29


def test_the_record_pins_the_script_and_passes_its_checks(record):
    assert record["failed_checks"] == []
    assert record["reproducibility"]["script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()).hexdigest()
    # The one line of `verify()` the record does not run is a reported diagnostic.
    patch = record["verifier_patch"]
    assert patch["replaced"] == "'reaction_depth': circuit.reaction_depth(),"
    assert "reaction_depth" in patch["why"]


def test_the_doc_states_the_score_and_the_budget():
    text = DOC.read_text()
    assert "Experiment E" in text
    assert "0.94906" in text
    assert "102,400" in text
    assert "reaction_depth" in text
    assert "29" in text


# ------------------------------------------- the arithmetic an external review asked for


def test_eps_plus_is_the_zero_error_bound_the_formula_predicts(record):
    """`eps_plus = 1 - 2^(log2_delta / n)` with zero observed errors.

    This is the quantity an external review quoted as `8.669587e-4` in one place and
    `8.660587e-4` in another. The record is the second: recomputing the bound from the
    budget it names reproduces the stored value and nothing else.
    """
    shots = record["instance"]["shots"]
    log2_delta = record["instance"]["log2_delta"]
    predicted = 1.0 - 2.0 ** (log2_delta / shots)
    for name, row in record["verified"].items():
        assert row["value_errors"] == row["phase_errors"] == 0, name
        assert row["effective_error_rate"] == pytest.approx(predicted, rel=1e-12), name
        assert row["effective_error_rate"] == pytest.approx(8.660587e-4, rel=1e-6)
    # The transposed digits are not a budget the verifier could produce at 102,400.
    assert record["verified"]["reference"]["effective_error_rate"] != pytest.approx(
        8.669587e-4, rel=1e-7)
    # The same bound at 256 shots is what refuses a zero-error run there.
    assert 1.0 - 2.0 ** (log2_delta / 256) == pytest.approx(0.2928932188, rel=1e-9)


def test_the_call_count_is_forced_by_the_static_delta(record):
    """32,408 calls, not 32,488: only one of them divides the measured static delta.

    The delta is fixed by the two emitted streams, so the call count is an identity
    rather than a free parameter. `32,488 x 9 = 292,392` and the delta is 291,672, so
    the review's reading of that product cannot stand while the streams do.
    """
    decomposition = record["decomposition"]
    per_call = decomposition["toffolis_per_call"]["delta"]
    calls = decomposition["multiply_calls"]
    assert per_call == 9
    assert decomposition["static_delta"] == decomposition["predicted_delta"]
    assert decomposition["static_delta"] % per_call == 0
    assert decomposition["static_delta"] // per_call == calls
    assert calls == 32_408
    # The arithmetic the review quoted, against the delta the record measured.
    assert 32_488 * 9 == 292_392
    assert 32_488 * 9 != decomposition["static_delta"]
    # The 42.45% share is that call count against the reference's stream, not an
    # independent estimate that could absorb the transposition.
    assert decomposition["multiply_share_of_static"] == pytest.approx(
        REFERENCE_PER_CALL * calls / REFERENCE_STATIC, rel=1e-12)


def test_the_executed_delta_reconciles_with_the_frozen_rows(record):
    """The displayed difference is recomputed from the stored rows, not transcribed."""
    reference = record["verified"]["reference"]["toffoli_count"]
    candidate = record["verified"]["candidate"]["toffoli_count"]
    assert record["decomposition"]["executed_delta"] == pytest.approx(
        reference - candidate, rel=1e-12)
    assert round(reference - candidate, 3) == 291_670.152
    assert record["verified"]["reference"]["toffoli_count"] == pytest.approx(
        5_725_108.382109375, rel=1e-15)


# ------------------------------------------------ the verifier shim, proven rather than
# ---------------------------------------------------------------------------- described


def _patched_source(record) -> tuple[str, str, str]:
    """The shipped verifier, the patched text, and the one line that was replaced."""
    source = VERIFIER.read_text()
    patch = record["verifier_patch"]
    assert source.count(patch["replaced"]) == 1
    return source, source.replace(patch["replaced"], patch["with"]), patch["replaced"]


def _verify_return(tree: ast.Module) -> tuple[ast.FunctionDef, ast.Return]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "verify":
            returns = [child for child in ast.walk(node)
                       if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict)]
            assert len(returns) == 1, "`verify` should return one summary dict"
            return node, returns[0]
    raise AssertionError("no `verify` function in the verifier")


def test_the_recorded_patch_is_the_exact_patch_that_was_applied(record):
    source, patched, replaced = _patched_source(record)
    patch = record["verifier_patch"]
    assert patch["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert patch["patched_sha256"] == hashlib.sha256(patched.encode()).hexdigest()
    assert patch["replaced"] == "'reaction_depth': circuit.reaction_depth(),"
    assert replaced == patch["replaced"]  # verbatim, so the hash is checkable


def test_the_stub_is_one_diagnostic_key_and_cannot_reach_the_verdict(record):
    """An AST-level proof, not a promise: one dict value differs and nothing else.

    If the shim could touch the `PASS` decision, the value or phase checks, `eps_plus`,
    `R` or the score, it would have to appear somewhere outside this one node. It does
    not: `reaction_depth` occurs once in the whole module, and every statement and
    every other function is byte-identical between the shipped and patched trees.
    """
    source, patched, _ = _patched_source(record)
    original_tree, patched_tree = ast.parse(source), ast.parse(patched)

    assert source.count("circuit.reaction_depth()") == 1
    calls = [node for node in ast.walk(original_tree)
             if isinstance(node, ast.Attribute) and node.attr == "reaction_depth"]
    keys = [node for node in ast.walk(original_tree)
            if isinstance(node, ast.Constant) and node.value == "reaction_depth"]
    assert len(calls) == len(keys) == 1
    # In the patched tree the key survives with a `None` value and the call is gone,
    # which is the whole point: the diagnostic is absent, nothing else changes.
    assert not any(isinstance(node, ast.Attribute) and node.attr == "reaction_depth"
                   for node in ast.walk(patched_tree))
    patched_keys = [node for node in ast.walk(patched_tree)
                    if isinstance(node, ast.Constant) and node.value == "reaction_depth"]
    assert len(patched_keys) == 1

    original_fn, original_return = _verify_return(original_tree)
    patched_fn, patched_return = _verify_return(patched_tree)

    # Every top-level statement except `verify` is unchanged.
    assert len(original_tree.body) == len(patched_tree.body)
    for before, after in zip(original_tree.body, patched_tree.body, strict=True):
        if before is original_fn:
            continue
        assert ast.dump(before) == ast.dump(after)

    # Inside `verify`, every statement except the summary return is unchanged.
    assert len(original_fn.body) == len(patched_fn.body)
    differing = [index for index, (before, after) in
                 enumerate(zip(original_fn.body, patched_fn.body, strict=True))
                 if ast.dump(before) != ast.dump(after)]
    assert differing == [original_fn.body.index(original_return)]

    # The return differs in exactly the one key, and the score lives outside it.
    original_items = dict(zip((key.value for key in original_return.value.keys),
                              original_return.value.values, strict=True))
    patched_items = dict(zip((key.value for key in patched_return.value.keys),
                             patched_return.value.values, strict=True))
    assert original_items.keys() == patched_items.keys()
    changed = [key for key in original_items
               if ast.dump(original_items[key]) != ast.dump(patched_items[key])]
    assert changed == ["reaction_depth"]
    call = original_items["reaction_depth"]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert call.func.attr == "reaction_depth"
    assert isinstance(patched_items["reaction_depth"], ast.Constant)
    assert patched_items["reaction_depth"].value is None

    score = ast.unparse(original_items["score"])
    assert "rep_factor" in score and "toffolis" in score and "num_qubits" in score
    assert ast.unparse(original_items["status"]) == "'PASS'"


def test_the_docs_do_not_call_the_shimmed_run_unmodified():
    """The wording must say what ran: the verifier with a diagnostic-only shim."""
    doc = DOC.read_text()
    assert "unmodified, on both circuits" not in doc
    assert "every scoring line unmodified" in doc
    assert "diagnostic-only" in doc
    assert "not a literal unmodified invocation" in doc

    documentation = DOCUMENTATION.read_text()
    assert "unmodified, on both circuits" not in documentation
    assert "diagnostic-only" in documentation
