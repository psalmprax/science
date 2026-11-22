"""The circuit checker as the judge of the variation loop: the strict reader, the judge, the loop.

Status: audit-only

The competition's simulator is not in the repository, so the checker in these tests is a small
simulator of the same instructions written here: it runs a circuit on every input of a tiny
benchmark and fails it the way the real verifier does (the input changed, a working qubit not
back at zero, the phase outside the tolerance). The last test runs the real verifier and is
skipped unless ``APEDP_Q_KICKMIX_PYTHON`` names an interpreter that can load the pinned simulator.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_circuit_evolution as runner

from apedp_dyn.evolution import circuit_task, circuit_text, jail
from apedp_dyn.evolution import operator as loop
from apedp_dyn.evolution.circuit_task import RewriteTask, Scope, Start, Verdict
from apedp_dyn.evolution.circuit_text import CircuitTextError

INPUTS = 3
#: a circuit on three input qubits that turns the phase by an eighth of a turn if x0 and x1, and by a
#: 256th if x1 and x2. It names working qubits q5 and q9 although it never uses two at once.
START_BODY = (
    "R q5", "CCX q0 q1 q5", "Z_POW q5 0.25", "HMR q5 b0", "CZ q0 q1 if b0",
    "R q9", "CCX q1 q2 q9", "Z_POW q9 0.0078125", "HMR q9 b1", "CZ q1 q2 if b1",
)
SMALL = Scope("inputs up to 5", 5, 2)
ALL = Scope("inputs up to 7", 7, 4)
WHOLE = Scope("every input, eight shots each", None, None)


def simulate(ops, x: int, rng: random.Random):
    """One shot: the qubits at the end, the phase in half turns, the costly gates executed, whether an R met a one."""
    qubits: dict[int, int] = defaultdict(int)
    bits: dict[int, int] = defaultdict(int)
    for k in range(INPUTS):
        qubits[k] = (x >> k) & 1
    phase, costly, ruined, stack = Fraction(0), 0, False, []
    for op in ops:
        if op.name == "PUSH_CONDITION":
            stack.append(bits[op.condition])
            continue
        if op.name == "POP_CONDITION":
            stack.pop()
            continue
        if not all(stack) or (op.condition is not None and not bits[op.condition]) or op.name == "INFO_TAG":
            continue
        q, name = op.qubits, op.name
        costly += name in circuit_text.TOFFOLI_LIKE
        if name == "X":
            qubits[q[0]] ^= 1
        elif name == "CX":
            qubits[q[1]] ^= qubits[q[0]]
        elif name == "CCX":
            qubits[q[2]] ^= qubits[q[0]] & qubits[q[1]]
        elif name == "SWAP" or (name == "CSWAP" and qubits[q[0]]):
            a, b = q[-2], q[-1]
            qubits[a], qubits[b] = qubits[b], qubits[a]
        elif name == "Z":
            phase += qubits[q[0]]
        elif name == "CZ":
            phase += qubits[q[0]] & qubits[q[1]]
        elif name == "CCZ":
            phase += qubits[q[0]] & qubits[q[1]] & qubits[q[2]]
        elif name == "Z_POW":
            phase += Fraction(op.angle) * qubits[q[0]]
        elif name == "NEG":
            phase += 1
        elif name == "R":
            ruined = ruined or bool(qubits[q[0]])
            qubits[q[0]] = 0
        elif name == "HMR":
            outcome = rng.getrandbits(1)
            phase += outcome & qubits[q[0]]
            bits[op.bit], qubits[q[0]] = outcome, 0
        elif name == "BIT_INVERT":
            bits[op.bit] ^= 1
        elif name in ("BIT_STORE0", "BIT_STORE1"):
            bits[op.bit] = int(name[-1])
    return qubits, phase % 2, costly, ruined


class StandIn:
    """A checker for the tiny benchmark: what ``START_BODY`` does is what a circuit has to do."""

    def __init__(self, jitter: float = 0.0) -> None:
        self.reference = circuit_text.read(list(START_BODY))
        self.jitter = jitter
        self.runs: list[tuple[str, str]] = []

    @staticmethod
    def tolerance(x: int) -> Fraction:
        return Fraction(1, 64) if x < 6 else Fraction(1, 1024)       # the large inputs are the strict ones

    def __call__(self, path: Path, scope: Scope) -> Verdict:
        text = Path(path).read_text(encoding="utf-8")
        head, body = circuit_text.split(text)
        assert head == circuit_text.header(INPUTS)
        ops = circuit_text.read(body)
        digest = hashlib.sha256(text.encode()).hexdigest()
        rng = random.Random(digest)                                   # seeded by the circuit's bytes, like the real one
        last, shots = (7, 8) if scope.whole else (scope.max_x, scope.shots_per_x)
        executed = 0
        for x in range(1, last + 1):
            _, want, _, _ = simulate(self.reference, x, random.Random(0))
            for _ in range(shots):
                qubits, got, costly, ruined = simulate(ops, x, rng)
                executed += costly
                if sum(qubits[k] << k for k in range(INPUTS)) != x:
                    return self._said(scope, "x was not restored.")
                dirty = sorted(q for q, value in qubits.items() if q >= INPUTS and value)
                if dirty:
                    return self._said(scope, f"Scratch qubit q{dirty[0]} did not return to |0>.")
                error = min((got - want) % 2, (want - got) % 2)
                if ruined or error > self.tolerance(x):
                    return self._said(scope, f"x={x}: phase error {float(error) / 2:.3e} turns exceeds tolerance "
                                             f"{float(self.tolerance(x)) / 2:.3e}.")
        profile = circuit_text.profile(ops, INPUTS)
        toffoli = executed / (last * shots) + (profile["t_gates"] + 29 * profile["rotations"]) / 4
        noise = 1.0 + self.jitter * (int(digest[:8], 16) / 0xFFFFFFFF - 0.5)
        self.runs.append((scope.name, "PASS"))
        return Verdict("PASS", profile["width"] * toffoli * noise, profile["width"], toffoli)

    def _said(self, scope: Scope, said: str) -> Verdict:
        self.runs.append((scope.name, "FAIL"))
        return Verdict("FAIL", said=said)


def _task(tmp_path: Path | None = None, **options) -> tuple[RewriteTask, StandIn]:
    check = options.pop("check", None) or StandIn()
    start = Start("the tiny circuit", START_BODY, INPUTS)
    options.setdefault("scopes", (SMALL, ALL))
    return RewriteTask(start, check, seconds_per_call=10.0, keep=tmp_path, **options), check


# ---------------------------------------------------------------------------- the strict reader
@pytest.mark.parametrize("line", [
    "X q7", "Z q0", "R q4095", "CX q1 q2", "CZ q1 q2", "SWAP q3 q4", "CCX q1 q2 q3", "CCZ q1 q2 q3", "CSWAP q1 q2 q3",
    "Z_POW q7 0.125", "Z_POW q7 -1.000000000001818989403545856475830078125", "Z_POW q7 1", "HMR q7 b3", "NEG",
    "BIT_INVERT b3", "BIT_STORE0 b3", "BIT_STORE1 b3", "PUSH_CONDITION if b3", "POP_CONDITION", "CZ q1 q2 if b0",
    "X q1 if b9", "HMR q7 b3 if b4", "NEG if b2", "BIT_INVERT b3 if b4", "Z_POW q7 0.5 if b1", 'INFO_TAG "@enter:adder (2)"',
])
def test_every_instruction_of_the_format_is_read_and_written_back_as_it_was(line: str) -> None:
    assert circuit_text.parse_line(line).line() == line


@pytest.mark.parametrize(("line", "reason"), [
    ('REGISTER r0 "x"', "not an instruction"), ("APPEND_TO_REGISTER q0 r0", "not an instruction"),
    ("DEBUG_PRINT q1", "not an instruction"), ("MR q1 b0", "not an instruction"), ("x q1", "not an instruction"),
    ("X  q1", "takes 1 qubit"), ("X\tq1", "not an instruction"), ("X q1 ", "takes 1 qubit"), (" X q1", "not an instruction"),
    ("X q1 q2", "takes 1 qubit"), ("CX q1", "takes 2 qubits"), ("CX q1 q1", "same qubit twice"),
    ("CCX q1 q2 q1", "same qubit twice"), ("X q4096", "beyond the limit"), ("X q99999999", "expected a qubit"),
    ("X b1", "expected a qubit"), ("X q-1", "expected a qubit"), ("X q1x", "expected a qubit"), ("X q", "expected a qubit"),
    ("HMR q1", "and a bit"), ("HMR q1 q2", "expected a bit"), ("Z_POW q1", "and an angle"), ("Z_POW q1 1e-3", "expected an angle"),
    ("Z_POW q1 .5", "expected an angle"), ("Z_POW q1 5.", "expected an angle"), ("Z_POW q1 0x10", "expected an angle"),
    ("Z_POW q1 nan", "expected an angle"), ("Z_POW q1 1234.5", "expected an angle"), ("PUSH_CONDITION b3", "is written"),
    ("PUSH_CONDITION", "is written"), ("POP_CONDITION if b1", "takes nothing"), ("NEG q1", "takes no qubit"),
    ("BIT_INVERT q1", "expected a bit"), ("X q1 if q2", "expected a bit"), ("X q1 if", "takes 1 qubit"),
    ('INFO_TAG "a "quote""', "may hold"), ('INFO_TAG "hash # sign"', "may hold"), ('INFO_TAG "back\\\\slash"', "may hold"),
    ('INFO_TAG "' + "a" * 81 + '"', "may hold"), ("INFO_TAG note", "not an instruction"), ("", "not an instruction"),
    ("X q1\nX q2", "takes 1 qubit"), ("X q1 # note", "takes 1 qubit"), ("X q" + "1" * 500, "more than 400"),
])
def test_anything_else_is_refused_with_the_reason(line: str, reason: str) -> None:
    with pytest.raises(CircuitTextError, match=reason):
        circuit_text.parse_line(line)


def test_a_line_that_is_not_text_and_a_circuit_that_is_not_a_list_are_refused() -> None:
    with pytest.raises(CircuitTextError, match="must be text"):
        circuit_text.read(["X q1", 7])
    with pytest.raises(CircuitTextError, match="list of lines, found int"):
        circuit_text.read(42)
    with pytest.raises(CircuitTextError, match="line 2: "):
        circuit_text.read(["X q1", "Y q1"])
    with pytest.raises(CircuitTextError, match="at most 3"):
        circuit_text.read(["X q1"] * 4, max_lines=3)
    assert [op.line() for op in circuit_text.read("X q1\nNEG\n")] == ["X q1", "NEG"]


def test_conditions_have_to_be_closed_in_order() -> None:
    with pytest.raises(CircuitTextError, match="never closed"):
        circuit_text.read(["PUSH_CONDITION if b0", "X q1"])
    with pytest.raises(CircuitTextError, match="line 1: POP_CONDITION without"):
        circuit_text.read(["POP_CONDITION"])
    assert len(circuit_text.read(["PUSH_CONDITION if b0", "X q1", "POP_CONDITION"])) == 3


def test_the_register_lines_are_the_callers_and_a_trusted_file_splits_at_them() -> None:
    head = circuit_text.header(3)
    assert head == ["APPEND_TO_REGISTER q0 r0", "APPEND_TO_REGISTER q1 r0", "APPEND_TO_REGISTER q2 r0", 'REGISTER r0 "x"']
    text = "\n".join([*head, *START_BODY]) + "\n"
    assert circuit_text.split(text) == (head, list(START_BODY))
    with pytest.raises(CircuitTextError, match="register lines after"):
        circuit_text.split("\n".join([*head, "X q0", 'REGISTER r1 "y"']) + "\n")
    assert Start.from_text("tiny", text, INPUTS).body == START_BODY
    with pytest.raises(CircuitTextError, match="register lines are not those"):
        Start.from_text("tiny", text, 4)


def test_what_the_text_says_about_cost() -> None:
    ops = circuit_text.read(list(START_BODY))
    profile = circuit_text.profile(ops, INPUTS)
    assert (profile["width"], profile["held"], profile["working_qubits_named"]) == (10, 4, 2)
    assert (profile["toffoli_like"], profile["rotations"], profile["t_gates"], profile["measurements"]) == (2, 1, 1, 2)
    assert circuit_text.lifetimes(ops, INPUTS) == [(0, 3, 5), (5, 8, 9)]            # each ends at its measurement
    assert circuit_text.held(ops, INPUTS) == (1, 0)
    assert [circuit_text.rotation_kind(a) for a in ("0.5", "-1", "0.25", "0.75", "0.125", "0.0078125")] == \
        ["free", "free", "t", "t", "rotation", "rotation"]
    block = circuit_text.read(["PUSH_CONDITION if b0", "CCX q0 q1 q3", "POP_CONDITION", "CCX q0 q1 q3 if b1", "CCX q0 q1 q3"])
    assert circuit_text.under_a_condition(block) == 2
    tagged = circuit_text.read(['INFO_TAG "@enter:first"', "CCX q0 q1 q3", 'INFO_TAG "@exit:first"', "X q0"])
    assert [(row["stage"], row["lines"], row["toffoli_like"]) for row in circuit_text.profile(tagged, INPUTS)["stages"]] == \
        [("first", 2, 1)]


def test_a_qubit_measured_under_a_condition_is_not_taken_to_be_free() -> None:
    """``HMR ... if b`` may not happen, so the qubit may still be in use after it."""
    ops = circuit_text.read(["CX q0 q3", "HMR q3 b0 if b1", "CX q0 q3", "CX q1 q4"])
    assert circuit_text.lifetimes(ops, INPUTS) == [(0, 2, 3), (3, 3, 4)]


# ---------------------------------------------------------------------------- the stand-in itself
def test_the_stand_in_fails_a_circuit_the_way_the_real_verifier_does() -> None:
    def said(body: list[str], scope: Scope = ALL) -> str:
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "c.kmx"
            file.write_text("\n".join([*circuit_text.header(INPUTS), *body]) + "\n", encoding="utf-8")
            verdict = StandIn()(file, scope)
        return verdict.status if verdict.status == "PASS" else verdict.said

    assert said(list(START_BODY)) == "PASS"
    assert said(["X q0", *START_BODY]) == "x was not restored."
    assert said([*START_BODY, "CX q0 q4"]).startswith("Scratch qubit q4")
    assert "phase error" in said(list(START_BODY[:4]) + list(START_BODY[5:]))       # the correction after a measurement
    assert "phase error" in said([*START_BODY, "CX q0 q4", "R q4"])                 # declared zero while holding a one


# ---------------------------------------------------------------------------- the judge
def test_the_seed_passes_and_a_renaming_that_frees_qubits_leads(tmp_path: Path) -> None:
    task, _ = _task(tmp_path)
    seed = task.evaluate(circuit_task.SEED_CODE)
    assert seed.ran and seed.score[0] == 2.0 and seed.details["profile"]["width"] == 10
    packed = task.evaluate(runner.PACK)
    assert packed.ran and packed.score[0] == 2.0 and packed.details["profile"]["width"] == 4
    assert -packed.score[1] == pytest.approx(-seed.score[1] * 4 / 10)               # the same gates on four qubits
    assert packed.score > seed.score
    assert "4 qubits named" in packed.feedback and "10 qubits named" in packed.feedback
    kept = sorted(p.name for p in tmp_path.iterdir())
    assert kept == sorted(f"{e.details['sha256'][:16]}.kmx" for e in (seed, packed))
    text = (tmp_path / f"{packed.details['sha256'][:16]}.kmx").read_text(encoding="utf-8")
    assert text.startswith("APPEND_TO_REGISTER q0 r0\n") and "q9" not in text and "q5" not in text


def test_a_wrong_circuit_ranks_below_every_circuit_that_passes_and_is_told_why() -> None:
    task, _ = _task()
    seed = task.evaluate(circuit_task.SEED_CODE)
    broken = task.evaluate(runner.DROP_A_GATE)
    assert broken.ran and broken.score == (0.0, 0.0) < seed.score
    assert "failed on inputs up to 5" in broken.summary and "phase error" in broken.feedback
    assert broken.details["verdicts"][0]["status"] == "FAIL" and len(broken.details["verdicts"]) == 1


def test_a_circuit_that_passes_the_small_check_only_is_caught_by_the_larger_one() -> None:
    """Without its finest rotation the circuit is cheaper and right on the inputs up to 5; it is wrong at 6."""
    task, _ = _task()
    seed = task.evaluate(circuit_task.SEED_CODE)
    cheaper = task.evaluate(runner.DROP_A_ROTATION)
    assert cheaper.score[0] == 1.0 and -cheaper.score[1] < seed.details["verdicts"][0]["cost"]
    assert cheaper.score < seed.score
    assert "then failed on inputs up to 7" in cheaper.summary and "x=6" in cheaper.feedback
    alone, _ = _task(scopes=(SMALL,))
    alone.evaluate(circuit_task.SEED_CODE)
    assert alone.evaluate(runner.DROP_A_ROTATION).score > alone.evaluate(circuit_task.SEED_CODE).score   # the small check alone is fooled


@pytest.mark.parametrize(("code", "reason"), [
    (runner.OUTSIDE_THE_FORMAT, "the returned circuit was refused: line 1: 'REGISTER' is not an instruction"),
    ("def rewrite(lines, facts):\n    return lines + ['X q999999']\n", "line 11: a qubit q999999 is beyond the limit"),
    ("def rewrite(lines, facts):\n    return 42\n", "a circuit is a list of lines, found int"),
    ("def rewrite(lines, facts):\n    return [1, 2]\n", "line 1: a line must be text"),
    ("def rewrite(lines, facts):\n    return open('/etc/passwd').read().split()\n", "refused before running"),
    ("def rewrite(lines, facts):\n    import os\n    return lines\n", "refused before running"),
    ("def rewrite(lines):\n    return lines\n", "refused before running"),
    ("def rewrite(lines, facts):\n    return lines[100]\n", "rewrite() failed: IndexError"),
    ("def rewrite(lines, facts):\n    while True:\n        pass\n", "rewrite() failed"),
])
def test_what_is_not_a_circuit_never_reaches_the_checker(code: str, reason: str) -> None:
    task, check = _task()
    task._seconds = 1.0
    evaluation = task.evaluate(code)
    assert not evaluation.ran and reason in evaluation.error
    assert check.runs == []


def test_the_same_circuit_is_checked_once(tmp_path: Path) -> None:
    task, check = _task()
    first = task.evaluate(circuit_task.SEED_CODE)
    again = task.evaluate("def rewrite(lines, facts):\n    return list(lines)\n")
    assert again.score == first.score and "the circuit you were given, unchanged" in again.feedback
    packed = task.evaluate(runner.PACK)
    repeat = task.evaluate(runner.PACK + "\n# the same circuit by other code\n")
    assert repeat.score == packed.score and "a circuit an earlier version returned" in repeat.feedback
    assert len(check.runs) == 4 and sum(task.checked.values()) == 4


def test_two_circuits_that_behave_alike_count_as_equal() -> None:
    """The checker's cost is a sample. Without the band a pair of flips that cancel can lead; with it, it cannot."""
    variants = [f"def rewrite(lines, facts):\n    return ['X q{q}', 'X q{q}'] * {n} + lines\n" for q in range(3) for n in (1, 2, 3)]

    def outcome(min_gain: float) -> tuple[int, int]:
        task, _ = _task(check=StandIn(jitter=1e-4), min_gain=min_gain)
        seed = task.evaluate(circuit_task.SEED_CODE)
        scored = [task.evaluate(code) for code in variants]
        return sum(e.score > seed.score for e in scored), sum(e.score == seed.score for e in scored)

    ahead, _ = outcome(0.0)
    assert ahead >= 1                                   # the draw alone puts a like circuit ahead
    ahead, level = outcome(circuit_task.MIN_GAIN)
    assert ahead == 0 and level >= 1                    # inside the band it is counted as equal, never as cheaper
    task, _ = _task(check=StandIn(jitter=1e-4))
    task.evaluate(circuit_task.SEED_CODE)
    like = task.evaluate(variants[0])
    assert like.details["counted_as_equal_to_the_leading_circuit"] and "counts as equal to it" in like.feedback


def test_the_longest_check_is_run_only_for_a_circuit_that_would_lead() -> None:
    task, _ = _task(scopes=(SMALL, ALL, WHOLE))
    seed = task.evaluate(circuit_task.SEED_CODE)
    assert seed.score[0] == 3.0 and task.checked[WHOLE.name] == 1
    like = task.evaluate(runner.HARMLESS)
    assert like.score[0] == 2.0 and task.checked[WHOLE.name] == 1 and "not checked on every input" in like.feedback
    packed = task.evaluate(runner.PACK)
    assert packed.score[0] == 3.0 and task.checked[WHOLE.name] == 2
    assert packed.score > seed.score > like.score


def test_a_checker_that_cannot_be_run_is_an_error_and_never_a_pass() -> None:
    def broken(path: Path, scope: Scope) -> Verdict:
        raise OSError("no simulator here")

    task, _ = _task(check=broken)
    evaluation = task.evaluate(circuit_task.SEED_CODE)
    assert evaluation.ran and evaluation.score == (0.0, 0.0)
    assert "could not be checked" in evaluation.summary and "no simulator here" in evaluation.feedback


def test_the_judge_cannot_be_given_a_salt() -> None:
    """A salted run on every input is a confirmation the ledger allows once; a loop must not be able to ask for it."""
    assert [f.name for f in dataclasses.fields(Scope)] == ["name", "max_x", "shots_per_x"]
    for function in (circuit_task.official_checker, circuit_task.receipt_checker, RewriteTask.__init__):
        assert not {"salt", "seed"} & set(inspect.signature(function).parameters)
    assert "salt" not in inspect.getsource(circuit_task.official_checker).replace("without a salt", "")
    with pytest.raises(ValueError, match="at least one scope"):
        RewriteTask(Start("tiny", START_BODY, INPUTS), StandIn(), scopes=())


def test_what_the_model_is_told() -> None:
    task, _ = _task()
    contract = task.contract
    assert contract.function == "rewrite" and contract.parameters == 2
    for needed in ("q0 to q2", "CCX, CCZ and CSWAP", "HMR q7 b3", "PUSH_CONDITION if b3", "read strictly",
                   "inputs up to 5; then, if it passes, on inputs up to 7", "0.02%"):
        assert needed in contract.description, needed
    assert "10 lines, 10 qubits named (4 in use at once at most), 2 costly gates, 1 rotations" in contract.notes
    assert task.facts == {"inputs": 3, "lines": 10, "width": 10, "held": 4, "toffoli_like": 2, "rotations": 1}
    prompt = loop._prompt(task, loop.Archive(), loop.Version(0, None, "seed", circuit_task.SEED_CODE,
                                                              task.evaluate(circuit_task.SEED_CODE)), 3)
    assert prompt.startswith("You are improving a function that rewrites a quantum circuit so that it costs less.")
    assert "natural laws" not in prompt


def test_the_other_loop_is_still_told_what_it_was_told() -> None:
    assert loop.Contract("f", 1, "f(x)", "d").subject == "one function of a search for natural laws"


# ---------------------------------------------------------------------------- the loop
def test_the_loop_with_scripted_replies_ends_on_the_narrower_circuit(tmp_path: Path) -> None:
    task, _ = _task(tmp_path / "circuits")
    model = runner.ScriptedModel()
    report = runner.run_loop(model, tmp_path, task, steps=len(runner.DEMO_REPLIES), seconds=None, model="scripted")
    assert report["versions"] == 6 and report["ran"] == 5 and report["refused_or_failed"] == 1
    assert report["best"]["origin"] == "model" and report["leading_passed_every_scope"]
    assert report["saved"] == pytest.approx(0.6) and report["leading_cost"] == pytest.approx(report["start_cost"] * 0.4)
    assert report["checker_runs"] == {SMALL.name: 5, ALL.name: 4}
    assert (tmp_path / "leading_code.txt").read_text(encoding="utf-8") == runner.PACK
    rows = [json.loads(line) for line in (tmp_path / "versions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["ran"] for row in rows] == [True, True, True, True, True, False]
    assert "REGISTER" in rows[5]["error"] and rows[3]["score"] == [0.0, 0.0]
    assert "salt" in report["note"] and model.calls == 5


# ---------------------------------------------------------------------------- receipts and the ledger
def _receipt(status: str, scope: str, *, score: float | None = None, error: str = "", digest: str = "a" * 64):
    from apedp_dyn.quantum_compilation.candidates import VerifierReceipt

    metrics = {"verification_scope": scope, **({"error": error} if error else {}),
               **({"toffoli_count": 6.5, "worst_error_over_tolerance": 0.4, "verified_shots": 4096} if score else {})}
    return VerifierReceipt(status=status, score=score, toffoli_equivalent=7.0 if score else None,
                           num_qubits=4 if score else None, metrics=metrics, exit_code=0 if status == "PASS" else 1,
                           command=("python", "verify"), salt_id="public", verification_scope=scope,
                           verifier_sha256="v" * 64, artifact_sha256=digest, contract_sha256="c" * 64, duration_s=0.1)


def test_receipts_become_verdicts_and_every_one_is_recorded_without_consuming_a_confirmation(tmp_path: Path) -> None:
    from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

    answers = iter([_receipt("PASS", "development-prefix", score=28.0), _receipt("PASS", "full-domain-public", score=28.5),
                    _receipt("FAIL", "development-prefix", error="x=3: phase error", digest="b" * 64),
                    _receipt("ERROR", "development-prefix", digest="d" * 64)])
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    check = circuit_task.receipt_checker(lambda path, scope: next(answers), contract_sha256="c" * 64, ledger=ledger)
    path = tmp_path / "c.kmx"
    first, second, failed, errored = (check(path, scope) for scope in (SMALL, WHOLE, SMALL, SMALL))
    assert (first.status, first.cost, first.qubits, first.toffoli) == ("PASS", 28.0, 4, 7.0)
    assert first.facts["worst_error_over_tolerance"] == 0.4 and second.cost == 28.5
    assert (failed.status, failed.said) == ("FAIL", "x=3: phase error")
    assert errored.status == "ERROR" and errored.said == "no reason given"
    assert len(ledger.all_nodes()) == 3                                # one circuit checked twice is one candidate
    assert ledger.verdict_summary()["by_status"] == {"VERIFIED_NONIMPROVING": 1, "VERIFIER_REJECTED": 1, "UNVERIFIABLE": 1}
    assert ledger.burned_holdouts() == {} and ledger.incumbent("invsqrt") == (None, None)
    assert (tmp_path / "ledger.jsonl").stat().st_size > 0


# ---------------------------------------------------------------------------- the jail's longer answers
def test_a_long_answer_is_read_only_when_the_task_says_answers_are_long() -> None:
    code = "def many(n):\n    return ['X q1'] * n\n"
    assert not jail.run(code, "many", [[40_000]], parameters=1).calls[0].ok
    allowed = jail.run(code, "many", [[40_000]], parameters=1, result_chars=1_000_000)
    assert allowed.calls[0].ok and len(allowed.calls[0].result) == 40_000
    assert "at most 1000" in jail.run(code, "many", [[40_000]], parameters=1, result_chars=1000).calls[0].error


# ---------------------------------------------------------------------------- the real verifier, where there is one
def test_the_competitions_verifier_judges_the_reference_circuit(tmp_path: Path) -> None:
    """Opt-in, like the other tests that need the simulator: ``APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13``."""
    if not os.environ.get("APEDP_Q_KICKMIX_PYTHON"):
        pytest.skip("set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter that can load the pinned kickmix")
    from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger

    start = runner.read_start(None)
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    task = RewriteTask(start, circuit_task.official_checker(ledger=ledger), scopes=(circuit_task.QUICK,))
    assert (task.profile["width"], task.profile["held"], task.profile["toffoli_like"]) == (260, 249, 7735)
    seed = task.evaluate(circuit_task.SEED_CODE)
    packed = task.evaluate(runner.PACK)
    broken = task.evaluate(runner.DROP_A_GATE)
    assert seed.score[0] == 1.0 and seed.details["verdicts"][0]["qubits"] == 260
    assert packed.score > seed.score and packed.details["verdicts"][0]["qubits"] == 249
    assert broken.score == (0.0, 0.0) and "phase error" in broken.feedback
    assert ledger.burned_holdouts() == {}
