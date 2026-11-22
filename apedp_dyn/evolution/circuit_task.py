"""Google's circuit checker as the judge of the variation loop.

Status: experimental

The Kaggle competition "Google Verifiable Quantum Circuit Compilation" asks for
cheaper circuits and ships a verifier that simulates a submitted circuit on
every input: the input must come back, every working qubit must return to
zero, and the phase must be right to within the benchmark's tolerance. A
circuit that passes is correct on what was simulated, whoever wrote it and
however. That is the kind of judge a loop with a model as its proposer needs,
and ``evolution/operator.py`` is such a loop.

**What a candidate is.** Code in the restricted subset (``restricted.check``),
run in the jail, defining ``rewrite(lines, facts)``. It is handed the lines of
a circuit that passes (``circuit_text`` describes them: one instruction a
line, the competition's own text format) and returns the lines of a circuit.
It may change a few lines, reorder them, rename qubits, or write another
circuit altogether. The seed returns the lines as they are.

**What happens to what it returns, all of it outside its reach.**

1. ``circuit_text.read`` accepts the lines only if every one is an instruction
   of the format, written the way the simulator writes it; the lines that say
   which qubits are the input are added here and never read from a candidate.
2. The text goes to a file and the file to the checker, one scope after the
   other, cheapest first: a few thousand inputs, then a million, then, where
   asked for, every input of the benchmark. A circuit that fails a scope is
   not shown the next one, and the longest scope is run only for a circuit
   that would lead.
3. The score is ``(scopes passed, -cost)``: a circuit that fails never
   outranks one that passes, however cheap, and among circuits that passed the
   same scopes the cheaper one leads. The cost is the checker's.

The model is told what the checker said, in the checker's words, and how the
returned circuit differs from the starting one in width and in costly gates.

**The checker here is ``quantum_compilation.referee``** running the
competition's verifier in its own process against a byte-pinned contract
(``official_checker``). It is never given a salt: the repository's ledger
treats a salted run on every input as a confirmation that can be asked once,
and a loop that asked it for every version would turn it into a development
set. What leads at the end of a run is therefore a circuit that passed the
unsalted checks, and a claim still needs the one salted run, made once, by
hand, outside the loop.

**The checker's cost is a sample.** How many costly gates a shot executes
depends on what its measurements give, and those are drawn from a seed made of
the circuit's own bytes. Ten copies of the repository's 89-qubit circuit that
differ in a pair of flips which cancel cost between 157,592 and 157,617 on
4,096 inputs and between 157,600.0 and 157,601.6 on a million. A loop that
took the cheaper of two such copies for an improvement would climb on the
draw. A circuit therefore leads only if it costs less than the leading one by
``MIN_GAIN`` of its cost (a third of what one costly gate is worth there), and
one inside that band counts as equal to it.

**Limits.** A model is handed a circuit it cannot read in one piece (the
repository's best has 27,381 lines and no notes in it); it sees counts, and
has to write a program that finds what it wants to change. Passing a scope of
a few thousand inputs says little about the others, which is why the scopes
grow: with its finest rotation left out, that circuit passes on 4,096 inputs,
cheaper, and fails at x = 685,219. The cost on a part of the inputs differs a
little from the cost on all of them, so costs are compared at the same scope
only. And nothing here has been run with a model.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apedp_dyn.evolution import circuit_text, jail, restricted
from apedp_dyn.evolution.circuit_text import CircuitTextError
from apedp_dyn.evolution.operator import Contract, Evaluation

FUNCTION = "rewrite"
SUBJECT = "a function that rewrites a quantum circuit so that it costs less"
SEED_CODE = "def rewrite(lines, facts):\n    return lines\n"
RESULT_CHARS = 32_000_000            # a circuit of 400,000 lines comes back as one answer
SECONDS_PER_CALL = 60.0
INPUTS = 22                          # Benchmark I: x has 2 * 10 + 2 bits
GENERATOR = "circuit-rewrite-v1"
MIN_GAIN = 2e-4                      # of the leading cost: what a circuit has to save to lead (see "a sample" above)
PASS_STATUS, FAIL_STATUS, ERROR_STATUS = "PASS", "FAIL", "ERROR"


@dataclass(frozen=True)
class Scope:
    """How much of the benchmark one check simulates."""

    name: str
    max_x: int | None                 #: inputs 1 .. max_x; ``None`` for every input of the benchmark
    shots_per_x: int | None           #: ``None`` for the benchmark's own number

    @property
    def whole(self) -> bool:
        return self.max_x is None and self.shots_per_x is None


QUICK = Scope("the first 4,096 inputs, one shot each", 4096, 1)
WIDE = Scope("every input up to 1,048,576, one shot each", 1 << 20, 1)
WHOLE = Scope("every input of the benchmark, 64 shots each", None, None)
DEVELOPMENT: tuple[Scope, ...] = (QUICK, WIDE)
COMPLETE: tuple[Scope, ...] = (QUICK, WIDE, WHOLE)


@dataclass
class Verdict:
    """What a checker says about one circuit file on one scope."""

    status: str                       #: "PASS", "FAIL" or "ERROR"
    cost: float | None = None         #: the competition's score; lower is better
    qubits: int | None = None
    toffoli: float | None = None      #: costly gates a shot, the rotations' share included
    said: str = ""                    #: the checker's own words, for the model
    facts: dict[str, Any] = field(default_factory=dict)


Checker = Callable[[Path, Scope], Verdict]


@dataclass(frozen=True)
class Start:
    """The circuit a run starts from: one that passes, as the lines a candidate is handed."""

    name: str
    body: tuple[str, ...]
    inputs: int = INPUTS

    @classmethod
    def from_text(cls, name: str, text: str, inputs: int = INPUTS) -> Start:
        """From the text of a circuit file. Its register lines must be the benchmark's, its body readable."""
        head, body = circuit_text.split(text)
        if head != circuit_text.header(inputs):
            raise CircuitTextError(f"the register lines are not those of a circuit with one input register of {inputs} qubits")
        return cls(name, tuple(circuit_text.written(circuit_text.read(body))), inputs)

    @property
    def header(self) -> list[str]:
        return circuit_text.header(self.inputs)


def _numbers(profile: dict[str, Any]) -> str:
    return (f"{profile['lines']:,} lines, {profile['width']} qubits named ({profile['held']} in use at once at most), "
            f"{profile['toffoli_like']:,} costly gates, {profile['rotations']} rotations")


def _table(profile: dict[str, Any]) -> str:
    if profile["stages"]:
        rows = [f"    {'  ' * row['depth']}{row['stage']}: from line {row['line']}, {row['lines']:,} lines, "
                f"{row['in_use']} qubits in use at its start, {row['toffoli_like']:,} costly gates" for row in profile["stages"]]
        return "  its parts, as its own notes mark them (INFO_TAG \"@enter:name\" ... \"@exit:name\"):\n" + "\n".join(rows)
    rows = [f"    from line {row['line']}: {row['in_use']} qubits in use, {row['toffoli_like']:,} costly gates in this twelfth"
            for row in profile["along"]]
    return "  it carries no notes; in twelve equal pieces:\n" + "\n".join(rows)


class RewriteTask:
    """The judge of ``rewrite`` versions: a strict reader and a checker behind a starting circuit."""

    def __init__(self, start: Start, check: Checker, *, scopes: Sequence[Scope] = DEVELOPMENT, keep: Path | None = None,
                 seconds_per_call: float = SECONDS_PER_CALL, min_gain: float = MIN_GAIN) -> None:
        if not scopes:
            raise ValueError("a judge needs at least one scope")
        self.start = start
        self.scopes = tuple(scopes)
        self._check = check
        self._keep = keep
        self._seconds = float(seconds_per_call)
        self._min_gain = float(min_gain)
        self._judged: dict[str, Evaluation] = {}
        self._leader: list[float] | None = None        #: the leading circuit's cost on every scope
        self.checked: Counter[str] = Counter()         #: how often the checker was run, by scope
        self.profile = circuit_text.profile(circuit_text.read(list(start.body)), start.inputs)
        self.start_sha256 = circuit_text.sha256([*start.header, *start.body])
        self.facts = {"inputs": start.inputs, "lines": self.profile["lines"], "width": self.profile["width"],
                      "held": self.profile["held"], "toffoli_like": self.profile["toffoli_like"],
                      "rotations": self.profile["rotations"]}
        last = start.inputs - 1
        steps = "; then, if it passes, on ".join(scope.name for scope in self.scopes)
        self.contract = Contract(
            function=FUNCTION, parameters=2, signature="rewrite(lines, facts)", subject=SUBJECT,
            description=(
                "rewrite(lines, facts) receives a quantum circuit as a list of text lines, one instruction a line, and "
                "returns the lines of a circuit that does the same and costs less (a list of lines, or one text with a "
                "line each). It may change lines, drop them, reorder them, rename qubits, or build another circuit. "
                f"facts is a dictionary of counts about the circuit it is given: {sorted(self.facts)}.\n\n"
                f"What the circuit must do. Qubits q0 to q{last} hold a number x, q0 its lowest bit. Every other qubit "
                "starts at zero. For every x from 1 to 3,133,452 the circuit must leave x as it found it, leave every "
                "other qubit at zero, and turn the phase by 1/sqrt(x) of a full turn, to within 2^-24 of that amount.\n\n"
                "The cost. (qubits) x (costly gates executed a run + T gates / 4). qubits is the highest qubit number "
                "named, plus one. The costly gates are CCX, CCZ and CSWAP; one under a condition that is false is not "
                "executed. Z_POW by a multiple of a quarter turn is free, by an odd eighth of a turn it is one T gate, "
                "by any finer angle about 29 T gates (more as the circuit's worst phase error nears the tolerance). "
                "Everything else is free.\n\n"
                "The instructions, exactly as they have to be written:\n"
                "  X q7 | Z q7 | R q7            flip the qubit | flip the phase if it is 1 | declare it zero (if it is "
                "not, the phase is ruined)\n"
                "  CX q1 q2 | CZ q1 q2 | SWAP q1 q2      flip q2 if q1 | flip the phase if both | exchange\n"
                "  CCX q1 q2 q3 | CCZ q1 q2 q3 | CSWAP q1 q2 q3      flip q3 if q1 and q2 | phase if all three | exchange "
                "q2 and q3 if q1\n"
                "  Z_POW q7 0.125                turn the phase by that many half turns if the qubit is 1\n"
                "  HMR q7 b3                     measure q7 across, keep the outcome in bit b3, leave q7 at zero. If q7 "
                "held a 1 and the outcome is 1 the phase is flipped, so a qubit holding (q1 and q2) is cleared without a "
                "costly gate by: HMR q7 b3, then CZ q1 q2 if b3\n"
                "  NEG                           flip the phase\n"
                "  BIT_INVERT b3 | BIT_STORE0 b3 | BIT_STORE1 b3      classical bits\n"
                "  PUSH_CONDITION if b3 ... POP_CONDITION      what lies between happens only if the bit is 1\n"
                "  any gate or bit instruction may end in: if b3\n"
                "  INFO_TAG \"text\"               a note, no effect\n\n"
                f"The judge. The returned lines are read strictly (a line that is none of the above is refused, with its "
                f"number) and handed to the competition's own verifier, which simulates the circuit on {steps}. A circuit "
                "that fails ranks below every circuit that passes; among circuits that passed the same checks the cheaper "
                f"one leads, if it is cheaper by at least {self._min_gain:.2%} of the cost: the verifier's count of executed "
                "gates depends on its random measurement outcomes, and two circuits that behave alike differ by less "
                "than that."),
            notes=(f"The circuit you are given ({start.name}): {_numbers(self.profile)}.\n{_table(self.profile)}\n"
                   "Because the cost is a product, one qubit less is worth as many costly gates as the circuit executes "
                   "per qubit it holds. Lines are text: 'CCX q1 q2 q3'.split() gives the words, int(word[1:]) the number."))

    # ------------------------------------------------------------------ the judge
    def evaluate(self, code: str) -> Evaluation:
        try:
            run = jail.run(code, FUNCTION, [[list(self.start.body), dict(self.facts)]], parameters=2,
                           seconds_per_call=self._seconds, result_chars=RESULT_CHARS)
        except restricted.RejectedCodeError as exc:
            return Evaluation(False, error=f"refused before running: {exc}")
        if not run.defined:
            return Evaluation(False, error=f"could not be defined: {run.error}")
        call = run.calls[0] if run.calls else None
        if call is None or not call.ok:
            return Evaluation(False, error=f"rewrite() failed: {call.error if call else run.error}")
        try:
            ops = circuit_text.read(call.result)
        except CircuitTextError as exc:
            return Evaluation(False, error=f"the returned circuit was refused: {exc}")
        body = circuit_text.written(ops)
        digest = circuit_text.sha256([*self.start.header, *body])
        known = self._judged.get(digest)
        if known is not None:
            which = "the circuit you were given, unchanged" if digest == self.start_sha256 else "a circuit an earlier version returned"
            return dataclasses.replace(known, feedback=f"  this is {which}; it was not checked again\n{known.feedback}")
        self._judged[digest] = self._judge(ops, body, digest)
        return self._judged[digest]

    def _checked(self, path: Path, scope: Scope) -> Verdict:
        self.checked[scope.name] += 1
        try:
            return self._check(path, scope)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
            return Verdict(ERROR_STATUS, said=f"the checker could not be run: {type(exc).__name__}: {exc}"[:300])

    def _leads(self, cost: float, level: int) -> bool:
        """Cheaper than the leading circuit on scope ``level`` by more than two like circuits differ."""
        return self._leader is None or cost < self._leader[level] * (1.0 - self._min_gain)

    def _judge(self, ops: list[circuit_text.Op], body: list[str], digest: str) -> Evaluation:
        profile = circuit_text.profile(ops, self.start.inputs)
        text = "\n".join([*self.start.header, *body]) + "\n"
        verdicts: list[tuple[Scope, Verdict]] = []
        skipped = ""
        with tempfile.TemporaryDirectory(prefix="circuit-") as directory:
            path = Path(directory) / f"{digest[:16]}.kmx"
            path.write_text(text, encoding="utf-8")
            for index, scope in enumerate(self.scopes):
                if scope.whole and verdicts and self._leader is not None:
                    before = verdicts[-1][1].cost
                    if before is None or not self._leads(before, index - 1):
                        skipped = (f"not checked on {scope.name}: on the check before, it does not cost less than the "
                                   "leading circuit by more than two like circuits differ")
                        break
                verdict = self._checked(path, scope)
                verdicts.append((scope, verdict))
                if verdict.status != PASS_STATUS or verdict.cost is None:
                    break
        passed = [(scope, verdict) for scope, verdict in verdicts if verdict.status == PASS_STATUS and verdict.cost is not None]
        complete = len(passed) == len(self.scopes)
        level = len(passed) - 1
        last_cost = float(passed[-1][1].cost or 0.0) if passed else 0.0
        leader_cost = self._leader[level] if passed and self._leader is not None else None
        tied = leader_cost is not None and abs(last_cost - leader_cost) <= leader_cost * self._min_gain
        if complete:
            costs = [float(verdict.cost) for _, verdict in passed if verdict.cost is not None]
            if self._leads(costs[-1], level):
                self._leader = costs
            if self._keep is not None:
                self._keep.mkdir(parents=True, exist_ok=True)
                (self._keep / f"{digest[:16]}.kmx").write_text(text, encoding="utf-8")
        details = {"sha256": digest, "passed": len(passed), "scopes": len(self.scopes), "complete": complete,
                   "cost": last_cost if passed else None, "counted_as_equal_to_the_leading_circuit": bool(tied),
                   "profile": {k: v for k, v in profile.items() if k not in ("along", "stages", "instructions")},
                   "verdicts": [{"scope": scope.name, "status": v.status, "cost": v.cost, "qubits": v.qubits,
                                 "toffoli": v.toffoli, "said": v.said, **v.facts} for scope, v in verdicts],
                   "skipped": skipped}
        change = (f"  the circuit you returned: {_numbers(profile)}\n"
                  f"  the circuit you were given: {_numbers(self.profile)}")
        if not passed:
            scope, verdict = verdicts[-1]
            said = verdict.said or "the checker gave no reason"
            word = "failed" if verdict.status == FAIL_STATUS else "could not be checked"
            return Evaluation(True, (0.0, 0.0), f"{word} on {scope.name}: {said}"[:300],
                              f"  the circuit {word} on {scope.name}. The checker said: {said}\n{change}", details=details)
        scope, verdict = passed[-1]
        cost = last_cost
        parts = (f"{verdict.qubits} qubits x {verdict.toffoli:,.2f} costly gates a run, rotations included"
                 if verdict.qubits is not None and verdict.toffoli is not None else "")
        price = (f"  one qubit less is worth {verdict.toffoli / verdict.qubits:,.1f} costly gates here\n"
                 if verdict.qubits and verdict.toffoli else "")
        reached = f"passed {len(passed)} of {len(self.scopes)} checks"
        summary = f"{reached}, cost {cost:,.2f} on {scope.name}" + (f" ({parts})" if parts else "")
        lines = [f"  {reached}; on {scope.name} it costs {cost:,.2f}" + (f": {parts}" if parts else "")]
        if len(passed) < len(verdicts):
            failed_scope, failed = verdicts[-1]
            word = "failed" if failed.status == FAIL_STATUS else "could not be checked"
            lines.append(f"  then it {word} on {failed_scope.name}. The checker said: {failed.said or 'no reason given'}")
            summary += f"; then {word} on {failed_scope.name}"
        if skipped:
            lines.append(f"  {skipped}")
        if tied and leader_cost is not None:
            lines.append(f"  the leading circuit costs {leader_cost:,.2f} on the same check; two circuits that behave alike "
                         f"differ by up to {self._min_gain:.2%}, so this one counts as equal to it")
            summary += "; counted as equal to the leading circuit"
        scored = leader_cost if tied and leader_cost is not None else cost
        return Evaluation(True, (float(len(passed)), -scored), summary[:300], "\n".join(lines) + f"\n{price}{change}",
                          details=details)


# ------------------------------------------------------------------ the competition's verifier as the checker
def receipt_checker(run: Callable[[Path, Scope], Any], *, contract_sha256: str = "", ledger: Any = None) -> Checker:
    """A checker from whatever returns the referee's receipt for a circuit file on a scope.

    With a ``QuantumCandidateLedger`` every circuit that reaches the verifier is registered there
    and every receipt recorded, under the ledger's own rules: a pass without a salt is development
    evidence, and nothing here can consume a confirmation.
    """
    from apedp_dyn.quantum_compilation import candidates

    registered: dict[str, Any] = {}

    def check(path: Path, scope: Scope) -> Verdict:
        receipt = run(path, scope)
        if ledger is not None:
            digest = receipt.artifact_sha256
            if digest not in registered:
                parameters = {"artifact_sha256": digest}
                record = candidates.CircuitCandidate(
                    candidate_id=candidates.candidate_id_for("invsqrt", GENERATOR, parameters, contract_sha256=contract_sha256),
                    benchmark="invsqrt", generator_id=GENERATOR, parameters=parameters, artifact_path=str(path),
                    artifact_sha256=digest, contract_sha256=contract_sha256)
                if record.candidate_id not in {node.candidate_id for node in ledger.all_nodes()}:
                    ledger.register(record)
                registered[digest] = record
            _, incumbent = ledger.incumbent("invsqrt")
            registered[digest] = ledger.record_verification(registered[digest], receipt, incumbent_score=incumbent)
        metrics = receipt.metrics
        facts = {key: metrics[key] for key in ("toffoli_count", "rotation_count", "worst_error_over_tolerance", "worst_x",
                                               "verified_shots", "circuit_sha256", "verification_scope") if key in metrics}
        if receipt.status == PASS_STATUS:
            return Verdict(PASS_STATUS, receipt.score, receipt.num_qubits, receipt.toffoli_equivalent, facts=facts)
        said = str(metrics.get("error") or receipt.stderr_tail[-300:] or "no reason given")
        return Verdict(receipt.status if receipt.status in (FAIL_STATUS, ERROR_STATUS) else ERROR_STATUS, said=said[:400],
                       facts=facts)

    return check


def official_checker(competition_root: Path | None = None, *, python_executable: str | None = None,
                     timeout_s: float = 7200.0, ledger: Any = None) -> Checker:
    """Benchmark I's verifier through ``quantum_compilation.referee``, without a salt.

    The contract is locked first: it refuses a verifier tree or a simulator build that is not the
    pinned one.
    """
    import os
    import sys

    from apedp_dyn.quantum_compilation import contracts, referee

    contract = contracts.load_contract("invsqrt", competition_root)
    interpreter = python_executable or os.environ.get("APEDP_Q_KICKMIX_PYTHON") or sys.executable

    def run(path: Path, scope: Scope) -> Any:
        return referee.run_invsqrt_verifier(contract, path, python_executable=interpreter, max_x=scope.max_x,
                                            shots_per_x=scope.shots_per_x, timeout_s=timeout_s)

    return receipt_checker(run, contract_sha256=contract.contract_sha256, ledger=ledger)


def circuit_file_as_text(path: Path, competition_root: Path | None = None, *, python_executable: str | None = None,
                         timeout_s: float = 600.0) -> str:
    """The text of a circuit file (``.kmb`` or ``.kmx``), written by the pinned simulator in a process of its own."""
    import os
    import sys

    from apedp_dyn.quantum_compilation import contracts

    contract = contracts.load_contract("invsqrt", competition_root)
    contract.assert_locked()
    interpreter = python_executable or os.environ.get("APEDP_Q_KICKMIX_PYTHON") or sys.executable
    program = ("import sys\nsys.path.insert(0, sys.argv[1])\nimport kickmix as km\n"
               "km.Circuit.from_file(sys.argv[2]).to_file(sys.argv[3], format='kmx')\n")
    with tempfile.TemporaryDirectory(prefix="circuit-text-") as directory:
        out = Path(directory) / "circuit.kmx"
        done = subprocess.run(  # noqa: S603 - fixed program, the pinned simulator's directory and two paths, no shell
            [interpreter, "-c", program, str(contract.kickmix_root), str(Path(path).resolve()), str(out)],
            capture_output=True, text=True, timeout=timeout_s, check=False)
        if done.returncode != 0 or not out.is_file():
            raise RuntimeError(f"the circuit could not be read: {done.stderr[-300:]}")
        return out.read_text(encoding="utf-8")


def as_record(evaluation: Evaluation) -> str:
    """One line for a run's log."""
    return json.dumps({"score": list(evaluation.score), "summary": evaluation.summary, "details": evaluation.details},
                      sort_keys=True, default=str)
