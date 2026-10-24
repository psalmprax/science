"""Unit tests for the APEDP-Q quantum-compilation control plane.

Status: audit-only

The fast tests exercise contracts, candidate identity, the ledger adapter, and
the holdout one-shot rule without importing ``kickmix``.  The integration test
is opt-in because the shipped extension targets CPython 3.13:

    APEDP_Q_KICKMIX_PYTHON=/path/to/python3.13 pytest tests/test_quantum_compilation.py
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    HoldoutContaminationError,
)
from apedp_dyn.quantum_compilation.candidates import (
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
    candidate_id_for,
)
from apedp_dyn.quantum_compilation.contracts import (
    ContractIntegrityError,
    default_competition_root,
    load_contract,
    sha256_tree,
)
from apedp_dyn.quantum_compilation.invsqrt import (
    neighbor_precisions,
    refinement_precisions,
    subset_xs,
)
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger
from apedp_dyn.quantum_compilation.referee import (
    build_invsqrt_command,
    invsqrt_verification_scope,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPETITION_ROOT = default_competition_root()
KICKMIX_PYTHON = os.environ.get("APEDP_Q_KICKMIX_PYTHON")


def _candidate(contract, *, parent=None, parameters=None) -> CircuitCandidate:
    parameters = parameters or {"delta_bits": 14, "coef_bits": 17}
    return CircuitCandidate(
        candidate_id=candidate_id_for(
            "invsqrt",
            "test-generator",
            parameters,
            contract_sha256=contract.contract_sha256,
        ),
        benchmark="invsqrt",
        generator_id="test-generator",
        parameters=parameters,
        artifact_path="",
        artifact_sha256="0" * 64,
        contract_sha256=contract.contract_sha256,
        parent_candidate_id=parent,
    )


def _receipt(*, status="PASS", scope="full-domain", salt_id="a" * 64, score=1.0):
    return VerifierReceipt(
        status=status,
        score=score,
        toffoli_equivalent=1.0,
        num_qubits=260,
        metrics={"verified_shots": 200_000_000},
        exit_code=0,
        command=("python", "verify_submission_invsqrt.py"),
        salt_id=salt_id,
        verification_scope=scope,
        verifier_sha256="b" * 64,
        artifact_sha256="c" * 64,
        contract_sha256="d" * 64,
        duration_s=1.0,
    )


def test_contract_loads_and_is_byte_locked():
    contract = load_contract("invsqrt")
    assert contract.benchmark == "invsqrt"
    assert contract.constants["N_P"] == 10
    assert contract.constants["ETA"] == 218
    assert contract.constants["T_PER_TOFFOLI"] == 4
    assert contract.constants["DEFAULT_SHOTS_PER_X"] == 64
    assert contract.verifier_sha256 == sha256_tree(contract.verifier_root)
    assert contract.contract_sha256
    contract.assert_locked()


def test_contract_digest_binds_benchmark_and_artifact():
    first = load_contract("invsqrt").contract_sha256
    second = load_contract("invsqrt").contract_sha256
    assert first == second

    dqi = load_contract("dqi")
    assert dqi.contract_sha256 != first


def test_candidate_id_depends_on_contract():
    first = load_contract("invsqrt")
    second = load_contract("dqi")
    parameters = {"delta_bits": 14, "coef_bits": 17}
    assert candidate_id_for(
        "invsqrt", "g", parameters, contract_sha256=first.contract_sha256
    ) != candidate_id_for(
        "invsqrt", "g", parameters, contract_sha256=second.contract_sha256
    )


def test_screen_rejection_is_recorded_not_promoted(tmp_path):
    contract = load_contract("invsqrt")
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate(contract)
    ledger.register(candidate)
    rejected = ledger.record_screen(
        candidate,
        passed=False,
        metrics={"error_tolerance_ratio": 1.4},
        score_estimate=None,
        rejection_reason="model error exceeds tolerance",
    )
    node = ledger.node(rejected.candidate_id)
    assert node.state is CandidateLifecycleState.PRE_SCREEN_REJECTED
    assert node.rejection_reason == "model error exceeds tolerance"
    assert node.metrics["screen_passed"] is False
    assert ledger.incumbent("invsqrt") == (None, None)


def test_development_prefix_pass_is_not_a_promotion(tmp_path):
    contract = load_contract("invsqrt")
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate(contract)
    ledger.register(candidate)
    ledger.record_screen(
        candidate, passed=True, metrics={"estimated_score": 2.0e6}, score_estimate=2.0e6
    )
    updated = ledger.record_verification(
        candidate,
        _receipt(scope="development-prefix", score=2.0e6),
        incumbent_score=None,
    )
    assert updated.status is CandidateStatus.VERIFIED_NONIMPROVING
    assert ledger.node(candidate.candidate_id).state is CandidateLifecycleState.DEVELOPMENT_PASSED
    assert ledger.incumbent("invsqrt") == (None, None)


def test_final_salt_is_one_shot_even_after_a_pass(tmp_path):
    contract = load_contract("invsqrt")
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    first = _candidate(contract)
    second = _candidate(
        contract,
        parent=first.candidate_id,
        parameters={"delta_bits": 13, "coef_bits": 17},
    )
    ledger.register(first)
    ledger.register(second)

    updated = ledger.record_verification(
        first, _receipt(scope="full-domain"), incumbent_score=None
    )
    assert updated.status is CandidateStatus.INSUFFICIENT_EVIDENCE

    with pytest.raises(HoldoutContaminationError):
        ledger.record_verification(
            second, _receipt(scope="full-domain"), incumbent_score=3.0
        )


def test_failed_final_salt_is_burned(tmp_path):
    contract = load_contract("invsqrt")
    ledger = QuantumCandidateLedger(tmp_path / "ledger.jsonl")
    candidate = _candidate(contract)
    ledger.register(candidate)
    updated = ledger.record_verification(
        candidate,
        _receipt(status="FAIL", scope="full-domain", score=None),
        incumbent_score=3.0,
    )
    assert updated.status is CandidateStatus.VERIFIER_REJECTED
    burned = ledger.burned_holdouts()
    assert ("invsqrt:full-domain", f"{'b' * 64}:{'a' * 64}") in burned


def test_contract_tamper_is_detected(tmp_path):
    contract = load_contract("invsqrt")
    target = tmp_path / "verify_submission_invsqrt.py"
    target.write_text(contract.entrypoint.read_text(encoding="utf-8") + "\n# tamper\n")
    tampered = type(contract)(
        **{
            **contract.__dict__,
            "entrypoint": target,
            "verifier_sha256": "0" * 64,
        }
    )
    with pytest.raises(ContractIntegrityError):
        tampered.assert_locked()


def test_verification_scope_distinguishes_development_and_final():
    contract = load_contract("invsqrt")
    assert invsqrt_verification_scope(
        contract, max_x=None, shots_per_x=None, salt=""
    )[0] == "full-domain-public"
    assert invsqrt_verification_scope(
        contract, max_x=None, shots_per_x=None, salt="secret"
    )[0] == "full-domain"
    assert invsqrt_verification_scope(
        contract, max_x=512, shots_per_x=1, salt=""
    )[0] == "development-prefix"
    assert invsqrt_verification_scope(
        contract, max_x=None, shots_per_x=1, salt="secret"
    )[0] == "full-domain-reduced-shots"


def test_verifier_command_keeps_the_salt_out_of_the_ledger():
    contract = load_contract("invsqrt")
    command = build_invsqrt_command(
        contract,
        Path("candidate.kmb"),
        python_executable="python3.13",
        salt="very-secret",
    )
    assert "--salt" in command
    assert command[command.index("--salt") + 1] == "very-secret"


def test_subset_xs_is_dense_and_bounded():
    xs = subset_xs(10_000, prefix=32, stride=37)
    assert xs[0] == 1
    assert xs[-1] == 10_000
    assert xs.size == len(set(xs.tolist()))
    assert set(range(1, 33)).issubset(set(xs.tolist()))


def test_refinement_covers_three_field_combinations():
    base = {"delta_bits": 14, "coef_bits": 17, "square_bits": 28, "product_bits": 28, "result_bits": 29}
    rows = list(refinement_precisions(base))
    target = {"delta_bits": 12, "coef_bits": 17, "square_bits": 28, "product_bits": 27, "result_bits": 28}
    assert target in rows
    assert len(rows) == 124


def test_neighbour_precisions_change_one_field_at_a_time():
    base = {"delta_bits": 14, "coef_bits": 17, "square_bits": 28, "product_bits": 28, "result_bits": 29}
    for neighbour in neighbor_precisions(base):
        changed = [name for name in base if neighbour[name] != base[name]]
        assert len(changed) == 1
        assert neighbour[changed[0]] != base[changed[0]]


@pytest.mark.skipif(not KICKMIX_PYTHON, reason="set APEDP_Q_KICKMIX_PYTHON to a CPython 3.13 interpreter")
def test_kickmix_screen_runs_in_the_matching_interpreter():
    script = """
import json
import tempfile
from pathlib import Path

from apedp_dyn.quantum_compilation.contracts import load_contract
from apedp_dyn.quantum_compilation.invsqrt import (
    base_precision,
    screen_precision,
    subset_xs,
    write_circuit,
)

contract = load_contract("invsqrt")
params = base_precision(contract)
screen = screen_precision(contract, params, xs=subset_xs(1024))
with tempfile.TemporaryDirectory() as tmp:
    artifact = Path(tmp) / "baseline.kmb"
    digest = write_circuit(contract, params, artifact)
print(json.dumps({
    "passed": screen.passed,
    "reason": screen.reason,
    "score": screen.score_estimate,
    "domain": screen.metrics.get("domain_kind"),
    "artifact_sha256": digest,
    "sample_sha256": contract.sample_sha256,
    "matches_sample": digest == contract.sample_sha256,
}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [KICKMIX_PYTHON, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["passed"] is True, payload
    assert payload["domain"] == "subset"
    assert payload["score"] is not None
    assert payload["matches_sample"] is True
