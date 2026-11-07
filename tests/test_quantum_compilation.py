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
from apedp_dyn.quantum_compilation.experiment import (
    Candidate,
    CandidateArtifact,
    CandidateScreen,
    DfthcsosProposer,
    DqiProposer,
    ExperimentRunner,
    Proposer,
)
from apedp_dyn.quantum_compilation.invsqrt import (
    calibrate_phase_angles,
    neighbor_precisions,
    refinement_precisions,
    rewrite_self_inverse_toffoli_pairs,
    subset_xs,
)
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import (
    DirectPolyProposer,
    PolyPrecision,
    evaluate_direct_poly_model,
)
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import (
    QPrecision,
    QromLineProposer,
    evaluate_qrom_line_model,
)
from apedp_dyn.quantum_compilation.ledger import (
    REFERENCE_COSTS,
    QuantumCandidateLedger,
    aggregate_score,
    chemistry_cost,
    dqi_cost,
    invsqrt_cost,
    score_ledger_portfolio,
    score_verified_portfolio,
)
from apedp_dyn.quantum_compilation.qrom_invsqrt import (
    qrom_base_precision,
    qrom_neighbor_precisions,
)
from apedp_dyn.quantum_compilation.referee import (
    build_dfthcsos_command,
    build_dqi_command,
    build_invsqrt_command,
    dfthcsos_verification_scope,
    dqi_verification_scope,
    invsqrt_verification_scope,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# `kickmix.so` is a compiled artifact, not source: the index carries the directory's
# LICENSE and the vendored tree, never the binary. A fresh clone therefore cannot load
# the pinned contract, so these tests skip with the build command instead of failing 18
# times for a file the repository was never meant to ship.
_KICKMIX_ROOT = (
    REPO_ROOT
    / "apedp_dyn"
    / "sandbox"
    / "google-verifiable-quantum-circuit-compilation"
    / "kickmix"
)
if not (_KICKMIX_ROOT / "kickmix.so").is_file():
    pytest.skip(
        f"kickmix extension not built at {_KICKMIX_ROOT / 'kickmix.so'}; "
        "run `python scripts/setup_verifier_env.py` to build it",
        allow_module_level=True,
    )
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


def test_qrom_line_model_and_resource_accounting_fix():
    import dataclasses

    contract = load_contract("invsqrt")
    n_p = int(contract.constants["N_P"])
    prec = QPrecision(
        index_bits=7,
        coef_bits=16,
        slope_bits=12,
        delta_bits=10,
        square_bits=28,
        product_bits=27,
        result_bits=28,
    )
    xs = subset_xs(4096, prefix=64, stride=127)
    valid, reason, metrics = evaluate_qrom_line_model(n_p, prec, xs=xs)
    assert valid is True, reason
    assert metrics["min_phase_margin_units"] > 0
    assert metrics["error_tolerance_ratio"] < 1.0

    proposer = QromLineProposer()
    artifact = CandidateArtifact(
        path=Path("dummy_qrom.kmb"),
        sha256="0" * 64,
        metadata={
            "num_qubits": 250,
            "toffoli_like": 5_000,
            "t_like": 120,
            "rotations": 35,
            "min_phase_margin_units": metrics["min_phase_margin_units"],
            "precision": dataclasses.asdict(prec),
        },
    )
    cost = proposer.measure_resources(artifact, contract)
    # Must reflect t_like (120 / 4 = 30 base Toffoli equiv) and single 2**64 division
    assert cost > 250 * 5030.0
    assert cost < 250 * 6000.0


def test_qrom_invsqrt_wrapper_has_no_broken_imports_or_placeholders():
    contract = load_contract("invsqrt")
    base = qrom_base_precision(contract)
    assert base["index_bits"] == 7
    neighbors = list(qrom_neighbor_precisions(base))
    assert len(neighbors) > 0
    for nb in neighbors:
        changed = [k for k in base if nb[k] != base[k]]
        assert len(changed) == 1


def test_direct_poly_quadratic_model_and_resource_accounting():
    import dataclasses

    contract = load_contract("invsqrt")
    n_p = int(contract.constants["N_P"])
    prec = PolyPrecision(index_bits=9, coef_bits=28, delta_bits=19)
    assert set(dataclasses.asdict(prec).keys()) == {
        "index_bits",
        "coef_bits",
        "delta_bits",
    }

    xs = subset_xs(4096, prefix=64, stride=127)
    _valid, _reason, metrics = evaluate_direct_poly_model(n_p, prec, xs=xs)
    assert "min_phase_margin_units" in metrics
    assert "error_tolerance_ratio" in metrics

    proposer = DirectPolyProposer()
    artifact = CandidateArtifact(
        path=Path("dummy_poly.kmb"),
        sha256="0" * 64,
        metadata={
            "num_qubits": 200,
            "toffoli_like": 2_000,
            "t_like": 80,
            "rotations": 35,
            "min_phase_margin_units": max(1, metrics["min_phase_margin_units"]),
            "precision": dataclasses.asdict(prec),
        },
    )
    cost = proposer.measure_resources(artifact, contract)
    assert cost > 200 * 2020.0
    assert cost < 200 * 5000.0


def test_multi_benchmark_referee_scopes_and_commands(tmp_path):
    dqi_contract = load_contract("dqi")
    scope, shots = dqi_verification_scope(dqi_contract, salt="")
    assert scope == "full-domain-public"
    assert shots == 102_400

    scope_salted, _ = dqi_verification_scope(dqi_contract, salt="secret-salt")
    assert scope_salted == "full-domain"

    scope_reduced, dev_shots = dqi_verification_scope(
        dqi_contract, num_shots=256, salt="secret-salt"
    )
    assert scope_reduced == "full-domain-reduced-shots"
    assert dev_shots == 256

    scope_dev, _ = dqi_verification_scope(
        dqi_contract, m=63, num_shots=256, salt="secret-salt"
    )
    assert scope_dev == "development-prefix"

    dqi_cmd = build_dqi_command(
        dqi_contract,
        tmp_path / "dqi.kmb",
        python_executable="python3.13",
        num_shots=512,
        batch_size=256,
        salt="dqi-secret",
    )
    assert "--num-shots" in dqi_cmd
    assert "--batch-size" in dqi_cmd
    assert "--salt" in dqi_cmd

    chem_contract = load_contract("dfthcsos")
    chem_scope, chem_shots = dfthcsos_verification_scope(chem_contract, salt="chem-salt")
    assert chem_scope == "full-domain"
    assert chem_shots == 64

    chem_dev_scope, _ = dfthcsos_verification_scope(
        chem_contract, shots_per_address=8, salt="chem-salt"
    )
    assert chem_dev_scope == "full-domain-reduced-shots"

    # When artifact points to submission.npz and factorization is None,
    # build_dfthcsos_command passes the parent directory as positional arg.
    npz_file = tmp_path / "chem_sub" / "submission.npz"
    chem_cmd = build_dfthcsos_command(
        chem_contract,
        npz_file,
        python_executable="python3.13",
        shots_per_address=16,
    )
    assert str(npz_file.resolve().parent) in chem_cmd
    assert "--shots-per-address" in chem_cmd


def test_experiment_runner_records_in_quantum_candidate_ledger(tmp_path):
    contract = load_contract("invsqrt")
    ledger = QuantumCandidateLedger(tmp_path / "exp_ledger.jsonl")

    class _StubProposer(Proposer):
        def generate(self, contract, incumbent, budget):
            c_bad = Candidate(
                candidate_id=candidate_id_for(
                    contract.benchmark,
                    "stub",
                    {"variant": "invalid"},
                    contract_sha256=contract.contract_sha256,
                ),
                generator="stub",
                parameters={"variant": "invalid"},
            )
            c_good = Candidate(
                candidate_id=candidate_id_for(
                    contract.benchmark,
                    "stub",
                    {"variant": "valid"},
                    contract_sha256=contract.contract_sha256,
                ),
                generator="stub",
                parameters={"variant": "valid"},
            )
            return [c_bad, c_good][:budget]

        def screen(self, candidate, contract):
            if candidate.parameters["variant"] == "invalid":
                return CandidateScreen(
                    valid=False,
                    reason="stub screen failure",
                    metrics={"margin_units": -5},
                )
            return CandidateScreen(
                valid=True,
                reason=None,
                metrics={"margin_units": 1000},
            )

        def compile(self, candidate, contract):
            p = tmp_path / "stub.kmb"
            p.write_bytes(b"stub-circuit")
            return CandidateArtifact(path=p, sha256="1" * 64)

        def measure_resources(self, artifact, contract):
            return 1_500_000.0

        def verify(self, artifact, contract, **kwargs):
            return _receipt(
                status="PASS",
                scope="full-domain",
                salt_id="salt-stub1234",
                score=1_500_000.0,
            )

    runner = ExperimentRunner(
        _StubProposer(),
        contract,
        incumbent_score=2_000_000.0,
        ledger=ledger,
    )
    outcome = runner.run(budget=2, salt="stub-salt")
    assert len(outcome.results) == 2
    assert outcome.results[0].status == "SCREENED_INVALID"
    assert outcome.results[1].status == "VERIFIED"
    assert outcome.incumbent_score == 1_500_000.0

    inc_id, inc_score = ledger.incumbent("invsqrt")
    assert inc_id == outcome.results[1].candidate.candidate_id
    assert inc_score == 1_500_000.0


def test_dqi_and_dfthcsos_proposers_screen_and_measure():
    dqi_contract = load_contract("dqi")
    dqi_proposer = DqiProposer()
    dqi_candidates = dqi_proposer.generate(dqi_contract, incumbent=None, budget=1)
    assert len(dqi_candidates) == 1
    dqi_screen = dqi_proposer.screen(dqi_candidates[0], dqi_contract)
    assert dqi_screen.valid is True
    dqi_artifact = dqi_proposer.compile(dqi_candidates[0], dqi_contract)
    dqi_est = dqi_proposer.measure_resources(dqi_artifact, dqi_contract)
    assert dqi_est == pytest.approx(REFERENCE_COSTS["dqi"], rel=1e-2)

    chem_contract = load_contract("dfthcsos")
    chem_proposer = DfthcsosProposer(
        ensemble_sizes=[54, 50],
        batch_sizes=[8, 7],
        angle_bits=12,
    )
    chem_candidates = chem_proposer.generate(chem_contract, incumbent=None, budget=3)
    assert len(chem_candidates) == 3
    chem_screen = chem_proposer.screen(chem_candidates[0], chem_contract)
    assert chem_screen.valid is True
    chem_artifact = chem_proposer.compile(chem_candidates[0], chem_contract)
    chem_est = chem_proposer.measure_resources(chem_artifact, chem_contract)
    assert chem_est == pytest.approx(REFERENCE_COSTS["dfthcsos"], rel=1e-2)

    # Low angle_bits (< 12) must fail screen
    bad_chem = Candidate(
        candidate_id="dfthcsos-bad",
        generator="dfthcsos-ensemble-v1",
        parameters={
            **chem_candidates[0].parameters,
            "angle_bits": 11,
        },
    )
    bad_screen = chem_proposer.screen(bad_chem, chem_contract)
    assert bad_screen.valid is False
    assert "angle_bits" in (bad_screen.reason or "")


def test_phase_calibration_and_self_inverse_toffoli_rewrites():
    import numpy as np

    signed_errors = np.array([1.0e-12, -1.0e-12, 0.5e-12, -0.5e-12])
    contribution_bits = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    tolerances = np.full(4, 5.0e-12)
    cal = calibrate_phase_angles(
        signed_errors,
        contribution_bits,
        tolerances,
    )
    assert cal.passed_all_constraints is True
    assert cal.status == "PASS_AFTER_QUANTIZATION"
    assert cal.quantized_margin > 0.0

    gates = [
        ("CCX", (0, 1, 2)),
        ("CCX", (1, 0, 2)),  # adjacent self-inverse pair -> cancels
        ("CCX", (2, 3, 4)),
        ("H", (0,)),
    ]
    report = rewrite_self_inverse_toffoli_pairs(gates)
    assert report["cancelled_toffoli_equivalents"] == 2
    assert report["rewritten_count"] == 2
    assert report["and_ancillas_balanced"] is True
    assert report["rewritten_operations"][0] == ("CCX", (2, 3, 4))


def test_three_benchmark_portfolio_scoring(tmp_path):
    assert invsqrt_cost(260, 7735.0, 1009.42) == pytest.approx(
        REFERENCE_COSTS["invsqrt"], rel=1e-3
    )
    assert dqi_cost(1.1950554, 1893, 5_707_104.0) == pytest.approx(
        REFERENCE_COSTS["dqi"], rel=1e-3
    )
    assert chemistry_cost(49.36120289, 730, 93_124.0) == pytest.approx(
        REFERENCE_COSTS["dfthcsos"], rel=1e-3
    )

    # Reference vs Reference -> aggregate score = 1.0
    ref_agg = aggregate_score(REFERENCE_COSTS, REFERENCE_COSTS)
    assert ref_agg == pytest.approx(1.0)

    # Reported portfolio receipts (~5.432)
    receipts = {
        "invsqrt": _receipt(scope="full-domain", salt_id="salt-1", score=182_600.86),
        "dqi": _receipt(
            scope="full-domain", salt_id="salt-2", score=2_300_794_066_593.29
        ),
        "dfthcsos": _receipt(
            scope="full-domain", salt_id="salt-3", score=2_529_044_626.52
        ),
    }
    proj_report = score_verified_portfolio(receipts, require_full_domain=True)
    assert proj_report.all_verified is True
    assert proj_report.all_full_domain is True
    assert proj_report.aggregate_score == pytest.approx(5.432, rel=1e-2)

    # Ledger-backed portfolio scoring requires all 3 benchmarks confirmed
    ledger = QuantumCandidateLedger(tmp_path / "portfolio_ledger.jsonl")
    inv_contract = load_contract("invsqrt")
    c_inv = _candidate(inv_contract)
    ledger.register(c_inv)
    ledger.record_verification(
        c_inv,
        _receipt(scope="full-domain", salt_id="salt-inv", score=1_941_280.56),
        incumbent_score=2_076_774.23,
    )
    with pytest.raises(ValueError, match="No CONFIRMED_VALID incumbent"):
        score_ledger_portfolio(ledger)


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


def test_kickmix_screen_skips_without_configured_interpreter():
    """An unset APEDP_Q_KICKMIX_PYTHON must report a skip, not a silent pass."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ)
    env.pop("APEDP_Q_KICKMIX_PYTHON", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "--no-header", "-p", "no:cacheprovider",
         "-k", "test_kickmix_screen_runs_in_the_matching_interpreter"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "1 skipped" in completed.stdout, completed.stdout


def test_invsqrt_signed_digit_canonical_and_truth_table():
    from apedp_dyn.quantum_compilation import invsqrt_signed_digit as sd

    desc = sd.selected_descriptor()
    assert desc["variant"] == "kozy_table_rudra_split_alex_schedule_borrow_and_domain"
    assert desc["selected_canonical_sha256"] == "63f70e623dd995f4f87ca13254f46f991a5ae545d4c20c80e8510f5ff0a49bbd"
    assert desc["measured_counts"]["qubits"] == 92
    assert desc["measured_counts"]["static_toffoli"] == 1472
    assert desc["measured_counts"]["rotations"] == 45
    assert desc["estimated_cost"] < 165000.0

    # Truth table checks on folded Gidney adder
    assert sd.truth_table_check(max_width=4, borrowed_count=0) > 0
    assert sd.truth_table_check(max_width=4, borrowed_count=1) > 0


def test_faros_isq_portfolio_and_sable_profiler():
    from apedp_dyn.quantum_compilation.faros_isq import (
        R1_NONUNIFORM_CONFIGS,
        R2_PREFIX_RESIDUAL_CONFIGS,
        REFERENCE_COST_90Q,
        REFERENCE_COST_90Q_AFFINE_OPT,
        FarosParetoTournament,
        Sable90Profiler,
        generate_evaluation_subset,
    )

    profile = Sable90Profiler.profile_90q_reference()
    assert profile["num_qubits"] == 90
    assert profile["static_toffoli"] == 1449
    assert profile["rotation_count"] == 44
    assert profile["cost_nominal_90q"] == REFERENCE_COST_90Q
    assert profile["cost_affine_opt_90q"] == REFERENCE_COST_90Q_AFFINE_OPT

    xs = generate_evaluation_subset(max_x=512)
    # Test R1 and R2 configs
    configs = [R1_NONUNIFORM_CONFIGS[0], R2_PREFIX_RESIDUAL_CONFIGS[0]]
    report = FarosParetoTournament.run_tournament(candidates=configs, xs=xs)
    assert report.total_screened == 2
    assert report.passed_f0_count == 2
    assert report.strictly_below_affine_opt_count == 2
    assert report.best_candidate is not None
    assert report.best_candidate.score < REFERENCE_COST_90Q_AFFINE_OPT


