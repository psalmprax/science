"""Level-5 Pillar 4 pins: confidential-enclave adjudication layer.

The attestation must be VERIFYABLE (Ed25519 over a payload bound to the
request hash), tampering must fail verification, and every infrastructure
failure must land on UNRESOLVED — never a synthesized verdict. Also pinned:
the honest executed_by claim ("process-isolated" without a declared TEE).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from apedp_dyn.epistemic.enclave_referee import (
    EnclaveAttestationError,
    is_tee_environment,
    run_enclave_adjudication,
    verify_attestation_payload,
)


def _clean_pair(rng: np.random.Generator) -> dict[str, list[float]]:
    good = 0.01 * rng.standard_normal(200)
    biased = 0.3 + 0.05 * rng.standard_normal(200)
    return {"candidate_a": [float(v) for v in good], "candidate_b": [float(v) for v in biased]}


def test_adjudication_runs_isolated_and_attestation_verifies(monkeypatch):
    monkeypatch.setenv("APEDX_TEE_ENCLAVE", "0")
    rng = np.random.default_rng(17)
    att = run_enclave_adjudication(_clean_pair(rng), bootstrap=False, data_origin="real")
    assert att.verified is True, att.reason
    assert att.verification_error == ""
    assert att.verdict, att.verdict  # a real referee status string, not empty
    assert att.executed_by == "process-isolated"  # honest: no TEE hardware here
    assert att.signature_hex and att.public_key_hex
    assert att.aggregates["n_best"] == 200.0
    assert att.aggregates["best_holdout_rmse"] < att.aggregates["rival_holdout_rmse"]


def test_pair_reproducibility_same_input_same_attested_aggregates():
    pair = _clean_pair(np.random.default_rng(17))
    att1 = run_enclave_adjudication(pair, bootstrap=False, data_origin="real")
    att2 = run_enclave_adjudication(pair, bootstrap=False, data_origin="real")
    assert att1.verified and att2.verified
    assert att1.verdict == att2.verdict
    assert att1.aggregates == att2.aggregates


def test_mutated_aggregates_fail_signature_verification():
    # A forged payload (same fields, different aggregates, fake signature
    # minted with a DIFFERENT key) must fail both the hash-binding check and
    # the signature check — tamper-evidence by construction.
    from apedp_dyn.epistemic.governance import sign_spec_sha256

    real_request_sha = "0" * 64
    forged = json.dumps({
        "worker": "enclave_referee_worker_v1",
        "verdict": "unique_support_some_status",
        "request_sha256": real_request_sha,
        "aggregates": {"best_holdout_rmse": 1e-30},
        "executed_by": "tee",
    }, sort_keys=True)
    forged_sig = sign_spec_sha256("ab" * 32, forged)
    ok, _verdict, err = verify_attestation_payload(forged, forged_sig, "cd" * 32, real_request_sha)
    assert not ok and err


def test_wrong_request_hash_binding_fails_before_signature():
    payload = json.dumps({"verdict": "x", "request_sha256": "1" * 64}, sort_keys=True)
    ok, verdict, err = verify_attestation_payload(payload, "00", "11", "2" * 64)
    assert not ok
    assert "does not bind" in err
    assert verdict == ""


def test_worker_timeout_fails_closed_unresolved(monkeypatch):
    monkeypatch.setenv("APEDX_TEE_ENCLAVE", "0")
    pair = _clean_pair(np.random.default_rng(17))
    att = run_enclave_adjudication(pair, bootstrap=False, data_origin="real", timeout_s=1e-9)
    assert att.verified is False
    assert att.verdict == "UNRESOLVED"
    assert att.reason and "failed closed" in att.reason
    assert att.signature_hex == ""


def test_pair_arity_contract_is_enforced():
    with pytest.raises(EnclaveAttestationError, match="exactly 2"):
        run_enclave_adjudication({"a": [0.1] * 10, "b": [0.2] * 10, "c": [0.3] * 10})
    with pytest.raises(EnclaveAttestationError, match="too short"):
        run_enclave_adjudication({"a": [0.1], "b": [0.2]})


def test_tee_claim_is_only_by_declaration(monkeypatch):
    monkeypatch.delenv("APEDX_TEE_ENCLAVE", raising=False)
    assert is_tee_environment() is False
    monkeypatch.setenv("APEDX_TEE_ENCLAVE", "1")
    assert is_tee_environment() is True
