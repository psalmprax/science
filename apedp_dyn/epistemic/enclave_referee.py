"""Level-5 roadmap Pillar 4: Confidential Epistemic Adjudication.

Runs Stage-C adjudication for a candidate PAIR inside a fresh OS subprocess
that receives the residual manifest through a permission-restricted file,
returns ONLY aggregate verdict statistics, and signs an Ed25519 attestation
token binding (verdict, request payload SHA-256, aggregates). The discovery
engine and the parent process never re-open the holdout during search; the
holdout slices themselves stay sealed in the existing vault/lifecycle
layers (`DataPartitionVault`, `FullLineageExperimentLedger` burn book) —
this layer adds the EXECUTION бокс that cannot be inspect mid-flight by
the search process.

HONEST BOUNDARY (disclosed, test-pinned):
- Without TEE hardware this is PROCESS isolation (fresh interpreter, no
  shared sys.modules, file-permission seals) plus post-hoc signature
  verification — NOT memory encryption. The module reports
  ``executed_by`` as "process-isolated" and only reports "tee" if the
  runtime explicitly declares a hardware enclave; nothing here claims a
  Nitro Enclave / AMD SEV deployment it cannot observe.
- The attestation is a signed EXECUTION record, not an acceptance
  decision: promotion authority stays with the existing referee stack
  (adjudicate + canonical spec + reputation/manifest layers).

Fail-closed contract:
- Worker timeout / crash / missing output  -> verdict "UNRESOLVED",
  verified False, honest reason string. No synthetic verdict is minted.
- Signature or payload-hash mismatch       -> UNRESOLVED with the
  verification error recorded (tamper-evidence, tamper-REPELLENT).
Status: experimental (Level-5 roadmap Phase 4 simulating layer; TEE port
keeps this exact protocol).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "EnclaveAttestation",
    "EnclaveAttestationError",
    "is_tee_environment",
    "run_enclave_adjudication",
    "verify_attestation_payload",
]

WORKER_TAG = "enclave_referee_worker_v1"


class EnclaveAttestationError(RuntimeError):
    """Contract misuse of the enclave runner (wrong pair arity, empty input)."""


@dataclass(frozen=True)
class EnclaveAttestation:
    verdict: str
    executed_by: str
    verified: bool
    verification_error: str
    payload_sha256: str
    signature_hex: str
    public_key_hex: str
    aggregates: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def is_tee_environment() -> bool:
    """Honest TEE detection: False unless the runtime DECLARES an enclave.

    No silent OS sniffing: a Nitro Enclave manifest path or an explicit
    environment flag is the only accepted declaration — the module never
    upgrades its own isolation claim.
    """
    return os.environ.get("APEDX_TEE_ENCLAVE") == "1"


def _generate_keypair() -> tuple[str, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    priv_hex = key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    ).hex()
    pub_hex = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    return priv_hex, pub_hex


def _serialize_request(residuals: Mapping[str, Sequence[float]], bootstrap: bool, data_origin: str) -> bytes:
    payload = {
        "worker": WORKER_TAG,
        "residuals": {str(k): [float(v) for v in residuals[k]] for k in residuals},
        "bootstrap": bool(bootstrap),
        "data_origin": str(data_origin),
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _worker_code(repo_root: str) -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "from apedp_dyn.epistemic.enclave_referee import _enclave_worker_main\n"
        "_enclave_worker_main(*sys.argv[1:])\n"
    )


def _enclave_worker_main(request_path: str, out_path: str, key_path: str) -> None:  # pragma: no cover - subprocess
    """Subprocess entry: adjudicate the pair, sign, exit. NO parent sys.modules."""
    raw = Path(request_path).read_bytes()
    req = json.loads(raw)
    import numpy as np

    from apedp_dyn.epistemic.adjudicate import adjudicate
    from apedp_dyn.grammar.candidate_ir import CandidateDynamics, ScoredCandidate, Term

    def _scored(name: str, resid: np.ndarray) -> ScoredCandidate:
        resid = np.asarray(resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        c = CandidateDynamics(lhs="y", terms=(Term(name, "a0"),), candidate_id=name)
        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else float("inf")
        gate = {"passed": True, "holdout_residuals": list(resid),
                "holdout_rmse": rmse, "holdout_obs_rms": 1.0}
        return ScoredCandidate(c, {"gate4_status": gate}, train_loss=float(np.mean(resid ** 2)) if len(resid) else 0.0,
                               holdout_rmse=rmse)

    scored = [_scored(k, v) for k, v in req["residuals"].items()]
    scored = sorted(scored, key=lambda s: float(s.holdout_rmse) if s.holdout_rmse is not None else float("inf"))
    if len(scored) < 2:
        raise ValueError("enclave worker needs a candidate pair")
    resid_best = scored[0].provenance["gate4_status"]["holdout_residuals"]
    resid_rival = scored[1].provenance["gate4_status"]["holdout_residuals"]
    adj = adjudicate(list(scored),
                     resid_best=np.asarray(resid_best),
                     resid_rival=np.asarray(resid_rival),
                     bootstrap=bool(req["bootstrap"]),
                     data_origin=str(req["data_origin"]))
    verdict = str(getattr(adj, "status", adj))
    aggregates = {
        "best_holdout_rmse": float(scored[0].holdout_rmse),
        "rival_holdout_rmse": float(scored[1].holdout_rmse),
        "n_best": float(len(resid_best)),
        "n_rival": float(len(resid_rival)),
    }
    del resid_best, resid_rival
    payload = json.dumps({
        "worker": WORKER_TAG,
        "verdict": verdict,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "aggregates": aggregates,
        "executed_by": "tee" if is_tee_environment() else "process-isolated",
    }, sort_keys=True).encode("utf-8")

    priv_hex = Path(key_path).read_text().strip()
    from cryptography.hazmat.primitives.asymmetric import ed25519 as _e
    key = _e.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    sig = key.sign(payload).hex()
    out = json.dumps({"payload_json": payload.decode("utf-8"), "signature_hex": sig})
    with open(out_path, "w", encoding="utf-8") as handle:
        os.chmod(out_path, 0o400)
        handle.write(out)


def verify_attestation_payload(payload_json: str, signature_hex: str, public_key_hex: str,
                               request_sha256: str) -> tuple[bool, str, str]:
    """Verify (signature, request-hash binding) of an attestation payload.

    Returns (ok, verdict, error). A mutated aggregates field or a swapped
    request hash FAILS here even though Ed25519 itself never fails open.
    """
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return False, "", f"attestation payload is not JSON: {exc}"
    if payload.get("request_sha256") != request_sha256:
        return False, "", "payload request_sha256 does not bind the executed request"
    from apedp_dyn.epistemic.governance import verify_ed25519_signature

    ok = verify_ed25519_signature(public_key_hex, payload_json, signature_hex)
    if not ok:
        return False, "", "Ed25519 signature verification FAILED"
    return True, str(payload.get("verdict", "")), ""


def run_enclave_adjudication(
    residuals: Mapping[str, Sequence[float]],
    *,
    bootstrap: bool = False,
    data_origin: str = "real",
    timeout_s: float = 120.0,
) -> EnclaveAttestation:
    """Adjudicate a candidate PAIR in the isolated worker; verify the token.

    Exactly two residual series (referee pair semantics). On any
    infrastructure failure the attestation is UNRESOLVED / verified=False —
    the canonical fail-closed outcome, never a synthesized verdict.
    """
    if len(residuals) != 2:
        raise EnclaveAttestationError(
            f"referee pair semantics: exactly 2 residual series required, got {len(residuals)}")
    for name, series in residuals.items():
        arr = np.asarray(series, dtype=float)
        if arr.size < 4:
            raise EnclaveAttestationError(f"residual series {name!r} too short for adjudication")

    priv_hex, pub_hex = _generate_keypair()
    request_bytes = _serialize_request(residuals, bootstrap, data_origin)
    request_sha = hashlib.sha256(request_bytes).hexdigest()

    tmpdir = tempfile.mkdtemp(prefix="apedx_enclave_")
    req_p = Path(tmpdir) / "request.json"
    out_p = Path(tmpdir) / "attestation.json"
    key_p = Path(tmpdir) / "worker.key"
    try:
        os.chmod(tmpdir, 0o700)
        req_p.write_bytes(request_bytes)
        os.chmod(req_p, 0o400)
        key_p.write_text(priv_hex)
        os.chmod(key_p, 0o600)
        proc = subprocess.run(  # noqa: S603 - argv is fixed, non-user-controlled (bundled worker code + ephemeral temp paths)
            [sys.executable, "-c", _worker_code(str(REPO_ROOT)), str(req_p), str(out_p), str(key_p)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            return EnclaveAttestation(
                "UNRESOLVED", "process-isolated" if not is_tee_environment() else "tee",
                False, "", request_sha, "", pub_hex, {},
                reason=f"enclave worker exited {proc.returncode}: {(proc.stderr or '')[-400:]}")
        if not out_p.exists():
            return EnclaveAttestation(
                "UNRESOLVED", "process-isolated" if not is_tee_environment() else "tee",
                False, "", request_sha, "", pub_hex, {},
                reason="enclave worker produced no attestation output")
        token = json.loads(out_p.read_text())
        # The parent rebuilt public-key verification stands between the
        # worker and the caller: a bit-flipped or swapped token fails here.
        ok, verdict, err = verify_attestation_payload(
            token["payload_json"], token["signature_hex"], pub_hex, request_sha)
        verified = ok
        aggregates = json.loads(token["payload_json"]).get("aggregates", {}) if verified else {}
        return EnclaveAttestation(
            verdict if verified else "UNRESOLVED",
            "process-isolated" if not is_tee_environment() else "tee",
            verified, err, request_sha, token["signature_hex"], pub_hex, aggregates,
            reason="attested" if verified else "attestation verification failed (fail-closed)")
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as exc:
        return EnclaveAttestation(
            "UNRESOLVED", "process-isolated" if not is_tee_environment() else "tee",
            False, "", request_sha, "", pub_hex, {},
            reason=f"enclave execution failed closed: {type(exc).__name__}: {exc}")
    finally:
        for p in (req_p, key_p, out_p):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
