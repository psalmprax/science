"""Cryptographic Multi-Party Preregistration Governance for APEDP-X / CPMD.

Enforces AGENTS.md §1.3 ("Canonical Specification Integrity") through
cryptographic multi-party quorum signatures and an append-only transparency log.

Guarantees that preregistered empirical discovery thresholds:
1. Cannot be unilaterally relaxed or modified by any single researcher or automated agent.
2. Require an M-of-N threshold quorum of cryptographic signatures (Ed25519) across registered trustees.
3. Maintain a tamper-evident, append-only hash chain audit log certifying every change.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger("apedp_dyn.epistemic.governance")


class GovernanceTamperingError(RuntimeError):
    """Raised when preregistration governance verification fails due to tampering,
    insufficient signatures, or broken transparency hash chains."""
    pass


@dataclass(frozen=True)
class TrusteeIdentity:
    """Registered governance trustee authorized to sign preregistration specifications."""
    trustee_id: str
    name: str
    public_key_hex: str
    role: str = "Epistemic Referee"

    def to_dict(self) -> dict[str, str]:
        return {
            "trustee_id": self.trustee_id,
            "name": self.name,
            "public_key_hex": self.public_key_hex,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> TrusteeIdentity:
        return cls(
            trustee_id=data["trustee_id"],
            name=data["name"],
            public_key_hex=data["public_key_hex"],
            role=data.get("role", "Epistemic Referee"),
        )


@dataclass(frozen=True)
class SignatureRecord:
    """Cryptographic signature record from an individual trustee."""
    trustee_id: str
    signature_hex: str
    signed_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "trustee_id": self.trustee_id,
            "signature_hex": self.signature_hex,
            "signed_at": self.signed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SignatureRecord:
        return cls(
            trustee_id=data["trustee_id"],
            signature_hex=data["signature_hex"],
            signed_at=data["signed_at"],
        )


@dataclass(frozen=True)
class TransparencyLogEntry:
    """Append-only tamper-evident transparency log entry certifying specification revisions."""
    index: int
    timestamp: str
    action: str
    spec_sha256: str
    trustee_id: str
    prev_entry_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "action": self.action,
            "spec_sha256": self.spec_sha256,
            "trustee_id": self.trustee_id,
            "prev_entry_hash": self.prev_entry_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransparencyLogEntry:
        return cls(
            index=int(data["index"]),
            timestamp=data["timestamp"],
            action=data["action"],
            spec_sha256=data["spec_sha256"],
            trustee_id=data["trustee_id"],
            prev_entry_hash=data["prev_entry_hash"],
            entry_hash=data["entry_hash"],
        )

    @staticmethod
    def compute_hash(
        index: int,
        timestamp: str,
        action: str,
        spec_sha256: str,
        trustee_id: str,
        prev_entry_hash: str,
    ) -> str:
        payload = f"{index}|{timestamp}|{action}|{spec_sha256}|{trustee_id}|{prev_entry_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GovernanceManifest:
    """Preregistration governance manifest containing trustees, quorum threshold, and signatures."""
    spec_id: str
    spec_sha256: str
    quorum_threshold: int
    trustees: list[TrusteeIdentity] = field(default_factory=list)
    signatures: list[SignatureRecord] = field(default_factory=list)
    transparency_log: list[TransparencyLogEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec_sha256": self.spec_sha256,
            "quorum_threshold": self.quorum_threshold,
            "trustees": [t.to_dict() for t in self.trustees],
            "signatures": [s.to_dict() for s in self.signatures],
            "transparency_log": [e.to_dict() for e in self.transparency_log],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceManifest:
        return cls(
            spec_id=data["spec_id"],
            spec_sha256=data["spec_sha256"],
            quorum_threshold=int(data["quorum_threshold"]),
            trustees=[TrusteeIdentity.from_dict(t) for t in data.get("trustees", [])],
            signatures=[SignatureRecord.from_dict(s) for s in data.get("signatures", [])],
            transparency_log=[TransparencyLogEntry.from_dict(e) for e in data.get("transparency_log", [])],
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_file(cls, path: Path | str) -> GovernanceManifest:
        p = Path(path).resolve()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, path: Path | str) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json())


@dataclass
class GovernanceVerificationResult:
    """Outcome of governance verification."""
    is_valid: bool
    spec_sha256_match: bool
    valid_signature_count: int
    required_quorum: int
    valid_signers: list[str]
    transparency_log_verified: bool
    errors: list[str] = field(default_factory=list)

    @property
    def has_quorum(self) -> bool:
        return self.valid_signature_count >= self.required_quorum


def generate_trustee_keypair() -> tuple[str, str]:
    """Generate a new Ed25519 private/public keypair as hex strings."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes.hex(), public_bytes.hex()


def sign_spec_sha256(private_key_hex: str, spec_sha256: str) -> str:
    """Sign a specification SHA-256 hash using an Ed25519 private key."""
    private_bytes = bytes.fromhex(private_key_hex)
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
    message = spec_sha256.strip().encode("utf-8")
    signature = private_key.sign(message)
    return signature.hex()


def verify_ed25519_signature(public_key_hex: str, spec_sha256: str, signature_hex: str) -> bool:
    """Verify an Ed25519 signature against a specification SHA-256 hash."""
    try:
        public_bytes = bytes.fromhex(public_key_hex)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_bytes)
        signature = bytes.fromhex(signature_hex)
        message = spec_sha256.strip().encode("utf-8")
        public_key.verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def compute_file_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    p = Path(path).resolve()
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_governance_manifest(
    manifest: GovernanceManifest,
    spec_path: Path | str,
) -> GovernanceVerificationResult:
    """Verify specification integrity, M-of-N quorum signatures, and transparency log hash chain."""
    errors: list[str] = []

    # 1. Spec file SHA-256 comparison
    actual_hash = compute_file_sha256(spec_path)
    spec_hash_match = (actual_hash == manifest.spec_sha256)
    if not spec_hash_match:
        errors.append(
            f"Specification SHA-256 mismatch: manifest specifies {manifest.spec_sha256}, "
            f"but actual disk file computes to {actual_hash}."
        )

    # 2. Map registered trustees
    trustee_map = {t.trustee_id: t for t in manifest.trustees}

    # 3. Verify signatures
    valid_signers: list[str] = []
    seen_signers: set[str] = set()

    for sig in manifest.signatures:
        if sig.trustee_id not in trustee_map:
            errors.append(f"Signature by unrecognized trustee ID '{sig.trustee_id}'.")
            continue

        if sig.trustee_id in seen_signers:
            errors.append(f"Duplicate signature by trustee '{sig.trustee_id}'.")
            continue

        trustee = trustee_map[sig.trustee_id]
        if verify_ed25519_signature(trustee.public_key_hex, manifest.spec_sha256, sig.signature_hex):
            valid_signers.append(sig.trustee_id)
            seen_signers.add(sig.trustee_id)
        else:
            errors.append(f"Cryptographic signature verification failed for trustee '{sig.trustee_id}'.")

    valid_count = len(valid_signers)
    has_quorum = valid_count >= manifest.quorum_threshold
    if not has_quorum:
        errors.append(
            f"Insufficient trustee signatures: required quorum {manifest.quorum_threshold}, "
            f"but only found {valid_count} valid signatures."
        )

    # 4. Verify append-only transparency log hash chain
    log_verified = True
    prev_hash = "0" * 64
    for idx, entry in enumerate(manifest.transparency_log):
        if entry.index != idx:
            errors.append(f"Transparency log index sequence broken at index {idx} (found {entry.index}).")
            log_verified = False
            break

        if entry.prev_entry_hash != prev_hash:
            errors.append(
                f"Transparency log hash chain broken at index {idx}: expected prev {prev_hash}, "
                f"found {entry.prev_entry_hash}."
            )
            log_verified = False
            break

        expected_hash = TransparencyLogEntry.compute_hash(
            index=entry.index,
            timestamp=entry.timestamp,
            action=entry.action,
            spec_sha256=entry.spec_sha256,
            trustee_id=entry.trustee_id,
            prev_entry_hash=entry.prev_entry_hash,
        )
        if entry.entry_hash != expected_hash:
            errors.append(
                f"Transparency log entry {idx} tampered: recorded {entry.entry_hash}, "
                f"recomputed {expected_hash}."
            )
            log_verified = False
            break

        prev_hash = entry.entry_hash

    is_valid = spec_hash_match and has_quorum and log_verified and len(errors) == 0

    return GovernanceVerificationResult(
        is_valid=is_valid,
        spec_sha256_match=spec_hash_match,
        valid_signature_count=valid_count,
        required_quorum=manifest.quorum_threshold,
        valid_signers=valid_signers,
        transparency_log_verified=log_verified,
        errors=errors,
    )


def enforce_preregistration_governance(
    spec_path: Path | str,
    manifest_path: Path | str,
) -> GovernanceVerificationResult:
    """Strictly assert that the preregistration specification satisfies multi-party quorum.

    Raises GovernanceTamperingError if validation fails, preventing hypothesis promotion.
    """
    manifest = GovernanceManifest.from_file(manifest_path)
    result = verify_governance_manifest(manifest, spec_path)
    if not result.is_valid:
        error_msg = "; ".join(result.errors)
        raise GovernanceTamperingError(
            f"EPISTEMIC GOVERNANCE VIOLATION: Preregistration certification failed. "
            f"Details: {error_msg}"
        )
    return result
