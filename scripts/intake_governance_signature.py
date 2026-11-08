"""Intake a trustee's returned Ed25519 signature for the FLAW-01 re-certification.

The ceremony (``scripts/run_governance_remint_ceremony.py``) can *mint* a
re-signed manifest, but until now the signature a trustee sends back by email
had to be pasted into the ceremony state by hand. That is the one step where a
maintainer could record a signature that was never verified against the
registered public key — the exact failure the ceremony exists to prevent.

This tool performs only that intake, and refuses everything else:

  - the signature must verify against the manifest's REGISTERED public key for
    that trustee ID (a wrong key or a forger's key is refused);
  - the trustee ID must be in the roster (unregistered signers are refused);
  - the signature must cover the request's spec digest, and the request must
    still match the spec bytes on disk (a stale request is refused);
  - the signature must be exactly 128 lowercase hex characters (a 64-byte
    Ed25519 signature);
  - ``signed_at`` must be a timezone-aware ISO-8601 timestamp;
  - a trustee may appear only once in the ledger.

It never generates a key, never accepts a private key, never writes the spec
JSON, and never writes the manifest. Minting remains the ceremony's job.

Status: audit-only (trustee signature intake for the governance layer).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apedp_dyn.epistemic.governance import (  # noqa: E402
    GovernanceManifest,
    GovernanceTamperingError,
    compute_file_sha256,
    verify_ed25519_signature,
)

SPEC_PATH = _REPO_ROOT / "specifications" / "canonical_preregistered_spec_v1.json"
REQUEST_PATH = _REPO_ROOT / "specifications" / "governance_remint_request.json"
MANIFEST_PATH = _REPO_ROOT / "specifications" / "governance_manifest.json"
LEDGER_PATH = _REPO_ROOT / "specifications" / "governance_remint_signatures.json"

_SIGNATURE_HEX = re.compile(r"^[0-9a-f]{128}$")


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_signed_at(signed_at: str | None) -> str:
    """Accept only an explicitly timezone-aware timestamp.

    A naive timestamp is ambiguous across trustees in different zones, and the
    transparency log is append-only — an ambiguous stamp cannot be corrected
    afterwards, so it is refused at intake instead.
    """
    if signed_at is None or signed_at.strip() == "":
        return _utcnow_iso()
    try:
        parsed = datetime.datetime.fromisoformat(signed_at.strip())
    except ValueError as exc:
        raise GovernanceTamperingError(
            f"signed_at '{signed_at}' is not ISO-8601: {exc}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GovernanceTamperingError(
            f"signed_at '{signed_at}' has no timezone offset; every signature "
            "record must state an absolute instant (e.g. "
            "2026-11-07T14:05:00+00:00).")
    return parsed.isoformat()


def validate_signature_record(
    *,
    request: dict[str, Any],
    manifest: GovernanceManifest,
    trustee_id: str,
    signature_hex: str,
    signed_at: str | None = None,
    spec_path: Path | str = SPEC_PATH,
) -> dict[str, Any]:
    """Verify one returned signature. Returns the record to append; refuses anything else."""
    digest = request.get("spec_sha256", "")
    if not digest:
        raise GovernanceTamperingError(
            "Request carries no spec_sha256; regenerate it with the ceremony's "
            "`request` mode.")
    if request.get("message_trustees_sign", digest) != digest:
        raise GovernanceTamperingError(
            "Request's message_trustees_sign disagrees with spec_sha256; the "
            "request artifact is malformed.")

    on_disk = compute_file_sha256(spec_path)
    if on_disk != digest:
        raise GovernanceTamperingError(
            f"Request is STALE: it asks trustees to sign {digest}, but the spec "
            f"on disk now computes {on_disk}. Regenerate the request; do not "
            "collect signatures for a digest that is no longer current.")

    trustee_map = {t.trustee_id: t for t in manifest.trustees}
    if trustee_id not in trustee_map:
        raise GovernanceTamperingError(
            f"Trustee '{trustee_id}' is not REGISTERED in the manifest; an "
            "unregistered signer can never count toward quorum.")
    registered_public_key = trustee_map[trustee_id].public_key_hex

    normalized = signature_hex.strip().lower()
    if not _SIGNATURE_HEX.match(normalized):
        raise GovernanceTamperingError(
            "signature_hex must be exactly 128 lowercase hex characters "
            f"(a 64-byte Ed25519 signature); got {len(normalized)} characters.")

    if not verify_ed25519_signature(registered_public_key, digest, normalized):
        raise GovernanceTamperingError(
            f"Signature from '{trustee_id}' does not verify against the "
            "REGISTERED public key over the requested digest — refusing to "
            "record it.")

    return {
        "trustee_id": trustee_id,
        "spec_sha256": digest,
        "signature_hex": normalized,
        "verified_against": registered_public_key,
        "signed_at": _normalize_signed_at(signed_at),
        "intake": (
            "verified by scripts/intake_governance_signature.py against the "
            "registered public key; mint re-verifies from scratch"
        ),
    }


def load_ledger(ledger_path: Path | str) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        return {"kind": "ceremony_signature_ledger", "request_spec_sha256": None,
                "signatures": [],
                "disclosure": ("signature ledger for an in-flight ceremony — "
                               "ephemeral; private keys are NEVER written here")}
    return json.loads(path.read_text(encoding="utf-8"))


def append_signature_record(
    ledger_path: Path | str,
    request: dict[str, Any],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append one verified record, refusing duplicates and a mismatched request."""
    ledger = load_ledger(ledger_path)
    digest = request.get("spec_sha256")
    if ledger.get("request_spec_sha256") not in (None, digest):
        raise GovernanceTamperingError(
            "Ledger belongs to a different request digest "
            f"({ledger['request_spec_sha256']}); a new request needs a new "
            "ledger, not a mixed one.")
    signatures = ledger.get("signatures", [])
    if any(s.get("trustee_id") == record["trustee_id"] for s in signatures):
        raise GovernanceTamperingError(
            f"Duplicate signature from '{record['trustee_id']}' is already in "
            "the ledger; one record per trustee.")
    signatures.append(record)
    ledger["request_spec_sha256"] = digest
    ledger["signatures"] = signatures
    Path(ledger_path).write_text(json.dumps(ledger, indent=2) + "\n",
                                 encoding="utf-8")
    return signatures


def record_signature(
    *,
    ledger_path: Path | str,
    manifest_path: Path | str,
    request_path: Path | str,
    trustee_id: str,
    signature_hex: str,
    signed_at: str | None = None,
    spec_path: Path | str = SPEC_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Full intake: load, validate, then append unless ``dry_run``."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    manifest = GovernanceManifest.from_file(manifest_path)
    record = validate_signature_record(
        request=request, manifest=manifest, trustee_id=trustee_id,
        signature_hex=signature_hex, signed_at=signed_at, spec_path=spec_path)
    if dry_run:
        return record
    append_signature_record(ledger_path, request, record)
    return record


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trustee-id", required=True)
    parser.add_argument("--signature-hex", required=True,
                        help="the 128-character signature the trustee emailed back")
    parser.add_argument("--signed-at", default=None,
                        help="trustee-declared ISO-8601 UTC instant (default: now)")
    parser.add_argument("--request", default=str(REQUEST_PATH))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--spec", default=str(SPEC_PATH))
    parser.add_argument("--ledger", default=str(LEDGER_PATH))
    parser.add_argument("--dry-run", action="store_true",
                        help="verify and report, write nothing")
    args = parser.parse_args(argv)

    try:
        record = record_signature(
            ledger_path=args.ledger, manifest_path=args.manifest,
            request_path=args.request, trustee_id=args.trustee_id,
            signature_hex=args.signature_hex, signed_at=args.signed_at,
            spec_path=args.spec, dry_run=args.dry_run)
    except GovernanceTamperingError as exc:
        print(f"INTAKE REFUSED (nothing recorded): {exc}", file=sys.stderr)
        return 2

    ledger = load_ledger(args.ledger)
    count = len(ledger.get("signatures", []))
    print(json.dumps(record, indent=2))
    if args.dry_run:
        print("dry-run: signature verifies; ledger not written.")
        return 0
    print(f"recorded: {record['trustee_id']} — ledger now holds {count} "
          "live signature(s).")
    print("Next: scripts/run_governance_remint_ceremony.py mint --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
