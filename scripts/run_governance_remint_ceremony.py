"""Trustee re-certification ceremony for the preregistration governance chain
(audit-only deciding-layer infrastructure).

The structural gap this closes: ``specifications/governance_manifest.json``
still seals the GENESIS specification SHA (a6d026…) while the canonical spec
on disk — legitimately revised post-certification and re-pinned by
``apedp_dyn/conf/canonical_spec.py`` (PINNED_SPEC_SHA256 = 5bb42f…) — hashes
differently. The verification side is sound (``enforce_preregistration_governance``
fails closed, the tamper check trips live); what is MISSING is the
formalized procedure to re-close the chain. Per AGENTS.md §1.2/§1.3 this
repository must never bypass, re-mint, or forge the manifest — so this
script does none of those things. It automates the LAWFUL ceremony:

  1. ``--status``   — report the desync and exactly what the quorum owes.
  2. ``--request``  — mint a CERTIFICATION REQUEST artifact: the only thing
     a trustee should see.  Anchors to ``canonical_spec.py``'s import-time
     pin (fail-closed if disk and pin disagree) and freezes the prior
     manifest's SHA for lineage.
  3. ``--collect``  — a trustee's PRIVATE key (supplied via environment —
     never stored, never generated here) signs the request's spec SHA; the
     signature is verified AGAINST THE MANIFEST'S REGISTERED PUBLIC KEY
     for that trustee ID before it is recorded.  Unregistered signers and
     forgeries are refused; duplicates are refused.
  4. ``--mint``     — re-verify every collected signature cryptographically
     from scratch, require the manifest's own quorum threshold, verify the
     PRIOR manifest's transparency log is intact, then extend the chain
     with ``PREREGISTRATION_RECERTIFICATION`` entries via the production
     ``TransparencyLogEntry.compute_hash`` and verify the FULL new manifest
     with the production ``verify_governance_manifest`` against the disk
     spec BEFORE writing it (``--dry-run`` to only report).

Invariants kept, permanently:
  - the tool NEVER generates a keypair and NEVER accepts ``--generate-key``;
    keys live with trustees, outside this repository;
  - the canonical spec JSON itself is NEVER written by this tool;
  - an unmet quorum, a forgery, or any pin disagreement exits NONZERO and
    writes nothing;
  - the resulting manifest's ``spec_id`` and trustee roster are carried
    through UNCHANGED from the prior manifest — a roster change is a
    separate ceremony, out of scope here.

Status: audit-only (ceremony harness for the governance layer).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apedp_dyn.conf.canonical_spec import PINNED_SPEC_SHA256  # noqa: E402
from apedp_dyn.epistemic.governance import (  # noqa: E402
    GovernanceManifest,
    GovernanceTamperingError,
    SignatureRecord,
    TransparencyLogEntry,
    compute_file_sha256,
    verify_ed25519_signature,
    verify_governance_manifest,
)

SPEC_PATH = _REPO_ROOT / "specifications" / "canonical_preregistered_spec_v1.json"
MANIFEST_PATH = _REPO_ROOT / "specifications" / "governance_manifest.json"
REQUEST_PATH = _REPO_ROOT / "specifications" / "governance_remint_request.json"
SIGNATURES_PATH = _REPO_ROOT / "specifications" / "governance_remint_signatures.json"


# ─────────────────────────────────────────────────────────────────────────────
# Ceremony primitives
# ─────────────────────────────────────────────────────────────────────────────

def _load_pinned_sha() -> str:
    """The import-time pin is the single source of the lawful spec identity."""
    return PINNED_SPEC_SHA256


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_request(
    spec_path: Path,
    prior_manifest_path: Path,
    pinned_sha: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Create the certification request trustees sign. Fail-closed on pin drift."""
    pinned = pinned_sha if pinned_sha is not None else _load_pinned_sha()
    disk_sha = compute_file_sha256(spec_path)
    if disk_sha != pinned:
        raise GovernanceTamperingError(
            "SPEC PIN DISAGREEMENT: the canonical spec on disk computes "
            f"{disk_sha} but canonical_spec.py pins {pinned}; the repo's own "
            "pins are out of sync — fix that first, no ceremony can proceed.")
    prior_manifest = GovernanceManifest.from_file(prior_manifest_path)
    prior_verification = verify_governance_manifest(prior_manifest, spec_path)
    return {
        "request_kind": "governance_remint_request",
        "spec_id": prior_manifest.spec_id,
        "spec_sha256": disk_sha,
        "prior_spec_sha256": prior_manifest.spec_sha256,
        "prior_manifest_transparency_log_intact": prior_verification.transparency_log_verified,
        "prior_manifest_valid_for_old_spec": prior_verification.is_valid,
        "quorum_threshold": prior_manifest.quorum_threshold,
        "trustees": [t.to_dict() for t in prior_manifest.trustees],
        # Each trustee signs EXACTLY the hex string of spec_sha256 — the same
        # message the genesis signatures covered.
        "message_trustees_sign": disk_sha,
        "rationale": rationale or (
            "Spec revised post-certification and re-pinned by "
            "canonical_spec.PINNED_SPEC_SHA256 (see CHANGELOG); the manifest "
            "quorum must re-seal the current hash."),
        "created_at": _utcnow_iso(),
        "pinned_sha_source": "apedp_dyn.conf.canonical_spec.PINNED_SPEC_SHA256",
    }


def collect_signature(
    request: dict[str, Any],
    prior_manifest_path: Path,
    trustee_id: str,
    private_key_hex: str,
) -> dict[str, Any]:
    """Sign the request hash with a trustee key and verify against the
    REGISTERED public key. Refuses unknown IDs, forgeries, duplicates via the
    caller's ledger discipline (the state file is checked in ``mint`` too)."""
    if private_key_hex.strip() == "":
        raise ValueError(
            "Refusing to sign with an EMPTY private key: no keys are "
            "generated or stored here; a trustee supplies their own via "
            "--private-key-env.")
    prior_manifest = GovernanceManifest.from_file(prior_manifest_path)
    trustee_map = {t.trustee_id: t for t in prior_manifest.trustees}
    if request["spec_id"] != prior_manifest.spec_id:
        raise GovernanceTamperingError(
            "Request spec_id does not match the manifest's spec_id.")
    if trustee_id not in trustee_map:
        raise GovernanceTamperingError(
            f"Trustee '{trustee_id}' is not REGISTERED in the prior manifest; "
            "unregistered signers can never close a quorum here.")
    registered_pub = trustee_map[trustee_id].public_key_hex
    signature_hex = _sign_with_key(private_key_hex, request["message_trustees_sign"])
    if not verify_ed25519_signature(
            registered_pub, request["message_trustees_sign"], signature_hex):
        raise GovernanceTamperingError(
            f"Signature from '{trustee_id}' does NOT verify against the "
            "REGISTERED public key — refusing to record a potential forgery.")
    return {
        "trustee_id": trustee_id,
        "spec_sha256": request["spec_sha256"],
        "signature_hex": signature_hex,
        "verified_against": registered_pub,
        "signed_at": _utcnow_iso(),
    }


def _sign_with_key(private_key_hex: str, message: str) -> str:
    """Delegate to the production signer (same message convention as genesis)."""
    from apedp_dyn.epistemic.governance import sign_spec_sha256
    return sign_spec_sha256(private_key_hex, message)


def mint_manifest(
    request: dict[str, Any],
    prior_manifest_path: Path,
    collected: list[dict[str, Any]],
    spec_path: Path,
) -> GovernanceManifest:
    """Re-certify the manifest if and only if every gate passes. Fail-closed:
    raises GovernanceTamperingError on ANY shortfall — writes nothing."""
    prior_manifest = GovernanceManifest.from_file(prior_manifest_path)
    prior_verification = verify_governance_manifest(prior_manifest, spec_path)
    if not prior_verification.transparency_log_verified:
        raise GovernanceTamperingError(
            "The PRIOR manifest's transparency log fails its hash-chain "
            "verification — extending a tampered chain is refused; the "
            "certification chain cannot be re-sealed on this basis.")

    new_spec_sha = request["spec_sha256"]
    if compute_file_sha256(spec_path) != new_spec_sha:
        raise GovernanceTamperingError(
            "Request SHA does not match the disk spec; the request is stale.")

    trustee_map = {t.trustee_id: t for t in prior_manifest.trustees}
    seen: set[str] = set()
    verified: list[tuple[str, dict[str, Any]]] = []
    for record in collected:
        tid = record["trustee_id"]
        if tid not in trustee_map:
            raise GovernanceTamperingError(
                f"Collected signature by UNREGISTERED trustee '{tid}'.")
        if tid in seen:
            raise GovernanceTamperingError(f"Duplicate signature by '{tid}'.")
        if record.get("spec_sha256") != new_spec_sha:
            raise GovernanceTamperingError(
                f"Signature by '{tid}' covers {record.get('spec_sha256')}, "
                f"not the request's spec SHA {new_spec_sha}.")
        pub = trustee_map[tid].public_key_hex
        if not verify_ed25519_signature(pub, new_spec_sha, record["signature_hex"]):
            raise GovernanceTamperingError(
                f"Collected signature from '{tid}' FAILED re-verification "
                "against the registered public key.")
        seen.add(tid)
        verified.append((tid, record))

    draft = GovernanceManifest(
        spec_id=prior_manifest.spec_id,
        spec_sha256=new_spec_sha,
        quorum_threshold=prior_manifest.quorum_threshold,
        trustees=prior_manifest.trustees,
        signatures=[
            SignatureRecord(
                trustee_id=tid,
                signature_hex=rec["signature_hex"],
                signed_at=rec.get("signed_at", _utcnow_iso()),
            ) for tid, rec in verified
        ],
        transparency_log=list(prior_manifest.transparency_log),
    )
    log_index = len(prior_manifest.transparency_log)
    prev_hash = (
        prior_manifest.transparency_log[-1].entry_hash
        if prior_manifest.transparency_log else "0" * 64)
    for tid, rec in verified:
        ts = rec.get("signed_at", _utcnow_iso())
        entry = TransparencyLogEntry(
            index=log_index,
            timestamp=ts,
            action="PREREGISTRATION_RECERTIFICATION",
            spec_sha256=new_spec_sha,
            trustee_id=tid,
            prev_entry_hash=prev_hash,
            entry_hash=TransparencyLogEntry.compute_hash(
                index=log_index, timestamp=ts,
                action="PREREGISTRATION_RECERTIFICATION",
                spec_sha256=new_spec_sha, trustee_id=tid,
                prev_entry_hash=prev_hash),
        )
        draft.transparency_log.append(entry)
        log_index += 1
        prev_hash = entry.entry_hash

    result = verify_governance_manifest(draft, spec_path)
    if not result.is_valid or not result.has_quorum:
        raise GovernanceTamperingError(
            "Draft manifest does NOT satisfy production verification; "
            "the ceremony writes nothing. Details: " + "; ".join(result.errors))
    return draft


def status_report() -> dict[str, Any]:
    """Human-readable state of the desync and the outstanding quorum."""
    disk_sha = compute_file_sha256(SPEC_PATH)
    pinned = _load_pinned_sha()
    manifest = GovernanceManifest.from_file(MANIFEST_PATH)
    verification = verify_governance_manifest(manifest, SPEC_PATH)
    registered = {t.trustee_id for t in manifest.trustees}
    pending = sorted(registered - set(verification.valid_signers))
    return {
        "ceremony": "governance_remint",
        "disk_spec_sha256": disk_sha,
        "canonical_spec_pinned_sha256": pinned,
        "inline_pins_agree": disk_sha == pinned,
        "manifest_sealed_sha256": manifest.spec_sha256,
        "desync": manifest.spec_sha256 != disk_sha,
        "quorum_threshold": manifest.quorum_threshold,
        "valid_signature_count_on_current_manifest": verification.valid_signature_count,
        "trustees_awaiting_re_signature": pending,
        "trustee_keys_are_held": "externally (never in this repository) — "
                                 "the ceremony can only record signatures "
                                 "that verify against the registered keys",
        "repo_governance_currently": "FAIL-CLOSED RED (correct): repository "
                                     "governance tests fail until trustees "
                                     "re-sign — bypassing it is forbidden",
    }


def _load_signatures_state(path: Path) -> list[dict[str, Any]]:
    """Only LIVE collected records reach mint; the ledger itself carries no
    credentials — every record is re-verified cryptographically in ``mint``."""
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    return [r for r in state.get("signatures", [])
            if r.get("status", "live") == "live"]


def cli() -> int:
    parser = argparse.ArgumentParser(
        description="Trustee re-certification ceremony (audit-only; fail-closed).")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("status", help="report the desync and outstanding quorum")
    req = sub.add_parser("request", help="write the certification request artifact")
    req.add_argument("--out", default=str(REQUEST_PATH))
    req.add_argument("--rationale", default=None)
    col = sub.add_parser("collect", help="record one verified trustee signature")
    col.add_argument("--trustee-id", required=True)
    col.add_argument("--private-key-env", required=True,
                     help="ENV VAR NAME holding the trustee's private key hex")
    col.add_argument("--signatures-state", default=str(SIGNATURES_PATH))
    mint = sub.add_parser("mint", help="re-certify the manifest if the quorum closes")
    mint.add_argument("--dry-run", action="store_true")
    mint.add_argument("--signatures-state", default=str(SIGNATURES_PATH))
    args = parser.parse_args()

    if args.mode == "status":
        print(json.dumps(status_report(), indent=2))
        print("\nExit stays 0 for status (informational); --mint enforces "
              "fail-closed when run.")
        return 0

    if args.mode == "request":
        request = build_request(SPEC_PATH, MANIFEST_PATH, rationale=args.rationale)
        out_path = Path(args.out)
        out_path.write_text(json.dumps(request, indent=2))
        print(f"request written: {out_path}")
        print(f"trustees sign exactly: {request['message_trustees_sign']}")
        return 0

    if args.mode == "collect":
        import os
        key = os.environ.get(args.private_key_env, "")
        if not key:
            print(f"REFUSED: env '{args.private_key_env}' holds no key value "
                  "(keys are never generated here).", file=sys.stderr)
            return 2
        request = json.loads(REQUEST_PATH.read_text())
        record = collect_signature(request, MANIFEST_PATH,
                                   args.trustee_id, key)
        state = _load_signatures_state(Path(args.signatures_state))
        for prior in state:
            if prior["trustee_id"] == record["trustee_id"]:
                print(f"REFUSED: duplicate signature from '{record['trustee_id']}' "
                      "is already in the ledger.", file=sys.stderr)
                return 2
        state.append(record)
        Path(args.signatures_state).write_text(json.dumps(
            {"kind": "ceremony_signature_ledger",
             "request_spec_sha256": request["spec_sha256"],
             "signatures": state,
             "disclosure": ("signature ledger for an in-flight ceremony — "
                            "ephemeral; private keys are NEVER written here")},
            indent=2))
        print(f"collected + verified: {record['trustee_id']} "
              f"({len(state)} live signature(s))")
        return 0

    if args.mode == "mint":
        try:
            request = json.loads(REQUEST_PATH.read_text())
            collected = _load_signatures_state(Path(args.signatures_state))
            draft = mint_manifest(request, MANIFEST_PATH, collected, SPEC_PATH)
        except GovernanceTamperingError as exc:
            print(f"MINT REFUSED (fail-closed writes nothing):\n{exc}",
                  file=sys.stderr)
            print(json.dumps(status_report(), indent=2), file=sys.stderr)
            return 1
        if args.dry_run:
            print("dry-run: draft manifest VALIDATES against production "
                  "verification and quorum; not written (--mint reruns the "
                  "check atomically, re-verifying from scratch).")
            return 0
        draft.save_to_file(MANIFEST_PATH)
        post = verify_governance_manifest(
            GovernanceManifest.from_file(MANIFEST_PATH), SPEC_PATH)
        print(f"manifest re-certified & re-verified valid={post.is_valid} "
              f"quorum={post.valid_signature_count}/{post.required_quorum}")
        return 0
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(cli())
