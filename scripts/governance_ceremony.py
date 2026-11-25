#!/usr/bin/env python3
"""The trustees' signing ceremony for the governance manifest: what to sign, and putting the signatures in.

Status: audit-only

This script never signs, never makes a key, and never changes the roster or the quorum. It does two
things:

    python scripts/governance_ceremony.py message
        prints the digests of the two files to be sealed and the one message each trustee signs.

    python scripts/governance_ceremony.py assemble \\
        --signature referee_lead_01:<128 hex>:<signed at, ISO-8601 UTC> \\
        --signature referee_stat_02:<128 hex>:<signed at, ISO-8601 UTC>
        builds the new manifest from the one in the repository (same trustees, same quorum, the log
        appended to, never rewritten), verifies it, and writes it only if it verifies.

**What is sealed** (owner's decision, 2026-11-25): the canonical specification, as before, and
beside it the discovery checks' thresholds (``specifications/discovery_checks_v1.json``). One message
binds both: ``apedp_dyn.epistemic.governance.combined_digest``.

A trustee signs with their own key, on their own machine:

    python -c "from apedp_dyn.epistemic.governance import sign_spec_sha256; \\
               print(sign_spec_sha256(input('private key hex: ').strip(), '<the message>'))"

and sends back three things: their trustee id, the 128 hex characters printed, and the time. The
private key is never sent, pasted into this repository, or given to an agent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.epistemic.governance import (  # noqa: E402
    GovernanceManifest,
    SealedFile,
    SignatureRecord,
    TransparencyLogEntry,
    combined_digest,
    compute_file_sha256,
    verify_ed25519_signature,
    verify_governance_manifest,
)

SPECIFICATIONS = ROOT / "specifications"
SPEC = SPECIFICATIONS / "canonical_preregistered_spec_v1.json"
MANIFEST = SPECIFICATIONS / "governance_manifest.json"
ALSO = (("discovery_checks_v1", "discovery_checks_v1.json"),)


def to_be_sealed(specifications: Path = SPECIFICATIONS) -> tuple[str, list[SealedFile]]:
    """The canonical specification's digest and the further files, each with its digest as on disk."""
    also = [SealedFile(spec_id, name, compute_file_sha256(specifications / name)) for spec_id, name in ALSO]
    return compute_file_sha256(specifications / SPEC.name), also


def message(manifest: GovernanceManifest, specifications: Path = SPECIFICATIONS) -> str:
    """The 64 hex characters each trustee signs."""
    spec_sha256, also = to_be_sealed(specifications)
    return combined_digest(manifest.spec_id, spec_sha256, also)


def assemble(manifest: GovernanceManifest, signatures: list[tuple[str, str, str]],
             specifications: Path = SPECIFICATIONS) -> GovernanceManifest:
    """The new manifest: the same trustees and quorum, the new digests and signatures, the log appended to.

    Raises ``ValueError`` if a signature is from no registered trustee, is given twice, or does not
    verify against that trustee's public key over the message; nothing is written by this function.
    """
    spec_sha256, also = to_be_sealed(specifications)
    signed = combined_digest(manifest.spec_id, spec_sha256, also)
    keys = {trustee.trustee_id: trustee.public_key_hex for trustee in manifest.trustees}
    seen: set[str] = set()
    for trustee_id, signature_hex, _signed_at in signatures:
        if trustee_id not in keys:
            raise ValueError(f"'{trustee_id}' is not a registered trustee; the roster is not changed here")
        if trustee_id in seen:
            raise ValueError(f"two signatures were given for '{trustee_id}'")
        seen.add(trustee_id)
        if not verify_ed25519_signature(keys[trustee_id], signed, signature_hex):
            raise ValueError(f"the signature of '{trustee_id}' does not verify over the message {signed}")
    log = list(manifest.transparency_log)
    previous = log[-1].entry_hash if log else "0" * 64
    for position, (trustee_id, _signature_hex, signed_at) in enumerate(signatures):
        action = "SPEC_REVISION" if position == 0 else "QUORUM_CO_SIGNATURE"
        index = len(log)
        entry_hash = TransparencyLogEntry.compute_hash(index, signed_at, action, signed, trustee_id, previous)
        log.append(TransparencyLogEntry(index=index, timestamp=signed_at, action=action, spec_sha256=signed,
                                        trustee_id=trustee_id, prev_entry_hash=previous, entry_hash=entry_hash))
        previous = entry_hash
    return GovernanceManifest(
        spec_id=manifest.spec_id, spec_sha256=spec_sha256, quorum_threshold=manifest.quorum_threshold,
        trustees=list(manifest.trustees),
        signatures=[SignatureRecord(trustee_id, signature_hex, signed_at) for trustee_id, signature_hex, signed_at in signatures],
        transparency_log=log, also_sealed=also)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("command", choices=("message", "assemble"))
    parser.add_argument("--signature", action="append", default=[],
                        help="trustee_id:signature_hex:signed_at, once for each trustee who signed")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    manifest = GovernanceManifest.from_file(args.manifest)
    spec_sha256, also = to_be_sealed()
    signed = message(manifest)
    if args.command == "message":
        print(f"{manifest.spec_id}: {spec_sha256}")
        for entry in also:
            print(f"{entry.spec_id} ({entry.file}): {entry.sha256}")
        print(f"quorum: {manifest.quorum_threshold} of {len(manifest.trustees)} "
              f"({', '.join(trustee.trustee_id for trustee in manifest.trustees)})")
        print(f"the message each trustee signs:\n{signed}")
        return 0
    given = []
    for text in args.signature:
        parts = text.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise SystemExit(f"--signature needs trustee_id:signature_hex:signed_at, got '{text[:40]}'. Nothing was written.")
        given.append((parts[0], parts[1], parts[2]))
    if len(given) < manifest.quorum_threshold:
        raise SystemExit(f"{len(given)} signatures given; the quorum is {manifest.quorum_threshold}. Nothing was written.")
    try:
        new = assemble(manifest, given)
    except ValueError as exc:
        raise SystemExit(f"{exc}. Nothing was written.") from exc
    result = verify_governance_manifest(new, SPEC)
    if not result.is_valid:
        raise SystemExit("the assembled manifest does not verify: " + "; ".join(result.errors) + ". Nothing was written.")
    new.save_to_file(args.manifest)
    print(f"written: {args.manifest} ({result.valid_signature_count} valid signatures of {result.required_quorum} "
          f"required, over {signed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
