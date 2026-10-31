"""Chained seal over the burned-holdout registry.

The anti-snooping guard in :mod:`apedp_dyn.epistemic.experiment_ledger` is only as
strong as the durability of the file it lives in. That file is an append-only
JSONL inside the repository, so before this module the registry could be
truncated, edited, or restored from an earlier commit and every burned slice
would silently come back alive: the ledger rebuilds its burned set by *reading*
the file, and a file that no longer contains a burn reads as a file that never
had one.

This module makes that visible. Each line of the registry is folded into a
running SHA-256 chain, and the chain's terminal digest is written to a small
sidecar. :func:`verify_seal` recomputes the chain and fails closed if the
line count, the byte length or the digest disagree with the seal. The three
properties it buys:

* **Truncation** (deleting the trailing burn lines) changes both the count and
  the chain.
* **An edited burn** (changing a candidate id, a slice id or an outcome inside a
  line) changes the chain from that line onward.
* **A rewind** (restoring the registry from an earlier commit while keeping the
  current seal) disagrees on the count.

What it deliberately does not claim: someone who reverts *both* the registry and
its sidecar, or who edits the seal to match, defeats it — there is no key and no
external witness. The seal is a local integrity check against accident and
casual reversion, and the sidecar's digest is itself pinned by
``tests/test_holdout_seal.py``, so a rewind has to change a test to pass. An
external append-only store (``FullLineageExperimentLedger`` accepts a
``storage_backend``) is the stronger form and is not what this module is.

Status: audit-only
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SEAL_VERSION = 1
#: Domain-separated genesis block, so an empty registry has a defined chain head
#: and a chain cannot be confused with one computed over different content.
_GENESIS = hashlib.sha256(b"apedp.holdout_seal.v1\x00").digest()


class HoldoutSealError(RuntimeError):
    """Raised when the burned-holdout registry does not match its seal."""


@dataclass(frozen=True)
class Seal:
    """One verified state of the registry."""

    artifact: str
    lines: int
    byte_length: int
    chain_sha256: str

    def to_dict(self) -> dict:
        return {
            "seal_version": SEAL_VERSION,
            "artifact": self.artifact,
            "lines": self.lines,
            "byte_length": self.byte_length,
            "chain_sha256": self.chain_sha256,
        }


def seal_path(path: Path | str) -> Path:
    """The sidecar holding the registry's terminal chain digest."""
    return Path(str(path) + ".seal")


def _chain(path: Path) -> tuple[str, int, int]:
    """Fold every non-empty line into the chain; return (digest, lines, bytes)."""
    digest = _GENESIS
    lines = 0
    length = 0
    with open(path, "rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            length += len(raw)
            lines += 1
            digest = hashlib.sha256(digest + raw).digest()
    return digest.hex(), lines, length


def compute_seal(path: Path | str) -> Seal:
    """The seal the registry on disk implies right now."""
    artifact = Path(path)
    digest, lines, length = _chain(artifact)
    return Seal(artifact=str(artifact), lines=lines, byte_length=length,
                chain_sha256=digest)


def write_seal(path: Path | str) -> Seal:
    """Recompute the chain and write the sidecar. Returns what was written."""
    seal = compute_seal(path)
    target = seal_path(path)
    target.write_text(json.dumps(seal.to_dict(), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return seal


def read_seal(path: Path | str) -> Seal:
    """Load the recorded seal, raising if it is absent or malformed."""
    target = seal_path(path)
    if not target.is_file():
        raise HoldoutSealError(
            f"the burned-holdout registry at {path} has no seal at {target}; a "
            f"registry that cannot be checked cannot be trusted"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HoldoutSealError(f"unreadable seal at {target}: {exc}") from exc
    if int(payload.get("seal_version", 0)) != SEAL_VERSION:
        raise HoldoutSealError(
            f"seal at {target} declares version {payload.get('seal_version')!r}, "
            f"expected {SEAL_VERSION}"
        )
    return Seal(artifact=str(payload.get("artifact")),
                lines=int(payload["lines"]),
                byte_length=int(payload["byte_length"]),
                chain_sha256=str(payload["chain_sha256"]))


def verify_seal(path: Path | str) -> Seal:
    """Fail closed unless the registry matches its seal exactly.

    A registry that does not exist yet is a fresh registry, not a tampered one:
    the chain is defined from a domain-separated genesis and the seal reports
    zero lines. Once the file exists, an absent seal is an error.
    """
    artifact = Path(path)
    if not artifact.is_file():
        return Seal(artifact=str(artifact), lines=0, byte_length=0,
                    chain_sha256=_GENESIS.hex())

    recorded = read_seal(artifact)
    actual = compute_seal(artifact)
    problems: list[str] = []
    if actual.lines != recorded.lines:
        direction = "fewer" if actual.lines < recorded.lines else "more"
        problems.append(
            f"{actual.lines} burn records present but the seal names "
            f"{recorded.lines} ({direction} lines than sealed)"
        )
    if actual.byte_length != recorded.byte_length:
        problems.append(
            f"registry is {actual.byte_length} bytes, seal names "
            f"{recorded.byte_length}"
        )
    if actual.chain_sha256 != recorded.chain_sha256:
        problems.append(
            f"chain digest {actual.chain_sha256} does not match the sealed "
            f"{recorded.chain_sha256}"
        )
    if problems:
        raise HoldoutSealError(
            "BURNED-HOLDOUT REGISTRY FAILED ITS SEAL (the anti-snooping record "
            "has been edited, truncated, or restored from an earlier state): "
            + "; ".join(problems)
        )
    return actual
