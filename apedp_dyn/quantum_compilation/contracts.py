"""Hash-pinned contracts for the verifiable-circuit-compilation benchmarks.

Status: experimental

The competition verifier is the authoritative specification.  This module never
duplicates its thresholds: it hashes the verifier source tree and reads the
module-level constants that describe the benchmark instance directly from the
shipped source with :mod:`ast`.  A contract is locked before every verifier run;
any byte change to the verifier, the kickmix extension, or the recorded sample
artifact raises :class:`ContractIntegrityError`.

The hash is a *reproducibility* device, not a proof of correctness.  It ensures
that the evidence in the ledger was produced by the exact verifier version that
was reviewed, and that a later run cannot silently substitute another one.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "ContractIntegrityError",
    "VerifierContract",
    "default_competition_root",
    "extract_module_constants",
    "load_contract",
    "sha256_file",
    "sha256_tree",
]

BENCHMARK_ENTRYPOINTS = {
    "invsqrt": "verify_submission_invsqrt.py",
    "dqi": "verify_submission_dqi.py",
    "dfthcsos": "verify.py",
}

BENCHMARK_CONSTANTS = {
    "invsqrt": (
        "N_P",
        "ETA",
        "T_PER_TOFFOLI",
        "DEFAULT_SHOTS_PER_X",
        "DEFAULT_BATCH_SIZE",
    ),
    "dqi": (
        "DEFAULT_M",
        "DEFAULT_N",
        "DEFAULT_DEGREE",
        "DEFAULT_R",
        "DEFAULT_NUM_SHOTS",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_LOG2_DELTA",
    ),
    "dfthcsos": (
        "LAMBDA_FLOOR",
        "E_GAP_FLOOR",
        "QUBIT_CAP",
        "KEEP_BITS",
        "SUBMISSION_FILE",
    ),
}

SAMPLE_ARTIFACTS = {
    "invsqrt": ("sample_submission", "invsqrt.kmb"),
    "dqi": ("sample_submission", "dqi.kmb"),
    "dfthcsos": ("sample_submission", "dfthcsos", "submission.npz"),
}

# Hashing a very large sample on every run is unnecessary and slow.  The
# verifier source and kickmix extension are always hashed; the sample is hashed
# when it is small enough to bind without a noticeable I/O cost.
_SAMPLE_HASH_LIMIT_BYTES = 128 * 1024 * 1024

_RELATIVE_COMPETITION_ROOT = Path(
    "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation"
)


class ContractIntegrityError(RuntimeError):
    """Raised when a pinned verifier artifact has changed since contract creation."""


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory tree deterministically.

    Relative path names and file bytes are both included, so moving a file or
    changing its contents changes the digest.  ``__pycache__`` and bytecode
    files are ignored because they are not part of the specification.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Contract tree not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def extract_module_constants(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    """Read module-level literal assignments without executing the module.

    Only assignments of the form ``NAME = <literal>`` are accepted.  Anything
    computed at import time is deliberately skipped (and therefore not
    "pinned" by this helper); callers that need such values must obtain them
    from the verifier in its own runtime.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return found


def default_competition_root() -> Path:
    """Locate the competition bundle.

    Priority:
    1. ``APEDP_Q_COMPETITION_ROOT``
    2. ``<repository>/apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation``
    3. the current working directory (for a notebook launched inside the bundle)
    """
    override = os.environ.get("APEDP_Q_COMPETITION_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"APEDP_Q_COMPETITION_ROOT points to a missing directory: {candidate}"
            )
        return candidate

    repository_root = Path(__file__).resolve().parents[2]
    candidate = repository_root / _RELATIVE_COMPETITION_ROOT
    if candidate.is_dir():
        return candidate

    cwd_candidate = Path.cwd() / "google-verifiable-quantum-circuit-compilation"
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the verifiable-circuit-compilation bundle. Set "
        "APEDP_Q_COMPETITION_ROOT or place the bundle at "
        f"{candidate}."
    )


@dataclasses.dataclass(frozen=True)
class VerifierContract:
    """A byte-pinned view of one benchmark's authoritative verifier."""

    benchmark: str
    competition_root: Path
    verifier_root: Path
    entrypoint: Path
    kickmix_root: Path
    sample_artifact: Path | None
    verifier_sha256: str
    kickmix_sha256: str
    sample_sha256: str | None
    constants: dict[str, Any]
    contract_sha256: str

    def assert_locked(self) -> None:
        """Recompute the pin and raise if any bound artifact changed."""
        current_verifier = sha256_tree(self.verifier_root)
        if current_verifier != self.verifier_sha256:
            raise ContractIntegrityError(
                f"{self.benchmark} verifier tree changed: expected "
                f"{self.verifier_sha256}, found {current_verifier}"
            )
        current_kickmix = sha256_tree(self.kickmix_root)
        if current_kickmix != self.kickmix_sha256:
            raise ContractIntegrityError(
                f"kickmix tree changed: expected {self.kickmix_sha256}, "
                f"found {current_kickmix}"
            )
        if self.sample_artifact is not None and self.sample_sha256 is not None:
            current_sample = sha256_file(self.sample_artifact)
            if current_sample != self.sample_sha256:
                raise ContractIntegrityError(
                    f"sample artifact changed: expected {self.sample_sha256}, "
                    f"found {current_sample}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "competition_root": str(self.competition_root),
            "verifier_root": str(self.verifier_root),
            "entrypoint": str(self.entrypoint),
            "kickmix_root": str(self.kickmix_root),
            "sample_artifact": str(self.sample_artifact) if self.sample_artifact else None,
            "verifier_sha256": self.verifier_sha256,
            "kickmix_sha256": self.kickmix_sha256,
            "sample_sha256": self.sample_sha256,
            "constants": dict(self.constants),
            "contract_sha256": self.contract_sha256,
        }


def _contract_digest(fields: dict[str, Any]) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_contract(
    benchmark: str,
    competition_root: Path | str | None = None,
) -> VerifierContract:
    """Build and lock the contract for ``benchmark``."""
    if benchmark not in BENCHMARK_ENTRYPOINTS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {sorted(BENCHMARK_ENTRYPOINTS)}")

    root = Path(competition_root).expanduser().resolve() if competition_root else default_competition_root()
    verifier_root = root / "verifiers" / benchmark
    entrypoint = verifier_root / BENCHMARK_ENTRYPOINTS[benchmark]
    kickmix_root = root / "kickmix"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Verifier entrypoint not found: {entrypoint}")
    if not (kickmix_root / "kickmix.so").is_file():
        raise FileNotFoundError(f"kickmix extension not found under {kickmix_root}")

    sample_parts = SAMPLE_ARTIFACTS[benchmark]
    candidate_sample = root.joinpath(*sample_parts)
    sample_artifact: Path | None = candidate_sample if candidate_sample.is_file() else None
    sample_sha = None
    if sample_artifact is not None and sample_artifact.stat().st_size <= _SAMPLE_HASH_LIMIT_BYTES:
        sample_sha = sha256_file(sample_artifact)

    constants = extract_module_constants(
        entrypoint, BENCHMARK_CONSTANTS[benchmark]
    )
    verifier_sha = sha256_tree(verifier_root)
    kickmix_sha = sha256_tree(kickmix_root)
    fields = {
        "benchmark": benchmark,
        "competition_root": str(root),
        "verifier_root": str(verifier_root),
        "entrypoint": str(entrypoint),
        "kickmix_root": str(kickmix_root),
        "sample_artifact": str(sample_artifact) if sample_artifact else None,
        "verifier_sha256": verifier_sha,
        "kickmix_sha256": kickmix_sha,
        "sample_sha256": sample_sha,
        "constants": constants,
    }
    contract = VerifierContract(
        benchmark=benchmark,
        competition_root=root,
        verifier_root=verifier_root,
        entrypoint=entrypoint,
        kickmix_root=kickmix_root,
        sample_artifact=sample_artifact,
        verifier_sha256=verifier_sha,
        kickmix_sha256=kickmix_sha,
        sample_sha256=sample_sha,
        constants=constants,
        contract_sha256="",
    )
    locked = dataclasses.replace(contract, contract_sha256=_contract_digest(fields))
    return locked
