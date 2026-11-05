"""Pinned ``kickmix`` builds: the extensions a verifier run may use.

Why this module exists
----------------------
:mod:`apedp_dyn.quantum_compilation.contracts` hashes the whole ``kickmix`` *tree*, so a
receipt cannot come from an unreviewed verifier bundle. That is the right pin for the
shipped competition build, but it cannot express the second build the DQI work needs: the
shipped ``kickmix.so`` has no Z-basis measurement (``mr``), so every MR-gated arm and
every measured clear has to run on an extension compiled from the vendored sources. Until
now that build had no identity — it was a file under a gitignored ``build/`` directory,
and a receipt naming it proved nothing about which bytes ran.

Two builds are allowed for a verifier run, and nothing else:

``shipped``
    ``kickmix/kickmix.so`` as the competition bundle ships it. No ``mr``.
``mr``
    ``scripts/build_kickmix_mr.py``: the vendored tree at the recorded commit plus
    ``scripts/kickmix_mr_opcode.patch``, compiled with CPython 3.13 and pybind11 2.12.0.
    Byte-reproducible: two builds in different output directories produce the same
    digest, recorded in ``benchmarks/quantum_compilation/kickmix_mr_build.json``.

A third, older local build (``e57dcc47…``) is recorded with ``allowed_for_verifier=False``.
It was compiled from a mutated working tree rather than from the pinned commit plus patch,
so it cannot be regenerated from the repository — and several earlier MR receipts carry its
digest. It is behaviourally identical to the pinned MR build on the probe recorded in that
receipt (same 60 public methods, same per-instruction counts on a GF(2¹²) inverse + ``mr``
circuit), which is why those arms remain valid measurements; it is not a build a new
receipt may cite.

Status: audit-only
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_VERIFIER_BUILDS",
    "LEGACY_LOCAL_BUILD",
    "MR_BUILD",
    "MR_BUILD_RECIPE",
    "PinnedBuild",
    "SHIPPED_BUILD",
    "UnpinnedKickmixBuildError",
    "assert_allowed",
    "identify",
    "sha256_file",
]


class UnpinnedKickmixBuildError(RuntimeError):
    """Raised when a ``kickmix`` extension is not one of the pinned verifier builds."""


@dataclasses.dataclass(frozen=True)
class PinnedBuild:
    """One ``kickmix`` extension, identified by the SHA-256 of its bytes."""

    name: str
    sha256: str
    allowed_for_verifier: bool
    has_mr: bool
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "allowed_for_verifier": self.allowed_for_verifier,
            "has_mr": self.has_mr,
            "provenance": dict(self.provenance),
        }


#: The competition bundle's extension. Hashes to the ``pinned_kickmix_sha256`` recorded in
#: ``benchmarks/quantum_compilation/dqi_official_compatibility.json``.
SHIPPED_BUILD = PinnedBuild(
    name="shipped",
    sha256="c66b3216c4029e0cdfd09054306aa0feadf53ab2f50c1164e37f752874e5005e",
    allowed_for_verifier=True,
    has_mr=False,
    provenance={
        "file": "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/kickmix.so",
        "source": "shipped with the competition bundle; no source in this repository",
        "how_built": "not built here",
    },
)

#: The reproducible MR-capable build. Both digests below were produced by two independent
#: runs of ``scripts/build_kickmix_mr.py`` into different output directories.
MR_BUILD = PinnedBuild(
    name="mr",
    sha256="36b1e6ef63686b0fe872d89ecf178ea3e72fb4afce956397ad484ebee8ddb939",
    allowed_for_verifier=True,
    has_mr=True,
    provenance={
        "file": "build/kickmix_mr/lib/kickmix.cpython-313-x86_64-linux-gnu.so (gitignored)",
        "script": "scripts/build_kickmix_mr.py",
        "source_commit": "638777fac4379bc62e2a1e4a52cf1a1753895d48",
        "source_tree": "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix/_qualtran_src",
        "patch": "scripts/kickmix_mr_opcode.patch",
        "patch_sha256": "9d96fd3a7099791d0896fe984299aa0e32f4c96a6a8977d02609078470264b0f",
        "interpreter": "CPython 3.13.14",
        "pybind11": "2.12.0",
        "public_methods": 60,
        "mr_if": False,
    },
)

#: Not a verifier build: compiled from a mutated working tree, so it cannot be regenerated
#: from the repository. Kept named because earlier receipts cite its digest.
LEGACY_LOCAL_BUILD = PinnedBuild(
    name="legacy-working-tree",
    sha256="e57dcc471213e43f79c767e585514aaa8315168235d4c5fd5539f59d2217c378",
    allowed_for_verifier=False,
    has_mr=True,
    provenance={
        "file": "build/kickmix_mr/lib (earlier build, replaced)",
        "how_built": "copied from the vendored working tree, whose checkout carries uncommitted "
                     "edits; not reproducible from the pinned commit plus patch",
        "cited_by": [
            "kickmix_mr_arm_costs.json",
            "gf54_mul_kickmix.json",
            "designer_directions.json",
            "dqi_region_ledger.json",
            "dqi_schedule_campaign.json",
        ],
        "equivalence": "same 60 public methods and same per-instruction counts as the pinned "
                       "mr build on the GF(2^12) inverse + mr probe recorded in "
                       "benchmarks/quantum_compilation/kickmix_mr_build.json",
    },
)

#: What a verifier run may load. Anything else raises.
ALLOWED_VERIFIER_BUILDS: tuple[PinnedBuild, ...] = (SHIPPED_BUILD, MR_BUILD)

#: The recipe the ``mr`` pin was produced by, for callers that want to rebuild it.
MR_BUILD_RECIPE: dict[str, Any] = {
    "script": "scripts/build_kickmix_mr.py",
    "command": "{python} scripts/build_kickmix_mr.py --out build/kickmix_mr --record "
               "benchmarks/quantum_compilation/kickmix_mr_build.json",
    "requires": {"python": "3.13", "pybind11": "2.12.0", "setuptools": "installed",
                 "tools": ["git", "tar", "patch", "c++"]},
    "expected_sha256": MR_BUILD.sha256,
}


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify(path: Path | str) -> PinnedBuild | None:
    """Return the pinned build whose digest matches ``path``, or ``None``."""
    digest = sha256_file(path)
    for build in (SHIPPED_BUILD, MR_BUILD, LEGACY_LOCAL_BUILD):
        if build.sha256 == digest:
            return build
    return None


def assert_allowed(path: Path | str, *, role: str | None = None) -> PinnedBuild:
    """Return the pinned build for ``path``, or raise if a verifier run may not use it.

    ``role`` restricts the check to ``"shipped"`` or ``"mr"`` when a caller needs a
    specific instruction surface (``mr`` for a measured clear, ``shipped`` for a bundle
    contract).
    """
    build = identify(path)
    if build is None:
        allowed = ", ".join(f"{b.name}={b.sha256[:12]}…" for b in ALLOWED_VERIFIER_BUILDS)
        raise UnpinnedKickmixBuildError(
            f"{path} hashes to {sha256_file(path)[:16]}…, which is not a pinned kickmix "
            f"build; a verifier run may use only {allowed}"
        )
    if not build.allowed_for_verifier:
        raise UnpinnedKickmixBuildError(
            f"{path} is the {build.name!r} build ({build.sha256[:12]}…), which is recorded "
            f"for provenance but is not reproducible from the repository and may not "
            f"produce a verifier receipt; rebuild with scripts/build_kickmix_mr.py"
        )
    if role is not None and build.name != role:
        raise UnpinnedKickmixBuildError(
            f"{path} is the {build.name!r} build, but this call requires {role!r} "
            f"({build.sha256[:12]}… does not provide it)"
            + ("" if build.has_mr else "; the shipped build has no `mr`")
        )
    return build
