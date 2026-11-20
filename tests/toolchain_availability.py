"""Guards for tests that drive an external toolchain: Lean, Cargo.

A test that runs ``lean`` or ``cargo build`` cannot pass where the toolchain is
absent, and until now such tests simply failed there: three universal-bridge
tests on any machine without Lean, two sandbox tests on any machine where the
pinned Rust toolchain is not installed. A failure that means "nothing was
installed" trains people to read past failures.

The rule is the one ``tests/conftest.py`` already applies to the native kernel
and torch. Locally, a missing toolchain is a skip with the reason. Where
``APEDP_REQUIRE_NATIVE=1`` (CI sets it), it is a failure, because there a skip
would mean the check silently stopped running: CI is expected to provide the
toolchain, and the workflow does.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import native_builds_required

__all__ = ["require_cargo", "require_lean"]

ROOT = Path(__file__).resolve().parents[1]


def _absent(what: str, how: str) -> None:
    message = f"{what} is not available here ({how})"
    if native_builds_required():
        pytest.fail(f"{message}, and APEDP_REQUIRE_NATIVE is set: this environment must provide it")
    pytest.skip(message)


def require_lean() -> str:
    """The ``lean`` executable, or skip (fail where native builds are required)."""
    lean = shutil.which("lean")
    if lean is None:
        _absent("the Lean toolchain", "no `lean` on PATH")
        raise AssertionError("unreachable")  # pragma: no cover
    return lean


def require_cargo() -> str:
    """A ``cargo`` that can run in this repository, or skip.

    ``cargo`` being on PATH is not enough: ``rust-toolchain.toml`` pins a
    version, and where rustup has not installed that version every cargo
    command first tries to download it. Asking for the version is the cheapest
    command that goes through the same resolution.
    """
    cargo = shutil.which("cargo")
    if cargo is None:
        _absent("the Rust toolchain", "no `cargo` on PATH")
        raise AssertionError("unreachable")  # pragma: no cover
    probe = subprocess.run(
        [cargo, "--version"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        lines = [
            line.strip() for line in (probe.stderr or probe.stdout).splitlines() if line.strip()
        ]
        errors = [line for line in lines if line.startswith("error")]
        detail = (errors or lines or ["no output"])[0][:200]
        _absent(
            "the pinned Rust toolchain", f"`cargo --version` failed in the repository: {detail}"
        )
    return cargo
