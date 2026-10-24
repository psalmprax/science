"""Subprocess adapter for the official competition verifiers.

Status: experimental

The proposer never runs the verifier in-process.  A separate process with the
verifier's required interpreter (Python 3.13 for the shipped ``kickmix``
extension) is started from the verifier directory, its JSON summary is parsed,
and the full command, exit code, hashes, duration, and stream tails are
recorded.  The grader salt is never written to the receipt: only its SHA-256
short id is kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.candidates import (
    VerifierReceipt,
    sha256_file,
)
from apedp_dyn.quantum_compilation.contracts import VerifierContract

__all__ = [
    "OfficialVerifierResult",
    "build_invsqrt_command",
    "invsqrt_verification_scope",
    "run_invsqrt_verifier",
]

OfficialVerifierResult = VerifierReceipt

_STREAM_TAIL_CHARS = 4000


def _salt_id(salt: str) -> str:
    if not salt:
        return "public"
    return "salt-" + hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16]


def _full_invsqrt_max_x(n_p: int) -> int:
    # Benchmark I defines x = |nu - nu'|^2 with components in
    # [-(2^(n_p-1)-1), 2^(n_p-1)-1], hence x <= 3 (2^n_p - 2)^2.
    return int(3 * (2**n_p - 2) ** 2)


def invsqrt_verification_scope(
    contract: VerifierContract,
    *,
    max_x: int | None,
    shots_per_x: int | None,
    salt: str,
) -> tuple[str, int, int]:
    """Return ``(scope, effective_max_x, effective_shots)``.

    ``full-domain`` is reserved for the complete input domain at the published
    shot count *and* a non-empty salt.  Anything weaker is development
    evidence and is labelled as such.
    """
    n_p = int(contract.constants["N_P"])
    full_max_x = _full_invsqrt_max_x(n_p)
    full_shots = int(contract.constants["DEFAULT_SHOTS_PER_X"])
    effective_max_x = full_max_x if max_x is None else int(max_x)
    effective_shots = full_shots if shots_per_x is None else int(shots_per_x)
    if not 1 <= effective_max_x <= full_max_x:
        raise ValueError(f"max_x must lie in [1, {full_max_x}], got {effective_max_x}")
    if effective_shots < 1:
        raise ValueError(f"shots_per_x must be positive, got {effective_shots}")
    if effective_max_x >= full_max_x and effective_shots == full_shots:
        scope = "full-domain" if salt else "full-domain-public"
    elif effective_max_x >= full_max_x:
        scope = "full-domain-reduced-shots"
    else:
        scope = "development-prefix"
    return scope, effective_max_x, effective_shots


def build_invsqrt_command(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    max_x: int | None = None,
    shots_per_x: int | None = None,
    salt: str = "",
    seed: int | None = None,
) -> list[str]:
    """Build the verifier command line without executing it."""
    command = [
        python_executable,
        str(contract.entrypoint),
        str(Path(artifact).resolve()),
    ]
    if max_x is not None:
        command.extend(["--max-x", str(int(max_x))])
    if shots_per_x is not None:
        command.extend(["--shots-per-x", str(int(shots_per_x))])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if salt:
        command.extend(["--salt", salt])
    return command


def _display_command(command: list[str], salt: str) -> tuple[str, ...]:
    if not salt:
        return tuple(command)
    return tuple("<redacted>" if token == salt else token for token in command)


def _extract_json(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def run_invsqrt_verifier(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    max_x: int | None = None,
    shots_per_x: int | None = None,
    salt: str = "",
    seed: int | None = None,
    timeout_s: float = 1800.0,
) -> VerifierReceipt:
    """Run the official invsqrt verifier and return a sealed receipt.

    The verifier is run in its own Python 3.13 process with the bundled
    ``kickmix`` on ``PYTHONPATH``.  A timeout or an unparseable summary is an
    ``ERROR`` receipt, never a pass.
    """
    contract.assert_locked()
    if salt and seed is not None:
        raise ValueError(
            "salt and seed are mutually exclusive: a fixed seed bypasses the "
            "hash-derived secret-salt schedule and cannot back a final claim."
        )
    scope, effective_max_x, effective_shots = invsqrt_verification_scope(
        contract, max_x=max_x, shots_per_x=shots_per_x, salt=salt
    )
    batch_size = int(contract.constants["DEFAULT_BATCH_SIZE"])
    num_shots = effective_max_x * effective_shots
    if num_shots % batch_size != 0:
        raise ValueError(
            f"max_x * shots_per_x ({effective_max_x} * {effective_shots} = "
            f"{num_shots}) must be a multiple of the verifier batch size "
            f"({batch_size})."
        )

    command = build_invsqrt_command(
        contract,
        artifact,
        python_executable=python_executable,
        max_x=max_x,
        shots_per_x=shots_per_x,
        salt=salt,
        seed=seed,
    )
    env = os.environ.copy()
    kickmix_path = str(contract.kickmix_root)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = kickmix_path if not existing else f"{kickmix_path}{os.pathsep}{existing}"

    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
            command,
            cwd=str(contract.verifier_root.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        raw_stdout: str | bytes = exc.stdout or ""
        raw_stderr: str | bytes = exc.stderr or ""
        stdout = (
            raw_stdout.decode("utf-8", errors="replace")
            if isinstance(raw_stdout, bytes)
            else raw_stdout
        )
        decoded_stderr = (
            raw_stderr.decode("utf-8", errors="replace")
            if isinstance(raw_stderr, bytes)
            else raw_stderr
        )
        stderr = decoded_stderr + f"\nTIMEOUT after {timeout_s}s"
        exit_code = -1
    duration = time.monotonic() - started

    payload = _extract_json(stdout)
    if payload is None:
        payload = {
            "status": "ERROR",
            "error": "official verifier produced no parseable JSON summary",
        }
    reported = str(payload.get("status", "ERROR")).upper()
    if reported == "PASS" and exit_code == 0:
        status = "PASS"
    elif reported == "FAIL" and exit_code != 0:
        status = "FAIL"
    else:
        status = "ERROR"

    score = payload.get("score")
    toffoli_equivalent = payload.get("toffoli_equivalent")
    num_qubits = payload.get("num_qubits")
    payload_with_scope = dict(payload)
    payload_with_scope["verification_scope"] = scope
    payload_with_scope["effective_max_x"] = effective_max_x
    payload_with_scope["effective_shots_per_x"] = effective_shots

    return VerifierReceipt(
        status=status,
        score=float(score) if isinstance(score, (int, float)) else None,
        toffoli_equivalent=(
            float(toffoli_equivalent)
            if isinstance(toffoli_equivalent, (int, float))
            else None
        ),
        num_qubits=int(num_qubits) if isinstance(num_qubits, int) else None,
        metrics=payload_with_scope,
        exit_code=exit_code,
        command=_display_command(command, salt),
        salt_id=_salt_id(salt),
        verification_scope=scope,
        verifier_sha256=contract.verifier_sha256,
        artifact_sha256=sha256_file(artifact),
        contract_sha256=contract.contract_sha256,
        duration_s=duration,
        stdout_tail=stdout[-_STREAM_TAIL_CHARS:],
        stderr_tail=stderr[-_STREAM_TAIL_CHARS:],
    )
