"""Subprocess adapter for the official competition verifiers (Benchmarks I, II, III).

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
from apedp_dyn.quantum_compilation.contracts import VerifierContract, sha256_tree

__all__ = [
    "DEFAULT_DFTHCSOS_SHOTS_PER_ADDRESS",
    "OfficialVerifierResult",
    "ReceiptArtifactHashMismatchError",
    "StructuralEquivalenceError",
    "build_dfthcsos_command",
    "build_dqi_command",
    "build_invsqrt_command",
    "dfthcsos_verification_scope",
    "dqi_verification_scope",
    "enforce_structural_diversity",
    "invsqrt_verification_scope",
    "run_dfthcsos_verifier",
    "run_dqi_verifier",
    "run_invsqrt_verifier",
    "run_verifier",
]

OfficialVerifierResult = VerifierReceipt


class ReceiptArtifactHashMismatchError(RuntimeError):
    """Raised when an official receipt does not cryptographically bind to the evaluated artifact."""


class StructuralEquivalenceError(RuntimeError):
    """Raised when a candidate's structural DAG fingerprint equals the incumbent champion (anti-clone gate)."""


def enforce_structural_diversity(
    candidate_structural_fp: str,
    champion_structural_fp: str,
) -> None:
    """Enforce that candidate is structurally distinct from champion to prevent renamed baselines."""
    if not candidate_structural_fp:
        raise ValueError("candidate_structural_fp cannot be empty")
    if not champion_structural_fp:
        return
    if candidate_structural_fp == champion_structural_fp:
        raise StructuralEquivalenceError(
            f"Candidate structural fingerprint ({candidate_structural_fp}) matches incumbent champion. "
            "Renamed baseline circuits and trivial gate re-orderings are barred from promotion."
        )


_STREAM_TAIL_CHARS = 4000
DEFAULT_DFTHCSOS_SHOTS_PER_ADDRESS = 64
DEFAULT_DQI_NUM_SHOTS = 400 * 256


def _salt_id(salt: str) -> str:
    if not salt:
        return "public"
    return "salt-" + hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16]


def _artifact_sha256(path: Path) -> str:
    target = Path(path)
    if target.is_dir():
        return sha256_tree(target)
    return sha256_file(target)


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


def dqi_verification_scope(
    contract: VerifierContract,
    *,
    num_shots: int | None = None,
    m: int | None = None,
    n: int | None = None,
    degree: int | None = None,
    r: int | None = None,
    log2_delta: float | None = None,
    salt: str = "",
) -> tuple[str, int]:
    """Return ``(scope, effective_num_shots)`` for Benchmark II (DQI)."""
    full_shots = int(
        contract.constants.get("DEFAULT_NUM_SHOTS", DEFAULT_DQI_NUM_SHOTS)
    )
    default_m = int(contract.constants["DEFAULT_M"])
    default_n = int(contract.constants["DEFAULT_N"])
    default_degree = int(contract.constants["DEFAULT_DEGREE"])
    default_r = int(contract.constants["DEFAULT_R"])
    default_log2_delta = float(contract.constants["DEFAULT_LOG2_DELTA"])

    effective_shots = full_shots if num_shots is None else int(num_shots)
    if effective_shots < 1:
        raise ValueError(f"num_shots must be positive, got {effective_shots}")

    eff_m = default_m if m is None else int(m)
    eff_n = default_n if n is None else int(n)
    eff_degree = default_degree if degree is None else int(degree)
    eff_r = default_r if r is None else int(r)
    eff_log2_delta = default_log2_delta if log2_delta is None else float(log2_delta)

    full_instance = (
        eff_m == default_m
        and eff_n == default_n
        and eff_degree == default_degree
        and eff_r == default_r
        and eff_log2_delta <= default_log2_delta
    )
    if not full_instance:
        scope = "development-prefix"
    elif effective_shots >= full_shots:
        scope = "full-domain" if salt else "full-domain-public"
    else:
        scope = "full-domain-reduced-shots"
    return scope, effective_shots


def dfthcsos_verification_scope(
    contract: VerifierContract,
    *,
    shots_per_address: int | None = None,
    salt: str = "",
) -> tuple[str, int]:
    """Return ``(scope, effective_shots_per_address)`` for Benchmark III (DFTHCSOS)."""
    full_shots = int(
        contract.constants.get(
            "SHOTS_PER_ADDRESS", DEFAULT_DFTHCSOS_SHOTS_PER_ADDRESS
        )
    )
    effective_shots = full_shots if shots_per_address is None else int(shots_per_address)
    if effective_shots < 1:
        raise ValueError(f"shots_per_address must be positive, got {effective_shots}")
    if effective_shots >= full_shots:
        scope = "full-domain" if salt else "full-domain-public"
    else:
        scope = "full-domain-reduced-shots"
    return scope, effective_shots


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
    """Build the Benchmark I verifier command line without executing it."""
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


def build_dqi_command(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    num_shots: int | None = None,
    batch_size: int | None = None,
    m: int | None = None,
    n: int | None = None,
    degree: int | None = None,
    r: int | None = None,
    log2_delta: float | None = None,
    max_workers: int | None = None,
    salt: str = "",
    seed: int | None = None,
) -> list[str]:
    """Build the Benchmark II (DQI) verifier command line without executing it."""
    command = [
        python_executable,
        str(contract.entrypoint),
        str(Path(artifact).resolve()),
    ]
    if m is not None:
        command.extend(["--m", str(int(m))])
    if n is not None:
        command.extend(["--n", str(int(n))])
    if degree is not None:
        command.extend(["--degree", str(int(degree))])
    if r is not None:
        command.extend(["--r", str(int(r))])
    if num_shots is not None:
        command.extend(["--num-shots", str(int(num_shots))])
    if batch_size is not None:
        command.extend(["--batch-size", str(int(batch_size))])
    if log2_delta is not None:
        command.extend(["--log2-delta", str(float(log2_delta))])
    if max_workers is not None:
        command.extend(["--max-workers", str(int(max_workers))])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    if salt:
        command.extend(["--salt", salt])
    return command


def build_dfthcsos_command(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    factorization: Path | None = None,
    tree_edges: Path | None = None,
    shots_per_address: int | None = None,
    salt: str = "",
    seed: int | None = None,
) -> list[str]:
    """Build the Benchmark III (DFTHCSOS) verifier command line without executing it."""
    command = [
        python_executable,
        str(contract.entrypoint),
    ]
    resolved_artifact = Path(artifact).resolve()
    if factorization is not None:
        command.append(str(Path(factorization).resolve()))
        command.append(str(resolved_artifact))
    elif resolved_artifact.name == "submission.npz":
        command.append(str(resolved_artifact.parent))
    else:
        command.append(str(resolved_artifact))
    if tree_edges is not None:
        command.extend(["--tree-edges", str(Path(tree_edges).resolve())])
    if shots_per_address is not None:
        command.extend(["--shots-per-address", str(int(shots_per_address))])
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


def _execute_verifier_subprocess(
    contract: VerifierContract,
    artifact: Path,
    command: list[str],
    *,
    scope: str,
    salt: str,
    timeout_s: float,
    extra_metrics: dict[str, Any] | None = None,
) -> VerifierReceipt:
    """Execute ``command`` in an isolated subprocess and seal the receipt."""
    env = os.environ.copy()
    kickmix_path = str(contract.kickmix_root)
    verifier_parent = str(contract.verifier_root.parent)
    existing = env.get("PYTHONPATH")
    prefix = f"{kickmix_path}{os.pathsep}{verifier_parent}"
    env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"

    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
            command,
            cwd=verifier_parent,
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
    toffoli_equivalent = payload.get(
        "toffoli_equivalent", payload.get("toffoli_count")
    )
    num_qubits = payload.get("num_qubits")
    payload_with_scope = dict(payload)
    payload_with_scope["verification_scope"] = scope
    if extra_metrics:
        payload_with_scope.update(extra_metrics)

    expected_artifact_sha256 = _artifact_sha256(artifact)
    payload_artifact_sha256 = payload.get("artifact_sha256") or payload.get("circuit_sha256")
    if payload_artifact_sha256 and payload_artifact_sha256 != expected_artifact_sha256:
        raise ReceiptArtifactHashMismatchError(
            f"Receipt artifact hash ({payload_artifact_sha256}) does not match "
            f"expected artifact SHA-256 ({expected_artifact_sha256})"
        )

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
        artifact_sha256=expected_artifact_sha256,
        contract_sha256=contract.contract_sha256,
        duration_s=duration,
        stdout_tail=stdout[-_STREAM_TAIL_CHARS:],
        stderr_tail=stderr[-_STREAM_TAIL_CHARS:],
    )


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
    return _execute_verifier_subprocess(
        contract,
        artifact,
        command,
        scope=scope,
        salt=salt,
        timeout_s=timeout_s,
        extra_metrics={
            "effective_max_x": effective_max_x,
            "effective_shots_per_x": effective_shots,
        },
    )


def run_dqi_verifier(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    num_shots: int | None = None,
    batch_size: int | None = None,
    m: int | None = None,
    n: int | None = None,
    degree: int | None = None,
    r: int | None = None,
    log2_delta: float | None = None,
    max_workers: int | None = None,
    salt: str = "",
    seed: int | None = None,
    timeout_s: float = 1800.0,
) -> VerifierReceipt:
    """Run the official Benchmark II (DQI) verifier and return a sealed receipt."""
    contract.assert_locked()
    if salt and seed is not None:
        raise ValueError(
            "salt and seed are mutually exclusive: a fixed seed bypasses the "
            "hash-derived secret-salt schedule and cannot back a final claim."
        )
    scope, effective_shots = dqi_verification_scope(
        contract,
        num_shots=num_shots,
        m=m,
        n=n,
        degree=degree,
        r=r,
        log2_delta=log2_delta,
        salt=salt,
    )
    eff_batch = (
        int(contract.constants["DEFAULT_BATCH_SIZE"])
        if batch_size is None
        else int(batch_size)
    )
    if eff_batch < 1:
        raise ValueError(f"batch_size must be positive, got {eff_batch}")
    if effective_shots % eff_batch != 0:
        raise ValueError(
            f"num_shots ({effective_shots}) must be a multiple of the verifier "
            f"batch size ({eff_batch})."
        )

    command = build_dqi_command(
        contract,
        artifact,
        python_executable=python_executable,
        num_shots=num_shots,
        batch_size=batch_size,
        m=m,
        n=n,
        degree=degree,
        r=r,
        log2_delta=log2_delta,
        max_workers=max_workers,
        salt=salt,
        seed=seed,
    )
    return _execute_verifier_subprocess(
        contract,
        artifact,
        command,
        scope=scope,
        salt=salt,
        timeout_s=timeout_s,
        extra_metrics={
            "effective_num_shots": effective_shots,
            "effective_batch_size": eff_batch,
        },
    )


def run_dfthcsos_verifier(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    factorization: Path | None = None,
    tree_edges: Path | None = None,
    shots_per_address: int | None = None,
    salt: str = "",
    seed: int | None = None,
    timeout_s: float = 1800.0,
) -> VerifierReceipt:
    """Run the official Benchmark III (DFTHCSOS) verifier and return a sealed receipt."""
    contract.assert_locked()
    if salt and seed is not None:
        raise ValueError(
            "salt and seed are mutually exclusive: a fixed seed bypasses the "
            "hash-derived secret-salt schedule and cannot back a final claim."
        )
    scope, effective_shots = dfthcsos_verification_scope(
        contract, shots_per_address=shots_per_address, salt=salt
    )
    command = build_dfthcsos_command(
        contract,
        artifact,
        python_executable=python_executable,
        factorization=factorization,
        tree_edges=tree_edges,
        shots_per_address=shots_per_address,
        salt=salt,
        seed=seed,
    )
    return _execute_verifier_subprocess(
        contract,
        artifact,
        command,
        scope=scope,
        salt=salt,
        timeout_s=timeout_s,
        extra_metrics={
            "effective_shots_per_address": effective_shots,
        },
    )


def run_verifier(
    contract: VerifierContract,
    artifact: Path,
    *,
    python_executable: str,
    salt: str = "",
    seed: int | None = None,
    timeout_s: float = 1800.0,
    **kwargs: Any,
) -> VerifierReceipt:
    """Dispatch to the official verifier for ``contract.benchmark``."""
    benchmark = contract.benchmark
    if benchmark == "invsqrt":
        return run_invsqrt_verifier(
            contract,
            artifact,
            python_executable=python_executable,
            salt=salt,
            seed=seed,
            timeout_s=timeout_s,
            **kwargs,
        )
    if benchmark == "dqi":
        return run_dqi_verifier(
            contract,
            artifact,
            python_executable=python_executable,
            salt=salt,
            seed=seed,
            timeout_s=timeout_s,
            **kwargs,
        )
    if benchmark == "dfthcsos":
        return run_dfthcsos_verifier(
            contract,
            artifact,
            python_executable=python_executable,
            salt=salt,
            seed=seed,
            timeout_s=timeout_s,
            **kwargs,
        )
    raise ValueError(f"Unsupported benchmark {benchmark!r}")
