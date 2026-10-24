"""Budgeted APEDP-Q search for Benchmark I (inverse square roots).

Status: experimental

The controller screens precision points with the exact integer model, keeps an
append-only ledger of accepted and rejected candidates, and sends only the
short-listed artifacts to the official verifier.  Development verification is
explicitly labelled ``development-prefix``; a leaderboard claim requires the
salted full-domain run (which is intentionally opt-in because it is 200M
simulated shots).

Run with the Python 3.13 interpreter that matches ``kickmix``::

    python -m apedp_dyn.quantum_compilation.search --budget 80
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from apedp_dyn.quantum_compilation.candidates import (
    CircuitCandidate,
    candidate_id_for,
    sha256_file,
)
from apedp_dyn.quantum_compilation.contracts import VerifierContract, load_contract
from apedp_dyn.quantum_compilation.invsqrt import (
    ScreenResult,
    base_precision,
    full_domain_xs,
    neighbor_precisions,
    refinement_precisions,
    screen_precision,
    subset_xs,
    write_circuit,
)
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger
from apedp_dyn.quantum_compilation.referee import run_invsqrt_verifier

__all__ = ["SearchOutcome", "run_invsqrt_search"]

_GENERATOR_ID = "invsqrt-nr-precision-v1"


@dataclasses.dataclass(frozen=True)
class SearchOutcome:
    status: str
    report: dict[str, Any]


def _key(parameters: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(parameters.items()))


def _as_params(key: tuple[tuple[str, int], ...]) -> dict[str, int]:
    return dict(key)


def _candidate_record(
    contract: VerifierContract,
    parameters: dict[str, int],
    *,
    parent_candidate_id: str | None,
    artifact_path: Path | None,
    screen: ScreenResult,
) -> CircuitCandidate:
    artifact_sha = sha256_file(artifact_path) if artifact_path else ""
    return CircuitCandidate(
        candidate_id=candidate_id_for(
            "invsqrt",
            _GENERATOR_ID,
            parameters,
            contract_sha256=contract.contract_sha256,
        ),
        benchmark="invsqrt",
        generator_id=_GENERATOR_ID,
        parameters=dict(parameters),
        artifact_path=str(artifact_path) if artifact_path else "",
        artifact_sha256=artifact_sha,
        contract_sha256=contract.contract_sha256,
        parent_candidate_id=parent_candidate_id,
        screen_metrics=screen.metrics,
        screen_score_estimate=screen.score_estimate,
    )


def _evaluate(
    contract: VerifierContract,
    ledger: QuantumCandidateLedger,
    candidates: dict[str, CircuitCandidate],
    parameters: dict[str, int],
    *,
    xs: np.ndarray,
    parent_candidate_id: str | None,
    artifact_dir: Path,
    write_artifact: bool,
) -> tuple[CircuitCandidate, ScreenResult]:
    screen = screen_precision(contract, parameters, xs=xs)
    candidate_id = candidate_id_for(
        "invsqrt",
        _GENERATOR_ID,
        parameters,
        contract_sha256=contract.contract_sha256,
    )
    existing = candidates.get(candidate_id)
    artifact_path: Path | None = None
    if existing is not None and existing.artifact_path:
        artifact_path = Path(existing.artifact_path)
    if screen.passed and write_artifact and artifact_path is None:
        artifact_path = artifact_dir / f"{candidate_id}.kmb"
        try:
            write_circuit(contract, parameters, artifact_path)
        except (OSError, ValueError, RuntimeError) as exc:
            screen = ScreenResult(False, f"artifact write failed: {exc}", None, screen.metrics)
            artifact_path = None

    if existing is None:
        candidate = _candidate_record(
            contract,
            parameters,
            parent_candidate_id=parent_candidate_id,
            artifact_path=artifact_path,
            screen=screen,
        )
        ledger.register(candidate)
        candidates[candidate_id] = candidate
    else:
        candidate = existing

    updated = ledger.record_screen(
        candidate,
        passed=screen.passed,
        metrics=screen.metrics,
        score_estimate=screen.score_estimate,
        rejection_reason=screen.reason,
    )
    candidates[candidate_id] = updated
    return updated, screen


def _fresh_run_dir(output_dir: Path, contract: VerifierContract) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"invsqrt-{stamp}-{contract.contract_sha256[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_invsqrt_search(
    *,
    competition_root: Path | str | None = None,
    output_dir: Path | str = Path("artifacts/quantum_compilation/invsqrt"),
    budget: int = 140,
    max_passes: int = 3,
    full_checks_per_pass: int = 3,
    dev_max_x: int = 1024,
    dev_shots: int = 2,
    verify_budget: int = 4,
    python_executable: str | None = None,
    timeout_s: float = 1800.0,
    final_salt: str = "",
    allow_full_run: bool = False,
    baseline_full_score: float | None = None,
) -> SearchOutcome:
    """Run the precision search and return a JSON-serialisable report."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if max_passes < 1 or full_checks_per_pass < 1:
        raise ValueError("max_passes and full_checks_per_pass must be positive")
    executable = python_executable or sys.executable

    contract = load_contract("invsqrt", competition_root)
    contract.assert_locked()
    run_dir = _fresh_run_dir(Path(output_dir), contract)
    artifact_dir = run_dir / "candidates"
    artifact_dir.mkdir()
    ledger = QuantumCandidateLedger(run_dir / "ledger.jsonl")
    candidates: dict[str, CircuitCandidate] = {}

    full_xs = full_domain_xs(contract)
    dev_xs = subset_xs(int(3 * (2 ** int(contract.constants["N_P"]) - 2) ** 2))

    baseline_params = base_precision(contract)
    baseline_candidate, baseline_screen = _evaluate(
        contract,
        ledger,
        candidates,
        baseline_params,
        xs=full_xs,
        parent_candidate_id=None,
        artifact_dir=artifact_dir,
        write_artifact=True,
    )
    if not baseline_screen.passed or baseline_screen.score_estimate is None:
        raise RuntimeError(f"reference precision failed its own screen: {baseline_screen.reason}")
    baseline_score = baseline_screen.score_estimate

    best_params = dict(baseline_params)
    best_score = baseline_score
    best_candidate = baseline_candidate
    evaluated: set[tuple[tuple[str, int], ...]] = {_key(baseline_params)}
    full_results: dict[tuple[tuple[str, int], ...], ScreenResult] = {
        _key(baseline_params): baseline_screen
    }
    full_candidates: dict[tuple[tuple[str, int], ...], CircuitCandidate] = {
        _key(baseline_params): baseline_candidate
    }

    used = 1
    passes_completed = 0

    def run_shortlist(
        parameter_iter: Iterable[dict[str, int]],
        parent_candidate_id: str,
        full_checks: int,
    ) -> tuple[float, dict[str, int], CircuitCandidate] | None:
        nonlocal used
        subset_rows: list[tuple[float, dict[str, int], CircuitCandidate, ScreenResult]] = []
        for parameters in parameter_iter:
            if used >= budget:
                break
            key = _key(parameters)
            if key in evaluated:
                continue
            used += 1
            evaluated.add(key)
            candidate, screen = _evaluate(
                contract,
                ledger,
                candidates,
                parameters,
                xs=dev_xs,
                parent_candidate_id=parent_candidate_id,
                artifact_dir=artifact_dir,
                write_artifact=True,
            )
            if screen.passed and screen.score_estimate is not None:
                subset_rows.append((screen.score_estimate, parameters, candidate, screen))

        subset_rows.sort(key=lambda row: row[0])
        best_row: tuple[float, dict[str, int], CircuitCandidate] | None = None
        checked = 0
        for _subset_score, parameters, candidate, _subset_screen in subset_rows:
            if checked >= full_checks:
                break
            key = _key(parameters)
            if key in full_results:
                continue
            checked += 1
            full_candidate, full_screen = _evaluate(
                contract,
                ledger,
                candidates,
                parameters,
                xs=full_xs,
                parent_candidate_id=candidate.parent_candidate_id,
                artifact_dir=artifact_dir,
                write_artifact=False,
            )
            full_results[key] = full_screen
            full_candidates[key] = full_candidate
            if full_screen.passed and full_screen.score_estimate is not None:
                if best_row is None or full_screen.score_estimate < best_row[0]:
                    best_row = (full_screen.score_estimate, parameters, full_candidate)
        return best_row

    # Interaction-aware opening move: precision fields trade against each
    # other, so a strictly one-coordinate walk can miss the best local basin.
    opening = run_shortlist(
        refinement_precisions(baseline_params),
        baseline_candidate.candidate_id,
        min(full_checks_per_pass, 3),
    )
    if opening is not None and opening[0] < best_score:
        best_score, best_params, best_candidate = opening

    for _ in range(max_passes):
        if used >= budget:
            break
        pass_best = run_shortlist(
            neighbor_precisions(best_params),
            best_candidate.candidate_id,
            full_checks_per_pass,
        )
        passes_completed += 1
        if pass_best is not None and pass_best[0] < best_score:
            best_score, best_params, best_candidate = pass_best
            continue
        break

    ranked = sorted(
        (
            (result.score_estimate, _as_params(key), full_candidates[key], result)
            for key, result in full_results.items()
            if result.passed and result.score_estimate is not None
        ),
        key=lambda row: row[0],
    )
    shortlist = ranked[: max(1, verify_budget)]
    if all(row[2].candidate_id != baseline_candidate.candidate_id for row in shortlist):
        shortlist.append((baseline_score, baseline_params, baseline_candidate, baseline_screen))

    verification_rows: list[dict[str, Any]] = []
    best_verified: tuple[float, CircuitCandidate, ScreenResult] | None = None
    for score, parameters, candidate, screen in shortlist:
        if not candidate.artifact_path:
            continue
        receipt = run_invsqrt_verifier(
            contract,
            Path(candidate.artifact_path),
            python_executable=executable,
            max_x=dev_max_x,
            shots_per_x=dev_shots,
            timeout_s=timeout_s,
        )
        verified = ledger.record_verification(candidate, receipt, incumbent_score=None)
        candidates[verified.candidate_id] = verified
        verification_rows.append(
            {
                "candidate_id": verified.candidate_id,
                "parameters": parameters,
                "model_score_estimate": score,
                "model_improvement_over_reference": baseline_score / score,
                "artifact_sha256": verified.artifact_sha256,
                "dev_verifier_status": receipt.status,
                "dev_verifier_score": receipt.score,
                "dev_verifier_scope": receipt.verification_scope,
                "dev_verified_shots": receipt.metrics.get("verified_shots"),
            }
        )
        if receipt.status == "PASS" and best_verified is None:
            best_verified = (score, verified, screen)

    final_receipt = None
    final_candidate = None
    if final_salt:
        if not allow_full_run:
            raise ValueError(
                "final_salt requires --allow-full-run: the full invsqrt protocol "
                "is 200M simulated shots and must be an explicit decision."
            )
        if baseline_full_score is None:
            raise ValueError(
                "the final salted run needs --baseline-full-score from a "
                "measured full-domain reference run; a projected baseline is "
                "not a comparable incumbent."
            )
        if best_verified is None:
            raise RuntimeError("no development-verified candidate is available for the final run")
        _, final_candidate, _ = best_verified
        final_receipt = run_invsqrt_verifier(
            contract,
            Path(final_candidate.artifact_path),
            python_executable=executable,
            salt=final_salt,
            timeout_s=timeout_s,
        )
        ledger.record_verification(
            final_candidate,
            final_receipt,
            incumbent_score=baseline_full_score,
        )

    best_model_score = best_score
    report: dict[str, Any] = {
        "status": (
            "SEARCH_COMPLETE_DEVELOPMENT_EVIDENCE"
            if verification_rows and any(row["dev_verifier_status"] == "PASS" for row in verification_rows)
            else "SEARCH_COMPLETE_NO_VERIFIED_CANDIDATE"
        ),
        "evidence_scope": "development-prefix (NOT a leaderboard result)",
        "final_verification_required": final_receipt is None,
        "contract": contract.to_dict(),
        "run_dir": str(run_dir),
        "ledger_path": str(run_dir / "ledger.jsonl"),
        "search": {
            "budget": budget,
            "used": used,
            "passes_completed": passes_completed,
            "full_checks": len(full_results) - 1,
            "dev_max_x": dev_max_x,
            "dev_shots_per_x": dev_shots,
            "python_executable": executable,
        },
        "baseline": {
            "candidate_id": baseline_candidate.candidate_id,
            "parameters": baseline_params,
            "artifact_sha256": baseline_candidate.artifact_sha256,
            "model_score_estimate": baseline_score,
            "screen_metrics": baseline_screen.metrics,
        },
        "best_model_candidate": {
            "candidate_id": best_candidate.candidate_id,
            "parameters": best_params,
            "artifact_sha256": best_candidate.artifact_sha256,
            "model_score_estimate": best_model_score,
            "model_improvement_over_reference": baseline_score / best_model_score,
        },
        "verification": verification_rows,
        "final_verification": (
            {
                "status": final_receipt.status,
                "score": final_receipt.score,
                "scope": final_receipt.verification_scope,
                "artifact_sha256": final_receipt.artifact_sha256,
                "salt_id": final_receipt.salt_id,
            }
            if final_receipt is not None
            else None
        ),
        "candidates_tested": len(evaluated),
        "candidates_registered": len(candidates),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return SearchOutcome(status=report["status"], report=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/quantum_compilation/invsqrt"),
    )
    parser.add_argument("--budget", type=int, default=140)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--full-checks-per-pass", type=int, default=3)
    parser.add_argument("--dev-max-x", type=int, default=1024)
    parser.add_argument("--dev-shots", type=int, default=2)
    parser.add_argument("--verify-budget", type=int, default=4)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--final-salt", default="")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--baseline-full-score", type=float, default=None)
    args = parser.parse_args()

    try:
        outcome = run_invsqrt_search(
            competition_root=args.competition_root,
            output_dir=args.output_dir,
            budget=args.budget,
            max_passes=args.max_passes,
            full_checks_per_pass=args.full_checks_per_pass,
            dev_max_x=args.dev_max_x,
            dev_shots=args.dev_shots,
            verify_budget=args.verify_budget,
            python_executable=args.python_executable,
            timeout_s=args.timeout,
            final_salt=args.final_salt,
            allow_full_run=args.allow_full_run,
            baseline_full_score=args.baseline_full_score,
        )
    except (AssertionError, ValueError, RuntimeError, OSError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(outcome.report, indent=2, default=str))


if __name__ == "__main__":
    main()
