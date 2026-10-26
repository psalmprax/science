"""Multi-structure search for invsqrt: try NR, QROM, and Direct Poly approaches.

Status: experimental

Combines different circuit topologies (Newton-Raphson precision search, QROM
piecewise-linear seed search, and direct degree-2 polynomial compilation) and
records every proposed, screened, and verified candidate in
:class:`~apedp_dyn.quantum_compilation.ledger.QuantumCandidateLedger`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.contracts import load_contract
from apedp_dyn.quantum_compilation.experiment import ExperimentRunner
from apedp_dyn.quantum_compilation.faros_isq import run_faros_isq_search
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import DirectPolyProposer
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import QromLineProposer
from apedp_dyn.quantum_compilation.ledger import QuantumCandidateLedger
from apedp_dyn.quantum_compilation.search import run_invsqrt_search

__all__ = ["run_multi_structure_search"]


def run_multi_structure_search(
    competition_root: str | Path | None = None,
    output_dir: str | Path = Path("artifacts/quantum_compilation/multi_structure"),
    budget_per_structure: int = 40,
    *,
    python_executable: str | None = None,
    dev_max_x: int | None = 1024,
    dev_shots: int | None = 2,
    ledger: QuantumCandidateLedger | None = None,
) -> dict[str, Any]:
    """Run search over multiple circuit structures with full ledger tracking."""
    contract = load_contract("invsqrt", competition_root)
    contract.assert_locked()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = (
        python_executable
        or os.environ.get("APEDP_Q_KICKMIX_PYTHON")
        or sys.executable
    )
    shared_ledger = (
        ledger
        if ledger is not None
        else QuantumCandidateLedger(output_dir / "ledger.jsonl")
    )
    verifier_kwargs: dict[str, Any] = {}
    if dev_max_x is not None:
        verifier_kwargs["max_x"] = dev_max_x
    if dev_shots is not None:
        verifier_kwargs["shots_per_x"] = dev_shots

    results: dict[str, Any] = {
        "ledger_path": str(shared_ledger.persistence_path)
        if shared_ledger.persistence_path
        else None,
    }

    # 1. Newton-Raphson precision search
    try:
        nr_outcome = run_invsqrt_search(
            competition_root=competition_root,
            output_dir=output_dir / "nr",
            budget=budget_per_structure * 2,
            dev_max_x=dev_max_x or 1024,
            dev_shots=dev_shots or 2,
            python_executable=executable,
        )
        results["nr"] = nr_outcome.report
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        results["nr"] = {"error": str(exc)}

    # 2. QROM piecewise-linear seed approach
    try:
        qrom_runner = ExperimentRunner(
            proposer=QromLineProposer(),
            contract=contract,
            ledger=shared_ledger,
            python_executable=executable,
            verifier_kwargs=verifier_kwargs,
        )
        qrom_outcome = qrom_runner.run(budget=budget_per_structure)
        results["qrom"] = {
            "results": [
                {
                    "candidate_id": r.candidate.candidate_id,
                    "status": r.status,
                    "estimated_cost": r.estimated_cost,
                    "measured_cost": r.measured_cost,
                    "verification_status": r.verification.status
                    if r.verification
                    else None,
                    "verification_scope": r.verification.verification_scope
                    if r.verification
                    else None,
                }
                for r in qrom_outcome.results
            ]
        }
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        results["qrom"] = {"error": str(exc)}

    # 3. Direct degree-2 polynomial approach
    try:
        poly_runner = ExperimentRunner(
            proposer=DirectPolyProposer(),
            contract=contract,
            ledger=shared_ledger,
            python_executable=executable,
            verifier_kwargs=verifier_kwargs,
        )
        poly_outcome = poly_runner.run(budget=budget_per_structure)
        results["direct_poly"] = {
            "results": [
                {
                    "candidate_id": r.candidate.candidate_id,
                    "status": r.status,
                    "estimated_cost": r.estimated_cost,
                    "measured_cost": r.measured_cost,
                    "verification_status": r.verification.status
                    if r.verification
                    else None,
                    "verification_scope": r.verification.verification_scope
                    if r.verification
                    else None,
                }
                for r in poly_outcome.results
            ]
        }
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        results["direct_poly"] = {"error": str(exc)}

    # 4. FAROS-ISQ representation portfolio & Pareto tournament (R1-R5)
    try:
        faros_outcome = run_faros_isq_search(
            competition_root=competition_root,
            output_dir=output_dir / "faros_isq",
            budget=budget_per_structure,
            python_executable=executable,
            dev_max_x=dev_max_x or 4096,
            ledger=shared_ledger,
        )
        results["faros_isq"] = faros_outcome
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        results["faros_isq"] = {"error": str(exc)}

    results["ledger_summary"] = shared_ledger.summary()
    report_path = output_dir / "multi_structure_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return results
