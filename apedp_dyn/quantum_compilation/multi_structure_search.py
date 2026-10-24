"""Multi-structure search for invsqrt: try NR, QROM, and Direct Poly approaches.

Combines different circuit topologies to find the best solution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apedp_dyn.quantum_compilation.contracts import load_contract
from apedp_dyn.quantum_compilation.experiment import ExperimentRunner
from apedp_dyn.quantum_compilation.invsqrt_direct_poly import DirectPolyProposer
from apedp_dyn.quantum_compilation.invsqrt_qrom_line import QromLineProposer
from apedp_dyn.quantum_compilation.search import run_invsqrt_search


def run_multi_structure_search(
    competition_root: str | Path | None = None,
    output_dir: str | Path = Path("artifacts/quantum_compilation/multi_structure"),
    budget_per_structure: int = 40,
) -> dict[str, Any]:
    """Run search over multiple circuit structures."""
    contract = load_contract("invsqrt", competition_root)
    contract.assert_locked()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Try NR precision search (original approach)
    try:
        nr_outcome = run_invsqrt_search(
            competition_root=competition_root,
            output_dir=output_dir / "nr",
            budget=budget_per_structure * 2,  # give it more budget as it's the main one
        )
        results["nr"] = nr_outcome.report
    except Exception as e:
        results["nr"] = {"error": str(e)}
    
    # Try QROM line approach
    try:
        runner = ExperimentRunner(
            proposer=QromLineProposer(),
            contract=contract,
        )
        outcome = runner.run(budget=budget_per_structure)
        results["qrom"] = {
            "results": [
                {
                    "candidate_id": r.candidate.candidate_id,
                    "status": r.status,
                    "measured_cost": r.measured_cost,
                    "verification_status": r.verification.status if r.verification else None,
                }
                for r in outcome.results
            ]
        }
    except Exception as e:
        results["qrom"] = {"error": str(e)}
    
    # Try Direct Poly approach
    try:
        runner = ExperimentRunner(
            proposer=DirectPolyProposer(),
            contract=contract,
        )
        outcome = runner.run(budget=budget_per_structure)
        results["direct_poly"] = {
            "results": [
                {
                    "candidate_id": r.candidate.candidate_id,
                    "status": r.status,
                    "measured_cost": r.measured_cost,
                    "verification_status": r.verification.status if r.verification else None,
                }
                for r in outcome.results
            ]
        }
    except Exception as e:
        results["direct_poly"] = {"error": str(e)}
    
    report_path = output_dir / "multi_structure_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    
    return results
