"""Adapter from quantum candidates to the APEDP-X experiment ledger and portfolio scorer.

Status: experimental

The existing :class:`FullLineageExperimentLedger` already provides append-only
persistence, candidate lineage, and anti-snooping holdout burning.  This module
maps the quantum-specific evidence categories onto that machinery instead of
creating a second, weaker ledger, and provides the 3-benchmark competition cost
formulas and geometric-mean portfolio scorer:

.. math::

    S = \\left(\\prod_{i \\in \\{\\text{invsqrt}, \\text{dqi}, \\text{dfthcsos}\\}}
        \\frac{C_{i,\\text{ref}}}{C_i}\\right)^{1/3}

Two rules are enforced here:

* A proposer-side screen is recorded, but it can never promote a candidate.
* A salted full-domain verifier failure burns the ``(benchmark, verifier,
  salt)`` confirmation slice; a replacement candidate cannot retry the same
  final salt.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.quantum_compilation.candidates import (
    CORRECTNESS_CLAIMING_STATUSES,
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
    evidence_from_receipt,
    missing_correctness_checks,
)

__all__ = [
    "BENCHMARK_ORDER",
    "REFERENCE_COSTS",
    "PortfolioScoreReport",
    "PortfolioTaskSummary",
    "QuantumCandidateLedger",
    "aggregate_score",
    "chemistry_cost",
    "dqi_cost",
    "early_exit_resource_gate",
    "invsqrt_cost",
    "relative_reduction",
    "score_ledger_portfolio",
    "score_verified_portfolio",
]

BENCHMARK_ORDER: tuple[str, str, str] = ("invsqrt", "dqi", "dfthcsos")

# Official baseline reference costs from the competition reference circuits.
REFERENCE_COSTS: dict[str, float] = {
    "invsqrt": 2076774.233715282,
    "dqi": 24439473914136.668,
    "dfthcsos": 3355599588.3155365,
}

_STATUS_TO_CORE = {
    CandidateStatus.PROPOSED: CandidateLifecycleState.PROPOSED,
    CandidateStatus.SCREEN_REJECTED: CandidateLifecycleState.PRE_SCREEN_REJECTED,
    CandidateStatus.VERIFIER_REJECTED: CandidateLifecycleState.CONFIRMATION_REJECTED,
    CandidateStatus.VERIFIED_NONIMPROVING: CandidateLifecycleState.DEVELOPMENT_PASSED,
    CandidateStatus.VERIFIED_IMPROVEMENT: CandidateLifecycleState.CONFIRMED_VALID,
    CandidateStatus.INSUFFICIENT_EVIDENCE: CandidateLifecycleState.UNRESOLVED,
    CandidateStatus.UNRESOLVED: CandidateLifecycleState.UNRESOLVED,
}


def invsqrt_cost(
    qubits: int,
    toffoli_like: float,
    t_count: float,
    *,
    t_per_toffoli: float = 4.0,
) -> float:
    """Benchmark I competition cost: ``Q * (Toffoli-equivalents + T / 4)``."""
    if qubits <= 0 or toffoli_like < 0 or t_count < 0 or t_per_toffoli <= 0:
        raise ValueError(
            "Resource counts must be nonnegative and qubits/t_per_toffoli must be positive"
        )
    return float(qubits * (toffoli_like + t_count / t_per_toffoli))


def dqi_cost(
    repetition_factor: float,
    qubits: int,
    mean_toffoli_like: float,
) -> float:
    """Benchmark II competition cost: ``R(eps+) * Q^2 * mean_toffoli_like``."""
    if repetition_factor < 1.0 or qubits <= 0 or mean_toffoli_like < 0:
        raise ValueError("Invalid DQI resource values")
    return float(repetition_factor * (qubits**2) * mean_toffoli_like)


def chemistry_cost(
    lambda_eff: float,
    qubits: int,
    toffoli_like: float,
) -> float:
    """Benchmark III competition cost: ``lambda_eff * Q * T``."""
    if lambda_eff <= 0 or qubits <= 0 or toffoli_like < 0:
        raise ValueError("Invalid chemistry resource values")
    return float(lambda_eff * qubits * toffoli_like)


def relative_reduction(reference: float, candidate: float) -> float:
    """Return ``1 - candidate / reference`` for positive costs."""
    if not (math.isfinite(reference) and math.isfinite(candidate)):
        raise ValueError("Costs must be finite")
    if reference <= 0 or candidate <= 0:
        raise ValueError("Costs must be positive")
    return float(1.0 - candidate / reference)


def early_exit_resource_gate(
    estimated_min_cost: float,
    champion_cost: float,
    noise_margin: float = 50.0,
) -> bool:
    """Conservative early-exit resource gate for candidate synthesis.

    Returns True if the candidate's optimistic lower bound cost is strictly
    lower than (champion_cost - noise_margin), justifying synthesis and official
    verification. Returns False if even the optimistic bound cannot beat the
    champion outside a conservative empirical buffer (noise_margin=50.0).

    Note on statistical calibration:
    For N = 200,540,928 shots on Benchmark I (Q = 90, T_max = 1448), the
    per-shot independent sampling standard error of cost is bounded by
    SE(C) <= 90 * (1448 / (2 * sqrt(N))) ≈ 4.60 cost units.
    The 50.0 margin is an empirical operational buffer against run-to-run
    variation across environments and seeds, not the per-shot standard error.
    """
    if not math.isfinite(estimated_min_cost) or estimated_min_cost <= 0:
        raise ValueError(f"Invalid estimated_min_cost: {estimated_min_cost}")
    if not math.isfinite(champion_cost) or champion_cost <= 0:
        raise ValueError(f"Invalid champion_cost: {champion_cost}")
    return float(estimated_min_cost) < float(champion_cost - noise_margin)


def _normalize_cost_vector(
    costs: Iterable[float] | Mapping[str, float],
) -> list[float]:
    if isinstance(costs, Mapping):
        missing = [b for b in BENCHMARK_ORDER if b not in costs]
        if missing:
            raise ValueError(
                f"Missing benchmark cost(s) for {missing}; expected {list(BENCHMARK_ORDER)}"
            )
        if len(costs) != len(BENCHMARK_ORDER):
            extra = sorted(set(costs) - set(BENCHMARK_ORDER))
            raise ValueError(f"Unexpected benchmark keys in cost mapping: {extra}")
        return [float(costs[b]) for b in BENCHMARK_ORDER]
    values = [float(x) for x in costs]
    if len(values) != len(BENCHMARK_ORDER):
        raise ValueError(
            f"Expected exactly three task costs {list(BENCHMARK_ORDER)}, got {len(values)}"
        )
    return values


def aggregate_score(
    reference_costs: Iterable[float] | Mapping[str, float],
    candidate_costs: Iterable[float] | Mapping[str, float],
) -> float:
    """Compute the 3-benchmark geometric-mean score ``(prod_i C_{i,ref} / C_i)^(1/3)``."""
    refs = _normalize_cost_vector(reference_costs)
    cands = _normalize_cost_vector(candidate_costs)
    if any((not math.isfinite(x) or x <= 0) for x in refs + cands):
        raise ValueError("All reference and candidate costs must be finite and positive")
    return float(
        math.exp(sum(math.log(r / c) for r, c in zip(refs, cands, strict=True)) / 3.0)
    )


@dataclasses.dataclass(frozen=True)
class PortfolioTaskSummary:
    """Per-benchmark summary inside a 3-benchmark portfolio report."""

    benchmark: str
    reference_cost: float
    candidate_cost: float
    improvement_ratio: float
    relative_reduction: float
    verification_scope: str
    artifact_sha256: str
    contract_sha256: str
    salt_id: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PortfolioScoreReport:
    """Complete 3-benchmark portfolio score report."""

    aggregate_score: float
    all_verified: bool
    all_full_domain: bool
    tasks: dict[str, PortfolioTaskSummary]
    warning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_score": self.aggregate_score,
            "all_verified": self.all_verified,
            "all_full_domain": self.all_full_domain,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "warning": self.warning,
        }


def score_verified_portfolio(
    receipts: Mapping[str, VerifierReceipt],
    reference_costs: Mapping[str, float] = REFERENCE_COSTS,
    *,
    require_full_domain: bool = False,
) -> PortfolioScoreReport:
    """Score a 3-benchmark portfolio from official :class:`VerifierReceipt` objects."""
    missing = [b for b in BENCHMARK_ORDER if b not in receipts]
    if missing:
        raise ValueError(
            f"Portfolio requires receipts for all 3 benchmarks {list(BENCHMARK_ORDER)}; "
            f"missing {missing}"
        )

    tasks: dict[str, PortfolioTaskSummary] = {}
    cand_costs: dict[str, float] = {}
    all_full_domain = True

    for benchmark in BENCHMARK_ORDER:
        receipt = receipts[benchmark]
        if receipt.status.upper() != "PASS":
            raise ValueError(
                f"Benchmark {benchmark!r} receipt did not PASS (status={receipt.status!r})"
            )
        if (
            receipt.score is None
            or not math.isfinite(receipt.score)
            or receipt.score <= 0
        ):
            raise ValueError(
                f"Benchmark {benchmark!r} receipt has invalid measured score: {receipt.score!r}"
            )
        if require_full_domain and receipt.verification_scope != "full-domain":
            raise ValueError(
                f"Benchmark {benchmark!r} receipt has scope {receipt.verification_scope!r}, "
                "but require_full_domain=True requires 'full-domain'"
            )
        if receipt.verification_scope != "full-domain":
            all_full_domain = False

        ref_cost = float(reference_costs[benchmark])
        cand_cost = float(receipt.score)
        cand_costs[benchmark] = cand_cost
        tasks[benchmark] = PortfolioTaskSummary(
            benchmark=benchmark,
            reference_cost=ref_cost,
            candidate_cost=cand_cost,
            improvement_ratio=ref_cost / cand_cost,
            relative_reduction=relative_reduction(ref_cost, cand_cost),
            verification_scope=receipt.verification_scope,
            artifact_sha256=receipt.artifact_sha256,
            contract_sha256=receipt.contract_sha256,
            salt_id=receipt.salt_id,
        )

    score = aggregate_score(reference_costs, cand_costs)
    warning = (
        "All three benchmarks are backed by salted full-domain verifier receipts."
        if all_full_domain
        else "One or more receipts are development/public scope; not a final leaderboard confirmation."
    )
    return PortfolioScoreReport(
        aggregate_score=score,
        all_verified=True,
        all_full_domain=all_full_domain,
        tasks=tasks,
        warning=warning,
    )


def score_ledger_portfolio(
    ledger: QuantumCandidateLedger,
    reference_costs: Mapping[str, float] = REFERENCE_COSTS,
) -> PortfolioScoreReport:
    """Score the incumbent confirmed-valid candidates across all 3 benchmarks in ``ledger``."""
    receipts: dict[str, VerifierReceipt] = {}
    for benchmark in BENCHMARK_ORDER:
        cand_id, _score = ledger.incumbent(benchmark)
        if cand_id is None:
            raise ValueError(
                f"No CONFIRMED_VALID incumbent found in ledger for benchmark {benchmark!r}"
            )
        node = ledger.node(cand_id)
        receipt_dict = node.metrics.get("verifier_receipt")
        if not isinstance(receipt_dict, dict):
            raise ValueError(
                f"Incumbent {cand_id!r} for {benchmark!r} is missing verifier_receipt metadata"
            )
        receipts[benchmark] = VerifierReceipt(
            status=str(receipt_dict["status"]),
            score=float(receipt_dict["score"])
            if receipt_dict.get("score") is not None
            else None,
            toffoli_equivalent=float(receipt_dict["toffoli_equivalent"])
            if receipt_dict.get("toffoli_equivalent") is not None
            else None,
            num_qubits=int(receipt_dict["num_qubits"])
            if receipt_dict.get("num_qubits") is not None
            else None,
            metrics=dict(receipt_dict.get("metrics", {})),
            exit_code=int(receipt_dict.get("exit_code", 0)),
            command=tuple(receipt_dict.get("command", ())),
            salt_id=str(receipt_dict.get("salt_id", "")),
            verification_scope=str(receipt_dict.get("verification_scope", "")),
            verifier_sha256=str(receipt_dict.get("verifier_sha256", "")),
            artifact_sha256=str(receipt_dict.get("artifact_sha256", "")),
            contract_sha256=str(receipt_dict.get("contract_sha256", "")),
            duration_s=float(receipt_dict.get("duration_s", 0.0)),
            stdout_tail=str(receipt_dict.get("stdout_tail", "")),
            stderr_tail=str(receipt_dict.get("stderr_tail", "")),
        )
    return score_verified_portfolio(
        receipts, reference_costs, require_full_domain=True
    )


class QuantumCandidateLedger:
    """Thin quantum-evidence adapter over the APEDP-X full-lineage ledger."""

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self._ledger = FullLineageExperimentLedger(persistence_path=persistence_path)
        self._path = Path(persistence_path) if persistence_path else None

    @property
    def persistence_path(self) -> Path | None:
        return self._path

    def register(self, candidate: CircuitCandidate) -> CircuitCandidate:
        """Register a candidate without granting any correctness status."""
        numeric_parameters = {
            name: float(value)
            for name, value in candidate.parameters.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if any(
            node.candidate_id == candidate.candidate_id
            for node in self._ledger.get_all_candidates()
        ):
            raise ValueError(f"Candidate {candidate.candidate_id!r} is already registered")
        self._ledger.register_candidate(
            candidate_id=candidate.candidate_id,
            proposer_id=candidate.generator_id,
            formula=json.dumps(candidate.parameters, sort_keys=True, default=str),
            parent_candidate_id=candidate.parent_candidate_id,
            feature_set=tuple(sorted(candidate.parameters)),
            parameter_choices=numeric_parameters,
        )
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            CandidateLifecycleState.PROPOSED,
            metrics={"quantum_candidate": candidate.to_dict()},
        )
        return candidate

    def record_screen(
        self,
        candidate: CircuitCandidate,
        *,
        passed: bool,
        metrics: dict[str, Any],
        score_estimate: float | None,
        rejection_reason: str | None = None,
    ) -> CircuitCandidate:
        """Record the proposer-side screen; only the screen may be rejected here."""
        status = CandidateStatus.PROPOSED if passed else CandidateStatus.SCREEN_REJECTED
        updated = dataclasses.replace(
            candidate,
            status=status,
            screen_metrics=metrics,
            screen_score_estimate=score_estimate,
        )
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            _STATUS_TO_CORE[status],
            rejection_reason=rejection_reason,
            metrics={
                "quantum_candidate": updated.to_dict(),
                "screen_metrics": metrics,
                "screen_score_estimate": score_estimate,
                "screen_passed": passed,
            },
        )
        return updated

    def record_verification(
        self,
        candidate: CircuitCandidate,
        receipt: VerifierReceipt,
        *,
        incumbent_score: float | None,
    ) -> CircuitCandidate:
        """Record an official-verifier receipt and map it to a lifecycle state.

        Promotion to ``CONFIRMED_VALID`` requires a *salted full-domain* PASS
        whose measured score is strictly lower than the incumbent.  A
        development-prefix PASS is recorded as ``DEVELOPMENT_PASSED`` and is
        never treated as a leaderboard result.
        """
        scope = receipt.verification_scope
        is_final = scope == "full-domain"
        role = (
            DatasetRole.CONFIRMATION_HOLDOUT
            if is_final
            else DatasetRole.TRAINING_DEVELOPMENT
        )
        slice_id = f"{receipt.verifier_sha256}:{receipt.salt_id}"
        if is_final:
            self._assert_final_slice_unused(candidate.benchmark, scope, slice_id)
        self._ledger.record_data_exposure(
            candidate_id=candidate.candidate_id,
            dataset_id=f"{candidate.benchmark}:{scope}",
            slice_identifier=slice_id,
            role=role,
            outcome_status="PASSED" if receipt.status == "PASS" else "FAILED",
        )

        if receipt.status != "PASS":
            status = CandidateStatus.VERIFIER_REJECTED
            reason = str(receipt.metrics.get("error", "official verifier did not pass"))
        elif not is_final:
            status = CandidateStatus.VERIFIED_NONIMPROVING
            reason = (
                "development-prefix verification passed; final salted run is still required"
            )
        elif receipt.score is None or incumbent_score is None:
            status = CandidateStatus.INSUFFICIENT_EVIDENCE
            reason = "full-domain pass without a comparable measured score"
        elif receipt.score < incumbent_score:
            status = CandidateStatus.VERIFIED_IMPROVEMENT
            reason = None
        else:
            status = CandidateStatus.VERIFIED_NONIMPROVING
            reason = "full-domain verification passed but did not beat the incumbent"

        # A correctness claim needs value, junk and phase accounted for. The receipt is
        # the evidence: for a per-shot verifier its own error counts, for a cost-model
        # verifier an explicit statement that it does not measure them. A claiming
        # status with no evidence is not promoted; it is recorded as insufficient.
        evidence = evidence_from_receipt(receipt, candidate.benchmark)
        if status in CORRECTNESS_CLAIMING_STATUSES:
            missing = missing_correctness_checks(evidence, benchmark=candidate.benchmark)
            if missing:
                status = CandidateStatus.INSUFFICIENT_EVIDENCE
                reason = "the receipt does not establish correctness: " + "; ".join(missing)

        updated = candidate.with_status(status, receipt=receipt, evidence=evidence)
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            _STATUS_TO_CORE[status],
            rejection_reason=reason,
            metrics={
                "quantum_candidate": updated.to_dict(),
                "verifier_receipt": receipt.to_dict(),
                "verified_score": receipt.score,
                "verification_scope": scope,
            },
        )
        return updated

    def _assert_final_slice_unused(
        self, benchmark: str, scope: str, slice_id: str
    ) -> None:
        """A final confirmation salt is one-shot, even if the first run passed.

        ``FullLineageExperimentLedger`` burns only failed confirmation slices.
        A passed slice has still been consumed: reusing it would turn the
        grader salt into a development set.  This check makes that reuse raise
        before the second verifier run can happen.
        """
        dataset_id = f"{benchmark}:{scope}"
        for node in self._ledger.get_all_candidates():
            for exposure in node.data_exposures:
                if exposure.role is not DatasetRole.CONFIRMATION_HOLDOUT:
                    continue
                if (exposure.dataset_id, exposure.slice_identifier) == (
                    dataset_id,
                    slice_id,
                ):
                    raise HoldoutContaminationError(
                        "Final confirmation slice was already consumed by "
                        f"candidate {exposure.candidate_id!r}; refusing to reuse "
                        "a secret-salt holdout."
                    )

    def incumbent(self, benchmark: str) -> tuple[str | None, float | None]:
        """Return the best fully verified candidate id and score for a benchmark."""
        best_id: str | None = None
        best_score: float | None = None
        for node in self._ledger.get_all_candidates():
            quantum_meta = node.metrics.get("quantum_candidate", {})
            node_benchmark = (
                quantum_meta.get("benchmark")
                if isinstance(quantum_meta, dict)
                else None
            )
            if node_benchmark != benchmark and not node.candidate_id.startswith(
                f"{benchmark}-"
            ):
                continue
            if node.state is not CandidateLifecycleState.CONFIRMED_VALID:
                continue
            score = node.metrics.get("verified_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score < best_score:
                best_id, best_score = node.candidate_id, float(score)
        return best_id, best_score

    def portfolio_score(
        self, reference_costs: Mapping[str, float] = REFERENCE_COSTS
    ) -> PortfolioScoreReport:
        """Compute the 3-benchmark portfolio score from confirmed-valid incumbents."""
        return score_ledger_portfolio(self, reference_costs=reference_costs)

    def node(self, candidate_id: str) -> Any:
        return self._ledger.get_candidate(candidate_id)

    def all_nodes(self) -> list[Any]:
        return cast(list[Any], self._ledger.get_all_candidates())

    def burned_holdouts(self) -> dict[tuple[str, str], str]:
        return cast(dict[tuple[str, str], str], self._ledger.get_burned_holdouts())

    def summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in self._ledger.get_all_candidates():
            quantum = node.metrics.get("quantum_candidate", {})
            rows.append(
                {
                    "candidate_id": node.candidate_id,
                    "proposer_id": node.proposer_id,
                    "state": node.state.value,
                    "rejection_reason": node.rejection_reason,
                    "screen_score_estimate": node.metrics.get("screen_score_estimate"),
                    "verified_score": node.metrics.get("verified_score"),
                    "verification_scope": node.metrics.get("verification_scope"),
                    "parameters": quantum.get("parameters", node.parameter_choices),
                }
            )
        return rows
