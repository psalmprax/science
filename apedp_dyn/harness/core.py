"""APEDP Experimental Harness (AEH) Core Orchestrator.

Enforces experimental control, capability isolation, deterministic execution,
and cryptographic event auditing around APEDP-X.

Architectural Rule:
  - APEDP decides scientific status.
  - AEH decides experimental permissions and audits compliance.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from apedp_dyn.epistemic.claim_ladder import (
    ClaimLevel,
    EpistemicAdjudicationVerdict,
    EvidenceBundle,
)
from apedp_dyn.epistemic.eedr import EEDRAdjudicationResult, RepairOperatorType, execute_eedr_loop
from apedp_dyn.epistemic.frozen_spec import get_frozen_spec
from apedp_dyn.harness.ledger import EventLedger
from apedp_dyn.harness.manifest import (
    ExperimentManifest,
    compute_dataset_hash,
    get_git_commit_sha,
)
from apedp_dyn.harness.roles import (
    ComponentRole,
    DataFirewall,
    FirewallViolationError,
    IsolatedDataVault,
)


@dataclass(frozen=True)
class HarnessResultPackage:
    """Complete, self-contained, audit-ready experimental package."""
    experiment_id: str
    manifest: ExperimentManifest
    candidates: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    repair_lineage: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    ledger_hash: str
    package_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "manifest": self.manifest.to_dict(),
            "candidates": self.candidates,
            "decisions": self.decisions,
            "repair_lineage": self.repair_lineage,
            "failures": self.failures,
            "metrics": self.metrics,
            "provenance": self.provenance,
            "ledger_hash": self.ledger_hash,
            "package_hash": self.package_hash,
        }


class AEH:
    """APEDP Experimental Harness: External experimental-control and audit layer."""

    VERSION = "AEH-v1.0"

    def __init__(
        self,
        experiment_id: str,
        random_seed: int = 42,
        config: dict[str, Any] | None = None,
        dataset_hash: str | None = None,
        code_commit: str | None = None,
        journal_path: str | None = None,
        stage_timeout_sec: float | None = 60.0,
    ) -> None:
        self.experiment_id = experiment_id
        self.random_seed = random_seed
        self.config = config or {}
        self.code_commit = code_commit or get_git_commit_sha()
        self.dataset_hash = dataset_hash or "SYNTHETIC_DATASET_HASH"
        self.stage_timeout_sec = stage_timeout_sec

        # 1. Deterministic Execution Control with Isolated RNG instances
        self.py_rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed) if HAS_NUMPY else None
        self._init_deterministic_seeds(random_seed)

        # 2. Immutable Event Ledger with WAL Disk Journal
        self.ledger = EventLedger(journal_path=journal_path)

        # 3. Records Containers
        self._candidates: list[dict[str, Any]] = []
        self._decisions: list[dict[str, Any]] = []
        self._repair_lineage: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._vault: IsolatedDataVault | None = None

        # Record Genesis Harness Event
        self.ledger.record_event(
            role=ComponentRole.HARNESS_ORCHESTRATOR,
            action="HARNESS_INITIALIZED",
            payload_summary={
                "experiment_id": self.experiment_id,
                "random_seed": self.random_seed,
                "aeh_version": self.VERSION,
                "code_commit": self.code_commit,
                "wal_active": journal_path is not None,
            },
        )

    def _init_deterministic_seeds(self, seed: int) -> None:
        random.seed(seed)
        if HAS_NUMPY:
            np.random.seed(seed)

    def get_rng(self) -> Any:
        """Returns isolated RNG generator instance without mutating global state."""
        return self.np_rng if self.np_rng is not None else self.py_rng

    def attach_vault(self, vault: IsolatedDataVault) -> None:
        """Attaches an isolated data vault with firewall access controls."""
        self._vault = vault
        self.dataset_hash = compute_dataset_hash(vault._training_data)
        self.ledger.record_event(
            role=ComponentRole.HARNESS_ORCHESTRATOR,
            action="DATA_VAULT_ATTACHED",
            payload_summary={"firewall_active": True, "dataset_hash": self.dataset_hash},
        )

    def attach_dataset(self, data: Any) -> str:
        """Attaches raw dataset and updates cryptographic dataset hash."""
        self.dataset_hash = compute_dataset_hash(data)
        self.ledger.record_event(
            role=ComponentRole.HARNESS_ORCHESTRATOR,
            action="DATASET_ATTACHED",
            payload_summary={"dataset_hash": self.dataset_hash},
        )
        return self.dataset_hash

    def _execute_with_timeout(self, fn: Callable[[], Any], timeout_sec: float | None = None) -> Any:
        timeout = timeout_sec if timeout_sec is not None else self.stage_timeout_sec
        if timeout is None or timeout <= 0:
            return fn()

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as e:
                raise TimeoutError(f"STAGE TIMEOUT: Exceeded hard budget of {timeout}s.") from e

    # --------------------------------------------------------------------------
    # Capability-Isolated Execution Stages
    # --------------------------------------------------------------------------

    def run_proposer_stage(
        self,
        candidate_id: str,
        proposer_fn: Callable[[], str],
        claimed_tier: ClaimLevel,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Executes candidate generation under proposer permissions."""
        DataFirewall.verify_access(ComponentRole.PROPOSER, "training_observables")
        
        # Execute upstream proposer with timeout protection
        try:
            model_repr: str = self._execute_with_timeout(proposer_fn)
        except TimeoutError as te:
            self.ledger.record_event(
                role=ComponentRole.PROPOSER,
                action="PROPOSER_STAGE_TIMEOUT",
                payload_summary={"candidate_id": candidate_id, "timeout_sec": self.stage_timeout_sec},
            )
            raise FirewallViolationError(f"STAGE TIMEOUT: Proposer exceeded budget for {candidate_id}") from te

        cand_record = {
            "candidate_id": candidate_id,
            "model_repr": model_repr,
            "claimed_tier": claimed_tier.name,
            "metadata": metadata or {},
        }
        self._candidates.append(cand_record)

        self.ledger.record_event(
            role=ComponentRole.PROPOSER,
            action="CANDIDATE_PROPOSED",
            payload_summary=cand_record,
        )
        return model_repr

    def run_adjudication_stage(
        self,
        candidate_id: str,
        claimed_tier: ClaimLevel,
        evidence: EvidenceBundle,
        adjudicate_fn: Callable[[str, ClaimLevel, EvidenceBundle], Any],
    ) -> Any:
        """Executes epistemic referee adjudication under strict evidence non-mutation."""
        DataFirewall.verify_access(ComponentRole.FINAL_ADJUDICATOR, "complete_evidence_bundle")

        evidence_hash = evidence.fingerprint()

        # Execute APEDP adjudication with timeout protection
        try:
            verdict_result = self._execute_with_timeout(lambda: adjudicate_fn(candidate_id, claimed_tier, evidence))
        except TimeoutError as te:
            self.ledger.record_event(
                role=ComponentRole.FINAL_ADJUDICATOR,
                action="ADJUDICATION_STAGE_TIMEOUT",
                payload_summary={"candidate_id": candidate_id, "timeout_sec": self.stage_timeout_sec},
            )
            raise FirewallViolationError(f"STAGE TIMEOUT: Adjudication exceeded budget for {candidate_id}") from te


        # Enforce that referee did not alter evidence bundle
        post_hash = evidence.fingerprint()
        if post_hash != evidence_hash:
            raise FirewallViolationError(
                f"EPISTEMIC SECURITY VIOLATION: Adjudicator modified evidence during evaluation of {candidate_id}!"
            )

        promoted = (
            hasattr(verdict_result, "verdict")
            and verdict_result.verdict == EpistemicAdjudicationVerdict.PROMOTED
        )

        decision_record = {
            "candidate_id": candidate_id,
            "claimed_tier": claimed_tier.name,
            "promoted": promoted,
            "evidence_hash": evidence_hash,
            "justified_tier": getattr(verdict_result, "justified_claim", claimed_tier).name,
        }
        self._decisions.append(decision_record)

        if not promoted:
            fail_record = {
                "candidate_id": candidate_id,
                "claimed_tier": claimed_tier.name,
                "evidence_hash": evidence_hash,
                "missing_evidence": [
                    r.value for r in getattr(verdict_result, "missing_evidence", [])
                ],
            }
            self._failures.append(fail_record)
            self.ledger.record_event(
                role=ComponentRole.FINAL_ADJUDICATOR,
                action="INITIAL_ADJUDICATION_FAILED",
                payload_summary=fail_record,
            )
        else:
            self.ledger.record_event(
                role=ComponentRole.FINAL_ADJUDICATOR,
                action="CANDIDATE_PROMOTED",
                payload_summary={"candidate_id": candidate_id, "status": "PROMOTED_ORIGINAL"},
            )

        return verdict_result

    def run_eedr_repair_stage(
        self,
        failed_candidate_id: str,
        claimed_tier: ClaimLevel,
        evidence: EvidenceBundle,
        domain: str = "general_science",
    ) -> EEDRAdjudicationResult:
        """Executes EEDR under strict anti-p-hacking and isolation invariants."""
        DataFirewall.verify_access(ComponentRole.REPAIRER, "diagnosed_failure")

        init_evidence_hash = evidence.fingerprint()

        # Run EEDR Loop with timeout protection
        try:
            eedr_res: EEDRAdjudicationResult = self._execute_with_timeout(
                lambda: execute_eedr_loop(
                    candidate_id=failed_candidate_id,
                    proposed_claim=claimed_tier,
                    evidence=evidence,
                    domain=domain,
                )
            )
        except TimeoutError as te:
            self.ledger.record_event(
                role=ComponentRole.REPAIRER,
                action="REPAIR_STAGE_TIMEOUT",
                payload_summary={"candidate_id": failed_candidate_id, "timeout_sec": self.stage_timeout_sec},
            )
            raise FirewallViolationError(f"STAGE TIMEOUT: EEDR Repair exceeded budget for {failed_candidate_id}") from te


        # 1. Enforce Evidence Conservation
        post_evidence_hash = evidence.fingerprint()
        if init_evidence_hash != post_evidence_hash:
            raise FirewallViolationError(
                f"ANTI-P-HACKING BREACH: EEDR altered evidence bundle for {failed_candidate_id}!"
            )

        # 2. Record lineage
        repaired_id = (
            eedr_res.repair_candidate.repair_id if eedr_res.repair_candidate else None
        )
        op_applied = (
            eedr_res.repair_candidate.operator_applied.value
            if eedr_res.repair_candidate
            else None
        )

        lineage_record = {
            "original_candidate_id": failed_candidate_id,
            "repaired_candidate_id": repaired_id,
            "final_status": eedr_res.final_status,
            "operator_applied": op_applied,
            "original_claimed_tier": claimed_tier.name,
            "justified_tier": eedr_res.final_justified_claim.name,
            "evidence_conserved": eedr_res.evidence_fingerprint_preserved,
        }
        self._repair_lineage.append(lineage_record)

        self.ledger.record_event(
            role=ComponentRole.REPAIRER,
            action="EEDR_REPAIR_EXECUTED",
            payload_summary=lineage_record,
        )

        if repaired_id:
            # Record that de novo adjudication was run
            self.ledger.record_event(
                role=ComponentRole.FINAL_ADJUDICATOR,
                action="DE_NOVO_ADJUDICATION_EXECUTED",
                payload_summary={
                    "candidate_id": repaired_id,
                    "final_status": eedr_res.final_status,
                },
            )

        # Formally Audit All 8 Invariants in Ledger
        invariants = self.ledger.audit_eedr_invariants(
            original_candidate_id=failed_candidate_id,
            repaired_candidate_id=repaired_id,
            initial_evidence_hash=init_evidence_hash,
            post_repair_evidence_hash=post_evidence_hash,
            operator_applied=op_applied,
            authorized_operators=tuple(op.value for op in RepairOperatorType),
        )

        if not all(invariants.values()):
            violated = [k for k, v in invariants.items() if not v]
            raise FirewallViolationError(
                f"EEDR AUDIT INVARIANT VIOLATION: {violated}"
            )

        return eedr_res

    # --------------------------------------------------------------------------
    # Finalization & Cryptographic Result Packaging
    # --------------------------------------------------------------------------

    def finalize_experiment(
        self,
        evaluation_protocol: str = "AEH-STANDALONE-BLIND-EVAL",
        metrics: dict[str, Any] | None = None,
    ) -> HarnessResultPackage:
        """Compiles manifest, verifies chain integrity, and emits immutable result package."""
        frozen_spec = get_frozen_spec()

        # 1. Verify Ledger Chain
        if not self.ledger.verify_chain_integrity():
            raise RuntimeError("CRITICAL HARNESS CORRUPTION: Event ledger hash chain is broken!")

        event_log_hash = self.ledger.get_ledger_hash()

        # 2. Count statistics
        c_count = len(self._candidates)
        r_count = len([r for r in self._repair_lineage if r["repaired_candidate_id"]])
        t_abstain = len([r for r in self._repair_lineage if r["final_status"] == "TERMINAL_ABSTAIN"])
        p_count = len([d for d in self._decisions if d["promoted"]]) + len(
            [r for r in self._repair_lineage if r["final_status"] == "PROMOTED_REPAIRED"]
        )

        # 3. Create Manifest
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True).encode("utf-8")
        ).hexdigest()

        manifest = ExperimentManifest.create(
            experiment_id=self.experiment_id,
            dataset_hash=self.dataset_hash,
            code_commit=self.code_commit,
            configuration_hash=config_hash,
            random_seed=self.random_seed,
            proposer_version="PROPOSER_HETEROGENEOUS_v1",
            validator_version=frozen_spec["version"],
            repair_policy_version="EEDR_POLICY_v1.0",
            evaluation_protocol=evaluation_protocol,
            allowed_data=("training_observables", "validation_observables", "test_observables"),
            forbidden_data=("answer_key", "hidden_labels", "future_test_results"),
            candidate_count=c_count,
            repair_count=r_count,
            terminal_abstentions=t_abstain,
            promotion_count=p_count,
            event_log_hash=event_log_hash,
        )

        provenance = {
            "aeh_version": self.VERSION,
            "referee_kernel_sha256": frozen_spec["referee_kernel_sha256"],
            "claim_ladder_sha256": frozen_spec["claim_ladder_sha256"],
            "grammar_sha256": frozen_spec["grammar_sha256"],
        }

        # Compute Package Hash
        pkg_payload = {
            "manifest_hash": manifest.manifest_hash,
            "event_log_hash": event_log_hash,
            "candidate_count": c_count,
            "repair_count": r_count,
            "provenance": provenance,
        }
        pkg_hash = hashlib.sha256(
            json.dumps(pkg_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        package = HarnessResultPackage(
            experiment_id=self.experiment_id,
            manifest=manifest,
            candidates=list(self._candidates),
            decisions=list(self._decisions),
            repair_lineage=list(self._repair_lineage),
            failures=list(self._failures),
            metrics=metrics or {},
            provenance=provenance,
            ledger_hash=event_log_hash,
            package_hash=pkg_hash,
        )

        self.ledger.record_event(
            role=ComponentRole.HARNESS_ORCHESTRATOR,
            action="EXPERIMENT_FINALIZED_PACKAGE_EMITTED",
            payload_summary={"package_hash": pkg_hash, "manifest_hash": manifest.manifest_hash},
        )

        return package
