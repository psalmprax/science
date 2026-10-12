"""Full-Lineage Experiment & Candidate Ledger with Anti-Snooping Guard.

Maintains an immutable record of the complete hypothesis search process:
  - Lineage tracking: parent hypotheses, feature sets, parameter choices, and variants.
  - Multi-state lifecycle: PROPOSED, PRE_SCREEN_REJECTED, CALIBRATION_FAILED,
    DEVELOPMENT_PASSED, CONFIRMATION_REJECTED, ECONOMIC_EVALUATION_FAILED, CONFIRMED_VALID.
  - Dataset exposure tracking & Holdout Contamination Guard:
    Prevents data-snooping by permanently burning any confirmation holdout slice
    inspected by a failed candidate so it cannot be re-used by successor iterations.
"""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class CandidateLifecycleState(str, Enum):
    """Explicit lifecycle states for candidate hypotheses."""
    PROPOSED = "proposed"
    PRE_SCREEN_REJECTED = "pre_screen_rejected"
    CALIBRATION_FAILED = "calibration_failed"
    DEVELOPMENT_PASSED = "development_passed"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    ECONOMIC_EVALUATION_FAILED = "economic_evaluation_failed"
    CONFIRMED_VALID = "confirmed_valid"
    UNRESOLVED = "unresolved"


class DatasetRole(str, Enum):
    """Role of a dataset slice in the discovery and validation protocol."""
    TRAINING_DEVELOPMENT = "training_development"
    CONFIRMATION_HOLDOUT = "confirmation_holdout"
    PROSPECTIVE_LIVE = "prospective_live"


class HoldoutContaminationError(RuntimeError):
    """Raised when an iterative candidate attempts to inspect a burned/contaminated holdout slice."""
    pass


@dataclass(frozen=True)
class DatasetExposureRecord:
    """Audit record of a candidate inspecting a dataset slice."""
    candidate_id: str
    dataset_id: str
    slice_identifier: str
    role: DatasetRole
    inspected_at: str
    result_status: str  # "PASSED", "FAILED", "IN_PROGRESS"


@dataclass
class CandidateExperimentNode:
    """Full-lineage record for one candidate hypothesis."""
    candidate_id: str
    proposer_id: str
    formula: str
    version: int = 1
    parent_candidate_id: str | None = None
    feature_set: tuple[str, ...] = ()
    parameter_choices: dict[str, float] = field(default_factory=dict)
    attempted_variants: list[str] = field(default_factory=list)
    state: CandidateLifecycleState = CandidateLifecycleState.PROPOSED
    rejection_reason: str | None = None
    data_exposures: list[DatasetExposureRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["data_exposures"] = [
            {**asdict(exp), "role": exp.role.value} for exp in self.data_exposures
        ]
        return d


class FullLineageExperimentLedger:
    """Immutable experiment ledger managing hypothesis lineage and preventing holdout reuse."""

    def __init__(self, persistence_path: Path | str | None = None):
        self._nodes: dict[str, CandidateExperimentNode] = {}
        # Tracks burned holdout slices: (dataset_id, slice_identifier) -> failing_candidate_id
        self._burned_holdouts: dict[tuple[str, str], str] = {}
        self._persistence_path = Path(persistence_path) if persistence_path else None
        if self._persistence_path and self._persistence_path.is_file():
            self._load_from_disk()

    def register_candidate(
        self,
        candidate_id: str,
        proposer_id: str,
        formula: str,
        parent_candidate_id: str | None = None,
        feature_set: tuple[str, ...] = (),
        parameter_choices: dict[str, float] | None = None,
    ) -> CandidateExperimentNode:
        """Register a newly proposed candidate in the experiment ledger."""
        version = 1
        attempted_variants = []
        if parent_candidate_id and parent_candidate_id in self._nodes:
            parent = self._nodes[parent_candidate_id]
            version = parent.version + 1
            attempted_variants = list(parent.attempted_variants) + [parent.formula]

        node = CandidateExperimentNode(
            candidate_id=candidate_id,
            proposer_id=proposer_id,
            formula=formula,
            version=version,
            parent_candidate_id=parent_candidate_id,
            feature_set=tuple(feature_set),
            parameter_choices=parameter_choices or {},
            attempted_variants=attempted_variants,
            state=CandidateLifecycleState.PROPOSED,
        )
        self._nodes[candidate_id] = node
        self._persist_append(node)
        return node

    def record_data_exposure(
        self,
        candidate_id: str,
        dataset_id: str,
        slice_identifier: str,
        role: DatasetRole,
        outcome_status: str = "IN_PROGRESS",
    ) -> DatasetExposureRecord:
        """Record dataset exposure. Raises HoldoutContaminationError if holdout slice was previously burned."""
        if candidate_id not in self._nodes:
            raise KeyError(f"Candidate {candidate_id} not registered in ledger.")

        holdout_key = (dataset_id, slice_identifier)
        if role == DatasetRole.CONFIRMATION_HOLDOUT:
            if holdout_key in self._burned_holdouts:
                failing_cand = self._burned_holdouts[holdout_key]
                raise HoldoutContaminationError(
                    f"DATA SNOOPING VIOLATION: Dataset slice '{dataset_id}:{slice_identifier}' "
                    f"was previously burned by failing candidate '{failing_cand}'. "
                    f"Confirmation holdouts may not be reused for replacement hypotheses."
                )

        exposure = DatasetExposureRecord(
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            slice_identifier=slice_identifier,
            role=role,
            inspected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            result_status=outcome_status,
        )

        node = self._nodes[candidate_id]
        node.data_exposures.append(exposure)
        node.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._persist_append(node)
        return exposure

    def update_lifecycle_state(
        self,
        candidate_id: str,
        new_state: CandidateLifecycleState,
        rejection_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> CandidateExperimentNode:
        """Transition candidate state and burn confirmation holdouts if rejected."""
        if candidate_id not in self._nodes:
            raise KeyError(f"Candidate {candidate_id} not registered.")

        node = self._nodes[candidate_id]
        node.state = new_state
        node.rejection_reason = rejection_reason
        if metrics:
            node.metrics.update(metrics)
        node.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # If rejected at confirmation or economic stage, permanently burn all confirmation slices it inspected
        if new_state in (CandidateLifecycleState.CONFIRMATION_REJECTED,
                         CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED):
            for exp in node.data_exposures:
                if exp.role == DatasetRole.CONFIRMATION_HOLDOUT:
                    self._burned_holdouts[(exp.dataset_id, exp.slice_identifier)] = candidate_id

        self._persist_append(node)
        return node

    def get_candidate(self, candidate_id: str) -> CandidateExperimentNode:
        return self._nodes[candidate_id]

    def get_all_candidates(self) -> list[CandidateExperimentNode]:
        return list(self._nodes.values())

    def get_burned_holdouts(self) -> dict[tuple[str, str], str]:
        return dict(self._burned_holdouts)

    def _persist_append(self, node: CandidateExperimentNode) -> None:
        if self._persistence_path:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(node.to_dict()) + "\n")

    def _load_from_disk(self) -> None:
        assert self._persistence_path is not None
        with open(self._persistence_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                exposures = [
                    DatasetExposureRecord(
                        candidate_id=e["candidate_id"],
                        dataset_id=e["dataset_id"],
                        slice_identifier=e["slice_identifier"],
                        role=DatasetRole(e["role"]),
                        inspected_at=e["inspected_at"],
                        result_status=e["result_status"],
                    )
                    for e in d.get("data_exposures", [])
                ]
                node = CandidateExperimentNode(
                    candidate_id=d["candidate_id"],
                    proposer_id=d["proposer_id"],
                    formula=d["formula"],
                    version=d.get("version", 1),
                    parent_candidate_id=d.get("parent_candidate_id"),
                    feature_set=tuple(d.get("feature_set", ())),
                    parameter_choices=d.get("parameter_choices", {}),
                    attempted_variants=d.get("attempted_variants", []),
                    state=CandidateLifecycleState(d["state"]),
                    rejection_reason=d.get("rejection_reason"),
                    data_exposures=exposures,
                    metrics=d.get("metrics", {}),
                    created_at=d.get("created_at", ""),
                    updated_at=d.get("updated_at", ""),
                )
                self._nodes[node.candidate_id] = node
                if node.state in (CandidateLifecycleState.CONFIRMATION_REJECTED,
                                  CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED):
                    for exp in exposures:
                        if exp.role == DatasetRole.CONFIRMATION_HOLDOUT:
                            self._burned_holdouts[(exp.dataset_id, exp.slice_identifier)] = node.candidate_id
