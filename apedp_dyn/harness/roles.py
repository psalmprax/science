"""Capability Isolation & Permission Firewall for APEDP Experimental Harness (AEH).

Enforces what each scientific component is permitted and forbidden to see.
Maintains a strict boundary:
  - APEDP decides scientific validity.
  - AEH decides experimental permissions and isolation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _freeze_readonly(arr: Any) -> Any:
    """Clear the numpy write flag on ``arr`` and return it (isolation helper)."""
    arr.setflags(write=False)
    return arr


class ComponentRole(str, Enum):
    """Scientific pipeline execution roles."""
    PROPOSER = "proposer"
    VALIDATOR = "validator"
    EEDR_DIAGNOSER = "eedr_diagnoser"
    REPAIRER = "repairer"
    FORWARD_TESTER = "forward_tester"
    FINAL_ADJUDICATOR = "final_adjudicator"
    HARNESS_ORCHESTRATOR = "harness_orchestrator"


class FirewallViolationError(PermissionError):
    """Raised when a component attempts to access forbidden experimental data."""
    pass


@dataclass(frozen=True)
class RolePermissions:
    """Explicit read/write permission matrix for a pipeline role."""
    role: ComponentRole
    allowed_data_types: tuple[str, ...]
    forbidden_data_types: tuple[str, ...]
    can_mutate_evidence: bool = False
    can_access_answer_key: bool = False


# Capability Matrix defining what each component CAN and CANNOT see
CAPABILITY_MATRIX: dict[ComponentRole, RolePermissions] = {
    ComponentRole.PROPOSER: RolePermissions(
        role=ComponentRole.PROPOSER,
        allowed_data_types=("training_observables", "model_priors", "domain_constants"),
        forbidden_data_types=("holdout_data", "answer_key", "hidden_labels", "future_test_results"),
        can_mutate_evidence=False,
        can_access_answer_key=False,
    ),
    ComponentRole.VALIDATOR: RolePermissions(
        role=ComponentRole.VALIDATOR,
        allowed_data_types=("candidate_model", "validation_observables", "metric_specs"),
        forbidden_data_types=("answer_key", "hidden_labels", "future_test_results"),
        can_mutate_evidence=False,
        can_access_answer_key=False,
    ),
    ComponentRole.EEDR_DIAGNOSER: RolePermissions(
        role=ComponentRole.EEDR_DIAGNOSER,
        allowed_data_types=("failure_summary", "failed_candidate_id", "adjudication_verdict"),
        forbidden_data_types=("answer_key", "future_test_results", "unseen_holdout"),
        can_mutate_evidence=False,
        can_access_answer_key=False,
    ),
    ComponentRole.REPAIRER: RolePermissions(
        role=ComponentRole.REPAIRER,
        allowed_data_types=("diagnosed_failure", "allowed_repair_operators", "initial_evidence_fingerprint"),
        forbidden_data_types=("answer_key", "future_test_results", "eventual_test_score", "raw_test_data"),
        can_mutate_evidence=False,  # Anti-p-hacking invariant: repair cannot mutate evidence
        can_access_answer_key=False,
    ),
    ComponentRole.FORWARD_TESTER: RolePermissions(
        role=ComponentRole.FORWARD_TESTER,
        allowed_data_types=("frozen_candidate_model", "test_observables", "evaluation_ladder"),
        forbidden_data_types=("repair_history_heuristics", "answer_key"),
        can_mutate_evidence=False,
        can_access_answer_key=False,
    ),
    ComponentRole.FINAL_ADJUDICATOR: RolePermissions(
        role=ComponentRole.FINAL_ADJUDICATOR,
        allowed_data_types=("complete_evidence_bundle", "candidate_model", "claim_level", "frozen_ladder"),
        forbidden_data_types=("answer_key", "unauthorized_priors"),
        can_mutate_evidence=False,  # Adjudicator cannot modify upstream evidence
        can_access_answer_key=False,
    ),
    ComponentRole.HARNESS_ORCHESTRATOR: RolePermissions(
        role=ComponentRole.HARNESS_ORCHESTRATOR,
        allowed_data_types=("*",),
        forbidden_data_types=(),
        can_mutate_evidence=False,
        can_access_answer_key=True,  # Harness manages answer key for post-hoc scoring ONLY
    ),
}


class DataFirewall:
    """Audits and enforces data access boundaries across pipeline stages."""

    @staticmethod
    def verify_access(
        role: ComponentRole,
        data_type: str,
        attempting_mutation: bool = False,
    ) -> None:
        """Enforces capability permissions against CAPABILITY_MATRIX."""
        perms = CAPABILITY_MATRIX.get(role)
        if not perms:
            raise FirewallViolationError(f"FIREWALL DENIAL: Unknown role '{role}'")

        if perms.role == ComponentRole.HARNESS_ORCHESTRATOR:
            return

        # Check forbidden access
        for forbidden in perms.forbidden_data_types:
            if forbidden == data_type or forbidden in data_type:
                raise FirewallViolationError(
                    f"FIREWALL VIOLATION: Role '{role.value}' is strictly forbidden from accessing '{data_type}'."
                )

        # Check evidence mutation attempt
        if attempting_mutation and not perms.can_mutate_evidence:
            raise FirewallViolationError(
                f"FIREWALL VIOLATION: Role '{role.value}' is not permitted to mutate scientific evidence."
            )

        # Check allowed access
        if "*" not in perms.allowed_data_types:
            allowed = any(allowed_t == data_type or allowed_t in data_type for allowed_t in perms.allowed_data_types)
            if not allowed:
                raise FirewallViolationError(
                    f"FIREWALL DENIAL: Data type '{data_type}' is not within permitted scope for role '{role.value}'. "
                    f"Allowed: {perms.allowed_data_types}"
                )


class IsolatedDataVault:
    """Encapsulates experimental data with fine-grained access control and immutability protection."""

    def __init__(
        self,
        training_data: Mapping[str, Any],
        validation_data: Mapping[str, Any] | None = None,
        test_data: Mapping[str, Any] | None = None,
        answer_key: Mapping[str, Any] | None = None,
    ) -> None:
        self._training_data = self._deep_seal(dict(training_data))
        self._validation_data = self._deep_seal(dict(validation_data or {}))
        self._test_data = self._deep_seal(dict(test_data or {}))
        self._answer_key = self._deep_seal(dict(answer_key or {}))

    @staticmethod
    def _deep_seal(data: dict[str, Any]) -> dict[str, Any]:
        """Deeply copies and marks numpy arrays as read-only to prevent in-place mutation."""
        import copy
        try:
            import numpy as np
            has_np = True
        except ImportError:
            has_np = False

        sealed: dict[str, Any] = {}
        for k, v in data.items():
            if has_np and isinstance(v, np.ndarray):
                arr = v.copy()
                arr.setflags(write=False)
                sealed[k] = arr
            elif isinstance(v, (list, tuple)):
                if has_np and any(isinstance(x, np.ndarray) for x in v):
                    sealed[k] = tuple(
                        (_freeze_readonly(x.copy())
                         if isinstance(x, np.ndarray) else x.copy())
                        for x in v
                    )
                else:
                    sealed[k] = copy.deepcopy(v)
            elif isinstance(v, dict):
                sealed[k] = IsolatedDataVault._deep_seal(v)
            else:
                try:
                    sealed[k] = copy.deepcopy(v)
                except Exception:  # noqa: BLE001 -- deepcopy can fail on exotic objects;
                    # fall back to the original reference so sealing never aborts
                    sealed[k] = v
        return sealed

    def get_data(self, role: ComponentRole, requested_type: str) -> dict[str, Any]:
        """Retrieves data if and only if role possesses permitted capability.
        
        Returns deeply sealed, immutable copies so callers cannot mutate the underlying vault.
        """
        DataFirewall.verify_access(role, requested_type)
        if requested_type == "training_observables":
            return self._deep_seal(self._training_data)
        elif requested_type in ("validation_observables", "model_priors"):
            return self._deep_seal(self._validation_data)
        elif requested_type == "test_observables":
            return self._deep_seal(self._test_data)
        elif requested_type in ("answer_key", "hidden_labels"):
            if role != ComponentRole.HARNESS_ORCHESTRATOR:
                raise FirewallViolationError(
                    f"CRITICAL FIREWALL BREACH: Role '{role}' attempted direct access to '{requested_type}'."
                )
            return self._deep_seal(self._answer_key)
        else:
            raise KeyError(f"Unknown data type '{requested_type}'")

