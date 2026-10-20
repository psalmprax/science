r"""Verified Knowledge Module Library (Gap-3 realization, additive over ledgers).

Maps the verified-knowledge-library requirement onto the EXISTING single
sources of truth:

* ``FullLineageExperimentLedger`` — a candidate may only become a module if
  its ledger node state is CONFIRMED_VALID.
* ``PromotionTracker`` — a candidate may only become a module if promotion
  has passed STAGE_4_OUT_OF_SAMPLE_TRANSFER (untouched holdout transfer).
  Stages 5-6 (economic validity / real-market promotion) are deliberately
  NOT required for library entry: the library is domain-general and economic
  stages are trading-specific — this boundary is recorded as a disclosure on
  every exported module rather than silently applied.

Each module carries the Gap-3 metadata explicitly:
  preconditions      — what must hold for reuse (typed strings)
  validity_domain    — per-input finite ranges; composition requires domain
                       INTERSECTION over all inputs present in every module
  uncertainty        — metric -> {"value": float, "method": str}; naked
                       numbers without an estimation method are refused
  dependencies       — other module ids that must be registerable/known
  counterexamples    — REQUIRED non-empty list of documented failing contexts
  evidence           — ledger node reference + highest passed promotion stage

The library never writes to the ledgers (read-only additive layer) and never
confers promotion: entry is evidence-gated, not a pass.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from apedp_dyn.epistemic.experiment_ledger import CandidateLifecycleState, FullLineageExperimentLedger
from apedp_dyn.epistemic.promotion_lifecycle import PromotionStage, PromotionTracker


class ModuleIntegrityError(ValueError):
    """Raised when a module would be registered/exported beyond its evidence."""


class ModuleCompositionError(ValueError):
    """Raised when a composition chain fails precondition/validity checks."""


@dataclass(frozen=True)
class ValidatedModuleSpec:
    """Metadata contract for one verified, reusable module."""
    module_id: str
    candidate_id: str
    formula: str
    preconditions: tuple[str, ...]
    validity_domain: dict[str, tuple[float, float]]  # input -> (min, max), finite
    uncertainty: dict[str, dict[str, Any]]  # metric -> {"value": float, "method": str}
    counterexamples: tuple[str, ...]  # REQUIRED non-empty (documented failures)
    dependencies: tuple[str, ...] = ()  # other module ids
    export_version: int = 1

    def __post_init__(self) -> None:
        if not self.preconditions:
            raise ModuleIntegrityError(f"{self.module_id}: preconditions must be non-empty")
        for inp, (lo, hi) in self.validity_domain.items():
            if not (float(lo) <= float(hi)):
                raise ModuleIntegrityError(f"{self.module_id}: invalid range for {inp}")
        for metric, rec in self.uncertainty.items():
            if "value" not in rec or "method" not in rec or not rec["method"]:
                raise ModuleIntegrityError(
                    f"{self.module_id}: uncertainty[{metric}] requires value + method"
                )
        if not self.counterexamples:
            raise ModuleIntegrityError(
                f"{self.module_id}: counterexamples are REQUIRED (a module without a "
                "documented failure is not verified knowledge)"
            )


@dataclass
class RegisteredModule:
    spec: ValidatedModuleSpec
    evidence: dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class VerifiedModuleLibrary:
    """Evidence-gated registry; read-only over the ledgers (additive layer)."""

    # Library entry requires OOS transfer (Stage 4); economic stages 5-6 are
    # trading-specific and NOT required — disclosed on every export.
    MIN_PROMOTION_STAGE = PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER

    def __init__(self) -> None:
        self._modules: dict[str, RegisteredModule] = {}
        self._disclosure = (
            "Library entry gates: experiment-ledger CONFIRMED_VALID + promotion "
            "Stage 4 (OOS transfer). Stages 5-6 (economic/real-market) are NOT "
            "required for general-domain reuse and are not implied by entry."
        )

    def register_module(
        self,
        spec: ValidatedModuleSpec,
        ledger: FullLineageExperimentLedger,
        promotion: PromotionTracker,
    ) -> RegisteredModule:
        # Gate 1: experiment-ledger evidence (full-lineage, anti-snooping layer).
        try:
            node = ledger.get_candidate(spec.candidate_id)
        except KeyError as exc:
            raise ModuleIntegrityError(
                f"{spec.candidate_id}: no experiment-ledger node; refusing to "
                "register untracked knowledge"
            ) from exc
        if node.state is not CandidateLifecycleState.CONFIRMED_VALID:
            raise ModuleIntegrityError(
                f"{spec.candidate_id}: ledger state {node.state.value} is not "
                "confirmed_valid; refusing to promote it to reusable knowledge"
            )
        # Gate 2: promotion lifecycle evidence (separate referee, not fungible).
        if promotion.highest_stage_passed < self.MIN_PROMOTION_STAGE:
            stage_names = {
                PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER: "stage 4 (out-of-sample transfer)",
            }
            raise ModuleIntegrityError(
                f"{spec.candidate_id}: promotion stage "
                f"{promotion.highest_stage_passed.name} < "
                f"{stage_names[PromotionStage.STAGE_4_OUT_OF_SAMPLE_TRANSFER]}"
            )
        if spec.dependencies:
            missing = [d for d in spec.dependencies if d not in self._modules]
            if missing:
                raise ModuleIntegrityError(
                    f"{spec.module_id}: unresolved dependencies {missing}"
                )
        reg = RegisteredModule(
            spec=spec,
            evidence={
                "experiment_node_state": node.state.value,
                "promotion_stage_passed": promotion.highest_stage_passed.name,
                "disclosure": self._disclosure,
            },
        )
        self._modules[spec.module_id] = reg
        return reg

    def get(self, module_id: str) -> RegisteredModule:
        if module_id not in self._modules:
            raise ModuleCompositionError(f"unknown module {module_id}")
        return self._modules[module_id]

    def check_compatible(self, chain: list[str]) -> dict[str, Any]:
        """Composition validity: domain intersection + dependency closure.

        If a chain input is declared in DIFFERENT overlapping modules with
        OVERLAPPING ranges, the composed module's validity domain is the
        INTERSECTION (the safe subset); disjoint ranges refuse composition.
        Returns the effective composed validity domain.
        """
        if not chain:
            raise ModuleCompositionError("empty chain")
        missing_deps: set[str] = set()
        seen: set[str] = set()
        domain: dict[str, tuple[float, float]] = {}
        for mid in chain:
            reg = self.get(mid)
            if mid in seen:
                continue
            seen.add(mid)
            missing_deps |= {d for d in reg.spec.dependencies if d not in seen and d not in chain}
            for inp, rng_v in reg.spec.validity_domain.items():
                if inp not in domain:
                    domain[inp] = rng_v
                else:
                    lo = max(domain[inp][0], rng_v[0])
                    hi = min(domain[inp][1], rng_v[1])
                    if lo > hi:
                        raise ModuleCompositionError(
                            f"{mid}: validity domains for {inp} are DISJOINT "
                            f"({list(domain.items()) or ''}[{rng_v[0]}, {rng_v[1]}]) — "
                            "refusing composition outside verified evidence"
                        )
                    domain[inp] = (lo, hi)
        if missing_deps:
            raise ModuleCompositionError(
                f"chain {chain}: dependencies {sorted(missing_deps)} are not in the chain"
            )
        return {"chain": list(chain), "validity_domain": {k: list(v) for k, v in domain.items()}}

    def export(self) -> list[dict[str, Any]]:
        """Deterministic-order export; every record carries the disclosure."""
        out = []
        for mid in sorted(self._modules):
            reg = self._modules[mid]
            spec = reg.spec
            out.append({
                "module_id": spec.module_id,
                "candidate_id": spec.candidate_id,
                "formula": spec.formula,
                "preconditions": list(spec.preconditions),
                "validity_domain": {k: list(v) for k, v in spec.validity_domain.items()},
                "uncertainty": spec.uncertainty,
                "counterexamples": list(spec.counterexamples),
                "dependencies": list(spec.dependencies),
                "export_version": spec.export_version,
                "registered_at": reg.registered_at,
                "evidence": dict(reg.evidence),
            })
        return out
