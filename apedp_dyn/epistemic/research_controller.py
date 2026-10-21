"""EHI Phase 5: Scientific Reasoning & Research Control Layer (fifth pillar).

The four pillars discover, constrain, adjudicate and probe; they do not
decide WHAT to investigate next, WHY, or WHEN TO STOP. This controller owns
that layer under the two-state discipline the external EHI review demands:

A. Scientific BELIEF state — revisable with every observation: candidate
   mechanisms with their support, uncertainty, alternative explanations,
   contradictions, and what remains unidentifiable. Updated freely here.
B. Governance state — the referee's evidential configuration, LOCKED at
   session start (canonical-spec SHA + a fingerprint over the pinned
   referee constants) and re-verified before EVERY planned action. The
   controller may PROPOSE a protocol revision; it can never apply one to a
   running session (the proposal is versioned and returned, nothing more).
   When evidence contradicts a candidate, the CANDIDATE is reconsidered —
   the standard is never quietly lowered.

Every planned action carries the five-part epistemic ledger the review
requires: what the data establish, what the model assumes, what remains
unidentifiable, what new observation could resolve the uncertainty, and
what the system must NOT claim even if a candidate fits extremely well.

Next-action planning is DETERMINISTIC over the belief state (documented
utility model): inspect problem → test constraint hypotheses → design an
information-gain experiment → improve data quality → or ABSTAIN. The loop
has a controlled exit: no feasible discriminating experiment ⇒ abstention
is the correct scientific outcome, not a failure to be engineered around.
(The referee stays external: promotions remain adjudicate()'s authority.)

Status: experimental (EHI Phase 5 deliverable; frozen-benchmark loop, no
human steering inside the run).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from apedp_dyn.conf.canonical_spec import PINNED_SPEC_SHA256
from apedp_dyn.epistemic.active_experimentalist import (
    INFO_GAIN_STOP_EPS,
    ExperimentOrder,
    design_information_orders,
    record_order_results,
)
from apedp_dyn.orchestration.model_class_selector import (
    OUTCOME_INSUFFICIENT_METADATA,
    OUTCOME_MULTIPLE_CLASSES,
    ModelClassSelection,
    select_model_class,
)

__all__ = [
    "BeliefState",
    "GovernanceState",
    "InvestigationReport",
    "NextAction",
    "PlanJustification",
    "ResearchController",
]


class NextAction(str, Enum):
    INSPECT_PROBLEM = "inspect_problem"
    TEST_CONSTRAINTS = "test_constraint_hypotheses"
    DESIGN_EXPERIMENT = "design_information_experiment"
    IMPROVE_DATA_QUALITY = "improve_data_quality"
    ABSTAIN = "abstain"


# ---------------------------------------------------------------------------
# Governance fingerprint: hash over the pinned referee constants actually
# imported by the referee stack (documented subset — the fingerprint refuses
# to be a ceremony: any drift in these values fails the run at the next
# decision point, because the governance state is re-verified, not trusted).
# ---------------------------------------------------------------------------
_GOVERNED_KEYS = ("DSR_THRESHOLD", "REFEREE_MAX_FPR", "REFEREE_MIN_CAR", "REFEREE_MIN_VRR")


def _governance_fingerprint() -> str:
    from apedp_dyn.conf import canonical_spec as cs

    snapshot = {k: repr(getattr(cs, k)) for k in _GOVERNED_KEYS if hasattr(cs, k)}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GovernanceState:
    spec_sha256: str
    referee_fingerprint: str
    locked_round: int


@dataclass
class BeliefState:
    problem: str = ""
    modality_outcome: str = ""
    posteriors: dict[str, float] = field(default_factory=dict)
    constraint_notes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    unidentifiable: list[str] = field(default_factory=list)
    acquired_evidence: list[str] = field(default_factory=list)
    data_quality_rejections: int = 0


@dataclass(frozen=True)
class PlanJustification:
    action: NextAction
    why: str
    what_data_establish: str
    model_assumes: str
    remains_unidentifiable: str
    next_resolution: str
    must_not_claim: str
    # Every PlanJustification leaves plan_next_action() AFTER verify_governance()
    # has passed (a tampered session raises instead of planning), so the default
    # is honest for all paths through the planner; direct constructions that
    # bypass the planner must set it explicitly.
    governance_verified: bool = True


@dataclass(frozen=True)
class InvestigationReport:
    rounds: int
    actions: tuple[str, ...]
    outcome: str  # "CONCLUSION_SUPPORTED" | "ABSTAINED" | "INCONCLUSIVE"
    belief: BeliefState
    ledger: dict[str, str]
    governance_intact: bool


class ResearchController:
    """Belief/governance separated investigation controller (fifth pillar)."""

    def __init__(self, problem: str) -> None:
        self.problem = problem
        self.belief = BeliefState(problem=problem)
        self.governance = GovernanceState(
            spec_sha256=PINNED_SPEC_SHA256, referee_fingerprint=_governance_fingerprint(),
            locked_round=0)
        self.governance_intact = True

    def verify_governance(self) -> bool:
        intact = (
            PINNED_SPEC_SHA256 == self.governance.spec_sha256
            and _governance_fingerprint() == self.governance.referee_fingerprint
        )
        self.governance_intact &= intact
        return intact

    def propose_protocol_revision(self, revision: dict) -> dict:
        """A protocol change is a VERSIONED PROPOSAL, never an application.

        The running session's governance state is frozen; the returned
        revision carries the successor fingerprint and a new-version marker
        so an audit can see it was never applied mid-experiment.
        """
        return {
            "revision": revision,
            "status": "PROPOSED_NOT_APPLIED",
            "current_governance_fingerprint": self.governance.referee_fingerprint,
            "requires": "fresh confirmation dataset + governance review",
        }

    # -- planner ---------------------------------------------------------
    def plan_next_action(self, selection: ModelClassSelection | None,
                         orders: tuple[ExperimentOrder, ...], *,
                         design_attempted: bool = False) -> PlanJustification:
        if not self.verify_governance():
            raise RuntimeError("GOVERNANCE STATE CHANGED MID-RUN: refusing to plan further actions")
        if selection is None or selection.outcome in (OUTCOME_INSUFFICIENT_METADATA,):
            return PlanJustification(
                NextAction.INSPECT_PROBLEM,
                "no usable problem specification yet: axes/units/kind are unresolved",
                "the byte layout is known; the mathematical meaning is not",
                "rows are observations of SOME process kind — nothing narrower",
                "observation process; measurement units; sampling geometry",
                "declared axes or a kind annotation; a schema avatar of the columns",
                "no model-class claim until the observation process is identified")
        if orders:
            return PlanJustification(
                NextAction.DESIGN_EXPERIMENT,
                f"{len(orders)} feasible discriminating experiment(s) with positive expected information"
                f" (I*={orders[0].expected_information_gain:.4f} nats > stop eps {INFO_GAIN_STOP_EPS})",
                "competing hypotheses are observationally distinguishable somewhere feasible",
                "each candidate's predictions are correct in mean under the measured noise",
                "the true mechanism among the survivors; parameters off the measured range",
                "the ordered measurement itself (ranges + required precision on the order)",
                "no promotion of any candidate before the external referee's verdict")
        if self.belief.data_quality_rejections > 0 and not orders:
            return PlanJustification(
                NextAction.IMPROVE_DATA_QUALITY,
                "acquisition returned out-of-band or non-finite points: instrument honesty",
                "the instrument sometimes produces mistrustful values",
                "the acquisition channel is the one ordered",
                "whether rejections are instrument fault or domain truth",
                "calibration metadata + a re-run of the same ordered ranges",
                "no scientific claim built on rejected values")
        if selection.outcome == OUTCOME_MULTIPLE_CLASSES:
            return PlanJustification(
                NextAction.ABSTAIN,
                "ambiguity persists and no feasible experiment reduces the uncertainty below"
                " the documented stop floor: the loop exits with what is known",
                "the surviving hypothesis classes could not be separated",
                "the candidate profiles are correct approximations of each class",
                "the mechanism identity; any asymptote beyond the measured window",
                "a new feasible domain, another hypothesis family, or better calibration",
                "NOTHING beyond 'multiple mechanisms remain plausible'; Level-5 autonomy"
                " does not mean producing a discovery on every run")
        if design_attempted:
            return PlanJustification(
                NextAction.ABSTAIN,
                "the information-design layer found NO feasible experiment above the"
                " stop floor: fabricating an order would manufacture a discovery",
                "the survivors' predictions are observationally indistinguishable at"
                " every feasible, noise-relevant operating point",
                "the candidates' predictive profiles hold inside the measured window",
                "the mechanism identity beyond the measured window",
                "a wider feasible domain, an independent hypothesis family, or better"
                " measurement precision — outside this session's budget",
                "NO defended conclusion until new evidence arrives; abstention is the"
                " outcome, not a failure to be engineered around")
        return PlanJustification(
            NextAction.TEST_CONSTRAINTS,
            "decisive routing recorded; constraint hypotheses need held-out comparison",
            selection.reason,
            "the routed family generated the data",
            "parameter identifiability within the family",
            "held-out comparison of constrained vs unconstrained family members",
            "constraints stay hypotheses with provenance until independently justified")

    # -- one investigation round ------------------------------------------
    def run_investigation(
        self,
        frame: pd.DataFrame,
        hypotheses: dict[str, object] | None = None,
        *,
        domain: tuple[float, float] | None = None,
        sigma_noise: float | None = None,
        acquisition=None,
        x_col: str = "x",
        max_rounds: int = 4,
    ) -> InvestigationReport:
        """Frozen-benchmark loop: select → plan → act → re-plan → exit.

        ``hypotheses`` are candidate survival functions the CALLER supplies
        (proposer tournament outputs); the controller never generates its
        own promotions. ``acquisition(order) -> (x_new, y_new)`` is the lab.
        """
        actions: list[str] = []
        selection: ModelClassSelection | None = None
        orders: tuple[ExperimentOrder, ...] = ()
        for round_i in range(max_rounds):
            if selection is None:
                selection = select_model_class(frame)
                self.belief.modality_outcome = selection.outcome
                self.belief.posteriors = dict(selection.posteriors)
                if selection.outcome == OUTCOME_INSUFFICIENT_METADATA:
                    self.belief.open_questions.append("observation process and axes unannotated")
                plan = self.plan_next_action(selection, orders)
                if plan.action in (NextAction.ABSTAIN, NextAction.INSPECT_PROBLEM):
                    actions.append(plan.action.value)
                    if plan.action == NextAction.ABSTAIN:
                        actions.append(plan.action.value)
                        break
                    # INSPECT ask: the run has no data-acquisition channel for
                    # problem formulation — terminate without pretending to act.
                    break
                # The decisively-routed round still RECORDS its planning step:
                # the audit trail must show the belief-forming action, not skip
                # straight to the designed measurement.
                actions.append(plan.action.value)
            if hypotheses and len(hypotheses) >= 2 and domain and sigma_noise:
                # A bogus noise model is refused at the planner boundary, not
                # rescued by a silent epsilon clamp that would fake a signal.
                if not (np.isfinite(sigma_noise) and sigma_noise > 0):
                    raise ValueError("sigma_noise must be positive and finite")
                orders = design_information_orders(hypotheses, domain, sigma_noise,
                                                   variable=x_col)
                plan = self.plan_next_action(selection, orders,
                                             design_attempted=not orders)
                actions.append(plan.action.value)
                if plan.action == NextAction.DESIGN_EXPERIMENT and acquisition is not None:
                    x_new, y_new = acquisition(orders[0])
                    _x_ok, _y_ok, n_ok, n_rej = record_order_results(
                        orders[0], np.asarray(x_new, dtype=float), np.asarray(y_new, dtype=float))
                    self.belief.data_quality_rejections += n_rej
                    if n_ok == 0:
                        # Good instruments before more theory: the acquisition
                        # channel is the one this session ordered.
                        plan = self.plan_next_action(selection, ())
                        actions.append(plan.action.value)
                        break
                    self.belief.acquired_evidence.append(
                        f"acquired {n_ok} accepted points (round {round_i}, "
                        f"{n_rej} rejected by range/integrity gate)")
                    # One acquisition per session: incorporating the result
                    # requires a referee re-adjudication (caller_data_protocol
                    # / experiment_ledger services it); the controller never
                    # self-adjudicates its own order. The run ends here and
                    # reports evidence-integrated-awaiting-external-verdict.
                    break
                if plan.action in (NextAction.ABSTAIN, NextAction.IMPROVE_DATA_QUALITY):
                    break
                continue
            plan = self.plan_next_action(selection, ())
            actions.append(plan.action.value)
            break
        # The loop NEVER declares its own promotion: the only terminal states
        # are abstention, honest incompleteness, and evidence handed to the
        # external referee (adjudicate() stays the acceptance authority).
        if any(a == "abstain" for a in actions):
            outcome = "ABSTAINED"
        elif any(a == "improve_data_quality" for a in actions) and not self.belief.acquired_evidence:
            # Refused to conclude on rejected values — an honest stop, not a
            # produced conclusion.
            outcome = "ABSTAINED"
        elif self.belief.acquired_evidence:
            outcome = "EVIDENCE_INTEGRATED_AWAITING_REFEREE"
        elif any(a == "inspect_problem" for a in actions):
            outcome = "BLOCKED_ON_PROBLEM_SPECIFICATION"
        else:
            outcome = "INCONCLUSIVE"
        return InvestigationReport(
            rounds=len(actions), actions=tuple(actions), outcome=outcome,
            belief=self.belief, ledger=_ledger(self, selection),
            governance_intact=self.governance_intact)


def _ledger(controller: ResearchController, selection: ModelClassSelection | None) -> dict[str, str]:
    out = selection.outcome if selection else ""
    return {
        "what_data_establish": out or "layout only",
        "model_assumes": "documented class profiles (selector layer)",
        "remains_unidentifiable": "; ".join(controller.belief.unidentifiable)
        or "mechanism identity; ranges outside the measured window",
        "could_resolve": "; ".join(controller.belief.open_questions)
        or "a feasible discriminating measurement in the ordered ranges",
        "must_not_claim": "Must not claim: any mechanism promoted without the external "
                           "referee's verdict; any asymptote beyond the measured window; "
                           "constraints as physical facts rather than hypotheses",
    }
