"""P1 — LLM-guided topology selection (blueprint §10.2).

P1 may rank the next topology to try and explain the ranking. It may not decide
acceptance, change the tolerance, or hand a candidate to the referee: it returns
an ordered selection plus the usage record, and the shared optimiser and checkers
do everything after that. The class deliberately has no method that returns a
verdict — see `tests/test_mmlm_matter_001_model_interop.py`, which asserts that
this surface stays absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mmlm_matter_001.evaluation.budget_accountant import LLMUsage, usage_from_response
from mmlm_matter_001.models.adapters import ModelAdapter, ModelRequest
from mmlm_matter_001.models.contract import ModelContract
from mmlm_matter_001.models.outputs import (
    P1SelectionOutput,
    TopologySelection,
    parse_p1_selection,
)
from mmlm_matter_001.models.prompts import (
    p1_prompt,
    render_catalogue,
    render_envelope,
    render_feedback,
    render_target_curve,
)
from mmlm_matter_001.schemas import PublicFeedback, PublicTaskView, TopologyEntry


@dataclass(frozen=True, slots=True)
class P1Proposal:
    """An ordered topology selection. It is not an acceptance decision."""

    task_id: str
    selections: tuple[TopologySelection, ...]
    usage: LLMUsage
    raw_text: str
    request_hash: str

    @property
    def topology_ids(self) -> tuple[str, ...]:
        return tuple(selection.topology_id for selection in self.selections)


class P1TopologyProposer:
    """Ranks catalogue topologies for the next downstream evaluation."""

    arm = "P1"
    proposer_id = "p1-llm-topology-v1"

    def __init__(self, *, contract: ModelContract, adapter: ModelAdapter) -> None:
        self._contract = contract
        self._adapter = adapter

    @property
    def contract(self) -> ModelContract:
        return self._contract

    def rank(
        self,
        task: PublicTaskView,
        *,
        catalogue: Sequence[TopologyEntry],
        max_selections: int,
        remaining_credits: int | None = None,
        feedback: Sequence[PublicFeedback] = (),
    ) -> P1Proposal:
        if not isinstance(task, PublicTaskView):
            raise TypeError(
                "P1 may only be given a PublicTaskView; sealed task fields stay out of the "
                "prompt so the ranking cannot be informed by the planted answer")
        entries = tuple(catalogue)
        if not entries:
            raise ValueError("cannot rank an empty catalogue")
        if not 1 <= max_selections <= len(entries):
            raise ValueError(
                f"max_selections must be in [1, {len(entries)}], got {max_selections}")

        template = p1_prompt()
        self._contract.check_pins(template, P1SelectionOutput)
        user_prompt = template.render(
            task_id=task.task_id,
            family=task.family,
            target_kind=task.target_kind,
            tolerance=task.tolerance,
            units=task.units,
            input_range_deg=f"[{task.input_range_deg[0]!r}, {task.input_range_deg[1]!r}]",
            envelope_mm=render_envelope(task.envelope_mm),
            sample_count=len(task.target_points) or len(task.function_samples),
            target_curve=render_target_curve(task),
            catalogue_block=render_catalogue(entries),
            remaining_credits="unlimited" if remaining_credits is None else str(remaining_credits),
            max_selections=max_selections,
            feedback_block=render_feedback(feedback),
        )
        request = ModelRequest(
            contract=self._contract,
            system_prompt=template.system,
            user_prompt=user_prompt,
        )
        response = self._adapter.complete(request)
        output = parse_p1_selection(
            response.text,
            catalogue_ids=[entry.topology_id for entry in entries],
            max_selections=max_selections,
        )
        return P1Proposal(
            task_id=task.task_id,
            selections=tuple(output.selections),
            usage=usage_from_response(response, contract=self._contract, arm=self.arm),
            raw_text=response.text,
            request_hash=request.request_hash,
        )
