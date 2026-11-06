"""P2 — direct LLM design, the ablation arm (blueprint §10.3).

P2 proposes the topology and the dimensions without invoking the shared
numerical optimiser. It uses the same checkers and the same acceptance
threshold as P0/P1, and its results are reported separately: the primary
comparison is P1 versus P0, and P2 exists to show whether direct design is
useful at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mmlm_matter_001.evaluation.budget_accountant import LLMUsage, usage_from_response
from mmlm_matter_001.models.adapters import ModelAdapter, ModelRequest
from mmlm_matter_001.models.contract import ModelContract
from mmlm_matter_001.models.outputs import P2DesignOutput, parse_p2_design
from mmlm_matter_001.models.prompts import (
    p2_prompt,
    render_catalogue,
    render_envelope,
    render_target_curve,
)
from mmlm_matter_001.schemas import PublicTaskView, TopologyEntry


@dataclass(frozen=True, slots=True)
class P2Proposal:
    """A direct design. It is not an acceptance decision."""

    task_id: str
    topology_id: str
    link_lengths_mm: dict[str, float]
    coupler_point_mm: tuple[float, float]
    rationale: str
    usage: LLMUsage
    raw_text: str
    request_hash: str


class P2DirectDesignProposer:
    """Proposes one topology and its dimensions with no optimiser in the loop."""

    arm = "P2"
    proposer_id = "p2-llm-direct-v1"

    def __init__(self, *, contract: ModelContract, adapter: ModelAdapter) -> None:
        self._contract = contract
        self._adapter = adapter

    @property
    def contract(self) -> ModelContract:
        return self._contract

    def propose(
        self,
        task: PublicTaskView,
        *,
        catalogue: Sequence[TopologyEntry],
    ) -> P2Proposal:
        if not isinstance(task, PublicTaskView):
            raise TypeError(
                "P2 may only be given a PublicTaskView; sealed task fields stay out of the "
                "prompt so the design cannot be copied from the planted answer")
        entries = tuple(catalogue)
        if not entries:
            raise ValueError("cannot design against an empty catalogue")

        template = p2_prompt()
        self._contract.check_pins(template, P2DesignOutput)
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
        )
        request = ModelRequest(
            contract=self._contract,
            system_prompt=template.system,
            user_prompt=user_prompt,
        )
        response = self._adapter.complete(request)
        output = parse_p2_design(
            response.text, catalogue_ids=[entry.topology_id for entry in entries])
        return P2Proposal(
            task_id=task.task_id,
            topology_id=output.topology_id,
            link_lengths_mm=dict(output.link_lengths_mm),
            coupler_point_mm=tuple(output.coupler_point_mm),
            rationale=output.rationale,
            usage=usage_from_response(response, contract=self._contract, arm=self.arm),
            raw_text=response.text,
            request_hash=request.request_hash,
        )
