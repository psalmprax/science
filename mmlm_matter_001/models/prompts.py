"""Versioned P1/P2 prompt templates.

The template text *and* the system prompt are hashed together, because the
contract's `prompt_hash` has to describe everything that is actually sent. A
change to either is a protocol change: the contract pin will refuse to run
(see `ModelContract.check_pins`) rather than let a run report numbers under a
prompt it did not use.

Templates are provider-neutral — no vendor-specific phrasing, no tool syntax —
so the same pinned protocol works against any adapter in
`mmlm_matter_001.models.adapters`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mmlm_matter_001.models.contract import canonical_json, sha256_text
from mmlm_matter_001.schemas import PublicFeedback, PublicTaskView, TopologyEntry


class PromptRenderError(RuntimeError):
    """A template was rendered without one of its declared fields."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    prompt_id: str
    arm: str
    system: str
    template: str

    @property
    def prompt_hash(self) -> str:
        return sha256_text(canonical_json({"system": self.system, "template": self.template}))

    def render(self, **values: Any) -> str:
        try:
            return self.template.format(**values).strip()
        except KeyError as exc:
            raise PromptRenderError(
                f"prompt {self.prompt_id!r} is missing field {exc.args[0]!r}") from exc


_TASK_BLOCK = """PUBLIC TASK SPECIFICATION
task_id: {task_id}
family: {family}
target kind: {target_kind}
tolerance: {tolerance} {units}
input range (deg): {input_range_deg}
envelope (mm): {envelope_mm}
target samples ({sample_count} ordered): {target_curve}"""


P1_TOPOLOGY_RANKING = PromptTemplate(
    prompt_id="p1_topology_ranking_v1",
    arm="P1",
    system=(
        "You rank candidate topologies for a planar, one-degree-of-freedom, revolute-only "
        "linkage synthesis task. You decide only the order in which a shared numerical "
        "optimiser examines topologies. You never accept or reject a design: an independent "
        "referee validates every candidate the optimiser produces. Answer with exactly one "
        "JSON object matching the output contract, and nothing else."),
    template=(
        _TASK_BLOCK
        + """

ALLOWED TOPOLOGIES (rank only the exact ids listed)
{catalogue_block}

DOWNSTREAM BUDGET
shared topology-evaluation credits remaining: {remaining_credits}
select at most {max_selections} topologies, best first

PUBLIC FEEDBACK FROM EARLIER ATTEMPTS OF THIS SAME RUN
{feedback_block}

OUTPUT CONTRACT
Return exactly one JSON object, with no prose and no code fence:
{{"selections": [{{"topology_id": "<id copied from the allowed list>", "rank": 1, "rationale": "<at most 400 characters>"}}]}}
Rules: every topology_id is copied exactly from the allowed list; no duplicates; ranks are
distinct positive integers; at most {max_selections} selections; the rationale explains the
structural reason for the ordering, not a claim that a design works."""))

P2_DIRECT_DESIGN = PromptTemplate(
    prompt_id="p2_direct_design_v1",
    arm="P2",
    system=(
        "You design one planar, one-degree-of-freedom, revolute-only linkage for the task "
        "below. No numerical optimiser runs after you: the dimensions you return are checked "
        "as written by an independent referee. Answer with exactly one JSON object matching "
        "the output contract, and nothing else."),
    template=(
        _TASK_BLOCK
        + """

ALLOWED TOPOLOGIES (choose exactly one id)
{catalogue_block}

OUTPUT CONTRACT
Return exactly one JSON object, with no prose and no code fence:
{{"topology_id": "<allowed id>", "link_lengths_mm": {{"<link name>": 42.0}}, "coupler_point_mm": [0.0, 0.0], "rationale": "<at most 800 characters>"}}
Rules: link_lengths_mm covers every link of the chosen topology with positive millimetre
lengths; coupler_point_mm is the coupler point in the coupler frame, in millimetres."""))

PROMPTS: dict[str, PromptTemplate] = {
    template.prompt_id: template
    for template in (P1_TOPOLOGY_RANKING, P2_DIRECT_DESIGN)
}


def get_prompt(prompt_id: str) -> PromptTemplate:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown prompt {prompt_id!r}; known: {sorted(PROMPTS)}") from exc


def p1_prompt() -> PromptTemplate:
    return P1_TOPOLOGY_RANKING


def p2_prompt() -> PromptTemplate:
    return P2_DIRECT_DESIGN


def render_catalogue(entries: Sequence[TopologyEntry]) -> str:
    if not entries:
        raise ValueError("cannot render an empty catalogue")
    ordered = sorted(entries, key=lambda entry: entry.topology_id)
    return "\n".join(entry.describe() for entry in ordered)


def render_target_curve(task: PublicTaskView) -> str:
    if task.target_points:
        return ", ".join(f"({x!r}, {y!r})" for x, y in task.target_points)
    if task.function_samples:
        return ", ".join(f"(input {x!r} deg -> output {y!r} deg)" for x, y in task.function_samples)
    return "none supplied"


def render_feedback(feedback: Sequence[PublicFeedback]) -> str:
    if not feedback:
        return "none yet"
    lines = []
    for row in sorted(feedback, key=lambda item: item.topology_id):
        error = "unscored" if row.error_metric is None else f"{row.error_metric:.6g}"
        lines.append(
            f"- {row.topology_id}: {row.status}, error={error}, "
            f"credits_used={row.credits_used}")
    return "\n".join(lines)


def render_envelope(envelope: tuple[float, float, float] | None) -> str:
    if envelope is None:
        return "none declared"
    return " x ".join(f"{value!r}" for value in envelope)
