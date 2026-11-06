"""Strict output schemas and parsers for the P1 and P2 arms.

Blueprint §10.2: "Require a strict JSON output schema containing topology IDs
from the supplied catalogue. Reject unknown IDs, malformed output and duplicate
selections." Every rejection here carries a machine-readable reason so the
budget ledger and the phase report can count *why* proposals failed, rather than
collapsing all failures into "the model produced nothing".
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

P1_SCHEMA_ID = "p1_topology_selection_v1"
P2_SCHEMA_ID = "p2_direct_design_v1"

REASONS = frozenset({
    "not_json",
    "schema",
    "empty",
    "too_many",
    "unknown_topology",
    "duplicate_topology",
    "duplicate_rank",
    "non_physical",
})


class ModelOutputError(ValueError):
    """A model response that the frozen protocol does not accept."""

    def __init__(self, reason: str, detail: str, *, raw_text: str | None = None) -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown rejection reason {reason!r}; add it to REASONS first")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.raw_text = raw_text


class TopologySelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topology_id: str = Field(min_length=1, max_length=128)
    rank: int = Field(ge=1, le=10_000)
    rationale: str = Field(min_length=1, max_length=400)


class P1SelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selections: list[TopologySelection]


class P2DesignOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topology_id: str = Field(min_length=1, max_length=128)
    link_lengths_mm: dict[str, float] = Field(min_length=1)
    coupler_point_mm: tuple[float, float]
    rationale: str = Field(min_length=1, max_length=800)


def _strip_single_fence(text: str) -> tuple[str, str]:
    """Remove one whole-text Markdown fence, the only permitted normalisation.

    Anything else is left for the JSON parser to reject: silently repairing
    model output would make the frozen prompt/schema hashes describe a protocol
    that was not actually applied.
    """
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")) or len(stripped) < 6:
        return stripped, "none"
    inner = stripped[3:-3].strip()
    first_line, _, rest = inner.partition("\n")
    if first_line.strip().lower() in {"json", ""}:
        return rest.strip(), "fenced_block_stripped"
    return inner, "fenced_block_stripped"


def _load_json(text: str) -> Any:
    payload_text, _ = _strip_single_fence(text)
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            "not_json",
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            raw_text=text) from exc


def _first_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"{location}: {first['msg']}"


def parse_p1_selection(
    text: str,
    *,
    catalogue_ids: Iterable[str],
    max_selections: int,
) -> P1SelectionOutput:
    """Validate a P1 ranking against the catalogue it was given."""
    allowed = frozenset(catalogue_ids)
    if max_selections < 1:
        raise ValueError(f"max_selections must be positive, got {max_selections}")
    payload = _load_json(text)
    try:
        output = P1SelectionOutput.model_validate(payload)
    except ValidationError as exc:
        raise ModelOutputError("schema", _first_validation_error(exc), raw_text=text) from exc

    selections = output.selections
    if not selections:
        raise ModelOutputError(
            "empty", "the model returned zero selections", raw_text=text)
    if len(selections) > max_selections:
        raise ModelOutputError(
            "too_many",
            f"{len(selections)} selections exceed the requested maximum of {max_selections}",
            raw_text=text)

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for selection in selections:
        if selection.topology_id in seen_ids:
            raise ModelOutputError(
                "duplicate_topology",
                f"topology {selection.topology_id!r} was selected more than once",
                raw_text=text)
        seen_ids.add(selection.topology_id)
        if selection.topology_id not in allowed:
            raise ModelOutputError(
                "unknown_topology",
                f"{selection.topology_id!r} is not in the supplied catalogue",
                raw_text=text)
        if selection.rank in seen_ranks:
            raise ModelOutputError(
                "duplicate_rank", f"rank {selection.rank} was used more than once",
                raw_text=text)
        seen_ranks.add(selection.rank)

    ordered = sorted(selections, key=lambda selection: selection.rank)
    return output.model_copy(update={"selections": ordered})


def parse_p2_design(text: str, *, catalogue_ids: Iterable[str]) -> P2DesignOutput:
    """Validate a P2 direct design: a catalogue id plus physical dimensions."""
    allowed = frozenset(catalogue_ids)
    payload = _load_json(text)
    try:
        output = P2DesignOutput.model_validate(payload)
    except ValidationError as exc:
        raise ModelOutputError("schema", _first_validation_error(exc), raw_text=text) from exc

    if output.topology_id not in allowed:
        raise ModelOutputError(
            "unknown_topology",
            f"{output.topology_id!r} is not in the supplied catalogue",
            raw_text=text)
    for name, length in output.link_lengths_mm.items():
        if not name:
            raise ModelOutputError("schema", "a link name is empty", raw_text=text)
        if not math.isfinite(length) or length <= 0:
            raise ModelOutputError(
                "non_physical",
                f"link {name!r} has length {length!r}; millimetre lengths must be finite "
                "and positive",
                raw_text=text)
    if not all(math.isfinite(value) for value in output.coupler_point_mm):
        raise ModelOutputError(
            "non_physical",
            f"coupler point {output.coupler_point_mm!r} is not finite",
            raw_text=text)
    return output
