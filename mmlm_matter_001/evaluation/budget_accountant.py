"""LLM usage accounting for the Phase 1 arms (blueprint §11).

§11.1 lists what must be recorded per call: input/output/billed tokens where the
provider exposes them, inference time and monetary cost, and the external
model/version and prompt hash. §11.2 requires the LLM overhead to be reported
separately from the shared downstream search budget rather than treated as free.

The rule that shapes this module: a provider that does not expose a number gets
`None`, never `0`. Sums report completeness instead of pretending a partial
total is a total, and cost is `None` when either the token counts or the prices
are unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmlm_matter_001.models.adapters import ModelResponse
from mmlm_matter_001.models.contract import ModelContract


@dataclass(frozen=True, slots=True)
class LLMUsage:
    arm: str
    provider: str
    model: str
    contract_hash: str
    prompt_hash: str
    schema_hash: str
    request_hash: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    cached: bool = False
    price_input_usd_per_mtok: float | None = None
    price_output_usd_per_mtok: float | None = None

    @property
    def cost_usd(self) -> float | None:
        if self.cached:
            return 0.0
        if self.input_tokens is None or self.output_tokens is None:
            return None
        if (self.price_input_usd_per_mtok is None
                or self.price_output_usd_per_mtok is None):
            return None
        return (
            self.input_tokens * self.price_input_usd_per_mtok
            + self.output_tokens * self.price_output_usd_per_mtok
        ) / 1_000_000

    @property
    def cost_basis(self) -> str:
        if self.cached:
            return "cached"
        if self.input_tokens is None or self.output_tokens is None:
            return "provider_tokens_unexposed"
        if (self.price_input_usd_per_mtok is None
                or self.price_output_usd_per_mtok is None):
            return "unpriced"
        return "priced"

    def to_row(self, *, sequence: int) -> dict[str, Any]:
        return {
            "sequence": sequence,
            "arm": self.arm,
            "provider": self.provider,
            "model": self.model,
            "contract_hash": self.contract_hash,
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "request_hash": self.request_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "cost_basis": self.cost_basis,
            "cached": self.cached,
        }


def usage_from_response(
    response: ModelResponse,
    *,
    contract: ModelContract,
    arm: str,
) -> LLMUsage:
    """Attach a provider response to the contract that governed it."""
    return LLMUsage(
        arm=arm,
        provider=response.provider,
        model=response.model,
        contract_hash=response.contract_hash,
        prompt_hash=contract.prompt_hash,
        schema_hash=contract.schema_hash,
        request_hash=response.request_hash,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_s=response.latency_s,
        cached=response.cached,
        price_input_usd_per_mtok=contract.price_input_usd_per_mtok,
        price_output_usd_per_mtok=contract.price_output_usd_per_mtok,
    )


class BudgetLedger:
    """Append-only JSONL of LLM usage rows, one per frozen model call."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, usage: LLMUsage) -> dict[str, Any]:
        sequence = len(self.rows()) + 1
        row = usage.to_row(sequence=sequence)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def aggregate(self) -> dict[str, Any]:
        """Totals over every recorded call, with completeness flags."""
        rows = self.rows()
        return _aggregate(rows)


def _sum_or_none(values: list[int | float | None]) -> tuple[float | None, bool]:
    if not values or any(value is None for value in values):
        return None, False
    return float(sum(values)), True


def _aggregate(rows: list[dict[str, Any]], *, with_arms: bool = True) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in rows})
    by_arm = {
        arm: _aggregate([row for row in rows if row["arm"] == arm], with_arms=False)
        for arm in arms
    } if with_arms else {}
    input_total, input_complete = _sum_or_none([row["input_tokens"] for row in rows])
    output_total, output_complete = _sum_or_none([row["output_tokens"] for row in rows])
    cost_total, cost_complete = _sum_or_none([row["cost_usd"] for row in rows])
    return {
        "calls": len(rows),
        "cached_calls": sum(1 for row in rows if row["cached"]),
        "input_tokens": None if input_total is None else int(input_total),
        "input_tokens_complete": input_complete,
        "output_tokens": None if output_total is None else int(output_total),
        "output_tokens_complete": output_complete,
        "cost_usd": cost_total,
        "cost_complete": cost_complete,
        "latency_s": float(sum(row["latency_s"] for row in rows)),
        "cost_bases": sorted({row["cost_basis"] for row in rows}),
        "by_arm": by_arm,
    }
