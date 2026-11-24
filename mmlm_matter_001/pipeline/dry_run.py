"""Offline dry run: P1 against a stub catalogue, writing a real budget ledger.

The point is to exercise the Phase 1 plumbing before a model exists: the same
prompt template, the same strict parser, the same budget rows and the same
protocol pins, with a deterministic local stub in place of a provider. Nothing
here is a model result and nothing records a model identity as if it were one —
the stub calls itself ``stub`` in every row, and the contract it runs under is
priced at zero so the ledger's cost arithmetic is exercised rather than faked.

Run it directly::

    python -m mmlm_matter_001.pipeline --ledger build/mmlm_matter_001/budget_ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmlm_matter_001.evaluation.budget_accountant import BudgetLedger
from mmlm_matter_001.kinematics.fourbar import FourBar
from mmlm_matter_001.models.adapters import ModelAdapter, ModelRequest, ModelResponse
from mmlm_matter_001.models.contract import ModelContract, canonical_json, sha256_text
from mmlm_matter_001.models.outputs import P1SelectionOutput
from mmlm_matter_001.models.prompts import p1_prompt
from mmlm_matter_001.proposers.p1_llm_topology import P1Proposal, P1TopologyProposer
from mmlm_matter_001.protocol.freeze import FrozenProtocol
from mmlm_matter_001.schemas import PublicTaskView, TopologyEntry

STUB_PROVIDER = "stub"
STUB_MODEL = "stub/echo-rank-v1"
DEFAULT_LEDGER = Path("build/mmlm_matter_001/budget_ledger.jsonl")
#: Four-bar used to generate the default dry-run target path: a crank-rocker
#: with the shortest link adjacent to ground (§5.2).
STUB_MECHANISM = FourBar(
    ground_mm=40.0, input_mm=12.0, coupler_mm=45.0, output_mm=30.0,
    coupler_point_mm=(20.0, 0.0))


class StubError(RuntimeError):
    """The stub could not read the prompt it was given."""


def _catalogue_ids_from(prompt: str) -> list[str]:
    """Read the catalogue block the prompt actually contains."""
    ids: list[str] = []
    for line in prompt.splitlines():
        marker = line.strip()
        if marker.startswith("- id="):
            value = marker[len("- id="):].split()[0].strip()
            if value:
                ids.append(value)
    if not ids:
        raise StubError("the prompt contains no '- id=' catalogue lines to rank")
    return ids


def _requested_max(prompt: str, available: int) -> int:
    marker = "select at most "
    for line in prompt.splitlines():
        if marker in line:
            try:
                requested = int(line.split(marker, 1)[1].split()[0])
            except (IndexError, ValueError):
                break
            return max(1, min(requested, available))
    return max(1, min(3, available))


class StubRankingAdapter(ModelAdapter):
    """A deterministic local stand-in for a model. Never a stand-in for a result.

    It reads the catalogue out of the prompt it is given and returns a valid
    strict-JSON ranking, so the parser, the ledger and the protocol pins are all
    exercised. Its token counts are its own whitespace counts, labelled as such,
    because a stub has no provider to report usage.
    """

    adapter_id = "stub"

    def __init__(self, order: str = "catalogue") -> None:
        if order not in {"catalogue", "reverse"}:
            raise StubError(f"stub order must be 'catalogue' or 'reverse', got {order!r}")
        self.order = order

    def complete(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        ids = _catalogue_ids_from(request.user_prompt)
        if self.order == "reverse":
            ids = list(reversed(ids))
        limit = _requested_max(request.user_prompt, len(ids))
        chosen = ids[:limit]
        text = canonical_json({"selections": [
            {
                "topology_id": topology_id,
                "rank": rank,
                "rationale": (
                    "stub ranking by catalogue position; this is a plumbing fixture, not a "
                    "judgement about the mechanism"),
            }
            for rank, topology_id in enumerate(chosen, start=1)
        ]})
        latency_s = time.monotonic() - started
        return ModelResponse(
            text=text,
            request_hash=request.request_hash,
            contract_hash=request.contract.digest,
            provider=STUB_PROVIDER,
            model=request.contract.model,
            input_tokens=len((request.system_prompt + " " + request.user_prompt).split()),
            output_tokens=len(text.split()),
            latency_s=latency_s,
            raw_usage={"stub": True, "counter": "whitespace_tokens"},
        )


def build_stub_contract() -> ModelContract:
    """The contract a dry run uses: pinned prompt and schema, zero prices."""
    return ModelContract.freeze(
        provider=STUB_PROVIDER,
        model=STUB_MODEL,
        prompt=p1_prompt(),
        schema=P1SelectionOutput,
        max_output_tokens=512,
        price_input_usd_per_mtok=0.0,
        price_output_usd_per_mtok=0.0,
    )


def stub_catalogue(count: int = 6) -> tuple[TopologyEntry, ...]:
    """A small deterministic catalogue: four-link revolute chains, four inversions."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count!r}")
    inversions = ("crank-rocker", "double-crank", "double-rocker", "change-point")
    entries: list[TopologyEntry] = []
    for index in range(count):
        topology_id = f"T4-{index:03d}"
        inversion = inversions[index % len(inversions)]
        entries.append(TopologyEntry(
            topology_id=topology_id,
            link_count=4,
            joint_count=4,
            inversion=inversion,
            ground_link="ground",
            input_link="input",
            output_link="output",
            parent_chain_id="4R-0001",
            graph_hash=sha256_text(f"stub-catalogue:{topology_id}:{inversion}"),
        ))
    return tuple(entries)


def stub_task(
    task_id: str = "T1-stub-0001",
    *,
    samples: int = 24,
    mechanism: FourBar = STUB_MECHANISM,
) -> PublicTaskView:
    """A public T1 task whose target is a real four-bar coupler path."""
    trace = mechanism.trace((0.0, 360.0), samples=samples, branch="open")
    points = trace.points_mm
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    extent_x = max(xs) - min(xs)
    extent_y = max(ys) - min(ys)
    tolerance = round(max(0.5, math.hypot(extent_x, extent_y) / 100.0), 3)
    return PublicTaskView(
        task_id=task_id,
        family="T1",
        target_kind="ordered planar coupler path",
        tolerance=tolerance,
        input_range_deg=(0.0, 360.0),
        envelope_mm=(round(extent_x + 2.0, 3), round(extent_y + 2.0, 3), 20.0),
        target_points=points,
    )


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """What a dry run did, in the terms the phase report uses."""

    ledger_path: Path
    tasks: int
    calls: int
    catalogue_ids: tuple[str, ...]
    proposals: tuple[P1Proposal, ...]
    aggregate: dict[str, Any]
    protocol_hash: str | None

    def record(self) -> dict[str, Any]:
        return {
            "ledger_path": str(self.ledger_path),
            "tasks": self.tasks,
            "calls": self.calls,
            "catalogue_ids": list(self.catalogue_ids),
            "protocol_hash": self.protocol_hash,
            "topologies_by_task": {
                proposal.task_id: list(proposal.topology_ids) for proposal in self.proposals
            },
            "aggregate": self.aggregate,
        }


def run_dry_run(
    *,
    tasks: Sequence[PublicTaskView] | None = None,
    catalogue: Sequence[TopologyEntry] | None = None,
    ledger_path: Path | str = DEFAULT_LEDGER,
    protocol: FrozenProtocol | None = None,
    adapter: ModelAdapter | None = None,
    max_selections: int = 3,
    remaining_credits: int | None = None,
    contract: ModelContract | None = None,
) -> DryRunResult:
    """Run P1 over `tasks` with the stub and append every call to the ledger."""
    entries = tuple(stub_catalogue() if catalogue is None else catalogue)
    if not entries:
        raise ValueError("a dry run needs a non-empty catalogue")
    task_list = tuple(tasks) if tasks is not None else (stub_task(),)
    if not task_list:
        raise ValueError("a dry run needs at least one task")
    if max_selections < 1:
        raise ValueError(f"max_selections must be positive, got {max_selections!r}")
    selected_contract = contract or build_stub_contract()
    if protocol is not None:
        # Drift is checked before the first call, so a refused protocol cannot
        # leave a partially written ledger behind.
        protocol.check_contract("P1", selected_contract)
    proposer = P1TopologyProposer(
        contract=selected_contract,
        adapter=adapter if adapter is not None else StubRankingAdapter(),
    )
    ledger = BudgetLedger(ledger_path)
    proposals: list[P1Proposal] = []
    for task in task_list:
        proposal = proposer.rank(
            task,
            catalogue=entries,
            max_selections=max_selections,
            remaining_credits=remaining_credits,
        )
        ledger.append(proposal.usage)
        proposals.append(proposal)
    return DryRunResult(
        ledger_path=Path(ledger_path),
        tasks=len(task_list),
        calls=len(proposals),
        catalogue_ids=tuple(entry.topology_id for entry in entries),
        proposals=tuple(proposals),
        aggregate=ledger.aggregate(),
        protocol_hash=None if protocol is None else protocol.protocol_hash,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mmlm_matter_001.pipeline.dry_run",
        description="Exercise the Phase 1 P1 pipeline against a local stub (no network).")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                        help=f"JSONL budget ledger to append to (default: {DEFAULT_LEDGER})")
    parser.add_argument("--tasks", type=int, default=3, help="how many stub tasks to run")
    parser.add_argument("--catalogue", type=int, default=6, help="stub catalogue size")
    parser.add_argument("--max-selections", type=int, default=3,
                        help="how many topologies P1 may select per task")
    parser.add_argument("--protocol", type=Path, default=None,
                        help="frozen protocol bundle to check the contract against")
    args = parser.parse_args(argv)
    if args.tasks < 1:
        parser.error("--tasks must be at least 1")
    protocol = FrozenProtocol.load(args.protocol) if args.protocol else None
    result = run_dry_run(
        tasks=tuple(stub_task(task_id=f"T1-stub-{index:04d}")
                    for index in range(1, args.tasks + 1)),
        catalogue=stub_catalogue(args.catalogue),
        ledger_path=args.ledger,
        protocol=protocol,
        max_selections=args.max_selections,
    )
    print(json.dumps(result.record(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
