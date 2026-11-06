"""Offline pipeline fixtures: exercise the Phase 1 plumbing without a provider."""

from mmlm_matter_001.pipeline.dry_run import (
    DEFAULT_LEDGER,
    STUB_MECHANISM,
    STUB_MODEL,
    STUB_PROVIDER,
    DryRunResult,
    StubRankingAdapter,
    build_stub_contract,
    run_dry_run,
    stub_catalogue,
    stub_task,
)
from mmlm_matter_001.pipeline.gate import (
    CAPABILITY_REASONS,
    GATE_FAMILIES,
    GATE_REASONS,
    GateError,
    GateVerdict,
    run_gate,
)

__all__ = [
    "CAPABILITY_REASONS",
    "DEFAULT_LEDGER",
    "GATE_FAMILIES",
    "GATE_REASONS",
    "STUB_MECHANISM",
    "STUB_MODEL",
    "STUB_PROVIDER",
    "DryRunResult",
    "GateError",
    "GateVerdict",
    "StubRankingAdapter",
    "build_stub_contract",
    "run_dry_run",
    "run_gate",
    "stub_catalogue",
    "stub_task",
]
