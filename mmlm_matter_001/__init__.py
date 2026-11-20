"""MMLM-MATTER-001: planar one-DOF mechanism synthesis, Phase 1 arms.

Status: experimental

This package implements the software experiment in
`docs/MMLM-MATTER-001_Implementation_Blueprint.md`. It is not part of, and does
not import, `apedp_mmlm`: that package is a mathematical-language-model /
quantum-circuit-discovery stack, and the name overlap is the project label.

The slice implemented here is the model-interoperability layer the P1/P2 arms
need before a provider is chosen (blueprint §10.2, §10.3, §11.1, §12.6):

  - `models.contract` — the frozen `ModelContract`: provider, model, prompt and
    schema hashes, sampling limits and prices, with drift refusals.
  - `models.prompts` — versioned P1/P2 templates whose text and system prompt
    are hashed together.
  - `models.adapters` — one OpenAI-compatible wire adapter covering OpenRouter,
    OpenAI, DeepSeek, NVIDIA NIM, Aihubmix and any local vLLM/Ollama endpoint,
    plus a replay adapter that reruns a recorded cassette with no network and no
    spend.
  - `models.outputs` — strict JSON schemas that reject malformed, unknown,
    duplicated or non-physical selections with a machine-readable reason.
  - `proposers` — P1 (topology ranking) and P2 (direct design).
  - `evaluation.budget_accountant` — per-call usage rows and aggregation that
    keeps "not reported by the provider" distinct from zero.

Nothing in this package accepts a candidate. Verification is the referee's.
"""

from mmlm_matter_001.evaluation import BudgetLedger, LLMUsage, usage_from_response
from mmlm_matter_001.kinematics import (
    FourBar,
    FunctionCheck,
    GrashofResult,
    MechanismError,
    PathCheck,
    UnreachableError,
    check_function,
    check_path,
    discrete_frechet,
)
from mmlm_matter_001.models import (
    ModelAdapter,
    ModelContract,
    ModelOutputError,
    ModelRequest,
    ModelResponse,
    ModelUnavailableError,
    OpenAICompatibleAdapter,
    PromptDriftError,
    ReplayAdapter,
    ReplayMissError,
    SchemaDriftError,
    build_adapter,
    write_cassette,
)
from mmlm_matter_001.pipeline import (
    DryRunResult,
    StubRankingAdapter,
    build_stub_contract,
    run_dry_run,
    stub_catalogue,
    stub_task,
)
from mmlm_matter_001.proposers import (
    P1Proposal,
    P1TopologyProposer,
    P2DirectDesignProposer,
    P2Proposal,
)
from mmlm_matter_001.protocol import (
    ArmPin,
    FrozenProtocol,
    ProtocolDriftError,
    ProtocolError,
    ProtocolIntegrityError,
    freeze_protocol,
    load_protocol,
)
from mmlm_matter_001.schemas import (
    PublicFeedback,
    PublicTaskView,
    SealedTask,
    TopologyEntry,
)

__all__ = [
    "ArmPin",
    "BudgetLedger",
    "DryRunResult",
    "FourBar",
    "FrozenProtocol",
    "FunctionCheck",
    "GrashofResult",
    "LLMUsage",
    "MechanismError",
    "ModelAdapter",
    "ModelContract",
    "ModelOutputError",
    "ModelRequest",
    "ModelResponse",
    "ModelUnavailableError",
    "OpenAICompatibleAdapter",
    "P1Proposal",
    "P1TopologyProposer",
    "P2DirectDesignProposer",
    "P2Proposal",
    "PathCheck",
    "PromptDriftError",
    "ProtocolDriftError",
    "ProtocolError",
    "ProtocolIntegrityError",
    "PublicFeedback",
    "PublicTaskView",
    "ReplayAdapter",
    "ReplayMissError",
    "SchemaDriftError",
    "SealedTask",
    "StubRankingAdapter",
    "TopologyEntry",
    "UnreachableError",
    "build_adapter",
    "check_path",
    "discrete_frechet",
    "freeze_protocol",
    "load_protocol",
    "run_dry_run",
    "stub_catalogue",
    "stub_task",
    "usage_from_response",
    "write_cassette",
]
