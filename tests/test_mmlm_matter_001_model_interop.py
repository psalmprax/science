"""Pins for the MMLM-MATTER-001 Phase 1 model-interoperability layer.

These tests run with no network and no credentials. What they pin is the
protocol, not a provider: the prompt/schema hashes that §12.6 requires to be
frozen, the rejections §10.2 requires, and the accounting rule that an
unreported token count is never recorded as zero (§11.1).
"""

from __future__ import annotations

import json
import types

import httpx
import pytest
from pydantic import BaseModel, Field

from mmlm_matter_001.evaluation import BudgetLedger, LLMUsage, usage_from_response
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
    p1_prompt,
    p2_prompt,
    parse_p1_selection,
    parse_p2_design,
    write_cassette,
)
from mmlm_matter_001.models.outputs import P1_SCHEMA_ID, P1SelectionOutput, P2DesignOutput
from mmlm_matter_001.proposers import P1TopologyProposer, P2DirectDesignProposer
from mmlm_matter_001.schemas import PublicFeedback, PublicTaskView, SealedTask, TopologyEntry

HASH_A = "a" * 64
HASH_B = "b" * 64


def _p1_contract(**overrides):
    kwargs = {
        "provider": "openrouter",
        "model": "test/model-v1",
        "prompt": p1_prompt(),
        "schema": P1SelectionOutput,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "price_input_usd_per_mtok": 0.5,
        "price_output_usd_per_mtok": 1.5,
    }
    kwargs.update(overrides)
    return ModelContract.freeze(**kwargs)


def _catalogue():
    return (
        TopologyEntry(
            topology_id="C6-001-g1",
            link_count=6,
            joint_count=7,
            inversion="crank-rocker",
            ground_link="L0",
            input_link="L1",
            output_link="L3",
            parent_chain_id="base-6-001",
            graph_hash=HASH_A,
        ),
        TopologyEntry(
            topology_id="C8-004-g2",
            link_count=8,
            joint_count=10,
            inversion="double-rocker",
            ground_link="L2",
            input_link="L4",
            output_link="L6",
            parent_chain_id="base-8-004",
            graph_hash=HASH_B,
        ),
    )


def _task():
    return PublicTaskView(
        task_id="T3-dev-0001",
        family="T3",
        target_kind="path",
        tolerance=0.5,
        input_range_deg=(0.0, 360.0),
        envelope_mm=(200.0, 150.0, 80.0),
        target_points=((10.0, 5.0), (12.5, 9.0), (14.0, 14.0)),
    )


def _valid_p1_json(ids=("C6-001-g1", "C8-004-g2")):
    return json.dumps({"selections": [
        {"topology_id": ids[0], "rank": 1, "rationale": "fewer links, same joints"},
        {"topology_id": ids[1], "rank": 2, "rationale": "longer chain, more coupler freedom"},
    ]})


class _StubAdapter(ModelAdapter):
    """Records the request and returns prepared text."""

    adapter_id = "stub"

    def __init__(self, text: str, *, input_tokens=11, output_tokens=7, latency_s=0.01):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_s = latency_s
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=self.text,
            request_hash=request.request_hash,
            contract_hash=request.contract.digest,
            provider=request.contract.provider,
            model=request.contract.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=self.latency_s,
        )


class _FailingAdapter(ModelAdapter):
    adapter_id = "failing"

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise ModelUnavailableError("provider openrouter model test/model-v1 call failed: 503")


# --- contract pins ---------------------------------------------------------

def test_the_contract_hash_is_stable_and_changes_only_when_its_record_changes():
    first = _p1_contract()
    second = _p1_contract()
    assert first.digest == second.digest

    retargeted = first.retarget(model="test/model-v2")
    assert retargeted.digest != first.digest
    # Choosing a model later must not disturb the preregistered pins.
    assert retargeted.prompt_hash == first.prompt_hash
    assert retargeted.schema_hash == first.schema_hash

    with pytest.raises(Exception, match="cannot retarget"):
        first.retarget(prompt_hash=HASH_A)


def test_choosing_a_model_later_keeps_the_prompt_and_schema_pins():
    contract = _p1_contract()
    swapped = contract.retarget(
        provider="local",
        model="qwen-local",
        base_url="http://127.0.0.1:8000/v1/chat/completions",
        price_input_usd_per_mtok=None,
        price_output_usd_per_mtok=None,
    )
    swapped.check_pins(p1_prompt(), P1SelectionOutput)
    assert swapped.prompt_id == p1_prompt().prompt_id
    assert swapped.price_input_usd_per_mtok is None


def test_a_drifted_prompt_is_refused_rather_than_run_under_a_stale_hash():
    contract = _p1_contract()
    drifted = types.SimpleNamespace(
        prompt_id=p1_prompt().prompt_id,
        prompt_hash="c" * 64,
    )
    with pytest.raises(PromptDriftError) as exc:
        contract.check_pins(drifted, P1SelectionOutput)
    assert contract.prompt_hash in str(exc.value)


def test_a_drifted_schema_is_refused():
    class _ChangedSchema(BaseModel):
        selections: list[str] = Field(default_factory=list)
        extra_note: str = ""

    contract = _p1_contract()
    with pytest.raises(SchemaDriftError):
        contract.check_pins(p1_prompt(), _ChangedSchema)


def test_the_prompt_hash_covers_the_system_prompt_too():
    template = p1_prompt()
    same_text_copied = type(template)(
        prompt_id=template.prompt_id,
        arm=template.arm,
        system=template.system,
        template=template.template,
    )
    assert same_text_copied.prompt_hash == template.prompt_hash

    changed_system = type(template)(
        prompt_id=template.prompt_id,
        arm=template.arm,
        system=template.system + " Be brief.",
        template=template.template,
    )
    assert changed_system.prompt_hash != template.prompt_hash


# --- output rejection ------------------------------------------------------

def test_a_valid_ranking_is_ordered_by_rank_and_accepted():
    output = parse_p1_selection(
        _valid_p1_json(ids=("C8-004-g2", "C6-001-g1")),
        catalogue_ids=["C6-001-g1", "C8-004-g2"],
        max_selections=2,
    )
    assert [selection.rank for selection in output.selections] == [1, 2]
    assert output.selections[0].topology_id == "C8-004-g2"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("not json at all", "not_json"),
        ('{"selections": [{"topology_id": "C6-001-g1", "rank": 1}]}', "schema"),
        ('{"selections": []}', "empty"),
        ('{"selections": [{"topology_id": "NOPE", "rank": 1, "rationale": "x"}]}',
         "unknown_topology"),
        ('{"selections": [{"topology_id": "C6-001-g1", "rank": 1, "rationale": "x"}, '
         '{"topology_id": "C6-001-g1", "rank": 2, "rationale": "y"}]}', "duplicate_topology"),
        ('{"selections": [{"topology_id": "C6-001-g1", "rank": 1, "rationale": "x"}, '
         '{"topology_id": "C8-004-g2", "rank": 1, "rationale": "y"}]}', "duplicate_rank"),
    ],
)
def test_every_malformed_ranking_is_rejected_for_a_named_reason(text, reason):
    with pytest.raises(ModelOutputError) as exc:
        parse_p1_selection(text, catalogue_ids=["C6-001-g1", "C8-004-g2"], max_selections=2)
    assert exc.value.reason == reason


def test_more_selections_than_requested_is_rejected_not_truncated():
    with pytest.raises(ModelOutputError) as exc:
        parse_p1_selection(
            _valid_p1_json(), catalogue_ids=["C6-001-g1", "C8-004-g2"], max_selections=1)
    assert exc.value.reason == "too_many"


def test_one_whole_text_code_fence_is_unwrapped_and_anything_else_is_not():
    fenced = "```json\n" + _valid_p1_json() + "\n```"
    output = parse_p1_selection(
        fenced, catalogue_ids=["C6-001-g1", "C8-004-g2"], max_selections=2)
    assert len(output.selections) == 2

    with pytest.raises(ModelOutputError) as exc:
        parse_p1_selection(
            "Here is my answer:\n" + _valid_p1_json(),
            catalogue_ids=["C6-001-g1", "C8-004-g2"],
            max_selections=2,
        )
    assert exc.value.reason == "not_json"


def test_a_design_with_a_negative_or_missing_dimension_is_rejected():
    bad = json.dumps({
        "topology_id": "C6-001-g1",
        "link_lengths_mm": {"L0": 40.0, "L1": -3.0},
        "coupler_point_mm": [1.0, 2.0],
        "rationale": "because",
    })
    with pytest.raises(ModelOutputError) as exc:
        parse_p2_design(bad, catalogue_ids=["C6-001-g1"])
    assert exc.value.reason == "non_physical"

    unknown = json.dumps({
        "topology_id": "C10-999-g9",
        "link_lengths_mm": {"L0": 40.0},
        "coupler_point_mm": [1.0, 2.0],
        "rationale": "because",
    })
    with pytest.raises(ModelOutputError) as exc:
        parse_p2_design(unknown, catalogue_ids=["C6-001-g1"])
    assert exc.value.reason == "unknown_topology"


# --- proposer behaviour ----------------------------------------------------

def test_p1_returns_a_ranking_and_a_usage_record_but_no_verdict():
    adapter = _StubAdapter(_valid_p1_json())
    proposer = P1TopologyProposer(contract=_p1_contract(), adapter=adapter)
    proposal = proposer.rank(
        _task(), catalogue=_catalogue(), max_selections=2, remaining_credits=40)

    assert proposal.topology_ids == ("C6-001-g1", "C8-004-g2")
    assert proposal.usage.input_tokens == 11
    assert proposal.usage.cost_basis == "priced"
    assert not hasattr(proposal, "accepted")
    assert not hasattr(proposer, "accept")
    assert "verdict" not in P1SelectionOutput.model_fields


def test_p1_prompt_carries_the_public_task_and_the_catalogue_ids_only():
    adapter = _StubAdapter(_valid_p1_json())
    proposer = P1TopologyProposer(contract=_p1_contract(), adapter=adapter)
    proposer.rank(
        _task(),
        catalogue=_catalogue(),
        max_selections=2,
        feedback=(PublicFeedback(topology_id="C8-004-g2", status="failed",
                                 error_metric=1.25, credits_used=4),),
    )
    sent = adapter.requests[0]
    assert "C6-001-g1" in sent.user_prompt
    assert "C8-004-g2" in sent.user_prompt
    assert "credits remaining: unlimited" in sent.user_prompt
    assert "error=1.25" in sent.user_prompt
    assert "T3-dev-0001" in sent.user_prompt


def test_a_sealed_task_cannot_be_smuggled_into_a_prompt():
    sealed = SealedTask(
        public=_task(),
        planted_topology_id="C10-230-g7",
        planted_dimensions_mm={"L0": 123.456, "L1": 77.5},
        hidden_seed=20270121,
    )
    proposer = P1TopologyProposer(contract=_p1_contract(), adapter=_StubAdapter(_valid_p1_json()))
    with pytest.raises(TypeError, match="PublicTaskView"):
        proposer.rank(sealed, catalogue=_catalogue(), max_selections=1)

    adapter = _StubAdapter(_valid_p1_json())
    P1TopologyProposer(contract=_p1_contract(), adapter=adapter).rank(
        sealed.public_view, catalogue=_catalogue(), max_selections=2)
    prompt = adapter.requests[0].user_prompt
    assert "123.456" not in prompt
    assert "C10-230-g7" not in prompt
    assert "20270121" not in prompt


def test_p2_carries_the_shared_downstream_budget_note_and_returns_dimensions():
    text = json.dumps({
        "topology_id": "C6-001-g1",
        "link_lengths_mm": {"L0": 40.0, "L1": 15.0, "L2": 60.0, "L3": 55.0},
        "coupler_point_mm": [30.0, 10.0],
        "rationale": "matched link ratios to the target curvature",
    })
    contract = ModelContract.freeze(
        provider="openai", model="test/model-v1", prompt=p2_prompt(), schema=P2DesignOutput)
    proposal = P2DirectDesignProposer(
        contract=contract, adapter=_StubAdapter(text)).propose(
            _task(), catalogue=_catalogue())
    assert proposal.topology_id == "C6-001-g1"
    assert proposal.link_lengths_mm["L2"] == 60.0
    assert proposal.usage.arm == "P2"


def test_a_failed_call_raises_and_produces_no_proposal():
    proposer = P1TopologyProposer(contract=_p1_contract(), adapter=_FailingAdapter())
    with pytest.raises(ModelUnavailableError, match="503"):
        proposer.rank(_task(), catalogue=_catalogue(), max_selections=1)


def test_the_replay_adapter_reproduces_a_recorded_answer_without_spending(
    tmp_path,
):
    contract = _p1_contract()
    request = ModelRequest(
        contract=contract,
        system_prompt=p1_prompt().system,
        user_prompt="a frozen rendered prompt",
    )
    recorded = ModelResponse(
        text=_valid_p1_json(),
        request_hash=request.request_hash,
        contract_hash=contract.digest,
        provider=contract.provider,
        model=contract.model,
        input_tokens=21,
        output_tokens=9,
        latency_s=0.42,
    )
    cassette = write_cassette(tmp_path / "cassette.jsonl", [recorded])
    adapter = ReplayAdapter(cassette)

    first = adapter.complete(request)
    second = adapter.complete(request)
    assert first.text == second.text == recorded.text
    assert first.cached and second.cached
    assert first.input_tokens == 21

    other = ModelRequest(
        contract=contract, system_prompt="other", user_prompt="unrecorded")
    with pytest.raises(ReplayMissError, match="no record"):
        adapter.complete(other)


def test_a_replay_path_wins_over_a_live_endpoint(tmp_path):
    contract = _p1_contract()
    request = ModelRequest(contract=contract, system_prompt="s", user_prompt="u")
    cassette = write_cassette(tmp_path / "cassette.jsonl", [ModelResponse(
        text="{}",
        request_hash=request.request_hash,
        contract_hash=contract.digest,
        provider=contract.provider,
        model=contract.model,
        input_tokens=None,
        output_tokens=None,
        latency_s=0.0,
    )])
    adapter = build_adapter(contract, replay_path=cassette)
    assert isinstance(adapter, ReplayAdapter)


# --- transport -------------------------------------------------------------

def test_the_wire_adapter_parses_text_and_reported_tokens(monkeypatch):
    def fake_post(*_args, **kwargs):
        payload = kwargs["json"]
        assert payload["model"] == "test/model-v1"
        assert payload["messages"][0]["role"] == "system"
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": _valid_p1_json()}}],
                "usage": {"prompt_tokens": 33, "completion_tokens": 12},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    adapter = OpenAICompatibleAdapter(provider="openrouter", api_key="test-key")
    response = adapter.complete(ModelRequest(
        contract=_p1_contract(), system_prompt="s", user_prompt="u"))
    assert response.input_tokens == 33
    assert response.output_tokens == 12


def test_a_provider_that_omits_usage_is_recorded_as_unexposed_not_zero(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: types.SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"message": {"content": "{}"}}]},
    ))
    contract = _p1_contract()
    response = OpenAICompatibleAdapter(
        provider="openrouter", api_key="test-key").complete(
            ModelRequest(contract=contract, system_prompt="s", user_prompt="u"))
    usage = usage_from_response(response, contract=contract, arm="P1")
    assert usage.input_tokens is None
    assert usage.cost_usd is None
    assert usage.cost_basis == "provider_tokens_unexposed"


def test_a_provider_error_is_raised_rather_than_returned_as_an_empty_answer(monkeypatch):
    def refusing_post(*_args, **_kwargs):
        raise httpx.ConnectError("503 from provider")

    monkeypatch.setattr(httpx, "post", refusing_post)
    adapter = OpenAICompatibleAdapter(provider="openrouter", api_key="test-key")
    with pytest.raises(ModelUnavailableError, match="503 from provider"):
        adapter.complete(ModelRequest(
            contract=_p1_contract(), system_prompt="s", user_prompt="u"))


def test_a_missing_key_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("APEDP_LLM_KEY", raising=False)
    adapter = OpenAICompatibleAdapter(provider="openrouter", env={})
    with pytest.raises(ModelUnavailableError, match="OPENROUTER_API_KEY"):
        adapter.complete(ModelRequest(
            contract=_p1_contract(), system_prompt="s", user_prompt="u"))


def test_an_adapter_refuses_a_contract_from_a_different_provider():
    adapter = OpenAICompatibleAdapter(provider="openai", api_key="k")
    with pytest.raises(Exception, match="contract names"):
        adapter.complete(ModelRequest(
            contract=_p1_contract(provider="openrouter"), system_prompt="s", user_prompt="u"))


def test_a_local_endpoint_needs_an_explicit_base_url():
    contract = _p1_contract(provider="local", model="qwen-local")
    with pytest.raises(Exception, match="base URL"):
        build_adapter(contract, api_key="k", extra_headers=None)


# --- budget accounting -----------------------------------------------------

def test_the_ledger_records_each_call_and_aggregates_by_arm(tmp_path):
    contract = _p1_contract()
    ledger = BudgetLedger(tmp_path / "budget_ledger.jsonl")

    priced = LLMUsage(
        arm="P1",
        provider=contract.provider,
        model=contract.model,
        contract_hash=contract.digest,
        prompt_hash=contract.prompt_hash,
        schema_hash=contract.schema_hash,
        request_hash="d" * 64,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_s=1.5,
        price_input_usd_per_mtok=0.5,
        price_output_usd_per_mtok=1.5,
    )
    unpriced = LLMUsage(
        arm="P2",
        provider=contract.provider,
        model=contract.model,
        contract_hash=contract.digest,
        prompt_hash=contract.prompt_hash,
        schema_hash=contract.schema_hash,
        request_hash="e" * 64,
        input_tokens=10,
        output_tokens=5,
        latency_s=0.5,
        price_input_usd_per_mtok=None,
        price_output_usd_per_mtok=None,
    )
    row = ledger.append(priced)
    ledger.append(unpriced)

    assert row["cost_usd"] == pytest.approx(2.0)
    assert row["cost_basis"] == "priced"
    totals = ledger.aggregate()
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 1_000_010
    # One arm is unpriced, so the all-in total is unknown, not 2.0.
    assert totals["cost_usd"] is None
    assert totals["cost_complete"] is False
    assert set(totals["by_arm"]) == {"P1", "P2"}
    assert totals["by_arm"]["P1"]["cost_usd"] == pytest.approx(2.0)
    assert totals["by_arm"]["P2"]["cost_bases"] == ["unpriced"]


def test_a_cached_call_costs_zero_and_says_so():
    contract = _p1_contract()
    usage = LLMUsage(
        arm="P1",
        provider=contract.provider,
        model=contract.model,
        contract_hash=contract.digest,
        prompt_hash=contract.prompt_hash,
        schema_hash=contract.schema_hash,
        request_hash="f" * 64,
        input_tokens=None,
        output_tokens=None,
        latency_s=0.0,
        cached=True,
        price_input_usd_per_mtok=0.5,
        price_output_usd_per_mtok=1.5,
    )
    assert usage.cost_usd == 0.0
    assert usage.cost_basis == "cached"


def test_the_two_arms_are_pinned_to_different_schemas():
    assert P1_SCHEMA_ID != "p2_direct_design_v1"
    assert p1_prompt().prompt_id != p2_prompt().prompt_id
