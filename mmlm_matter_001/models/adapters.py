"""Provider-agnostic model adapters for the Phase 1 arms.

Two adapters cover the experiment:

* :class:`OpenAICompatibleAdapter` — one OpenAI-style chat-completions wire
  against any provider that speaks it: OpenRouter, OpenAI, DeepSeek, NVIDIA
  NIM, Aihubmix, and any local vLLM/Ollama/llama.cpp server through `base_url`.
  Because OpenRouter fronts Anthropic, Google and open-weight models with the
  same schema, choosing the model later is a contract retarget, not a code
  change.
* :class:`ReplayAdapter` — deterministic replay of a recorded cassette. A
  rerun costs nothing, needs no network, and reproduces byte-identical output,
  which is what §12.6's "freeze and hash" is worth without it.

Failure discipline follows the lesson recorded in
`apedp_dyn/grammar/llm_proposer.py`: a refused request, a missing key and a
malformed response must never be reported as an empty proposal. Everything that
is not a model answer raises :class:`ModelUnavailableError`.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mmlm_matter_001.models.contract import (
    ContractError,
    ModelContract,
    canonical_json,
    sha256_text,
)


class ModelUnavailableError(RuntimeError):
    """A model call that was required to happen did not happen."""


class ReplayMissError(ModelUnavailableError):
    """A replay cassette has no record for this request hash."""


@dataclass(frozen=True, slots=True)
class ModelRequest:
    contract: ModelContract
    system_prompt: str
    user_prompt: str

    @property
    def request_hash(self) -> str:
        return sha256_text(canonical_json({
            "contract": self.contract.digest,
            "system": self.system_prompt,
            "user": self.user_prompt,
        }))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    request_hash: str
    contract_hash: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    cached: bool = False
    raw_usage: dict[str, Any] = field(default_factory=dict)


#: provider -> (default endpoint, environment variables that may hold the key).
#: `local` and `custom` have no default endpoint; they need `base_url`.
PROVIDER_ENDPOINTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "openrouter": (
        "https://openrouter.ai/api/v1/chat/completions",
        ("OPENROUTER_API_KEY", "APEDP_LLM_KEY")),
    "openai": (
        "https://api.openai.com/v1/chat/completions",
        ("OPENAI_API_KEY", "APEDP_LLM_KEY")),
    "deepseek": (
        "https://api.deepseek.com/v1/chat/completions",
        ("DEEPSEEK_API_KEY", "APEDP_LLM_KEY")),
    "nvidia": (
        "https://integrate.api.nvidia.com/v1/chat/completions",
        ("NVIDIA_API_KEY", "NIM_API_KEY", "APEDP_LLM_KEY")),
    "aihubmix": (
        "https://aihubmix.com/v1/chat/completions",
        ("AIHUBMIX_API_KEY", "APEDP_LLM_KEY")),
    "local": (None, ("APEDP_LLM_KEY", "LOCAL_LLM_KEY")),
    "custom": (None, ("APEDP_LLM_KEY", "CUSTOM_API_KEY")),
}


class ModelAdapter(ABC):
    """A transport that turns a :class:`ModelRequest` into a response or raises."""

    adapter_id: ClassVar[str]

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the model's text plus whatever usage the provider reported."""


def _as_int(value: Any) -> int | None:
    """Provider-reported integer or `None` for "not exposed".

    Zero is a real value and is kept; a missing field becomes `None`, because
    the budget protocol forbids inventing token counts (§11.1).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class OpenAICompatibleAdapter(ModelAdapter):
    adapter_id = "openai-compatible"

    def __init__(
        self,
        *,
        provider: str,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        provider = provider.lower()
        if provider not in PROVIDER_ENDPOINTS:
            raise ContractError(
                f"unknown provider {provider!r}; known: {sorted(PROVIDER_ENDPOINTS)}")
        endpoint, key_envs = PROVIDER_ENDPOINTS[provider]
        environ = os.environ if env is None else env
        resolved = base_url or endpoint or environ.get("APEDP_LLM_BASE")
        if not resolved:
            raise ContractError(
                f"provider {provider!r} has no default base URL; pass base_url= or set "
                "APEDP_LLM_BASE")
        self.provider = provider
        self.base_url = resolved
        self.key_envs = key_envs
        # Resolved eagerly so a misconfiguration names the variable, but only
        # enforced at call time so contracts can be frozen and hashed offline.
        self._api_key = api_key or next(
            (environ[name] for name in key_envs if environ.get(name)), None)
        self.extra_headers = dict(extra_headers or {})

    def complete(self, request: ModelRequest) -> ModelResponse:
        contract = request.contract
        if contract.provider.lower() != self.provider:
            raise ContractError(
                f"adapter {self.adapter_id!r} is configured for provider "
                f"{self.provider!r} but the contract names {contract.provider!r}")
        if not self._api_key:
            raise ModelUnavailableError(
                f"provider {self.provider!r} has no API key; set one of "
                f"{', '.join(self.key_envs)}")
        try:
            import httpx
        except ImportError as exc:
            raise ModelUnavailableError(
                "the model adapters need httpx; install it with `pip install httpx`") from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers.setdefault("HTTP-Referer", "https://github.com/APEDP")
            headers.setdefault("X-Title", "MMLM-MATTER-001")
        headers.update(self.extra_headers)
        payload = {
            "model": contract.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": contract.temperature,
            "max_tokens": contract.max_output_tokens,
        }

        started = time.monotonic()
        try:
            response = httpx.post(
                self.base_url, headers=headers, json=payload, timeout=contract.timeout_s)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"provider {self.provider!r} model {contract.model!r} call failed: "
                f"{type(exc).__name__}: {exc}") from exc
        latency_s = time.monotonic() - started

        try:
            body = response.json()
        except ValueError as exc:
            raise ModelUnavailableError(
                f"provider {self.provider!r} returned a body that is not JSON: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(
                f"provider {self.provider!r} response has no choices[0].message.content "
                f"({type(exc).__name__})") from exc
        if not isinstance(text, str) or not text.strip():
            raise ModelUnavailableError(
                f"provider {self.provider!r} model {contract.model!r} returned no text content")

        usage = body.get("usage") if isinstance(body, dict) else None
        usage_dict = dict(usage) if isinstance(usage, dict) else {}
        return ModelResponse(
            text=text,
            request_hash=request.request_hash,
            contract_hash=contract.digest,
            provider=self.provider,
            model=contract.model,
            input_tokens=_as_int(usage_dict.get("prompt_tokens")),
            output_tokens=_as_int(usage_dict.get("completion_tokens")),
            latency_s=latency_s,
            raw_usage=usage_dict,
        )


def _require_record_text(record: Mapping[str, Any], line_number: int, path: Path) -> str:
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ContractError(
            f"cassette {str(path)!r} line {line_number} has no non-empty 'text'")
    return text


def _load_cassette(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise ContractError(f"replay cassette {str(path)!r} does not exist")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"cassette {str(path)!r} line {line_number} is not JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ContractError(
                f"cassette {str(path)!r} line {line_number} is not a JSON object")
        request_hash = record.get("request_hash")
        if not isinstance(request_hash, str) or len(request_hash) != 64:
            raise ContractError(
                f"cassette {str(path)!r} line {line_number} has no 64-hex request_hash")
        _require_record_text(record, line_number, path)
        if request_hash in records:
            raise ContractError(
                f"cassette {str(path)!r} records request_hash {request_hash} twice "
                f"(line {line_number}); a replay must be unambiguous")
        records[request_hash] = record
    return records


class ReplayAdapter(ModelAdapter):
    """Deterministic replay of recorded responses, keyed by request hash."""

    adapter_id = "replay"

    def __init__(self, cassette: Path | str) -> None:
        self.cassette = Path(cassette)
        self._records = _load_cassette(self.cassette)

    def complete(self, request: ModelRequest) -> ModelResponse:
        record = self._records.get(request.request_hash)
        if record is None:
            raise ReplayMissError(
                f"cassette {str(self.cassette)!r} has no record for request "
                f"{request.request_hash}; record it with write_cassette() or run live")
        return ModelResponse(
            text=str(record["text"]),
            request_hash=request.request_hash,
            contract_hash=str(record.get("contract_hash") or request.contract.digest),
            provider=str(record.get("provider") or request.contract.provider),
            model=str(record.get("model") or request.contract.model),
            input_tokens=_as_int(record.get("input_tokens")),
            output_tokens=_as_int(record.get("output_tokens")),
            latency_s=float(record.get("latency_s") or 0.0),
            cached=True,
            raw_usage=dict(record.get("raw_usage") or {}),
        )


def cassette_record(response: ModelResponse) -> dict[str, Any]:
    return {
        "request_hash": response.request_hash,
        "contract_hash": response.contract_hash,
        "provider": response.provider,
        "model": response.model,
        "text": response.text,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_s": response.latency_s,
        "raw_usage": response.raw_usage,
    }


def write_cassette(path: Path | str, responses: Iterable[ModelResponse]) -> Path:
    """Append replay records for `responses`, one JSON object per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for response in responses:
            handle.write(canonical_json(cassette_record(response)) + "\n")
    return target


def build_adapter(
    contract: ModelContract,
    *,
    replay_path: Path | str | None = None,
    api_key: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> ModelAdapter:
    """Construct the adapter a contract asks for.

    A replay path always wins: a rerun of a frozen protocol must not silently
    reach the network.
    """
    if replay_path is not None:
        return ReplayAdapter(replay_path)
    return OpenAICompatibleAdapter(
        provider=contract.provider,
        base_url=contract.base_url,
        api_key=api_key,
        extra_headers=extra_headers,
    )
