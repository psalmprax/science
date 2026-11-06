"""Frozen model contract shared by the Phase 1 P1/P2 arms.

Blueprint §12.6 requires the P1 model/version, prompt template and output
schema to be frozen and hashed before hidden testing, and §11.1 requires the
external model/version and prompt hash on every budget row. This module is the
object that carries those pins.

The contract can be frozen before a provider is chosen: the prompt and schema
hashes are properties of this repository, not of the model. Choosing a model
later is `contract.retarget(model=..., provider=..., price_...)`, which leaves
the prompt and schema pins untouched. What must never happen is a run whose
recorded hash no longer describes the prompt it actually sent, so every call
site re-checks the pins through :meth:`ModelContract.check_pins`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

CONTRACT_VERSION = "1.0"

_HEX_DIGITS = frozenset("0123456789abcdef")


class ContractError(RuntimeError):
    """A contract that cannot be honoured as written."""


class PromptDriftError(ContractError):
    """The prompt template no longer matches the hash pinned in the contract."""


class SchemaDriftError(ContractError):
    """The output schema no longer matches the hash pinned in the contract."""


def canonical_json(payload: Any) -> str:
    """Serialise `payload` so equal content always hashes equally."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(schema: type[BaseModel]) -> str:
    """Hash a pydantic model's JSON schema.

    This is a drift detector, not a wire-format promise: a pydantic upgrade that
    changes the emitted schema is exactly the change Phase 1 must notice.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise ContractError(f"schema must be a pydantic BaseModel subclass, got {schema!r}")
    return sha256_text(canonical_json(schema.model_json_schema()))


def _require_digest(field: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or set(value.lower()) - _HEX_DIGITS:
        raise ContractError(
            f"{field} must be a 64-character lowercase SHA-256 hex digest, got {value!r}")


@dataclass(frozen=True, slots=True)
class ModelContract:
    """The pinned identity of one proposer arm's model call."""

    provider: str
    model: str
    prompt_id: str
    prompt_hash: str
    schema_id: str
    schema_hash: str
    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout_s: float = 60.0
    base_url: str | None = None
    #: Price per million input tokens in USD. `None` means "not published to
    #: this project", which the accountant reports as unpriced rather than 0.
    price_input_usd_per_mtok: float | None = None
    price_output_usd_per_mtok: float | None = None
    version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("provider", "model", "prompt_id", "schema_id"):
            if not getattr(self, name):
                raise ContractError(f"{name} must be non-empty")
        _require_digest("prompt_hash", self.prompt_hash)
        _require_digest("schema_hash", self.schema_hash)
        if not 0.0 <= self.temperature <= 2.0:
            raise ContractError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_output_tokens <= 0:
            raise ContractError(
                f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if self.timeout_s <= 0:
            raise ContractError(f"timeout_s must be positive, got {self.timeout_s}")
        for name in ("price_input_usd_per_mtok", "price_output_usd_per_mtok"):
            price = getattr(self, name)
            if price is not None and price < 0:
                raise ContractError(f"{name} must be non-negative when set, got {price}")

    @classmethod
    def freeze(
        cls,
        *,
        provider: str,
        model: str,
        prompt: Any,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> ModelContract:
        """Build a contract whose prompt and schema hashes describe this checkout."""
        return cls(
            provider=provider,
            model=model,
            prompt_id=prompt.prompt_id,
            prompt_hash=prompt.prompt_hash,
            schema_id=kwargs.pop("schema_id", schema.__name__),
            schema_hash=schema_hash(schema),
            **kwargs,
        )

    def retarget(self, **changes: Any) -> ModelContract:
        """Return a copy with transport/model fields changed and pins preserved.

        This is how the model is chosen later without touching the prompt or
        schema hashes the protocol was preregistered against.
        """
        forbidden = {"prompt_hash", "schema_hash"} & set(changes)
        if forbidden:
            raise ContractError(
                f"cannot retarget {sorted(forbidden)}; freeze a new contract instead")
        return replace(self, **changes)

    def check_pins(self, prompt: Any, schema: type[BaseModel]) -> None:
        """Refuse to run when the prompt or schema is not the one that was pinned."""
        if prompt.prompt_id != self.prompt_id:
            raise PromptDriftError(
                f"contract pins prompt {self.prompt_id!r} but the call site uses "
                f"{prompt.prompt_id!r}")
        actual_prompt = prompt.prompt_hash
        if actual_prompt != self.prompt_hash:
            raise PromptDriftError(
                f"prompt {self.prompt_id!r} drifted: contract pins {self.prompt_hash}, "
                f"checkout has {actual_prompt}")
        actual_schema = schema_hash(schema)
        if actual_schema != self.schema_hash:
            raise SchemaDriftError(
                f"output schema {self.schema_id!r} drifted: contract pins "
                f"{self.schema_hash}, checkout has {actual_schema}")

    def _hashed_fields(self) -> dict[str, Any]:
        """Everything the digest covers; deliberately excludes the digest itself."""
        return {
            "version": self.version,
            "provider": self.provider,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "schema_id": self.schema_id,
            "schema_hash": self.schema_hash,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_s": self.timeout_s,
            "base_url": self.base_url,
            "price_input_usd_per_mtok": self.price_input_usd_per_mtok,
            "price_output_usd_per_mtok": self.price_output_usd_per_mtok,
        }

    def record(self) -> dict[str, Any]:
        """The contract as it is written into protocol and budget records."""
        payload = self._hashed_fields()
        payload["contract_hash"] = self.digest
        return payload

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(self._hashed_fields()))
