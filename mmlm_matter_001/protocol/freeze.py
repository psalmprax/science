"""Frozen Phase 1 protocol bundle (blueprint §12.6).

§12.6 requires the P1 model/version, prompt template and output schema — and the
P2 configuration — to be frozen and hashed before hidden tasks are opened.
:class:`ModelContract` pins one arm's call; this module pins *the protocol*: the
set of arms, their prompt and schema hashes, and (once a model is chosen) the
contract hashes too, in one file whose own hash is stored inside it. Editing the
file by hand is therefore detectable, and a checkout whose prompt or schema has
drifted refuses to run rather than reporting numbers under a protocol it is no
longer implementing.

Two modes, because the model is chosen later:

* ``pin_models=True`` (default) — the contract hash is pinned. Any retarget after
  freezing is drift: re-freeze deliberately instead of sliding the model under a
  preregistered protocol.
* ``pin_models=False`` — only the prompt/schema hashes and the sampling limits
  are pinned, so the arms can be exercised (for example by the offline dry run)
  before a provider is selected. ``seal_model`` is then the deliberate step that
  adds the model pins.

The bundle is written atomically and never silently overwritten; ``freeze_protocol``
refuses unless ``overwrite=True`` is passed explicitly.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mmlm_matter_001.models.contract import (
    ContractError,
    ModelContract,
    canonical_json,
    schema_hash,
    sha256_text,
)
from mmlm_matter_001.models.outputs import (
    P1_SCHEMA_ID,
    P2_SCHEMA_ID,
    P1SelectionOutput,
    P2DesignOutput,
)
from mmlm_matter_001.models.prompts import get_prompt
from mmlm_matter_001.protocol.preregistration import (
    Phase0Preregistration,
    PreregistrationError,
)

PROTOCOL_VERSION = "1.0"
#: The two arms §12.6 names separately as P1 and P2 configuration.
ARMS = ("P1", "P2")
DEFAULT_BUNDLE_ID = "mmlm-matter-001/phase1"


class ProtocolError(RuntimeError):
    """A protocol bundle that cannot be read, written or honoured."""


class ProtocolDriftError(ProtocolError):
    """The checkout no longer implements the frozen protocol."""


class ProtocolIntegrityError(ProtocolError):
    """The bundle file itself is malformed or has been edited."""


#: schema id -> output model. Both the canonical id (`P1_SCHEMA_ID`) and the
#: class name are accepted, because `ModelContract.freeze` defaults the id to the
#: class name when the caller does not pass one.
SCHEMAS_BY_ID: dict[str, type[BaseModel]] = {
    P1_SCHEMA_ID: P1SelectionOutput,
    P2_SCHEMA_ID: P2DesignOutput,
    P1SelectionOutput.__name__: P1SelectionOutput,
    P2DesignOutput.__name__: P2DesignOutput,
}


def schema_for(schema_id: str) -> type[BaseModel]:
    try:
        return SCHEMAS_BY_ID[schema_id]
    except KeyError as exc:
        raise ProtocolError(
            f"unknown schema id {schema_id!r}; known: {sorted(SCHEMAS_BY_ID)}") from exc


def _require_digest(field: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolIntegrityError(
            f"{field} must be a 64-character SHA-256 hex digest, got {value!r}")
    if set(value.lower()) - frozenset("0123456789abcdef"):
        raise ProtocolIntegrityError(f"{field} is not hexadecimal: {value!r}")
    return value.lower()


@dataclass(frozen=True, slots=True)
class ArmPin:
    """One arm's frozen identity."""

    arm: str
    prompt_id: str
    prompt_hash: str
    schema_id: str
    schema_hash: str
    provider: str | None = None
    model: str | None = None
    contract_hash: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    base_url: str | None = None
    price_input_usd_per_mtok: float | None = None
    price_output_usd_per_mtok: float | None = None

    @classmethod
    def from_contract(cls, arm: str, contract: ModelContract) -> ArmPin:
        if not isinstance(contract, ModelContract):
            raise ContractError(
                f"arm {arm!r} must be pinned to a ModelContract, got {type(contract).__name__}")
        return cls(
            arm=arm,
            prompt_id=contract.prompt_id,
            prompt_hash=contract.prompt_hash,
            schema_id=contract.schema_id,
            schema_hash=contract.schema_hash,
            provider=contract.provider,
            model=contract.model,
            contract_hash=contract.digest,
            temperature=contract.temperature,
            max_output_tokens=contract.max_output_tokens,
            timeout_s=contract.timeout_s,
            base_url=contract.base_url,
            price_input_usd_per_mtok=contract.price_input_usd_per_mtok,
            price_output_usd_per_mtok=contract.price_output_usd_per_mtok,
        )

    def __post_init__(self) -> None:
        if not self.arm:
            raise ProtocolError("arm must be non-empty")
        if not self.prompt_id:
            raise ProtocolError(f"arm {self.arm!r}: prompt_id must be non-empty")
        _require_digest("prompt_hash", self.prompt_hash)
        if not self.schema_id:
            raise ProtocolError(f"arm {self.arm!r}: schema_id must be non-empty")
        _require_digest("schema_hash", self.schema_hash)
        if self.contract_hash is not None:
            _require_digest("contract_hash", self.contract_hash)

    @property
    def pins_model(self) -> bool:
        return self.contract_hash is not None

    def record(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "prompt_id": self.prompt_id,
            "prompt_hash": self.prompt_hash,
            "schema_id": self.schema_id,
            "schema_hash": self.schema_hash,
            "provider": self.provider,
            "model": self.model,
            "contract_hash": self.contract_hash,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_s": self.timeout_s,
            "base_url": self.base_url,
            "price_input_usd_per_mtok": self.price_input_usd_per_mtok,
            "price_output_usd_per_mtok": self.price_output_usd_per_mtok,
        }

    def to_contract(self) -> ModelContract:
        """Rebuild the pinned contract, for a run that starts from the bundle."""
        if not (self.provider and self.model):
            raise ProtocolError(
                f"arm {self.arm!r} does not pin a model; choose one and re-freeze, or run "
                "with an explicit contract")
        return ModelContract(
            provider=self.provider,
            model=self.model,
            prompt_id=self.prompt_id,
            prompt_hash=self.prompt_hash,
            schema_id=self.schema_id,
            schema_hash=self.schema_hash,
            temperature=0.0 if self.temperature is None else self.temperature,
            max_output_tokens=1024 if self.max_output_tokens is None else self.max_output_tokens,
            timeout_s=60.0 if self.timeout_s is None else self.timeout_s,
            base_url=self.base_url,
            price_input_usd_per_mtok=self.price_input_usd_per_mtok,
            price_output_usd_per_mtok=self.price_output_usd_per_mtok,
        )


def _bundled_payload(
    *,
    created_utc: str,
    bundle_id: str,
    pin_models: bool,
    arms: Mapping[str, ArmPin],
    notes: str,
    preregistration: Phase0Preregistration | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "bundle_id": bundle_id,
        "created_utc": created_utc,
        "pin_models": pin_models,
        "notes": notes,
        "arms": {name: pin.record() for name, pin in sorted(arms.items())},
        "preregistration": (
            None if preregistration is None else preregistration.record()),
    }


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(payload)))


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    """A protocol bundle loaded from disk and verified against itself."""

    path: Path
    bundle_id: str
    created_utc: str
    pin_models: bool
    notes: str
    arms: Mapping[str, ArmPin]
    protocol_hash: str
    preregistration: Phase0Preregistration | None = None

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: Path | str) -> FrozenProtocol:
        target = Path(path)
        if not target.exists():
            raise ProtocolError(f"protocol bundle {str(target)!r} does not exist")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProtocolIntegrityError(
                f"protocol bundle {str(target)!r} is not JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ProtocolIntegrityError(f"protocol bundle {str(target)!r} is not an object")
        version = payload.get("protocol_version")
        if version != PROTOCOL_VERSION:
            raise ProtocolIntegrityError(
                f"protocol bundle {str(target)!r} declares version {version!r}; this checkout "
                f"reads version {PROTOCOL_VERSION!r}")
        raw_arms = payload.get("arms")
        if not isinstance(raw_arms, dict) or not raw_arms:
            raise ProtocolIntegrityError(
                f"protocol bundle {str(target)!r} has no arms")
        arms: dict[str, ArmPin] = {}
        for name, record in raw_arms.items():
            if not isinstance(record, dict):
                raise ProtocolIntegrityError(
                    f"protocol bundle {str(target)!r} arm {name!r} is not an object")
            if record.get("arm") != name:
                raise ProtocolIntegrityError(
                    f"protocol bundle {str(target)!r} arm {name!r} records "
                    f"arm={record.get('arm')!r}")
            try:
                arms[name] = ArmPin(**record)
            except TypeError as exc:
                raise ProtocolIntegrityError(
                    f"protocol bundle {str(target)!r} arm {name!r} has unexpected fields: "
                    f"{exc}") from exc
        stored = payload.pop("protocol_hash", None)
        expected = _payload_hash(payload)
        if stored != expected:
            raise ProtocolIntegrityError(
                f"protocol bundle {str(target)!r} has been edited: stored hash {stored!r} does "
                f"not match the recomputed {expected!r}")
        if not isinstance(payload.get("pin_models"), bool):
            raise ProtocolIntegrityError(
                f"protocol bundle {str(target)!r} has no boolean pin_models")
        raw_preregistration = payload.get("preregistration")
        preregistration: Phase0Preregistration | None = None
        if raw_preregistration is not None:
            if not isinstance(raw_preregistration, dict):
                raise ProtocolIntegrityError(
                    f"protocol bundle {str(target)!r}: preregistration is not an object")
            try:
                preregistration = Phase0Preregistration.from_record(raw_preregistration)
            except PreregistrationError as exc:
                raise ProtocolIntegrityError(
                    f"protocol bundle {str(target)!r}: preregistration is invalid: "
                    f"{exc}") from exc
        return cls(
            path=target,
            bundle_id=str(payload.get("bundle_id") or ""),
            created_utc=str(payload.get("created_utc") or ""),
            pin_models=bool(payload["pin_models"]),
            notes=str(payload.get("notes") or ""),
            arms=arms,
            protocol_hash=expected,
            preregistration=preregistration,
        )

    # ------------------------------------------------------------------ access
    def arm(self, name: str) -> ArmPin:
        try:
            return self.arms[name]
        except KeyError as exc:
            raise ProtocolError(
                f"protocol bundle {str(self.path)!r} has no arm {name!r}; "
                f"frozen arms: {sorted(self.arms)}") from exc

    def contract(self, name: str) -> ModelContract:
        return self.arm(name).to_contract()

    # ----------------------------------------------------------- verification
    def verify_checkout(self) -> None:
        """Refuse when this checkout no longer implements the frozen protocol."""
        drift: list[str] = []
        for name, pin in sorted(self.arms.items()):
            try:
                template = get_prompt(pin.prompt_id)
            except KeyError:
                drift.append(
                    f"{name}: prompt {pin.prompt_id!r} no longer exists in this checkout")
            else:
                if template.prompt_hash != pin.prompt_hash:
                    drift.append(
                        f"{name}: prompt {pin.prompt_id!r} drifted "
                        f"(pinned {pin.prompt_hash}, checkout {template.prompt_hash})")
            try:
                schema = schema_for(pin.schema_id)
            except ProtocolError:
                drift.append(
                    f"{name}: schema {pin.schema_id!r} no longer exists in this checkout")
            else:
                actual = schema_hash(schema)
                if actual != pin.schema_hash:
                    drift.append(
                        f"{name}: output schema {pin.schema_id!r} drifted "
                        f"(pinned {pin.schema_hash}, checkout {actual})")
        if drift:
            raise ProtocolDriftError(
                "frozen protocol no longer matches this checkout:\n  " + "\n  ".join(drift))

    def check_contract(self, name: str, contract: ModelContract) -> None:
        """Refuse a contract that does not implement this arm's pins."""
        pin = self.arm(name)
        problems: list[str] = []
        if contract.prompt_id != pin.prompt_id:
            problems.append(
                f"prompt id {contract.prompt_id!r} != pinned {pin.prompt_id!r}")
        if contract.prompt_hash != pin.prompt_hash:
            problems.append(
                f"prompt hash {contract.prompt_hash} != pinned {pin.prompt_hash}")
        if contract.schema_hash != pin.schema_hash:
            problems.append(
                f"schema hash {contract.schema_hash} != pinned {pin.schema_hash}")
        if pin.temperature is not None and contract.temperature != pin.temperature:
            problems.append(
                f"temperature {contract.temperature} != pinned {pin.temperature}")
        if pin.max_output_tokens is not None and contract.max_output_tokens != pin.max_output_tokens:
            problems.append(
                f"max_output_tokens {contract.max_output_tokens} != "
                f"pinned {pin.max_output_tokens}")
        if self.pin_models or pin.pins_model:
            if contract.digest != pin.contract_hash:
                problems.append(
                    f"contract hash {contract.digest} != pinned {pin.contract_hash}; the model "
                    f"or its limits changed after the protocol was frozen")
        if problems:
            raise ProtocolDriftError(
                f"arm {name!r} does not implement the frozen protocol "
                f"{self.protocol_hash}:\n  " + "\n  ".join(problems))

    def require_preregistration(self) -> Phase0Preregistration:
        """The pinned Phase 0 thresholds, or a refusal if none were pinned."""
        if self.preregistration is None:
            raise ProtocolError(
                f"protocol bundle {str(self.path)!r} pins no Phase 0 preregistration; "
                "freeze the Grashof margin and T2 tolerance before hidden testing")
        return self.preregistration

    def verify_preregistration(self, expected: Phase0Preregistration) -> None:
        """Refuse when this bundle was frozen with different thresholds."""
        if not isinstance(expected, Phase0Preregistration):
            raise ProtocolError(
                f"expected a Phase0Preregistration, got {type(expected).__name__}")
        pinned = self.require_preregistration()
        if pinned.digest == expected.digest:
            return
        fields = [
            name
            for name in ("grashof_margin_mm", "function_tolerance_deg")
            if getattr(pinned, name) != getattr(expected, name)
        ]
        detail = "\n  ".join(
            f"{name}: pinned {getattr(pinned, name)!r}, supplied "
            f"{getattr(expected, name)!r}" for name in fields
        ) or "rationale or version differ"
        raise ProtocolDriftError(
            f"protocol bundle {str(self.path)!r} pins a different Phase 0 "
            f"preregistration:\n  {detail}")

    def record(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "bundle_id": self.bundle_id,
            "created_utc": self.created_utc,
            "pin_models": self.pin_models,
            "protocol_hash": self.protocol_hash,
            "arms": {name: pin.record() for name, pin in sorted(self.arms.items())},
            "preregistration": (
                None if self.preregistration is None
                else self.preregistration.record()),
        }


def freeze_protocol(
    path: Path | str,
    arms: Mapping[str, ModelContract],
    *,
    pin_models: bool = True,
    bundle_id: str = DEFAULT_BUNDLE_ID,
    notes: str = "",
    created_utc: str | None = None,
    require_all_arms: bool = True,
    overwrite: bool = False,
    preregistration: Phase0Preregistration | None = None,
) -> FrozenProtocol:
    """Write the protocol bundle and return it as loaded from disk.

    ``require_all_arms`` demands one contract per arm §12.6 names (P1 and P2).
    An existing bundle is never replaced unless ``overwrite=True``: a frozen
    protocol that can be rewritten silently is not frozen.
    """
    target = Path(path)
    if not isinstance(arms, Mapping) or not arms:
        raise ProtocolError("a protocol bundle needs at least one arm")
    names = set(arms)
    if require_all_arms and names != set(ARMS):
        raise ProtocolError(
            f"protocol bundle needs exactly the arms {sorted(ARMS)}; got {sorted(names)}")
    pins = {name: ArmPin.from_contract(name, contract) for name, contract in arms.items()}
    if not pin_models:
        pins = {
            name: ArmPin(
                arm=pin.arm,
                prompt_id=pin.prompt_id,
                prompt_hash=pin.prompt_hash,
                schema_id=pin.schema_id,
                schema_hash=pin.schema_hash,
                temperature=pin.temperature,
                max_output_tokens=pin.max_output_tokens,
                timeout_s=pin.timeout_s,
            )
            for name, pin in pins.items()
        }
    if target.exists() and not overwrite:
        raise ProtocolError(
            f"protocol bundle {str(target)!r} already exists; pass overwrite=True to replace a "
            "frozen protocol deliberately")
    payload = _bundled_payload(
        created_utc=created_utc or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        bundle_id=bundle_id,
        pin_models=pin_models,
        arms=pins,
        notes=notes,
        preregistration=preregistration,
    )
    payload["protocol_hash"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "protocol_hash"})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    frozen = FrozenProtocol.load(target)
    # A bundle that does not describe this checkout must not survive the write.
    try:
        frozen.verify_checkout()
    except ProtocolDriftError:
        target.unlink(missing_ok=True)
        raise
    return frozen


def load_protocol(path: Path | str) -> FrozenProtocol:
    return FrozenProtocol.load(path)
