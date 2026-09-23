"""Source-aware audit primitives for scientific datasets.

This module does not infer scientific provenance from a CSV.  It requires the
caller to declare the source and records what is known, unknown, or potentially
unsafe before discovery is run.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SourceKind = Literal["experiment", "publication", "database", "simulation", "fit", "unknown"]
ValueKind = Literal["measured", "simulated", "fitted", "mixed", "unknown"]


@dataclass(frozen=True)
class SourceMetadata:
    """Declared origin and lineage of a dataset or observation batch."""

    source_id: str
    source_kind: SourceKind = "unknown"
    value_kind: ValueKind = "unknown"
    citation: str = ""
    doi: str = ""
    database_id: str = ""
    dataset_id: str = ""
    parent_dataset_ids: tuple[str, ...] = ()
    independent_source_ids: tuple[str, ...] = ()
    experiment_id: str = ""
    laboratory_id: str = ""
    selection_criteria: str = ""
    preprocessing: str = ""
    unit_conventions: Mapping[str, str] = field(default_factory=dict)
    uncertainty_groups: Mapping[str, str] = field(default_factory=dict)
    covariance_available: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.source_kind not in {"experiment", "publication", "database", "simulation", "fit", "unknown"}:
            raise ValueError(f"unsupported source_kind: {self.source_kind}")
        if self.value_kind not in {"measured", "simulated", "fitted", "mixed", "unknown"}:
            raise ValueError(f"unsupported value_kind: {self.value_kind}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parent_dataset_ids"] = list(self.parent_dataset_ids)
        value["independent_source_ids"] = list(self.independent_source_ids)
        value["unit_conventions"] = dict(self.unit_conventions)
        value["uncertainty_groups"] = dict(self.uncertainty_groups)
        return value


@dataclass(frozen=True)
class DatasetAudit:
    """Machine-readable audit result; warnings are not proof of publication bias."""

    dataset_hash: str
    row_count: int
    source: SourceMetadata
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    duplicate_row_count: int = 0
    repeated_lineage: bool = False
    repeated_dataset: bool = False
    publication_bias_assessed: bool = False

    @property
    def safe_for_discovery(self) -> bool:
        return not self.errors and not self.repeated_dataset and not self.repeated_lineage

    @property
    def requires_human_review(self) -> bool:
        return bool(self.warnings or self.errors or self.repeated_dataset or self.repeated_lineage)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = self.source.to_dict()
        value["warnings"] = list(self.warnings)
        value["errors"] = list(self.errors)
        value["safe_for_discovery"] = self.safe_for_discovery
        value["requires_human_review"] = self.requires_human_review
        return value


class DatasetLineageRegistry:
    """Registry that detects repeated datasets and declared parent lineage."""

    def __init__(self) -> None:
        self._dataset_hashes: dict[str, str] = {}
        self._lineage_ids: dict[str, str] = {}

    def check_and_register(self, audit: DatasetAudit) -> DatasetAudit:
        dataset_id = audit.source.dataset_id or audit.dataset_hash
        repeated_dataset = audit.dataset_hash in self._dataset_hashes
        lineage_keys = set(audit.source.parent_dataset_ids)
        if audit.source.dataset_id:
            lineage_keys.add(audit.source.dataset_id)
        repeated_lineage = bool(lineage_keys & self._lineage_ids.keys())
        warnings = list(audit.warnings)
        if repeated_dataset:
            warnings.append("dataset hash already registered; do not count as independent evidence")
        if repeated_lineage:
            warnings.append("dataset lineage overlaps a registered dataset; independence is unproven")
        self._dataset_hashes.setdefault(audit.dataset_hash, dataset_id)
        for lineage_id in lineage_keys:
            self._lineage_ids.setdefault(lineage_id, dataset_id)
        return DatasetAudit(
            dataset_hash=audit.dataset_hash,
            row_count=audit.row_count,
            source=audit.source,
            warnings=tuple(dict.fromkeys(warnings)),
            errors=audit.errors,
            duplicate_row_count=audit.duplicate_row_count,
            repeated_lineage=repeated_lineage,
            repeated_dataset=repeated_dataset,
            publication_bias_assessed=audit.publication_bias_assessed,
        )


def _canonical_row(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    normalized: dict[str, Any] = {}
    for column in sorted(columns):
        value = row.get(column)
        if isinstance(value, float):
            value = format(value, ".17g")
        normalized[column] = value
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


def dataset_content_hash(rows: Iterable[Mapping[str, Any]], columns: Iterable[str] | None = None) -> str:
    """Hash canonical row content, independent of input row dictionary ordering."""
    materialized = list(rows)
    selected = list(columns or (materialized[0].keys() if materialized else ()))
    payload = "\n".join(_canonical_row(row, selected) for row in materialized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_dataset(
    rows: Iterable[Mapping[str, Any]],
    source: SourceMetadata,
    *,
    columns: Iterable[str] | None = None,
    required_columns: Iterable[str] = (),
    known_dataset_hashes: Iterable[str] = (),
    known_lineage_ids: Iterable[str] = (),
    registry: DatasetLineageRegistry | None = None,
) -> DatasetAudit:
    """Audit structured rows before equation/model discovery.

    The function deliberately reports missing declarations instead of guessing
    whether a value was measured, simulated, or fitted.  It also reports shared
    uncertainty groups without covariance metadata, because treating those rows
    as independent can make confidence intervals too narrow.
    """
    materialized = [dict(row) for row in rows]
    selected = list(columns or (materialized[0].keys() if materialized else ()))
    content_hash = dataset_content_hash(materialized, selected)
    required = set(required_columns)
    present = set(selected)
    warnings: list[str] = []
    errors: list[str] = []

    missing = sorted(required - present)
    if not materialized:
        errors.append("dataset contains no rows")
    if missing:
        errors.append(f"missing required columns: {missing}")
    if source.source_kind == "unknown":
        warnings.append("source kind is unknown")
    if source.value_kind == "unknown" or source.value_kind == "mixed":
        warnings.append("observation status is not a single declared measured/simulated/fitted class")
    if source.source_kind == "publication" and not (source.doi or source.citation):
        warnings.append("publication has no DOI or citation metadata")
    if source.source_kind == "publication" and not source.selection_criteria:
        warnings.append("publication selection/exclusion criteria were not supplied")
    if not source.unit_conventions:
        warnings.append("unit conventions were not supplied")
    if source.uncertainty_groups and not source.covariance_available:
        warnings.append("uncertainty groups exist but covariance metadata is unavailable")

    fingerprints = [_canonical_row(row, selected) for row in materialized]
    duplicate_count = len(fingerprints) - len(set(fingerprints))
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate observation row(s) detected")

    known_hashes = set(known_dataset_hashes)
    known_lineage = set(known_lineage_ids)
    lineage = set(source.parent_dataset_ids)
    if source.dataset_id:
        lineage.add(source.dataset_id)
    repeated_dataset = content_hash in known_hashes
    repeated_lineage = bool(lineage & known_lineage)
    if repeated_dataset:
        warnings.append("dataset hash matches a previously registered dataset")
    if repeated_lineage:
        warnings.append("dataset lineage overlaps a previously registered dataset")

    audit = DatasetAudit(
        dataset_hash=content_hash,
        row_count=len(materialized),
        source=source,
        warnings=tuple(dict.fromkeys(warnings)),
        errors=tuple(errors),
        duplicate_row_count=duplicate_count,
        repeated_lineage=repeated_lineage,
        repeated_dataset=repeated_dataset,
    )
    return registry.check_and_register(audit) if registry is not None else audit


__all__ = [
    "DatasetAudit",
    "DatasetLineageRegistry",
    "SourceMetadata",
    "audit_dataset",
    "dataset_content_hash",
]
