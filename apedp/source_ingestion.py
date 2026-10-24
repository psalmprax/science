"""Conservative adapters for publication and domain data ingestion.

Structured tables can be passed to equation discovery after auditing. PDF text
is exposed for review only; this module never pretends that arbitrary PDF text
or a table layout is a trustworthy numerical dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apedp.semf_validation import AmeDataset, parse_mas20
from apedp.source_audit import DatasetAudit, SourceMetadata, audit_dataset


@dataclass(frozen=True)
class StructuredSource:
    """A parsed table plus its source audit and raw-file hash."""

    path: str
    file_hash: str
    rows: tuple[dict[str, Any], ...]
    audit: DatasetAudit

    @property
    def ready_for_discovery(self) -> bool:
        return self.audit.safe_for_discovery and not self.audit.requires_human_review


@dataclass(frozen=True)
class PublicationManifest:
    """Declared publication metadata and its curation completeness."""

    source: SourceMetadata
    warnings: tuple[str, ...]
    doi_syntax_valid: bool

    @property
    def ready_for_curation(self) -> bool:
        return not self.warnings and self.doi_syntax_valid


@dataclass(frozen=True)
class PublicationText:
    """PDF text extraction result; numerical interpretation remains unverified."""

    path: str
    file_hash: str
    page_count: int
    text: str
    warnings: tuple[str, ...]

    @property
    def ready_for_discovery(self) -> bool:
        return False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_json_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, dict):
        for key in ("rows", "data", "records"):
            candidate = value.get(key)
            if isinstance(candidate, list) and all(isinstance(row, dict) for row in candidate):
                return [dict(row) for row in candidate]
    raise ValueError("JSON source must be a list of objects or contain rows/data/records")


def load_structured_source(
    path: str,
    source: SourceMetadata,
    *,
    required_columns: Iterable[str] = (),
    delimiter: str = ",",
) -> StructuredSource:
    """Load CSV/JSON records and audit them before discovery.

    Numeric conversion is intentionally not performed here: preserving the raw
    values lets the audit identify malformed cells instead of silently changing
    the source. Discovery callers can explicitly convert their validated rows.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        with file_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [dict(row) for row in csv.DictReader(stream, delimiter=delimiter)]
    elif suffix == ".json":
        with file_path.open(encoding="utf-8") as stream:
            rows = _coerce_json_rows(json.load(stream))
    else:
        raise ValueError("structured source must be .csv or .json")
    if not rows:
        raise ValueError(f"source contains no records: {path}")
    columns = list(rows[0])
    audit = audit_dataset(rows, source, columns=columns, required_columns=required_columns)
    return StructuredSource(
        path=str(file_path),
        file_hash=_file_hash(file_path),
        rows=tuple(rows),
        audit=audit,
    )


def load_publication_manifest(path: str) -> PublicationManifest:
    """Load a JSON publication manifest without contacting external services.

    A manifest must explicitly identify a publication. DOI resolution and
    citation correctness are intentionally not inferred locally.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise ValueError("publication manifest must be a .json file")
    with file_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("publication manifest must be a JSON object")
    source = SourceMetadata(**payload)
    if source.source_kind != "publication":
        raise ValueError("publication manifest source_kind must be 'publication'")
    warnings: list[str] = []
    doi_valid = bool(re.fullmatch(r"10\.\d{4,9}/\S+", source.doi)) if source.doi else False
    if not source.doi and not source.citation:
        warnings.append("DOI or citation is required")
    if source.doi and not doi_valid:
        warnings.append("DOI has invalid syntax")
    if not source.selection_criteria:
        warnings.append("selection_criteria is required")
    if not source.preprocessing:
        warnings.append("preprocessing declaration is required")
    if not source.independent_source_ids:
        warnings.append("independent_source_ids declaration is required")
    return PublicationManifest(
        source=source,
        warnings=tuple(warnings),
        doi_syntax_valid=doi_valid or (not source.doi and bool(source.citation)),
    )


def load_ame_mass_table(
    path: str,
    source: SourceMetadata | None = None,
) -> StructuredSource:
    """Load an AME/MAS20 mass table as audited nuclear observations.

    The existing fixed-width parser excludes rows marked as extrapolated and
    emits experimental binding energy per nucleon in MeV. The returned rows
    retain that status and include parser exclusions in the audit warnings.
    """
    data: AmeDataset = parse_mas20(path)
    metadata = source or SourceMetadata(
        source_id="ame2020-mass-table",
        source_kind="database",
        value_kind="measured",
        citation="Huang et al., Chinese Physics C 45 030002 (2021); Wang et al., 030003 (2021)",
        database_id="AME2020",
        dataset_id="AME2020:mass_1.mas20.txt",
        selection_criteria="experimental rows only; MAS20 rows marked # excluded",
        preprocessing="fixed-width parser; binding/A keV converted to MeV per nucleon",
        unit_conventions={"Z": "dimensionless", "N": "dimensionless", "A": "dimensionless", "BE_per_nucleon": "MeV"},
    )
    if metadata.source_kind != "database" or metadata.value_kind != "measured":
        raise ValueError("AME mass tables require source_kind='database' and value_kind='measured'")
    rows = tuple(
        {"Z": int(z), "N": int(n), "A": int(a), "BE_per_nucleon": float(be)}
        for z, n, a, be in zip(data.Z, data.N, data.A, data.BE_per_nucleon, strict=False)
    )
    audit = audit_dataset(rows, metadata, required_columns=("Z", "N", "A", "BE_per_nucleon"))
    warnings = list(audit.warnings)
    if data.n_skipped_extrapolated:
        warnings.append(f"excluded {data.n_skipped_extrapolated} extrapolated MAS20 row(s)")
    if data.n_skipped_missing:
        warnings.append(f"excluded {data.n_skipped_missing} missing/invalid MAS20 row(s)")
    audit = DatasetAudit(
        dataset_hash=audit.dataset_hash,
        row_count=audit.row_count,
        source=audit.source,
        warnings=tuple(dict.fromkeys(warnings)),
        errors=audit.errors,
        duplicate_row_count=audit.duplicate_row_count,
        repeated_lineage=audit.repeated_lineage,
        repeated_dataset=audit.repeated_dataset,
        publication_bias_assessed=audit.publication_bias_assessed,
    )
    return StructuredSource(
        path=str(Path(path)),
        file_hash=_file_hash(Path(path)),
        rows=rows,
        audit=audit,
    )


def extract_publication_text(path: str) -> PublicationText:
    """Extract PDF text for human review, never as an automatic data table.

    Scanned PDFs, layout-dependent tables, equations, and OCR output remain
    unverified. The warnings are part of the return value so callers cannot
    accidentally present this as a discovery-ready dataset.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("publication text extraction requires a .pdf file")
    if not file_path.is_file():
        raise FileNotFoundError(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires the pypdf dependency") from exc
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return PublicationText(
        path=str(file_path),
        file_hash=_file_hash(file_path),
        page_count=len(pages),
        text="\n\n".join(pages),
        warnings=(
            "PDF text extraction is not table extraction",
            "equations, units, OCR, and measured/simulated status require human verification",
            "convert and audit a structured table before equation discovery",
        ),
    )


__all__ = [
    "PublicationManifest",
    "PublicationText",
    "StructuredSource",
    "extract_publication_text",
    "load_ame_mass_table",
    "load_publication_manifest",
    "load_structured_source",
]
