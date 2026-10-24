import json

import pytest

from apedp.source_audit import SourceMetadata
from apedp.source_ingestion import (
    extract_publication_text,
    load_ame_mass_table,
    load_publication_manifest,
    load_structured_source,
)


def _source() -> SourceMetadata:
    return SourceMetadata(
        source_id="test-table",
        source_kind="database",
        value_kind="measured",
        dataset_id="test-table-1",
        unit_conventions={"x": "meter", "y": "joule"},
    )


def test_structured_json_preserves_rows_and_file_provenance(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({"rows": [{"x": 1.0, "y": 2.0}]}))
    loaded = load_structured_source(str(path), _source(), required_columns=("x", "y"))
    assert loaded.rows == ({"x": 1.0, "y": 2.0},)
    assert len(loaded.file_hash) == 64
    assert loaded.audit.errors == ()
    assert loaded.ready_for_discovery is True


def test_structured_csv_requires_declared_columns(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("x,y\n1,2\n")
    loaded = load_structured_source(str(path), _source(), required_columns=("x", "z"))
    assert loaded.ready_for_discovery is False
    assert any("missing required" in error for error in loaded.audit.errors)


def test_publication_manifest_requires_curation_declarations(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "source_id": "paper-1",
        "source_kind": "publication",
        "value_kind": "measured",
        "doi": "10.1234/example",
    }))
    manifest = load_publication_manifest(str(path))
    assert manifest.ready_for_curation is False
    assert any("selection_criteria" in warning for warning in manifest.warnings)
    assert any("preprocessing" in warning for warning in manifest.warnings)
    assert any("independent_source_ids" in warning for warning in manifest.warnings)
    assert manifest.doi_syntax_valid


def test_ame_mass_table_adapter_preserves_experimental_status(tmp_path):
    def row(n, z, a, binding, extrapolated=False):
        line = [" "] * 80
        line[4:9] = f"{n:5d}"
        line[9:14] = f"{z:5d}"
        line[14:19] = f"{a:5d}"
        line[54:67] = f"{binding:13.3f}"
        if extrapolated:
            line[20] = "#"
        return "".join(line) + "\n"

    path = tmp_path / "mass_1.mas20.txt"
    path.write_text(row(30, 26, 56, 87900) + row(31, 26, 57, 88000, True))
    loaded = load_ame_mass_table(str(path))
    assert loaded.rows == ({"Z": 26, "N": 30, "A": 56, "BE_per_nucleon": 87.9},)
    assert loaded.audit.source.database_id == "AME2020"
    assert any("extrapolated" in warning for warning in loaded.audit.warnings)


def test_pdf_extraction_is_explicitly_not_discovery_ready(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "paper.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as stream:
        writer.write(stream)
    extracted = extract_publication_text(str(path))
    assert extracted.page_count == 1
    assert extracted.ready_for_discovery is False
    assert any("not table extraction" in warning for warning in extracted.warnings)
