from apedp.source_audit import (
    DatasetLineageRegistry,
    SourceMetadata,
    audit_dataset,
)


def test_publication_audit_records_missing_curation_metadata():
    source = SourceMetadata(
        source_id="paper-1",
        source_kind="publication",
        value_kind="measured",
        citation="Example et al. (2024)",
        dataset_id="paper-1-table-2",
        unit_conventions={"x": "meter", "y": "joule"},
    )
    audit = audit_dataset(
        [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}],
        source,
        required_columns=("x", "y"),
    )
    assert audit.errors == ()
    assert audit.dataset_hash
    assert any("selection/exclusion" in warning for warning in audit.warnings)
    assert audit.requires_human_review
    assert not audit.publication_bias_assessed


def test_registry_detects_reused_dataset_and_lineage():
    registry = DatasetLineageRegistry()
    source = SourceMetadata(
        source_id="database",
        source_kind="database",
        value_kind="measured",
        dataset_id="ame-table",
        unit_conventions={"mass": "kilogram"},
    )
    rows = [{"mass": 1.0}]
    first = audit_dataset(rows, source, registry=registry)
    second = audit_dataset(rows, source, registry=registry)
    assert first.safe_for_discovery
    assert second.repeated_dataset
    assert second.repeated_lineage
    assert not second.safe_for_discovery


def test_shared_uncertainty_group_requires_covariance_metadata():
    source = SourceMetadata(
        source_id="experiment-run",
        source_kind="experiment",
        value_kind="measured",
        experiment_id="run-7",
        laboratory_id="lab-a",
        unit_conventions={"x": "meter", "y": "joule"},
        uncertainty_groups={"y": "sensor-1"},
        covariance_available=False,
    )
    audit = audit_dataset([{"x": 1.0, "y": 2.0}], source)
    assert any("covariance metadata" in warning for warning in audit.warnings)


def test_mixed_source_values_are_not_silently_classified():
    source = SourceMetadata(
        source_id="nuclear-collection",
        source_kind="database",
        value_kind="mixed",
        dataset_id="nuclide-set",
    )
    audit = audit_dataset([{"z": 26, "a": 56, "energy": 492.0}], source)
    assert any("single declared" in warning for warning in audit.warnings)
    assert any("unit conventions" in warning for warning in audit.warnings)
