"""MMLM benchmark protocol and metric contracts."""

from dataclasses import replace

import pytest

from apedp_mmlm.models import (
    BenchmarkMetrics,
    Milestone1BenchmarkSuite,
    MMLMBenchmarkDataset,
)


def _dataset(**overrides):
    values = {
        "name": "External sample corpus",
        "source_url": "https://example.org/corpus",
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "retrieved_at": "2026-11-06T00:00:00Z",
        "train_texts": ("ada studies optics .", "ben builds circuits ."),
        "dev_texts": ("cyra observes stars .",),
        "test_texts": ("dax uses models .",),
    }
    return MMLMBenchmarkDataset(**(values | overrides))


def _metrics(**overrides):
    values = {
        "model_id": "model",
        "held_out_perplexity": 2.0,
        "bits_per_byte": 1.0,
        "generation_energy": 0.5,
        "long_context_retrieval_accuracy": 0.5,
        "compositional_reasoning_accuracy": 0.5,
        "serialized_bytes": 100,
        "training_time_ms": 10.0,
        "inference_tokens_per_sec": 100.0,
    }
    return BenchmarkMetrics(**(values | overrides))


def test_external_dataset_requires_provenance_and_disjoint_nonempty_splits():
    with pytest.raises(ValueError, match="license_url"):
        _dataset(license_url="not-a-url")
    with pytest.raises(ValueError, match="disjoint"):
        _dataset(test_texts=("ada studies optics .",))
    with pytest.raises(ValueError, match="non-empty text"):
        _dataset(dev_texts=("",))


def test_external_dataset_hashes_and_benchmark_split_protocol():
    dataset = _dataset()
    provenance = dataset.provenance()
    assert provenance["license_id"] == "CC-BY-4.0"
    assert set(provenance["split_sha256"]) == {"train", "dev", "test"}
    assert all(len(value) == 64 for value in provenance["split_sha256"].values())

    suite = Milestone1BenchmarkSuite(dataset)
    assert suite.train_corpus == list(dataset.train_texts)
    assert suite.dev_corpus == list(dataset.dev_texts)
    assert suite.eval_corpus == list(dataset.test_texts)
    assert all(
        text not in suite.train_corpus
        for text in (*suite.dev_corpus, *suite.eval_corpus)
    )


def test_h5_quality_and_efficiency_axes_are_explicit_majority_comparisons():
    hybrid = _metrics(
        held_out_perplexity=1.0,
        bits_per_byte=0.5,
        long_context_retrieval_accuracy=0.75,
        compositional_reasoning_accuracy=0.75,
        serialized_bytes=50,
        training_time_ms=5.0,
        inference_tokens_per_sec=50.0,
    )
    baseline = _metrics()
    axes = Milestone1BenchmarkSuite._quality_efficiency_axes(hybrid, baseline)

    assert axes == {
        "quality": True,
        "efficiency": True,
        "quality_metric_wins": 4,
        "efficiency_metric_wins": 2,
    }
    no_quality = replace(
        hybrid,
        held_out_perplexity=3.0,
        bits_per_byte=2.0,
        long_context_retrieval_accuracy=0.25,
        compositional_reasoning_accuracy=0.25,
    )
    assert not Milestone1BenchmarkSuite._quality_efficiency_axes(no_quality, baseline)["quality"]


def test_milestone_report_measures_h2_and_h5_without_overclaiming():
    report = Milestone1BenchmarkSuite().run_milestone_1(seed=23)

    assert report.hypothesis_verdicts["H2"] in {
        "OBSERVED_COMPARISON_PASS",
        "OBSERVED_COMPARISON_FAIL",
    }
    assert report.hypothesis_verdicts["H5"] in {
        "OBSERVED_COMPARISON_PASS",
        "OBSERVED_COMPARISON_FAIL",
    }
    assert report.protocol_summary["streaming_retention_threshold"] == pytest.approx(0.70)
    assert set(report.protocol_summary["h5_quality_efficiency_axes"]) == {"B0", "C0"}
    assert "not a significance test" in report.hypothesis_evidence["H5"]
    assert report.protocol_summary["external_or_real_world_evidence"] is False
