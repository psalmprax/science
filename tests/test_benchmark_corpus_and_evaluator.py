"""Unit tests for Phase 4 Benchmark Corpus integrity and Independent Evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.independent_evaluator import IndependentEvaluator

CORPUS_DIR = Path(__file__).parent.parent / "benchmark_corpus"


def test_benchmark_corpus_exists_and_balanced():
    assert CORPUS_DIR.exists()
    manifest_path = CORPUS_DIR / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_problems"] == 30
    assert manifest["num_classes"] == 5
    assert manifest["problems_per_class"] == 6

    # Verify exactly 6 problems per class
    class_counts = {}
    for p in manifest["problems"].values():
        cid = p["class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1

    for cid in range(1, 6):
        assert class_counts[cid] == 6, f"Class {cid} has {class_counts.get(cid, 0)} problems, expected 6"


def test_benchmark_checksums_match():
    checksum_file = CORPUS_DIR / "checksums.sha256"
    assert checksum_file.exists()

    with open(checksum_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 30 * 5  # At least 150 files

    for line in lines:
        expected_hash, rel_path = line.strip().split("  ", 1)
        target = CORPUS_DIR / rel_path
        assert target.exists(), f"File {rel_path} in checksums does not exist on disk"

        h = hashlib.sha256()
        with open(target, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        assert h.hexdigest() == expected_hash, f"Hash mismatch for {rel_path}"


def test_public_and_hidden_separation():
    public_dir = CORPUS_DIR / "public"
    hidden_dir = CORPUS_DIR / "hidden"

    for i in range(1, 31):
        pid = f"P{i:02d}"
        pub_p = public_dir / pid
        hid_p = hidden_dir / pid

        assert pub_p.exists()
        assert hid_p.exists()

        # Public files: train, validation, near_ood, far_ood, problem_spec
        assert (pub_p / "train.csv").exists()
        assert (pub_p / "validation.csv").exists()
        assert (pub_p / "near_ood.csv").exists()
        assert (pub_p / "far_ood.csv").exists()
        assert (pub_p / "problem_spec.json").exists()

        # Ensure ground truth is NOT in public
        assert not (pub_p / "ground_truth.json").exists()

        # Ensure hidden contains ground truth
        assert (hid_p / "ground_truth.json").exists()

        # Verify problem spec has no leaked ground truth formula
        with open(pub_p / "problem_spec.json", "r", encoding="utf-8") as f:
            spec = json.load(f)
        assert "ground_truth_formula" not in spec
        assert "formula" not in spec


def test_independent_evaluator_synthetic_mock(tmp_path):
    evaluator = IndependentEvaluator(CORPUS_DIR)

    # Create mock competitor results directory
    mock_results = tmp_path / "mock_competitor"
    mock_results.mkdir(parents=True)

    # Generate perfect predictions for P01, P07, P19
    for pid, state in [("P01", "UNIQUE_SUPPORT"), ("P07", "MULTI_HYPOTHESIS"), ("P19", "INSUFFICIENT_EVIDENCE")]:
        prob_dir = mock_results / pid
        prob_dir.mkdir(parents=True)
        res_data = {
            "problem_id": pid,
            "epistemic_state": state,
            "top_expression": "a**1.5" if pid == "P01" else "2.0*x + 1.0",
            "near_ood_mse": 0.001,
            "dimensional_status": "VALID",
            "wall_clock_seconds": 2.5,
            "candidates_evaluated": 150,
        }
        with open(prob_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(res_data, f)

    eval_out = evaluator.evaluate_competitor("MockModel", mock_results)
    assert eval_out["competitor_name"] == "MockModel"
    assert eval_out["total_problems_evaluated"] == 30
    assert "evaluation_profile_M" in eval_out
    assert "epistemic_confusion_matrix" in eval_out
    assert "class_profiles" in eval_out

    # Check P01 epistemic match
    assert eval_out["per_problem_results"]["P01"]["epistemic_match"] is True
    # Check P07 epistemic match
    assert eval_out["per_problem_results"]["P07"]["epistemic_match"] is True
    # Check P19 epistemic match
    assert eval_out["per_problem_results"]["P19"]["epistemic_match"] is True
