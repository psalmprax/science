"""Regression suite for the population extrapolation benchmark (RIME, RIME-AD, GA term selection).

Until 2026-11-19 the benchmark ran RIME and reported it as RIME-AD (SYSTEM_FLAWS_AUDIT
M-05). These tests pin that each engine's row comes from that engine.
"""

import ast
import json
from pathlib import Path

RESULTS_JSON = Path("benchmarks/population_extrapolation_results.json")
REPORT_MD = Path("docs/population_extrapolation_rime_ad_vs_ga.md")
SCRIPT = Path("benchmarks/run_population_extrapolation_rime_ad_vs_ga.py")


def _methods():
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["methods"]


def test_extrapolation_artifacts_exist():
    assert RESULTS_JSON.exists(), f"Missing benchmark JSON: {RESULTS_JSON}"
    assert REPORT_MD.exists(), f"Missing markdown report: {REPORT_MD}"


def test_each_row_names_the_engine_that_produced_it():
    methods = _methods()
    assert methods["rime"]["engine_class"].endswith("rime_proposer.RimeProposer")
    assert methods["rime_ad"]["engine_class"].endswith("rime_ad_proposer.RimeAdProposer")
    # The script constructs both classes; a row cannot be one engine's numbers under the other's name.
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"RimeProposer", "RimeAdProposer"} <= called


def test_rime_recovers_the_lagged_interaction():
    rime = _methods()["rime"]
    assert rime["r2_A_te"] > 0.9990 and rime["r2_B"] > 0.9990 and rime["r2_D_heavy"] > 0.9999
    assert "x1[t-1] * x2[t-1]" in rime["formula"] and "x3[t-2]" in rime["formula"]


def test_rime_ad_does_not_recover_a_lagged_law():
    """RIME-AD builds features from contemporaneous columns only, so this law is
    outside what it searches; its measured fit is recorded, not claimed."""
    rime_ad = _methods()["rime_ad"]
    assert rime_ad["r2_A_te"] < 0.5
    assert "[t-" not in (rime_ad["formula"] or "")


def test_ga_large_needs_its_planted_library_and_does_not_simplify():
    methods = _methods()
    assert methods["ga_small"]["r2_A_tr"] < 0.75  # without the interaction column it cannot fit
    assert "x1[t-1]*x2[t-1]" in methods["ga_large"]["active_terms"]
    assert len(methods["ga_large"]["active_terms"]) > 3  # it fits, but keeps many more terms than the truth has


def test_distractor_noise_immunity():
    dist = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["distractor_challenge"]
    assert dist["engine"] == "RIME (RimeProposer)"
    assert dist["distractor_resilience_passed"] is True
    assert dist["distractors_admitted"] == []
    assert set(dist["inputs_used"]) == {"x1", "x2", "x3"}
    assert dist["dev_r2"] > 0.9990


def test_the_report_carries_the_correction():
    report = REPORT_MD.read_text(encoding="utf-8")
    assert "M-05" in report and "RIME-AD was" in report
    assert "file:///" not in report
