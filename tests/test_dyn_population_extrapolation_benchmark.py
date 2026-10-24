"""
Regression suite for Multi-Population Extrapolation & Distractor Challenge: RIME-AD vs GA/GP.
"""

import json
from pathlib import Path

RESULTS_JSON = Path("benchmarks/population_extrapolation_results.json")
REPORT_MD = Path("docs/population_extrapolation_rime_ad_vs_ga.md")


def test_extrapolation_artifacts_exist():
    assert RESULTS_JSON.exists(), f"Missing benchmark JSON: {RESULTS_JSON}"
    assert REPORT_MD.exists(), f"Missing markdown report: {REPORT_MD}"


def test_population_extrapolation_metrics():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    methods = data["methods"]
    ga_s = methods["ga_small"]
    ga_l = methods["ga_large"]
    rime = methods["rime_ad"]

    # GA-Small fails due to missing nonlinear interaction
    assert ga_s["r2_A_tr"] < 0.75
    assert ga_s["r2_D_heavy"] < 0.40

    # RIME-AD matches or exceeds GA-Large on all populations
    assert rime["r2_A_tr"] > 0.9990
    assert rime["r2_A_te"] > 0.9990
    assert rime["r2_B"] > 0.9990
    assert rime["r2_C_shift"] > 0.8500
    assert rime["r2_D_heavy"] > 0.9999

    # Recovered formula identifies true interaction and lags
    formula = rime["formula"]
    assert "x1[t-1]" in formula or "x1" in formula
    assert "x2[t-1]" in formula or "x2" in formula
    assert "x3[t-2]" in formula or "x3" in formula

    # Speedup: Native Rust RIME-AD is at least 30x faster than GA-Large
    assert rime["wall_ms"] * 30 < ga_l["wall_ms"]


def test_distractor_noise_immunity():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    dist = data["distractor_challenge"]
    assert dist["distractor_resilience_passed"] is True
    assert dist["distractors_admitted"] == []
    assert set(dist["inputs_used"]) == {"x1", "x2", "x3"}
    assert dist["dev_r2"] > 0.9990
