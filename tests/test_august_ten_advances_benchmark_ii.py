"""Pins for the August 1 OpenAI batch screened against Benchmark II.

The screen exists because that batch is the only OpenAI release whose disciplines
intersect the DQI decoder (coding theory, arithmetic circuit complexity, quantum
complexity). Its value is the negative result, so these tests pin the negative
result: no row is allowed to claim a cost reduction, the frozen benchmark
instance must match the official verifier's own defaults, and the cost model
must match the pinned reference/target workloads rather than a retyped copy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "benchmarks" / "openai_math_sandbox" / "catalog" / "august_ten_advances_screen.json"
VERIFIER = (
    ROOT
    / "apedp_dyn"
    / "sandbox"
    / "google-verifiable-quantum-circuit-compilation"
    / "verifiers"
    / "dqi"
    / "verify_submission_dqi.py"
)

VERDICTS = {"cost_reducing", "analyzable_not_cost_reducing", "cannot_reduce_cost", "opposite_direction"}


def _screen() -> dict:
    assert SCREEN.is_file(), f"missing screen: {SCREEN}"
    return json.loads(SCREEN.read_text(encoding="utf-8"))


def test_all_ten_results_are_screened_with_a_reason():
    d = _screen()
    results = d["results"]
    assert len(results) == 10
    assert len({r["id"] for r in results}) == 10
    for r in results:
        assert r["verdict"] in VERDICTS, r["id"]
        assert len(r["reason"]) > 80, r["id"]
        assert r["field"].strip() and r["result_kind"].strip(), r["id"]


def test_no_result_is_recorded_as_a_cost_reduction():
    d = _screen()
    assert not [r for r in d["results"] if r["verdict"] == "cost_reducing"]
    counts = {
        v: sum(1 for r in d["results"] if r["verdict"] == v) for v in VERDICTS
    }
    summary = d["summary"]
    for v, n in counts.items():
        assert summary[v] == n, (v, summary[v], n)
    assert summary["results_screened"] == 10
    assert summary["cost_reducing"] == 0


def test_the_only_analyzable_row_is_the_code_bound_and_is_not_a_cost_path():
    d = _screen()
    analyzable = [r for r in d["results"] if r["verdict"] == "analyzable_not_cost_reducing"]
    assert len(analyzable) == 1, [r["id"] for r in analyzable]
    row = analyzable[0]
    assert row["id"] == "binary_and_spherical_codes"
    assert "cannot reduce Q or T_mean" in row["reason"]
    assert len(row["evidence_needed"]) >= 3
    # The claim this row must never make: that the bound lowers the score.
    assert "floor" in row["use"]


def test_lower_bounds_are_not_filed_as_tools():
    """A lower bound or hardness result must never be filed as anything but opposite."""
    d = _screen()
    for r in d["results"]:
        if r["result_kind"] in {"lower_bound", "hardness"}:
            assert r["verdict"] == "opposite_direction", (r["id"], r["verdict"])
    lowerish = [r for r in d["results"] if r["result_kind"] in {"lower_bound", "hardness"}]
    assert len(lowerish) == 3


def test_frozen_instance_matches_the_official_verifier_defaults():
    d = _screen()["fixed_instance"]
    text = VERIFIER.read_text(encoding="utf-8")
    for name, key in (("DEFAULT_M", "m"), ("DEFAULT_N", "n"), ("DEFAULT_DEGREE", "degree"), ("DEFAULT_R", "r")):
        m = re.search(rf"^{name}\s*=\s*(\d+)", text, re.MULTILINE)
        assert m, f"{name} not found in the verifier"
        assert int(m.group(1)) == d[key], (name, int(m.group(1)), d[key])
    assert d["max_errors"] == d["n"] // 2 == 35
    assert d["source"].endswith("verify_submission_dqi.py")


def test_cost_model_matches_the_pinned_workloads():
    from apedp_dyn.quantum.dqi_photonic_resource import (
        DQI_GLUZDOV_V6_TARGET_WORKLOAD,
        DQI_REFERENCE_WORKLOAD,
    )

    cm = _screen()["benchmark_ii_cost_model"]
    assert cm["reference"]["qubits"] == DQI_REFERENCE_WORKLOAD.qubits
    assert cm["reference"]["mean_toffoli_like"] == DQI_REFERENCE_WORKLOAD.mean_toffoli_like
    assert cm["target"]["qubits"] == DQI_GLUZDOV_V6_TARGET_WORKLOAD.qubits
    assert cm["target"]["mean_toffoli_like"] == DQI_GLUZDOV_V6_TARGET_WORKLOAD.mean_toffoli_like
    assert cm["reference"]["label"] == DQI_REFERENCE_WORKLOAD.label
    assert cm["target"]["label"] == DQI_GLUZDOV_V6_TARGET_WORKLOAD.label

    q_ref, t_ref = cm["reference"]["qubits"], cm["reference"]["mean_toffoli_like"]
    q_tgt, t_tgt = cm["target"]["qubits"], cm["target"]["mean_toffoli_like"]
    gain = 1.0 / ((q_tgt / q_ref) ** 2 * (t_tgt / t_ref))
    assert abs(gain - cm["derived"]["target_gain"]) < 5e-4, (gain, cm["derived"]["target_gain"])
    # The quoted 14.396x and the recomputed 14.3973x must both stay on record.
    assert abs(gain - cm["derived"]["quoted_target_gain"]) < 0.01
    assert "rounding" in cm["derived"]["gain_rounding_note"]
    assert cm["derived"]["qubit_reduction"] == q_ref - q_tgt == 104


def test_the_documented_levers_are_the_only_levers():
    cm = _screen()["benchmark_ii_cost_model"]
    assert cm["score"] == "R(eps_plus) * Q^2 * T_mean"
    assert cm["only_levers"] == ["Q", "T_mean"]
    # R is measured from the submission, so it is not an input a candidate can set.
    assert "measured by the verifier" in cm["note"]
