"""Pins for the measured DQI cost structure (`scripts/measure_dqi_primitive_costs.py`).

The numbers here are the evidence behind the Benchmark II HOLD gate: the Forney
phase inverts a field element at all 4,095 positions, and that phase alone costs
2.9x the *entire* target T_mean. So the 12.86x target is not a matter of a
missing receipt — no optimisation of that phase can reach it.

The measurement needs the kickmix extension built for the running interpreter,
so the recorded artifact is what is pinned by default; when the extension is
importable the numbers are re-measured and compared against the record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "quantum_compilation" / "dqi_primitive_costs.json"
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"


def _record() -> dict:
    assert ARTIFACT.is_file(), f"missing measured artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_constant_multipliers_and_dividers_are_free():
    """The composite-field/Karatsuba idea has no target: these are CNOT-only maps."""
    p = _record()["primitives_per_call"]
    assert p["multiply_by_constant"] == 0.0
    assert p["divide_by_constant"] == 0.0
    assert p["quantum_multiply"] == 63.0


def test_inversion_is_exactly_five_quantum_multiplications():
    """Itoh-Tsujii at GF(2^12) costs 5 multiplications and is already tight."""
    p = _record()["primitives_per_call"]
    assert p["inverse_scaffold_pair"] == 5 * p["quantum_multiply"] == 315.0
    assert p["divide_scaffold_pair"] > p["inverse_scaffold_pair"]


def test_slopes_at_benchmark_widths():
    s = _record()["slopes"]
    assert s["per_iteration"] == 17208.0
    assert s["per_position"] == 351.0
    assert s["toffoli_iter2_err255"] == 192697
    assert s["toffoli_iter4_err255"] == 227113
    assert s["toffoli_iter2_err511"] == 282553


def test_forney_inversion_alone_exceeds_the_whole_target():
    """4,095 inversions cost 2.9x the target T_mean, before any EEA iteration."""
    d = _record()["derived"]
    tot = d["forney_inversion_total"]
    assert tot["toffoli"] == 1_289_925.0
    assert tot["multiple_of_entire_target_T"] > 2.0
    assert 20.0 < tot["share_of_T_mean_percent"] < 25.0


def test_no_optimisation_of_that_phase_reaches_the_target():
    d = _record()["derived"]
    assert d["target_factor_required"] > 12.0
    assert max(d["ceilings_if_phase_became"].values()) < 1.5, d["ceilings_if_phase_became"]


def test_batch_inversion_never_pays_at_this_qubit_price():
    d = _record()["derived"]
    assert d["batch_inversion_ever_pays"] is False
    assert all(row["pays"] is False for row in d["batch_inversion"])
    assert all(row["net_toffoli_equivalents"] < 0 for row in d["batch_inversion"])
    # Block size 4 is the least-bad; the qubit price kills it before the Toffoli saving pays.
    best = max(d["batch_inversion"], key=lambda r: r["net_toffoli_equivalents"])
    assert best["block"] == 4
    assert best["score_ratio"] > 1.0, "the score must get worse, not better"


def test_the_104_qubit_target_needs_most_of_the_transient_headroom():
    q = _record()["derived"]["qubit_floor"]
    assert q["persistent_registers"] == 2 * 73 * 12 == 1752
    assert q["measured"] == 1887
    assert q["transient_headroom"] == 135
    assert q["target_reduction"] == 104
    # 104 of 135 is 77% of every scratch, control and counter qubit in the circuit.
    assert q["target_reduction"] / q["transient_headroom"] > 0.75


def test_remeasurement_matches_the_record_when_kickmix_is_importable():
    sys.path.insert(0, str(BUNDLE / "kickmix"))
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    try:
        import kickmix  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure of this
        # extension means "not importable in this interpreter".
        pytest.skip(f"kickmix not importable here: {exc}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_dqi_primitive_costs as m

    km, DR = m._imports()
    primitives = m.measure_primitives(km)
    slopes = m.measure_slopes(km, DR)
    rec = _record()
    for key, value in primitives.items():
        assert value == rec["primitives_per_call"][key], key
    for key in ("per_iteration", "per_position", "toffoli_iter2_err255",
                "toffoli_iter4_err255", "toffoli_iter2_err511"):
        assert slopes[key] == rec["slopes"][key], key
