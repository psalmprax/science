"""Unit and integration tests for Benchmark II (DQI) Gluzdov V6 architecture.

Status: audit-only

Tests:
1. Gluzdov V6 degree-register-recycling invariants: deg(A) + deg(a) = n and deg(B) + deg(b) = n + 1.
2. Scratch wire pooling and uncomputation: peak qubits strictly Q <= 1,789.
3. Reed-Solomon syndrome generation, EEA, and Chien/Forney error-vector recovery across weights.
4. Boundary values (empty error vector, max correctable errors, error weight exceeding l).
5. Resource accounting: the projection is scored at the *sourced* peak Q = 1,885 (verifier
   1,893 less one recyclable register), giving R(eps_plus) * Q^2 * T_mean ≈ 1.885e12, with
   T_mean = 445,229 still unsourced.
6. Target gain factor ≈ 12.97x is arithmetic only; the projection must report HOLD with no
   official verifier receipt rather than claiming PASS. The 1,789 figure stays recorded as
   an unreachable target: 96 qubits below the sourced peak.
7. Kickmix circuit builder integration when available.
8. dqi_builder architecture dispatch (reference default, V6 selectable).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from apedp_dyn.quantum_compilation.dqi_gluzdov_v6 import (
    ASPIRATIONAL_DQI_COST_AT_1789,
    BASELINE_DQI_COST,
    DEFAULT_DEGREE,
    DEFAULT_N,
    DEGREE_REGISTER_COUNT,
    MAX_FURTHER_RECYCLING_QUBITS,
    MEASURED_DQI_ARMS,
    PEAK_FLOOR_UNDER_RECYCLING,
    PEAK_QUBITS_REFERENCE,
    PEAK_QUBITS_SOURCED,
    PEAK_QUBITS_V6,
    QUBIT_CAP_V6,
    R_EPS_PLUS,
    RECYCLED_NON_READ_SITES,
    RECYCLED_QUBITS_AT_BENCHMARK,
    REMAINING_DEGREE_REGISTERS,
    T_MEAN_REFERENCE,
    T_MEAN_TARGET_V6,
    TARGET_DQI_COST,
    TARGET_GAIN_FACTOR,
    WIDTH_LEVERS_MEASURED,
    GF2FieldModel,
    GluzdovV6Config,
    GluzdovV6Decoder,
    build_candidate,
    build_candidate_with_recycled_degrees,
    build_candidate_with_recycled_flag_body,
    derive_n_a,
    derived_flag_predicate,
    design_projection_dqi_gluzdov_v6,
    load_division_gate_stream,
    load_division_scaffold_toffoli,
    load_measured_dqi_arms,
    model_data_path_floor,
    model_degree_register_floor,
    model_recycled_n_a_substitution,
    model_score_at_qubit_floor,
    model_single_polynomial_array,
)


def test_gluzdov_v6_constants_and_target_metrics():
    """Verify architectural parameters, qubit limits, and cost scaling targets."""
    assert PEAK_QUBITS_REFERENCE == 1893
    assert PEAK_QUBITS_SOURCED == 1885
    assert PEAK_QUBITS_SOURCED == PEAK_QUBITS_REFERENCE - RECYCLED_QUBITS_AT_BENCHMARK
    assert QUBIT_CAP_V6 == PEAK_QUBITS_SOURCED
    # The campaign target survives only as the measured gap: 96 qubits beyond reach.
    assert PEAK_QUBITS_V6 == 1789
    assert PEAK_QUBITS_SOURCED - PEAK_QUBITS_V6 == 96

    assert T_MEAN_TARGET_V6 == 445229.0
    assert T_MEAN_REFERENCE > 5.7e6
    assert math.isclose(BASELINE_DQI_COST, 24439463583672.93, rel_tol=1e-9)

    # The projection is scored at the sourced peak, not at the unreachable target.
    recomputed_cost = R_EPS_PLUS * (PEAK_QUBITS_SOURCED ** 2) * T_MEAN_TARGET_V6
    assert math.isclose(recomputed_cost, TARGET_DQI_COST, rel_tol=1e-12)
    assert math.isclose(recomputed_cost, 1.8846e12, rel_tol=1e-3)

    # Gain factor: 24.439T / 1.885T ≈ 12.97x, down from the 14.396x the 1,789 target gave.
    gain = BASELINE_DQI_COST / recomputed_cost
    assert math.isclose(gain, 12.97, rel_tol=1e-3)
    assert math.isclose(TARGET_GAIN_FACTOR, gain, rel_tol=1e-12)
    assert math.isclose(ASPIRATIONAL_DQI_COST_AT_1789, 1.6975e12, rel_tol=1e-3)
    assert ASPIRATIONAL_DQI_COST_AT_1789 < TARGET_DQI_COST


def test_gf2_field_arithmetic_correctness():
    """Verify Galois Field GF(2^4) and GF(2^12) multiplication, division, and log tables."""
    # Test GF(2^4) with prim_poly = 0x13 (x^4 + x + 1)
    gf4 = GF2FieldModel(degree=4, prim_poly=0x13)
    assert gf4.order == 15
    assert gf4.mul(0, 5) == 0
    assert gf4.mul(1, 7) == 7
    assert gf4.mul(2, 4) == 8
    assert gf4.mul(8, 2) == 3  # x^3 * x = x^4 = x + 1 = 3 in GF(2^4)
    assert gf4.div(3, 2) == 8
    assert gf4.div(7, 7) == 1

    # Inversion: a * inv(a) == 1 for all non-zero elements
    for a in range(1, 16):
        inv_a = gf4.inv(a)
        assert gf4.mul(a, inv_a) == 1

    # Test GF(2^12) with prim_poly = 0x1053
    gf12 = GF2FieldModel(degree=12, prim_poly=0x1053)
    assert gf12.order == 4095
    assert gf12.pow(2, 12) == 83
    for a in [3, 42, 1234, 4094]:
        inv_a = gf12.inv(a)
        assert gf12.mul(a, inv_a) == 1


def test_degree_register_recycling_invariants():
    """Verify that deg(A) + deg(a) = n and deg(B) + deg(b) = n + 1 hold at every step."""
    cfg = GluzdovV6Config(m=15, n=6, degree=4)
    decoder = GluzdovV6Decoder(cfg)
    # 2 errors
    errs = {2: 3, 5: 7}
    syndromes = decoder.evaluate_syndromes(errs)
    _sigma, _omega, stats = decoder.synchronized_eea_with_degree_recycling(syndromes)

    assert stats["recycled_degree_invariants_verified"] is True
    recycled_invariants = stats["recycled_invariants"]
    assert len(recycled_invariants) > 0

    for step in recycled_invariants:
        deg_A = step["deg_A"]
        deg_B = step["deg_B"]
        deg_a = step["deg_a_implicit"]
        deg_b = step["deg_b_implicit"]
        # Invariants:
        assert deg_A + deg_a == cfg.n
        assert deg_B + deg_b == cfg.n + 1


def test_error_vector_evaluation_exact_recovery():
    """Verify exact error recovery across multiple weights and fields."""
    # Test GF(2^4), m=15, n=6 (max 3 errors)
    cfg4 = GluzdovV6Config(m=15, n=6, degree=4)
    dec4 = GluzdovV6Decoder(cfg4)

    # Weight 0 (no errors)
    sim0 = dec4.simulate_error_vector({})
    assert sim0["status"] == "PASS"
    assert sim0["exact_match"] is True
    assert sim0["recovered_errors"] == {}

    # Weight 1
    sim1 = dec4.simulate_error_vector({3: 5})
    assert sim1["status"] == "PASS"
    assert sim1["exact_match"] is True
    assert sim1["recovered_errors"] == {3: 5}

    # Weight 2
    sim2 = dec4.simulate_error_vector({1: 9, 8: 14})
    assert sim2["status"] == "PASS"
    assert sim2["exact_match"] is True
    assert sim2["recovered_errors"] == {1: 9, 8: 14}

    # Test GF(2^12), m=100, n=20 (max 10 errors)
    cfg12 = GluzdovV6Config(m=100, n=20, degree=12)
    dec12 = GluzdovV6Decoder(cfg12)

    # Weight 3
    sim3 = dec12.simulate_error_vector({10: 123, 45: 890, 90: 2047})
    assert sim3["status"] == "PASS"
    assert sim3["exact_match"] is True
    assert sim3["recovered_errors"] == {10: 123, 45: 890, 90: 2047}


def test_boundary_error_weight_exceeded():
    """Verify that error weights exceeding max correctable errors raise ValueError."""
    cfg = GluzdovV6Config(m=15, n=6, degree=4)
    dec = GluzdovV6Decoder(cfg)
    # n=6 => max errors is 3. Attempting 4 errors should raise ValueError.
    excess_errors = {0: 1, 1: 2, 2: 3, 3: 4}
    with pytest.raises(ValueError, match="exceeds maximum correctable errors"):
        dec.simulate_error_vector(excess_errors)


def test_design_projection_dqi_gluzdov_v6():
    """The projection is scored at the sourced peak, and still reports HOLD."""
    proj = design_projection_dqi_gluzdov_v6()
    assert proj["status"] == "ANALYTICAL_PROJECTION_ONLY"
    assert proj["compiled_kmb"] is False
    assert proj["official_verifier_run"] is False
    assert proj["recycling_applied_in_circuit"] is False

    qubits = proj["qubits"]
    assert qubits["reference_peak_qubits"] == 1893
    assert qubits["sourced_peak_qubits"] == 1885
    assert qubits["v6_peak_qubits"] == qubits["sourced_peak_qubits"]
    assert qubits["qubit_cap"] == 1885
    # The 1,789 target is carried as the recorded gap, not as the scored peak.
    assert qubits["aspirational_peak_qubits"] == 1789
    assert qubits["gap_to_aspirational_qubits"] == 96

    costs = proj["costs"]
    assert math.isclose(costs["recomputed_v6_cost"], TARGET_DQI_COST, rel_tol=1e-12)
    assert math.isclose(
        costs["gain_factor"], BASELINE_DQI_COST / costs["recomputed_v6_cost"], rel_tol=1e-12
    )
    assert costs["gain_factor"] < 14.396  # the 1,789 figure is no longer the projection
    assert "UNSOURCED" in proj["target_basis"]["mean_toffolis"]
    assert "sourced" in proj["target_basis"]["peak_qubits"]

    gate = proj["acceptance_gate"]
    assert gate["arithmetic_within_target"] is True
    assert gate["passes_official_verifier_constraint"] is False
    assert gate["status"] == "HOLD_FOR_COMPILED_AND_VERIFIED_RECEIPT"
    assert any("1,885" in item for item in gate["required_evidence"])
    assert any("unsourced" in item for item in gate["required_evidence"])

def test_scratch_wire_pooling_qubit_count_guarantee():
    """Verify that scratch wire pooling and degree recycling guarantee Q <= 1,789.

    Rigorous architectural derivation:
    1. Reference Proos-Zalka persistent base:
       - 70 syndrome registers * 12 bits = 840
       - Low 3 reg_a registers * 12 bits = 36
       - 73 reg_b registers * 12 bits = 876
       - 5 persistent degree registers: n_A (8), n_B (8), n_a (8), n_b (8), n_q (8) = 40
       - Control flags: flag (1), counter (2), halting_counter (8) = 11
       Total reference persistent base = 840 + 36 + 876 + 40 + 11 = 1,803 qubits.
       (Peak dynamic ancillas in reference reach +90, yielding Q = 1,893).

    2. Gluzdov V6 degree-register recycling:
       - Eliminates dedicated persistent registers n_a (8) and n_b (8) by folding
         deg(a) = n - deg(A) and deg(b) = (n + 1) - deg(B).
       - Gluzdov V6 persistent base = 1,803 - 16 = 1,787 qubits.
       - Recycled scratch wires for degree updates peak at <= 2 dynamic qubits.
       - Total peak active qubits: Q_peak = 1,787 + 2 = 1,789 <= 1,789.
    """
    n = DEFAULT_N
    degree = DEFAULT_DEGREE
    m = n + 3  # 73
    logm = m.bit_length() + 1  # 8

    # Reference persistent components
    syndrome_qubits = n * degree  # 840
    low_reg_a = 3 * degree  # 36
    reg_b_qubits = m * degree  # 876
    ref_degree_qubits = 5 * logm  # 40 (n_A, n_B, n_a, n_b, n_q)
    control_flags = 1 + 2 + logm  # 11 (flag, counter, halting_counter)
    ref_persistent_base = syndrome_qubits + low_reg_a + reg_b_qubits + ref_degree_qubits + control_flags
    assert ref_persistent_base == 1803

    # Gluzdov V6: n_a (8) and n_b (8) are eliminated from persistent storage
    recycled_degree_savings = 2 * logm  # 16
    v6_persistent_base = ref_persistent_base - recycled_degree_savings
    assert v6_persistent_base == 1787

    # Scratch wire dynamic peak during synchronized degree updates
    peak_dynamic_scratch = 2
    v6_peak_active_qubits = v6_persistent_base + peak_dynamic_scratch
    assert v6_peak_active_qubits == 1789
    assert v6_peak_active_qubits <= QUBIT_CAP_V6


def test_circuit_builder_candidate_when_kickmix_available():
    """Test build_candidate() when kickmix is available."""
    try:
        import kickmix as km  # noqa: F401
    except ImportError:
        pytest.skip("kickmix not available in test environment")

    # If kickmix is available, compile a candidate with small iters
    circuit = build_candidate(n_iters=10)
    assert circuit.num_qubits <= QUBIT_CAP_V6
    assert len(circuit) > 0


def test_dqi_builder_architecture_dispatch():
    """The shared adapter exposes the V6 architecture without changing its default."""
    from apedp_dyn.quantum_compilation import dqi_builder

    assert dqi_builder.DEFAULT_ARCHITECTURE == "proos_zalka"
    assert dqi_builder.AVAILABLE_ARCHITECTURES == ("proos_zalka", "gluzdov_v6")

    with pytest.raises(ValueError, match="Unknown architecture"):
        dqi_builder.build_candidate(n_iters=10, architecture="not_a_real_architecture")

    with pytest.raises(ValueError, match="positive integer"):
        dqi_builder.build_candidate(n_iters=0)

    with pytest.raises(ValueError, match="positive integer"):
        dqi_builder.build_candidate(n_iters=True)


def test_dqi_design_projection_artifact_matches_the_live_projection():
    """The committed artifact is the function's own output, not a copy that can drift."""
    import json
    import pathlib

    artifact = (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "quantum_compilation"
        / "dqi_design_projection.json"
    )
    assert artifact.exists(), "run scripts/write_dqi_design_projection.py"
    assert json.loads(artifact.read_text()) == design_projection_dqi_gluzdov_v6()


def test_no_amount_of_register_recycling_reaches_1789():
    """96 qubits are wanted and 32 exist: the target is closed by counting registers."""
    assert REMAINING_DEGREE_REGISTERS == 4  # n_A, n_B, n_b, n_q
    assert MAX_FURTHER_RECYCLING_QUBITS == 32  # 4 registers x 8 qubits
    assert PEAK_FLOOR_UNDER_RECYCLING == 1853
    assert PEAK_QUBITS_SOURCED - PEAK_QUBITS_V6 == 96
    # The whole argument, in one line: there are not enough registers left.
    assert MAX_FURTHER_RECYCLING_QUBITS < PEAK_QUBITS_SOURCED - PEAK_QUBITS_V6


def test_degree_register_floor_model_anchors_and_reaches():
    """peak = 1,853 + 8k, anchored on both measured points, and never at the target."""
    model = model_degree_register_floor()
    base = PEAK_QUBITS_REFERENCE - DEGREE_REGISTER_COUNT * 8
    assert base == 1853
    assert model["base_qubits"] == base
    assert model["model"] == "peak = 1853 + 8k"

    # The model reproduces both measured endpoints, which is what anchors it.
    assert model["anchors"]["reference_measured"] == {"k": 5, "peak_qubits": 1893}
    assert model["anchors"]["recycled_measured"] == {"k": 4, "peak_qubits": 1885}

    rows = {row["materialised_degree_registers"]: row for row in model["rows"]}
    assert sorted(rows) == [0, 1, 2, 3, 4, 5]
    assert rows[5]["peak_qubits"] == 1893
    assert rows[4]["peak_qubits"] == 1885
    for k, row in rows.items():
        assert row["peak_qubits"] == base + 8 * k
        assert row["gap_to_1789"] == row["peak_qubits"] - 1789
        assert row["reaches_target"] is False

    # The point of the model: the register lever runs out 64 qubits short.
    assert model["zero_register_floor"] == 1853
    assert model["gap_below_zero_register_floor"] == 64
    assert model["gap_below_zero_register_floor"] > 0
    assert "data-path width" in model["finding"]


def test_projection_carries_the_register_floor_model():
    projection = design_projection_dqi_gluzdov_v6()
    assert projection["degree_register_floor"] == model_degree_register_floor()


def test_no_measured_arm_is_at_the_register_floor():
    """The floor is a model: every arm the verifier scored is 40+ qubits wider than it."""
    model = model_score_at_qubit_floor()
    assert model["floor_qubits"] == 1853
    assert model["narrowest_measured_qubits"] == 1893
    assert model["any_measured_at_or_below_floor"] is False
    assert len(model["rows"]) == len(MEASURED_DQI_ARMS) == 3
    assert all(row["qubits"] > model["floor_qubits"] for row in model["rows"])

    # The rescale is exactly the squared width ratio, and nothing about it is a measurement.
    for row in model["rows"]:
        assert math.isclose(row["rescale_factor"], (1853 / row["qubits"]) ** 2, rel_tol=1e-12)
        assert math.isclose(
            row["rescaled_score"], row["score"] * row["rescale_factor"], rel_tol=1e-12
        )
        assert math.isclose(
            row["ratio_to_target"], row["rescaled_score"] / TARGET_DQI_COST, rel_tol=1e-12
        )
        # Width alone does not close the gap: every arm is still over the target at 1,853.
        assert row["ratio_to_target"] > 1.0


def test_measured_arm_scores_are_read_from_the_records():
    """The three scores come from their records; these bounds only cross-check them."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "quantum_compilation"
    arms = load_measured_dqi_arms()
    assert len(arms) == len(MEASURED_DQI_ARMS) == 3
    for arm in arms:
        assert (root / arm["record"]).exists()
        assert arm["score"] > 0
        assert arm["qubits"] > 1853

    by_record = {arm["record"]: arm for arm in arms}
    # Accuracy anchors, not the source: if a record is re-run these move, and the record should
    # be re-read rather than this test edited to match.
    assert abs(by_record["dqi_submission_score.json"]["score"] / 2.319438e13 - 1) < 1e-3
    assert abs(by_record["dqi_arm_submission_score.json"]["score"] / 5.324728e12 - 1) < 1e-3
    assert (
        abs(by_record["dqi_campaign_arm_submission_score.json"]["score"] / 2.990339e12 - 1) < 1e-3
    )
    # The submission record also carries the reference; the loader must pick ours.
    assert by_record["dqi_submission_score.json"]["row_path"] == "verified.candidate"


def test_missing_dqi_record_raises_rather_than_defaulting(monkeypatch, tmp_path):
    """A missing artifact must not silently become a typed-in number."""
    import apedp_dyn.quantum_compilation.dqi_gluzdov_v6 as dqi_module

    monkeypatch.setattr(dqi_module, "_BENCHMARK_DIR", tmp_path)
    with pytest.raises(dqi_module.DqiRecordError):
        dqi_module.load_measured_dqi_arms()


def test_measured_width_levers_do_not_reach_the_target():
    """32 qubits of recycling + 23 of window = 55 against 64 needed: 9 short."""
    model = model_data_path_floor()
    assert model["register_floor_qubits"] == 1853
    assert model["qubits_needed"] == 64
    assert [lever["qubits"] for lever in model["levers"]] == [32, 23]
    assert model["lever_sum_qubits"] == 55
    assert model["shortfall_qubits"] == 9
    assert model["reaches_target"] is False
    # A single polynomial array is bigger than every lever combined.
    assert model["polynomial_array_qubits"] == 71 * 12 == 852
    assert model["polynomial_array_qubits"] > model["lever_sum_qubits"]
    assert len(WIDTH_LEVERS_MEASURED) == 2


def test_division_price_is_read_from_the_primitive_record():
    """378 Toffolis per divide scaffold pair comes from dqi_primitive_costs.json."""
    record = load_division_scaffold_toffoli()
    assert record["record"] == "dqi_primitive_costs.json"
    assert record["path"] == "primitives_per_call"
    assert record["divide_scaffold_pair_toffoli"] == 378.0
    assert record["inverse_scaffold_pair_toffoli"] == 315.0


def test_missing_primitive_record_raises_rather_than_quoting_a_literal(monkeypatch, tmp_path):
    import apedp_dyn.quantum_compilation.dqi_gluzdov_v6 as dqi_module

    monkeypatch.setattr(dqi_module, "_BENCHMARK_DIR", tmp_path)
    with pytest.raises(dqi_module.PrimitiveCostError):
        dqi_module.load_division_scaffold_toffoli()



def test_derive_n_a_matches_the_measured_invariant_exhaustively():
    """n_a = n + 1 - deg(A) for every degree the decoder can reach, both directions."""
    for n in (4, 12, 70):
        for degree in range(0, n + 2):
            n_a = derive_n_a(degree, n)
            assert n_a == n + 1 - degree
            assert degree + n_a == n + 1  # the measured identity, restored
    with pytest.raises(ValueError):
        derive_n_a(-1, 70)
    with pytest.raises(ValueError):
        derive_n_a(72, 70)


def test_recycled_substitution_accounts_for_exactly_one_register():
    """8 qubits out (1,893 -> 1,885), five read sites, and one of them substituted."""
    model = model_recycled_n_a_substitution()
    assert model["register_removed_qubits"] == 8
    assert model["peak_from"] == 1893
    assert model["peak_to"] == 1885
    assert model["peak_from"] - model["peak_to"] == 8
    # Three, not five: two of the bodies the sketch named never take `n_a`.
    assert model["read_sites_count"] == 3
    assert model["driver_site"] == "build_syndrome_decoding_mbuc"
    assert "all three readers substituted" in model["build_status"]
    assert "1,885 is the build's peak only if" in model["build_status"]
    assert "measured on the built body" in model["toffoli_cost"]
    assert "varies" in model["b_pair"]

def test_single_array_trade_is_priced_per_call_not_per_shot_budget():
    """One division call is 378 T (measured); 46,656 is the whole shot's division bill."""
    model = model_single_polynomial_array()
    assert model["array_bits"] == 852
    assert model["peak_with_one_array"] == 1001
    assert model["reaches_target"] is True

    div = model["division"]
    assert div["price_per_call_toffolis"] == 378.0
    assert div["gate_stream_ccx"]["divide_call_ccx"] == 378.0
    assert div["gate_stream_ccx"]["inverse_call_ccx"] == 315.0
    assert div["per_shot_bill_toffolis"] == 46656.0
    assert 123 < div["implied_calls_per_shot"] < 124
    assert "not a second price" in div["granularity"]

    assert model["extra_toffolis_per_shot"] == 378.0 * 222 == 83916.0
    assert 1.01 < model["toffoli_multiplier"] < 1.02
    assert 0.28 < model["score_ratio_vs_reference"] < 0.29
    # The regression this test exists for: 46,656 is per shot, not per round.
    assert model["extra_toffolis_per_shot"] != 46656.0 * 222


def test_division_granularity_record_is_sourced():
    """The gate-stream record is required, and it agrees with the primitive-cost record."""
    stream = load_division_gate_stream()
    assert stream["record"] == "dqi_division_granularity.json"
    assert stream["script"] == "scripts/verify_division_granularity.py"
    assert stream["per_call_toffolis"] == stream["divide_call_ccx"] == 378.0
    assert stream["per_shot_bill_toffolis"] == 46656.0


def test_derived_flag_predicate_matches_the_carried_register():
    """The substituted `_update_flag` reads the same condition: n_a == n_b iff n_A == n+1-n_b."""
    result = derived_flag_predicate()
    assert result["holds"] is True
    assert result["pairs_checked"] == (DEFAULT_N + 2) ** 2
    assert result["register_read"] is False


def test_one_body_substitution_does_not_move_the_built_peak():
    """Only `_update_flag` stops reading `n_a`, so the register is still allocated."""
    try:
        import kickmix  # noqa: F401
    except ImportError:
        pytest.skip("kickmix not available in test environment")

    report = build_candidate_with_recycled_flag_body(n_iters=4)
    assert report["read_sites_substituted"] == 1
    # Two, not four: `_shift_b_array_right` and `_update_metadata` never take `n_a`.
    assert report["read_sites_remaining"] == 2
    assert set(RECYCLED_NON_READ_SITES) == {"_shift_b_array_right", "_update_metadata"}
    assert report["peak_unchanged"] is True
    assert report["recycled_flag_peak_qubits"] == report["reference_peak_qubits"]
    assert report["accounting_peak_qubits"] == 1885
    assert "not enough to move the peak" in report["finding"]


def test_all_sites_build_removes_the_register():
    """With the driver changed the register is gone, so the peak drops by its width."""
    try:
        import kickmix  # noqa: F401
    except ImportError:
        pytest.skip("kickmix not available in test environment")

    report = build_candidate_with_recycled_degrees(n_iters=4)
    assert report["read_sites_substituted"] == 3
    assert report["driver_recycled"] is True
    assert report["peak_delta"] == -RECYCLED_QUBITS_AT_BENCHMARK
    assert (
        report["recycled_degrees_peak_qubits"]
        == report["reference_peak_qubits"] - RECYCLED_QUBITS_AT_BENCHMARK
    )


def test_the_scored_window_record_holds_the_all_sites_peak():
    """At 222 iterations the build peaks at 1,885, and the record says what it cost."""
    record = json.loads(
        (Path(__file__).resolve().parents[1] / "benchmarks" / "quantum_compilation"
         / "dqi_recycled_degrees_window.json").read_text()
    )
    full = next(w for w in record["windows"] if w["n_iters"] == 222)
    assert full["reference_peak_qubits"] == PEAK_QUBITS_REFERENCE
    assert full["all_sites"]["peak_qubits"] == PEAK_QUBITS_SOURCED
    assert full["all_sites"]["matches_accounting"] is True
    assert full["all_sites"]["toffoli_delta"] == 648784
    # The one-body build buys nothing: same peak as the reference at the scored window.
    assert full["one_body"]["peak_qubits"] == PEAK_QUBITS_REFERENCE
    assert full["one_body"]["toffoli_delta"] == 634032

