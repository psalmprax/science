"""Behavioural pins for the MR-gated DQI arms on the rebuilt ``kickmix``.

``scripts/run_mr_gated_arms_kickmix.py`` runs the arms on a locally compiled,
MR-capable ``kickmix`` instead of the local model. That extension is a build
artifact (``scripts/build_kickmix_mr.py`` writes it to ``build/kickmix_mr/lib``),
so these pins skip — rather than fail — when it has not been built for this
interpreter. Nothing here touches the pinned competition binary.

The pins are the ones the runner's docstring promises: every arm reproduces the
reference on the real builder and simulator, and both falsifiers fail.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_mr_gated_arms_kickmix.py"
MEASURE = REPO_ROOT / "scripts" / "measure_mr_arm_costs_kickmix.py"
BUILD_DIR = REPO_ROOT / "build" / "kickmix_mr" / "lib"
COST_RECORD = REPO_ROOT / "benchmarks" / "quantum_compilation" / "kickmix_mr_arm_costs.json"
HEADROOM_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                   / "gf_mul_and_static_headroom.json")
DESIGNER_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                   / "designer_directions.json")
COMBINED = REPO_ROOT / "scripts" / "measure_dqi_combined_arm.py"
COMBINED_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                   / "dqi_combined_arm.json")
ARMS = ("swap", "sweep", "forney", "clear", "shift", "quotient", "all", "full")

#: Small instance for the cost pin: 0.7 s against the official run's minutes.
SMALL_INSTANCE = ("--m", "15", "--n", "6", "--degree", "4", "--shots", "8")

#: Toffoli per shot measured at `SMALL_INSTANCE`. This pins the gate stack, not the
#: decoder: a lever that stops firing (or one that skips work it needs) moves its row
#: by far more than the 1% tolerance, and the official 64-shot record need not be rerun
#: to see it.
SMALL_INSTANCE_T = {
    "reference": 33_235.5,
    "forney": 32_830.0,
    "forney+clear": 16_482.0,
    "forney+clear+sweep": 12_700.5,
    "forney+clear+sweep+swap": 11_647.0,
    "all": 10_890.0,
    "all+quotient": 10_409.5,
}


def _mr_extension_available() -> bool:
    """True when this interpreter can import a kickmix extension with ``mr``."""
    if not any(BUILD_DIR.glob("kickmix*.so")):
        return False
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import kickmix as km; "
        "raise SystemExit(0 if hasattr(km.CircuitBuilder, 'mr') else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(BUILD_DIR)],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


MR_EXTENSION = _mr_extension_available()
requires_mr = pytest.mark.skipif(
    not MR_EXTENSION,
    reason="MR-capable kickmix not built for this interpreter "
           "(run scripts/build_kickmix_mr.py)",
)


def _run(*args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--json", *args],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.stdout.strip(), f"runner produced no JSON:\n{proc.stderr}"
    return proc.returncode, json.loads(proc.stdout)


@requires_mr
@pytest.mark.parametrize("arm", ARMS)
def test_arm_reproduces_reference_on_kickmix(arm):
    """Each arm must restore the syndromes and leave the phase and scratch alone."""
    code, result = _run("--arm", arm)
    assert result["engine"] == "kickmix"
    assert "build/kickmix_mr" in result["kickmix"], "pinned build leaked into the test"
    assert result["syndromes_match_reference"] is True
    assert result["phase_matches_reference"] is True
    assert result["syndromes_restored"] is True
    assert result["junk_clean"] is True
    assert result["verdict"] == "PASS"
    assert code == 0


@requires_mr
def test_reference_alone_is_clean_on_kickmix():
    """The ungated reference is the baseline the arms are compared against."""
    code, result = _run("--arm", "none")
    assert result["syndromes_restored"] is True
    assert result["junk_clean"] is True
    assert result["verdict"] == "PASS"
    assert code == 0


@requires_mr
def test_reset_falsifier_fails_on_kickmix():
    """``R`` clears like ``mr`` but randomizes the phase, so the arm must fail."""
    code, result = _run("--arm", "swap", "--clear-with", "r")
    assert result["phase_matches_reference"] is False
    assert result["verdict"] == "FAIL"
    assert code == 1


@requires_mr
def test_under_gated_shift_fails_on_kickmix():
    """Dropping the ¬flag∧(c0∨c3) branch skips shifts the decoder needs."""
    code, result = _run("--arm", "shift", "--shift-under")
    assert result["syndromes_restored"] is False
    assert result["verdict"] == "FAIL"
    assert code == 1


@requires_mr
def test_small_instance_cost_pin_catches_gate_regressions(tmp_path):
    """Re-price the stacked gates at m=15, so a dead lever fails without the 64-shot run."""
    record = tmp_path / "costs.json"
    proc = subprocess.run(
        [sys.executable, str(MEASURE), *SMALL_INSTANCE, "--record", str(record)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(record.read_text())
    assert list(data["cumulative"]) == list(SMALL_INSTANCE_T)
    for name, pinned in SMALL_INSTANCE_T.items():
        entry = data["cumulative"][name]
        assert entry["correct"] is True, name
        assert entry["mean_toffoli_like"] == pytest.approx(pinned, rel=0.01), name
    # Every gate must still remove work, stage over stage.
    rows = [(name, data["cumulative"][name]["mean_toffoli_like"]) for name in SMALL_INSTANCE_T]
    from itertools import pairwise

    for (prev_name, prev), (name, count) in pairwise(rows):
        assert count < prev * 0.999, f"{name} does not improve on {prev_name}"
    # Reading a predicate must not cost qubits: the reads share one ancilla, so the
    # gated stacks pay at most one id over the reference instead of one per read.
    reference_qubits = data["cumulative"]["reference"]["qubits"]
    for name in SMALL_INSTANCE_T:
        assert data["cumulative"][name]["qubits"] <= reference_qubits + 1, name


def test_combined_control_arm_record_holds():
    """The stacked arm's record: 2,952 saved, not the 3,000 the two gates add to."""
    data = json.loads(COMBINED_RECORD.read_text())
    observation = data["observation"]
    assert observation["combined_delta_per_shot"] == pytest.approx(2_952.0, abs=0.5)
    assert observation["predicted_delta_per_shot"] == pytest.approx(2_952.0, abs=0.5)
    assert observation["naive_additive_per_shot"] == pytest.approx(3_000.0, abs=0.5)
    assert observation["model_matched"] == "interaction"
    assert observation["dead_rounds"] == 6
    arms = data["arms"]
    assert arms["control"]["mean_toffoli_like"] == pytest.approx(1_221_466.8, abs=0.5)
    assert arms["combined"]["mean_toffoli_like"] == pytest.approx(1_218_514.8, abs=0.5)
    for name in ("control", "halt_gated", "combined"):
        assert arms[name]["qubits"] == 1_913, name
        assert arms[name]["correct"] is True, name
    assert data["all_checks_passed"] is True
    assert data["verdict"] == "PROMOTE"
    assert data["scope"]["is_official_result"] is False


@requires_mr
def test_combined_control_arm_runs_on_the_rebuilt_extension(tmp_path):
    """Re-run the stack at the small instance so a dead gate fails without 64 shots."""
    record = tmp_path / "combined.json"
    proc = subprocess.run(
        [sys.executable, str(COMBINED), *SMALL_INSTANCE, "--record", str(record)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(record.read_text())
    assert data["all_checks_passed"] is True
    assert data["observation"]["model_matched"] == "interaction"
    # Both gates must still remove work, and the combined arm must remove more than
    # either one alone.
    arms = data["arms"]
    assert arms["combined"]["mean_toffoli_like"] < arms["halt_gated"]["mean_toffoli_like"]
    assert arms["halt_gated"]["mean_toffoli_like"] < arms["control"]["mean_toffoli_like"]
    assert arms["combined"]["correct"] is True


def test_designer_directions_record_still_holds():
    """The three-direction verdicts are pinned: A bounded, B designed, C refuted alone.

    The 54-AND multiply is the one open opportunity; it is recorded as a design, so this
    pin also keeps it from being quoted as a measured circuit until a circuit executes it.
    """
    data = json.loads(DESIGNER_RECORD.read_text())
    search = data["b_bilinear_search"]
    assert search["n2"]["rank"] == 3 and search["n2"]["proved"] is True
    assert search["n3"]["rank"] == 6 and search["n3"]["proved"] is True
    assert search["n3"]["mismatches"] == 0
    assert search["composed_and_count_12"] == 54
    assert search["karatsuba_builtin_and_count_12"] == 63
    assert search["composed_verified_against_field"]["mismatches"] == 0

    ceiling = data["a_predicate_ceiling"]
    assert ceiling["total_ceiling_percent"] == pytest.approx(2.71, abs=0.05)

    ladder = data["c_decoder_requirement"]["ladder"]
    assert ladder["63"]["ratio_vs_reference"] == pytest.approx(4.320, abs=0.01)
    # The 54-AND multiply is measured now (gf54_mul_kickmix.json), and the ladder
    # reproduces that measurement exactly: 54 ANDs is 4.652x, not the 4.928x the
    # AND-count scaling predicted, because a controlled multiply always pays its
    # 12-Toffoli add.
    assert ladder["54"]["ratio_vs_reference"] == pytest.approx(4.652, abs=0.01)
    derived = data["c_decoder_requirement"]["derived_from"]
    assert derived["multiply_calls_per_shot"] == 10_510
    assert derived["everything_else_per_shot"] == pytest.approx(536_980.0, abs=1.0)
    assert ladder["0"]["t_mean"] == pytest.approx(
        derived["everything_else_per_shot"]
        + derived["multiply_calls_per_shot"] * derived["controlled_add_per_call"])
    # The point of the ladder: no achievable multiply clears 445,229 on this decoder --
    # not even one that cost nothing but its controlled add.
    assert ladder["23"]["ratio_vs_target"] > 1.3
    assert ladder["0"]["ratio_vs_target"] > 1.0


def test_headroom_record_bounds_the_remaining_levers():
    """The multiply and the static stream are both at their floor, as recorded.

    A later "cheaper multiply" or "T-count optimiser" claim has to move one of these
    numbers. They are pinned here so it cannot be claimed without moving them, and so a
    regression (an accidental extra AND, a gate that stops being deleted) is visible
    without rerunning the 64-shot measurement.
    """
    data = json.loads(HEADROOM_RECORD.read_text())
    assert data["multiply"]["karatsuba_is_exact"] is True
    degrees = {int(degree): row for degree, row in data["multiply"]["degrees"].items()}
    for degree, row in degrees.items():
        assert row["measured"] == row["karatsuba"], degree
    assert degrees[12]["measured"] == 63
    splits = {k: v for k, v in data["splits_at_degree_12"].items() if k != "schoolbook"}
    assert min(splits.values()) == 63, splits
    assert data["splits_at_degree_12"]["schoolbook"] == 144

    static = data["static"]
    assert static["reference"]["toffoli_like"] == 5_725_461
    # Gating adds conditions, not deletions: the sixth lever must move static T by 0.
    assert static["six levers"]["toffoli_like"] == static["five levers"]["toffoli_like"]
    assert static["six levers"]["static_ratio_vs_reference"] == pytest.approx(1.642, abs=0.01)
    for label, row in data["peephole"].items():
        assert row["removable"] == 0, label

    # The share is measured by differencing two correct arms, not by deleting the
    # multiply: a deleted multiply leaves the registers wrong, so the predicates that
    # gate the rest of the EEA read garbage and the differenced circuit executes a
    # different amount of other work. The ablation's 86.33% was superseded here.
    executed = data["multiply"]["executed"]
    assert executed["multiply_share_percent"] == pytest.approx(59.48, abs=0.2)
    assert executed["executed_multiply_calls"] == 10_510
    assert executed["multiply_t_mean"] == pytest.approx(788_250.0, abs=1.0)
    assert executed["everything_else_t_mean"] == pytest.approx(536_980.0, abs=1.0)
    assert executed["ratio_with_free_multiply"] == pytest.approx(8.634, abs=0.02)
    at_45 = data["multiply"]["derived_if_cheaper"]["45_ands"]["ratio_vs_reference"]
    assert at_45 == pytest.approx(5.039, abs=0.02)


GF54_SCRIPT = REPO_ROOT / "scripts" / "measure_gf54_mul_kickmix.py"
GF54_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation" / "gf54_mul_kickmix.json")
NATIVE_SCRIPT = REPO_ROOT / "scripts" / "measure_gf54_native_opcode.py"
NATIVE_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation" / "gf54_native_opcode.json")


def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gf54_stack():
    """The MR-capable kickmix, the runner's builder, and the 54-AND mixin."""
    if str(BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(BUILD_DIR))
    import kickmix as km

    runner = _load_module(RUNNER, "mr_gated_arms_kickmix")
    mul = _load_module(REPO_ROOT / "scripts" / "gf54_mul.py", "gf54_mul")
    shim = runner.shim_for(km, runner.builder_class(km))
    return km, mul, shim, type("Gf54Builder", (mul.Gf54MulMixin, shim.CircuitBuilder), {})


def test_gf54_multiply_is_the_proved_algorithm():
    """The base cases are the exhaustive-search minima, composed to 54 ANDs at n = 12."""
    mul = _load_module(REPO_ROOT / "scripts" / "gf54_mul.py", "gf54_mul")
    assert mul.and_count(2) == 3
    assert mul.and_count(3) == 6
    assert mul.and_count(12) == 54
    products, outputs = mul.terms(tuple(frozenset([i]) for i in range(12)))
    assert len(products) == 54
    assert len(outputs) == 23
    # Carry-less product against schoolbook, and the reduced product against the field.
    for a in range(0, 4096, 37):
        for b in (0, 1, 0b101, 0b111111111111, (a * 7 + 3) & 0xFFF):
            expected = 0
            for i in range(12):
                if (a >> i) & 1:
                    expected ^= b << i
            assert mul.poly_product(a, b, 12) == expected


@requires_mr
def test_gf54_multiply_is_correct_as_a_circuit():
    """Correct on basis states, operands restored, workspace clean, phase untouched."""
    import numpy as np

    km, _mul, _shim, builder_cls = _gf54_stack()
    field = km.GF2Field(12)
    shots = 16
    rng = np.random.default_rng(11)
    lhs_in = rng.integers(0, 1 << 12, shots)
    rhs_in = rng.integers(0, 1 << 12, shots)
    control_in = rng.integers(0, 2, shots)

    builder = builder_cls()
    lhs = builder.create_quantum_register(12, name="L")
    rhs = builder.create_quantum_register(12, name="R")
    target = builder.create_quantum_register(12, name="T")
    control = builder.create_quantum_register(1, name="C")
    builder.ixor_gf2_mul(lhs, rhs, target=target, field=field, control=control[0])
    circuit = builder.finish_circuit()

    def bits(value):
        return np.array([(int(value) >> i) & 1 for i in range(12)], dtype=bool)

    sim = km.Simulator(batch_size=shots, seed=11)
    sim.use_same_registers_as(circuit)
    operands = np.array([np.concatenate([bits(lhs_in[i]), bits(rhs_in[i])])
                         for i in range(shots)])
    sim.write_across_shots(km.array([q for r in (lhs, rhs) for q in r]), operands.T)
    sim.write_across_shots(km.array([control[0]]), control_in.astype(bool).reshape(1, -1))
    sim.do(circuit)

    read = sim.read_across_shots(km.array(list(target)))
    got = (read.T * (1 << np.arange(12))).sum(axis=1)
    want = np.array([field.mul(int(lhs_in[i]), int(rhs_in[i])) if control_in[i] else 0
                     for i in range(shots)])
    assert np.array_equal(got, want)

    everything = sim.read_across_shots(
        km.array([km.q(i) for i in range(circuit.num_qubits)])).T
    kept = [q.id for r in (lhs, rhs) for q in r]
    known = set(kept) | {q.id for q in target} | {control[0].id}
    other = [i for i in range(circuit.num_qubits) if i not in known]
    assert np.array_equal(everything[:, kept], operands), "operands were not restored"
    assert not everything[:, other].any(), "workspace was not cleared"
    assert int(sim.read_phase_flipped_across_shots().sum()) == 0


@requires_mr
def test_gf54_multiply_costs_54_ands_and_66_controlled_toffolis():
    """The AND saving is real and the controlled add is not: 66 against the built-in 75."""
    km, _mul, shim, builder_cls = _gf54_stack()

    def cost(cls):
        builder = cls()
        lhs = builder.create_quantum_register(12, name="L")
        rhs = builder.create_quantum_register(12, name="R")
        target = builder.create_quantum_register(12, name="T")
        control = builder.create_quantum_register(1, name="C")
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=km.GF2Field(12),
                             control=control[0])
        circuit = builder.finish_circuit()
        stats = dict(circuit.operation_statistics())
        return int(stats.get("CCX", 0)), circuit.num_qubits

    gf54_ccx, gf54_qubits = cost(builder_cls)
    builtin_ccx, builtin_qubits = cost(shim.CircuitBuilder)
    assert gf54_ccx == 54 + 12, "54 ANDs plus the controlled add"
    assert builtin_ccx == 63 + 12, "63 ANDs plus the controlled add"
    # The saving is 12% of the multiply, not 14.3%: the add is in both circuits.
    assert (builtin_ccx - gf54_ccx) / builtin_ccx == pytest.approx(0.12, abs=0.005)
    # And it is paid for in workspace, which is why Q moves at all.
    assert gf54_qubits > builtin_qubits


@requires_mr
def test_gf54_runner_prices_both_arms(tmp_path):
    """The measurement script runs end to end and prices both arms on a small instance."""
    record = tmp_path / "gf54.json"
    proc = subprocess.run(
        [sys.executable, str(GF54_SCRIPT), "--m", "15", "--n", "6", "--degree", "4",
         "--shots", "4", "--checks", "4", "--record", str(record)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(record.read_text())
    assert data["function_check"]["gf54"]["value_mismatches"] == 0
    for label in ("gated_builtin", "gated_gf54"):
        assert data["arms"][label]["correct"] is True, label
    # At GF(2^4) both algorithms are 9 ANDs, so the two arms must be identical.
    assert data["arms"]["gated_gf54"]["mean_toffoli_like"] == pytest.approx(
        data["arms"]["gated_builtin"]["mean_toffoli_like"])


def test_gf54_record_matches_the_circuit():
    """The recorded swap must be the one the circuit performs, on the pinned instance."""
    data = json.loads(GF54_RECORD.read_text())
    assert data["unit_cost"]["gf54"]["ccx"] == 66
    assert data["unit_cost"]["builtin"]["ccx"] == 75
    assert data["function_check"]["gf54"]["value_mismatches"] == 0
    assert data["function_check"]["gf54"]["phase_flips"] == 0
    assert data["function_check"]["gf54"]["workspace_back_to_zero"] is True

    arms = data["arms"]
    assert arms["gated_builtin"]["correct"] is True
    assert arms["gated_gf54"]["correct"] is True
    assert arms["gated_builtin"]["mean_toffoli_like"] == pytest.approx(1_325_230.0, abs=1.0)
    assert arms["gated_builtin"]["ratio_vs_reference"] == pytest.approx(4.320, abs=0.01)
    assert arms["gated_gf54"]["mean_toffoli_like"] == pytest.approx(1_230_640.0, abs=1.0)
    assert arms["gated_gf54"]["ratio_vs_reference"] == pytest.approx(4.652, abs=0.01)

    decomposition = data["decomposition"]
    assert decomposition["executed_multiply_calls"] == 10_510
    assert decomposition["multiply_share_percent"] == pytest.approx(59.48, abs=0.2)
    assert decomposition["reconstructed_matches_measurement"] is True
    # The ladder in the same record is the arm priced for any multiply cost.
    ladder = data["ladder"]
    assert ladder["54"]["toffoli_per_shot"] == pytest.approx(
        arms["gated_gf54"]["mean_toffoli_like"], abs=1.0)
    assert ladder["63"]["toffoli_per_shot"] == pytest.approx(
        arms["gated_builtin"]["mean_toffoli_like"], abs=1.0)
    assert ladder["0"]["ratio_vs_target"] > 1.0


@requires_mr
def test_native_opcode_is_exposed_by_the_builder():
    """`gf_product` is the policy switch, and it reaches every product entry point.

    A Python mixin can only intercept `ixor_gf2_mul`; the reference's inversion and
    division emit their own products inside the extension. Selecting the generator
    through the builder is what makes those calls 54-AND too.
    """
    if str(BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(BUILD_DIR))
    import kickmix as km

    def cost(product):
        builder = km.CircuitBuilder()
        builder.gf_product = product
        lhs = builder.create_quantum_register(12, name="L")
        rhs = builder.create_quantum_register(12, name="R")
        target = builder.create_quantum_register(12, name="T")
        control = builder.create_quantum_register(1, name="C")
        builder.ixor_gf2_mul(lhs, rhs, target=target, field=km.GF2Field(12),
                             control=control[0])
        circuit = builder.finish_circuit()
        return int(dict(circuit.operation_statistics()).get("CCX", 0)), circuit.num_qubits

    karatsuba_ccx, karatsuba_qubits = cost("karatsuba")
    assert karatsuba_ccx == 63 + 12
    and54_ccx, and54_qubits = cost("and54")
    assert and54_ccx == 54 + 12
    # The workspace is the price: 19 bits of scratch plus m for the controlled temporary,
    # against the built-in 12. It was 54 until each sub-product started being reduced into
    # the target as it is computed instead of into a whole 2m-1 product window.
    assert and54_qubits - karatsuba_qubits == (12 + 19) - 12
    with pytest.raises(ValueError):
        km.CircuitBuilder().gf_product = "schoolbook"


@requires_mr
def test_native_opcode_prices_the_forney_inversion(tmp_path):
    """The opcode runs end to end and the inversion is where it reaches the reference."""
    record = tmp_path / "native.json"
    proc = subprocess.run(
        [sys.executable, str(NATIVE_SCRIPT), "--m", "15", "--n", "6", "--degree", "4",
         "--shots", "4", "--checks", "4", "--record", str(record)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(record.read_text())
    assert data["pinned_instance"] is False
    for name in ("multiply", "inversion", "division"):
        assert data["function_check"][name]["native_and54"]["value_mismatches"] == 0, name
    # The Forney arm is the same reference with only the inversion's products switched:
    # it must still decode, and its saving must be that block's own product count. At
    # GF(2^4) both algorithms are 9 ANDs, so both savings are legitimately zero here.
    isolated = data["forney_inversion"]["isolated"]
    assert isolated["check_toffoli_saved_is_nine_per_product"] is True
    assert isolated["percent_of_total_saving"] is None
    assert data["reference"]["saving_percent"] == pytest.approx(0.0)
    assert data["forney_inversion"]["everything_else_percent_of_total_saving"] is None


def test_native_opcode_record_is_internally_consistent():
    """The recorded native numbers, including the Forney split, must add up.

    No extension needed: this pins the record's arithmetic so the Forney share cannot
    drift back to an assertion. The block runs five products at every one of the
    reference's 4,095 positions, so the static site count is 20,475 of 9 Toffolis each;
    the isolated arm shows 4,093.9 positions execute on average, which is why the
    measured saving is 184,225 rather than the static 184,275.
    """
    data = json.loads(NATIVE_RECORD.read_text())
    assert data["pinned_instance"] is True
    units = data["unit_cost"]
    assert units["multiply"]["builtin"]["toffoli_like"] == 63
    assert units["multiply"]["and54"]["toffoli_like"] == 54
    assert units["inversion"]["builtin"]["toffoli_like"] == 315
    assert units["inversion"]["and54"]["toffoli_like"] == 270
    assert units["division"]["builtin"]["toffoli_like"] == 378
    assert units["division"]["and54"]["toffoli_like"] == 324
    # Scratch: one sub-product window at a time, not a whole 2m-1 product window.
    assert units["multiply"]["and54"]["workspace"] == 19
    assert units["controlled_multiply"]["and54"]["workspace"] == 12 + 19
    for name, row in data["function_check"].items():
        assert row["native_and54"]["value_mismatches"] == 0, name
        assert row["native_and54"]["operands_restored"] is True, name
        assert row["native_and54"]["workspace_back_to_zero"] is True, name
        assert row["native_and54"]["phase_flips"] == 0, name

    reference = data["reference"]
    assert reference["builtin"]["mean_toffoli_like"] == pytest.approx(5_725_071.6, abs=1.0)
    assert reference["and54"]["mean_toffoli_like"] == pytest.approx(5_201_222.5, abs=1.0)
    assert reference["saving_percent"] == pytest.approx(9.15, abs=0.02)
    assert reference["builtin"]["qubits"] == 1_893
    # 19 of workspace per multiply, not 42: 1,893 + 19.
    assert reference["and54"]["qubits"] == 1_912
    assert data["arm"]["and54"]["qubits"] == 1_913
    # Every executed product removes exactly nine, controlled or not.
    assert reference["mean_executed_product_calls"] * 9 == pytest.approx(
        reference["toffoli_saved_per_shot"], abs=1e-6)

    forney = data["forney_inversion"]
    assert forney["positions"] == 4_095
    assert forney["products_per_position"] == 5
    assert forney["product_sites"] == 20_475
    assert forney["static_toffoli_saved"] == 184_275
    isolated = forney["isolated"]
    assert 4_000 < isolated["mean_executed_positions"] <= 4_095
    assert isolated["toffoli_saved_per_shot"] == pytest.approx(
        isolated["mean_executed_product_calls"] * 9, abs=1e-6)
    assert 3.1 < isolated["percent_of_reference"] < 3.3
    assert 34.0 < isolated["percent_of_total_saving"] < 36.0
    # The split is the whole saving: Forney plus everything else, no remainder.
    assert (isolated["toffoli_saved_per_shot"]
            + forney["everything_else_toffoli_saved_per_shot"]) == pytest.approx(
        reference["toffoli_saved_per_shot"], abs=1e-6)

    arm = data["arm"]
    assert arm["builtin"]["mean_toffoli_like"] == pytest.approx(1_325_230.0, abs=1.0)
    assert arm["and54"]["mean_toffoli_like"] == pytest.approx(1_221_466.8, abs=1.0)
    assert arm["and54"]["ratio_vs_reference"] == pytest.approx(4.687, abs=0.01)
    assert arm["builtin"]["correct"] is True and arm["and54"]["correct"] is True
    # The Python emitter and the opcode differ: the opcode reaches the inversion too.
    assert arm["recorded_python_emitter_t_mean"] == pytest.approx(1_230_640.0, abs=1.0)
    assert arm["and54"]["mean_toffoli_like"] < arm["recorded_python_emitter_t_mean"]


RECONCILIATION_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                         / "dqi_resource_reconciliation.json")
TARGET_T_MEAN = 445_229.0


def test_the_arm_decomposes_into_three_measured_terms():
    """Products, controlled adds and the non-product stream are separate, measured parts.

    An earlier two-term model folded the 1,019.23 products inside the inversion and the
    division into its constant at 54 ANDs each, which made "free multiply" mean "free
    ixor only" and reported 1.186x where the honest figure is 1.062x. This pin makes the
    three terms explicit: every product executes 54 ANDs, only the 10,510 controlled
    multiplies also pay the 12-Toffoli add, and the residual is the EEA/predicate stream.
    """
    data = json.loads(NATIVE_RECORD.read_text())
    arm = data["arm"]
    iso = data["isolated_blocks"]["arm"]
    total_products = arm["mean_executed_product_calls"]
    inversion_products = iso["forney_inversion"]["mean_executed_products"]
    division_products = iso["division_and_sweep"]["mean_executed_products"]
    controlled = total_products - inversion_products - division_products
    assert controlled == pytest.approx(10_510.0, abs=1e-6)
    assert inversion_products == pytest.approx(155.234375, abs=1e-6)
    assert division_products == pytest.approx(864.0, abs=1e-6)

    residual = (arm["and54"]["mean_toffoli_like"]
                - controlled * (54 + 12)
                - (inversion_products + division_products) * 54)
    assert residual == pytest.approx(472_768.2, abs=0.5)
    # The three terms reproduce the measured arm, and the residual alone misses the
    # target: no field-arithmetic opcode can close the gap.
    assert (residual + controlled * 66
            + (inversion_products + division_products) * 54
            == pytest.approx(arm["and54"]["mean_toffoli_like"]))
    assert residual / TARGET_T_MEAN == pytest.approx(1.062, abs=0.001)
    assert residual > TARGET_T_MEAN
    # The ladder, from the same terms.
    at_23_ands = residual + controlled * 35 + (inversion_products + division_products) * 23
    assert at_23_ands == pytest.approx(864_060.6, abs=1.0)
    free_products = residual + controlled * 12
    assert free_products == pytest.approx(598_888.2, abs=1.0)
    assert at_23_ands / TARGET_T_MEAN == pytest.approx(1.941, abs=0.002)
    assert free_products / TARGET_T_MEAN == pytest.approx(1.345, abs=0.002)


LEDGER_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                 / "dqi_nonmultiply_ledger.json")


def test_the_nonmultiply_ledger_reconciles_without_double_counting():
    """Every executed Toffoli is in exactly one category, and the categories add up.

    The residual's 80% is deliberately *not* attributed: the simulator reports per-shot
    counts by operation type, not per operation, so an executed per-region split of the
    conditional EEA body cannot be read off. The ledger says that in a named bucket with
    its size instead of estimating it into the table.
    """
    data = json.loads(LEDGER_RECORD.read_text())
    ledger = data["executed_ledger"]
    total = ledger["total"]
    ids = [row["id"] for row in ledger["components"]]
    assert len(ids) == len(set(ids)) == 4
    assert ledger["component_sum"] == pytest.approx(total, abs=1e-3)
    assert ledger["reconciles"] is True
    for row in ledger["components"]:
        assert row["measured"] is True, row["id"]
        assert row["toffoli_per_shot"] > 0
    by_id = {row["id"]: row for row in ledger["components"]}
    assert by_id["ixor_gf2_mul"]["toffoli_per_shot"] == pytest.approx(693_660.0)
    assert by_id["non_product_residual"]["toffoli_per_shot"] == pytest.approx(472_768.2,
                                                                            abs=0.5)

    split = ledger["residual_split"]
    assert split["unconditional_total"] == 94_470
    assert split["unconditional_share_of_residual_percent"] == pytest.approx(20.0, abs=0.1)
    assert split["unattributed_executed"] == pytest.approx(378_298.2, abs=1.0)
    assert (split["unconditional_total"] + split["unattributed_executed"]
            == pytest.approx(by_id["non_product_residual"]["toffoli_per_shot"], abs=1e-6))
    assert "not" in split["unattributed_note"]

    static = data["static_ledger"]
    assert static["arm_total"] == 3_486_597
    assert static["reference_total"] == 5_725_461
    assert static["unconditional_regions"] + static["remainder"] == static["arm_total"]
    assert static["reconciles"] is True
    # The gaps are named, not hidden, and the first experiment carries its gates.
    assert len(data["measurement_gaps"]) >= 2
    experiment = data["first_experiment"]
    gates = experiment["promotion_gates"]
    assert any("phase" in gate for gate in gates)
    assert any("iteration" in gate for gate in gates)
    assert any("Q^2" in gate for gate in gates)
    # The fixed-iteration incumbent is the control, not early halting.
    assert "222" in experiment["control"]
    assert experiment["prediction_required_before_the_edit"] is True


def test_the_unconditional_region_ledger_is_the_largest_measured_lever_left():
    """The six unconditional regions are 94,470 Toffolis, 20% of the residual.

    They run on every shot, so their static count is their executed count, which makes the
    ledger an exact ceiling for any predicate on them -- the measurement the EEA plan's
    first lever rests on.
    """
    directions = json.loads(DESIGNER_RECORD.read_text())
    ceiling = directions["a_predicate_ceiling"]
    regions = ceiling["regions"]
    assert sum(row["ceiling_static_toffoli"] for row in regions.values()) == 94_470
    assert ceiling["total_ceiling_static_toffoli"] == 94_470
    assert ceiling["baseline_static_toffoli"] == 3_486_597

    data = json.loads(NATIVE_RECORD.read_text())
    decomposition = data["isolated_blocks"]["arm_decomposition"]
    assert decomposition["check_terms_sum_to_arm"] is True
    residual = decomposition["residual_non_product_toffoli"]
    assert 94_470 / residual == pytest.approx(0.200, abs=0.002)
    # The largest single region, and the two that are subsets of the same ledger.
    assert regions["_update_flag"]["ceiling_static_toffoli"] == 32_856
    assert (regions["_update_halting_counter"]["ceiling_static_toffoli"]
            + regions["_update_counter"]["ceiling_static_toffoli"]) == 10_878


def test_the_five_comparisons_are_labelled_and_consistent():
    """Each ratio names its baseline, and they reconcile with each other.

    The figures are easy to quote against the wrong baseline: 4.687x is the arm against
    the *unchanged* reference, while a same-policy comparison gives 4.258x, lower than
    the built-in 4.320x because the policy saves more on the reference (9.15%) than on
    the gated arm (7.83%). Mixing them makes the update look like a regression.
    """
    data = json.loads(NATIVE_RECORD.read_text())
    cmp = data["comparisons"]
    reference = data["reference"]
    arm = data["arm"]
    assert cmp["multiply_effect_on_reference_percent"] == pytest.approx(9.15, abs=0.01)
    assert cmp["multiply_effect_on_arm_percent"] == pytest.approx(7.83, abs=0.01)
    assert cmp["arm_vs_unchanged_reference"] == pytest.approx(4.687, abs=0.01)
    assert cmp["arm_vs_same_policy_reference"] == pytest.approx(4.258, abs=0.01)
    assert cmp["arm_q2t_reduction_vs_builtin_arm_percent"] == pytest.approx(5.97, abs=0.05)

    # The recorded ratios are the recorded means, not recomputations by hand.
    assert cmp["arm_vs_unchanged_reference"] == pytest.approx(
        reference["builtin"]["mean_toffoli_like"] / arm["and54"]["mean_toffoli_like"])
    assert cmp["arm_vs_same_policy_reference"] == pytest.approx(
        reference["and54"]["mean_toffoli_like"] / arm["and54"]["mean_toffoli_like"])
    assert cmp["arm_q2t_builtin"] == pytest.approx(
        arm["builtin"]["qubits"] ** 2 * arm["builtin"]["mean_toffoli_like"])
    assert cmp["arm_q2t_and54"] == pytest.approx(
        arm["and54"]["qubits"] ** 2 * arm["and54"]["mean_toffoli_like"])
    # The same-policy ratio is below the built-in arm's ratio: that is the finding, not
    # an error -- the reference gains more from the policy than the gated arm does.
    builtin_ratio = (reference["builtin"]["mean_toffoli_like"]
                     / arm["builtin"]["mean_toffoli_like"])
    assert cmp["arm_vs_same_policy_reference"] < builtin_ratio
    assert cmp["multiply_effect_on_reference_percent"] > cmp["multiply_effect_on_arm_percent"]


def test_official_row_and_local_row_are_reconciled():
    """The official 5,725,108.38 and this project's 5,725,071.6 are the same measurement.

    The official figure is a mean over the verifier's 102,400 shots; this project's is a
    64-shot mean. The record proves the shot counts from the figures' own fractional
    parts, and proves the two extensions count identically by running both on the same
    sampled syndromes and comparing per-shot arrays.
    """
    data = json.loads(RECONCILIATION_RECORD.read_text())
    official = data["official_reference_row"]
    assert official["qubits"] == 1_893
    assert official["t_mean"] == pytest.approx(5_725_108.382109375)
    assert official["mean_over_shots"] == 400 * 256
    assert official["is_an_integer_sum_at_that_shot_count"] is True
    # And it is not a 64-shot mean: its fractional part is not a multiple of 1/64.
    assert official["is_a_mean_over_this_scripts_shots"] is False

    reference = data["reference"]
    assert reference["same_shot_sample_across_builds"] is True
    assert reference["per_shot_arrays_identical_across_builds"] is True
    # Same decoder, same metric, same count -- on both binaries.
    assert reference["pinned"]["mean_toffoli_like"] == pytest.approx(5_725_071.609375)
    assert reference["local"]["mean_toffoli_like"] == pytest.approx(5_725_071.609375)
    assert reference["pinned"]["qubits"] == 1_893
    assert reference["local"]["qubits"] == 1_893
    assert reference["pinned"]["syndromes_restored"] is True
    assert reference["pinned"]["junk_clean"] is True
    # The difference from the official row is sampling: under one standard error.
    assert abs(reference["local_delta_in_standard_errors"]) < 1.0
    assert data["verdicts"]["difference_is_a_shot_sample_not_a_metric"][
        "explained"] is True


def test_the_arm_has_no_receipt_and_cannot_have_one():
    """The pinned binary cannot construct the arm; no receipt may come from another build."""
    data = json.loads(RECONCILIATION_RECORD.read_text())
    pinned = data["arm"]["pinned_build"]
    assert pinned["built"] is False
    assert pinned["error_type"] == "AttributeError"
    assert "mr" in pinned["error"]
    assert data["builds"]["pinned"]["has_mr"] is False
    assert data["builds"]["local"]["has_mr"] is True
    assert data["verdicts"]["arm_can_be_compiled_on_the_pinned_binary"] is False
    assert data["verdicts"]["arm_receipt_possible"] is False
    # The local arm numbers are recorded with the same extension the record measured.
    arm = data["arm"]["local_build"]
    assert arm["correct"] is True
    assert arm["mean_toffoli_like"] == pytest.approx(1_221_466.8, abs=1.0)
    assert arm["qubits"] == 1_913


def test_cost_record_is_internally_consistent():
    """The recorded cost measurement must reproduce the projection, stage by stage.

    This pin needs no extension: it checks the recorded arithmetic and that the fully
    gated stack lands on the oracle-gated 4.142x instead of below it, that the sixth
    lever is still worth its 4.07%, and that the reads cost no qubits. A record that
    drifts back to a shortfall (the first pass measured 3.556x), to a lost gate, or to
    one leaked qubit per read fails here.
    """
    data = json.loads(COST_RECORD.read_text())
    reference = data["cumulative"]["reference"]["mean_toffoli_like"]
    assert reference == pytest.approx(5_725_071.61, abs=1.0)
    for group in ("cumulative", "arms"):
        for entry in data[group].values():
            assert entry["correct"] is True
            assert entry["mean_toffoli_like"] > 0
            assert entry["ratio_vs_reference"] == pytest.approx(
                reference / entry["mean_toffoli_like"])
    gated = data["cumulative"]["all"]
    assert gated["ratio_vs_reference"] == pytest.approx(4.144, abs=0.01)
    assert gated["mean_toffoli_like"] == pytest.approx(
        data["projection"]["gated_t_mean"], rel=0.001)
    gaps = data["stage_gap_percent"]
    assert set(gaps) == {"forney", "forney+clear", "forney+clear+sweep",
                         "forney+clear+sweep+swap", "all"}
    assert max(abs(gap) for gap in gaps.values()) < 0.5, gaps
    assert data["agrees_with_projection"] is True
    assert "4.144x" in data["finding"]
    # The sixth lever must still be worth more than the 5.14% the shift gate buys.
    six = data["cumulative"]["all+quotient"]
    assert six["ratio_vs_reference"] == pytest.approx(4.320, abs=0.01)
    assert six["mean_toffoli_like"] < gated["mean_toffoli_like"] * 0.97
    # Reads share one ancilla: the gated stack must not pay Q for the predicates.
    reference_qubits = data["cumulative"]["reference"]["qubits"]
    for name in ("all", "all+quotient"):
        assert data["cumulative"][name]["qubits"] <= reference_qubits + 1, name


REGION_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                 / "dqi_region_ledger.json")
FLAG_REUSE_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                     / "dqi_flag_predicate_reuse.json")
ORACLE_RECORD = (REPO_ROOT / "benchmarks" / "quantum_compilation"
                 / "dqi_eea_oracle_ledger.json")


def test_the_region_ledger_splits_the_former_unattributed_bucket():
    """The 80% bucket is now a table whose rows sum to the residual.

    `attribute_dqi_nonmultiply_tcount.py` had to leave 378,298.2 unattributed because the
    simulator counted by operation type. Per-instruction counts plus `INFO_TAG` boundaries
    make the split exact, so a summary that drifts back to a named bucket, or a region that
    silently loses its attribution, fails here.
    """
    data = json.loads(REGION_RECORD.read_text())
    six = data["six_levers"]
    assert six["mean_toffoli_like"] == pytest.approx(1_221_466.8, abs=1.0)
    assert six["reconciles_with_measured_total"] is True
    assert six["correct"] is True

    rows = {row["region"]: row for row in six["regions"]}
    residuals = {name: row["executed_toffoli_per_shot"] - row["product_toffoli_per_shot"]
                 for name, row in rows.items()}
    assert sum(residuals.values()) == pytest.approx(472_768.2, abs=1.0)
    products = sum(row["product_toffoli_per_shot"] for row in rows.values())
    assert products == pytest.approx(748_698.7, abs=1.0)
    # The six unconditional regions of the older ledger are still individually visible.
    assert residuals["update_flag"] == pytest.approx(32_856)
    assert residuals["shift_step"] == pytest.approx(30_912)
    assert residuals["update_metadata"] == pytest.approx(12_432)
    assert residuals["update_halting_counter"] == pytest.approx(10_656)
    assert residuals["pre_align_reg_b"] + residuals["unalign_reg_b"] == pytest.approx(7_392)
    # The EEA decode row carries the 222-Toffoli counter plus the 4,440-Toffoli decode.
    assert residuals["eea_step"] == pytest.approx(4_662)
    # The conditional body, region by region, is where the residual actually is.
    assert residuals["shift_b_array_right"] == pytest.approx(162_479.6, abs=0.1)
    assert residuals["sweep_positions"] == pytest.approx(143_038.0, abs=0.1)
    assert residuals["swap_a_and_b"] == pytest.approx(64_224.0, abs=0.1)

    # Each lever's arm-level saving equals the sum of the region deltas it caused.
    for name, row in data["per_lever"].items():
        assert row["correct"] is True, name
        assert sum(row["regions_that_moved"].values()) == pytest.approx(
            row["arm_delta_per_shot"], abs=0.51), name
    assert data["per_lever_check"]["every_arm_correct"] is True
    assert data["per_lever_check"]["measured_deltas_match_region_deltas"] is True


def test_the_flag_reuse_experiment_ran_against_the_eight_gates():
    """The first controlled experiment: predicted before measuring, then gated.

    The prediction is computed from unit costs at the register width (8 bits, not the field
    degree), and the record has to show the measurement matching it -- a variant that merely
    ``looks smaller`` is not the claim.
    """
    data = json.loads(FLAG_REUSE_RECORD.read_text())
    prediction = data["prediction"]
    assert prediction["written_before_measurement"] is True
    assert prediction["register_width"] == 8
    assert prediction["toffoli_per_iteration_saved"] == 8
    assert prediction["predicted_delta_toffoli_per_shot"] == pytest.approx(1_776.0)

    control = data["control"]["mean_toffoli_like"]
    variant = data["variant"]["mean_toffoli_like"]
    assert control == pytest.approx(1_221_466.8, abs=1.0)
    assert control - variant == pytest.approx(1_776.0, abs=0.1)
    assert data["observation"]["within_prediction"] is True
    assert data["control"]["update_flag_toffoli_per_shot"] == pytest.approx(32_856)
    assert data["variant"]["update_flag_toffoli_per_shot"] == pytest.approx(31_080)
    assert data["variant"]["qubits"] == data["control"]["qubits"] == 1_913

    gates = data["promotion_gates"]
    assert len(gates) == 8
    assert all(gate["passed"] for gate in gates), [g["gate"] for g in gates
                                                   if not g["passed"]]
    assert data["all_gates_passed"] is True
    assert data["verdict"] == "PROMOTE"
    # Two instruments, one total: the per-instruction counts reconcile with num_toffoli_like.
    assert data["control"]["multiply_toffoli_per_shot"] + data["control"][
        "non_product_toffoli_per_shot"] == pytest.approx(control, abs=0.1)


def test_the_halt_gate_is_measured_and_correct():
    """`is_hc_zero` gates `_update_flag` and `_update_metadata`; the saving is the region rows.

    Both regions are controlled on the one-hot counter bits, so skipping them in a halted
    round is exact. The record has to show the arm still correct and the delta equal to the
    regions it removed -- a gate that also skipped live rounds would fail correctness.
    """
    data = json.loads(ORACLE_RECORD.read_text())
    control = data["control"]["mean_toffoli_like"]
    gated = data["halt_gated"]["mean_toffoli_like"]
    assert data["control"]["correct"] is True
    assert data["halt_gated"]["correct"] is True
    assert data["halt_gated"]["qubits"] == data["control"]["qubits"]
    assert control - gated == pytest.approx(1_224.0, abs=0.1)
    rows = {row["region"]: row for row in data["region_table"]}
    assert rows["update_flag"]["removable_by_halt_gate"] == pytest.approx(888.0)
    assert rows["update_metadata"]["removable_by_halt_gate"] == pytest.approx(336.0)
    assert data["totals"]["halt_gate_removable_per_shot"] == pytest.approx(1_224.0)
    assert rows["update_flag"]["executed_toffoli_per_shot"] == pytest.approx(32_856)
    assert data["iteration_control"]["fixed_iterations"] == 222
    assert data["checks"]["fixed_iterations_is_222"] is True
    assert data["checks"]["halt_gate_is_correct"] is True
    # The regions without a known predicate are named, not given a number.
    assert "pre_align_reg_b" in data["no_known_predicate"]


@requires_mr
def test_per_instruction_counts_reconcile_with_the_type_counters():
    """The new instrument must agree with `num_toffoli_like` on the same run.

    Checks the two things that are easy to get wrong: the SIMD lanes beyond `batch_size`
    must not be counted, and the per-instruction mask must be the same one the type
    counters use.
    """
    probe = (
        "import sys, numpy as np\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import kickmix as km\n"
        "b = km.CircuitBuilder()\n"
        "q = b.alloc_qubits(3)\n"
        "bit = b.alloc_bits(1)[0]\n"
        "with b.mark('r'):\n"
        "    b.cx(q[0], q[1])\n"
        "    b.ccx(q[0], q[1], q[2])\n"
        "c = b.finish_circuit()\n"
        "sim = km.Simulator(batch_size=4, counting_operations=True, seed=1)\n"
        "sim.counting_op_executions = True\n"
        "sim.use_same_registers_as(c)\n"
        "sim.write_across_shots(km.b(0), np.array([True, True, False, False]))\n"
        "sim.do(c)\n"
        "counts = np.asarray(sim.operation_execution_counts())\n"
        "mask = np.asarray(c.op_toffoli_like)\n"
        "print(int(counts[mask].sum()), int(sim.operation_statistics().num_toffoli_like),\n"
        "      int(counts.max()), len(counts), c.info_tags()[0][1])\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe, str(BUILD_DIR)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    per_instruction, type_counts, largest, length, tag = proc.stdout.split()
    assert int(per_instruction) == int(type_counts) == 4
    assert length == "4"
    # Four requested shots, not the 256 SIMD lanes.
    assert int(largest) == 4
    assert tag == "@enter:r"
