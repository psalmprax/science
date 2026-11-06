"""Pins for `scripts/probe_dqi_predicate_gating.py`: what predicate-gated execution and an MR clear are worth, and why neither ships.

Three claims are pinned here.

1. **Forney gating.** The body is semantically dead wherever `sigma(alpha^j) != 0`,
   so with the truth table supplied as an oracle classical register the official
   instance drops from 5,725,071.61 to 4,299,013.45 executed Toffoli-like gates per
   shot (1.332x) with 0 value and 0 phase errors and `Q` unchanged. That is the
   ceiling `docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5 lists as "if the phase were
   free" (1.335x), measured rather than estimated.

2. **Replacing the reverse EEA with a clear.** The EEA runs twice — forward, then
   its adjoint to uncompute — and half of its cost is the uncompute. Clearing the
   EEA scratch and the syndromes instead and re-preparing the syndromes from a
   classical readout gives the same T (2,158,651.45, both clears cost zero Toffoli)
   but only the **MR** clear is correct: `R` randomizes the tracked phase (35/64
   shots at the official instance, 254/512 at m=255), while measure-then-X clears
   give 0/0. With MR the design reaches **2.652x** on the official instance.

3. **Why neither can ship.** `push_condition` needs a classical bit; `hmr` is
   X-basis (31/64 ones on `|1>`); `push_condition(qubit)` and
   `bit_store(control=qubit)` are rejected; `R` is not a phase-safe clear. Z-basis
   measure-and-reset is the single missing instruction, and it is needed twice.

Re-measurement needs the kickmix extension built for the running interpreter, so
the recorded artifact is what is pinned by default; the cheap instance is
re-measured and compared exactly when the extension is importable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "quantum_compilation" / "dqi_predicate_gating.json"
BUNDLE = ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"


def _record() -> dict:
    assert ARTIFACT.is_file(), f"missing measured artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _instance(name: str) -> dict:
    return _record()["instances"][name]


def test_readout_of_a_computational_basis_predicate_is_impossible():
    """`hmr` is an X-basis measurement, and no other readout exists."""
    probe = _record()["readout_probe"]
    ones, shots = probe["hmr_on_one_ones_count"], probe["hmr_on_one_shots"]
    # A Z-basis readout of |1> would be all ones; HMR is random instead.
    assert shots > 32 and 0 < ones < shots, probe
    assert 0.25 < ones / shots < 0.75, "a biased result would mean HMR is not X-basis"
    assert probe["push_condition_accepts_qubit"] == "REJECTED"
    assert probe["bit_store_accepts_quantum_control"] == "REJECTED"


def test_reset_randomizes_the_phase_even_with_no_phase_in_the_circuit():
    """`R` is not a phase-safe clear: this is what blocks the reverse-EEA removal."""
    probe = _record()["readout_probe"]
    assert probe["reset_only"] == 0, "no qubit dirtied, nothing to randomize"
    assert probe["no_reset_after_x"] == 0, "no reset, phase stays definite"
    assert 0 < probe["reset_after_x"] < probe["hmr_on_one_shots"], probe


def test_official_instance_forney_body_is_a_quarter_of_the_cost():
    """The measured body share agrees with the 22.5% inversion-only bound in §5.5."""
    rec = _instance("official")
    assert (rec["m"], rec["n"], rec["degree"]) == (4095, 70, 12)
    assert round(rec["baseline"]["toffoli_per_shot"], 2) == 5725071.61
    assert round(rec["gated"]["toffoli_per_shot"], 2) == 4299013.45
    assert rec["baseline"]["static_toffoli"] == rec["gated"]["static_toffoli"]
    assert 0.20 < rec["forney_body_share"] < 0.26, rec["forney_body_share"]
    assert 0.007 < rec["root_density"] < 0.008, rec["root_density"]


def test_gating_the_body_reaches_the_free_phase_ceiling_and_stays_correct():
    """1.332x, against the 1.335x ceiling recorded for a free Forney phase."""
    rec = _instance("official")
    assert rec["semantics_preserved"] is True
    assert rec["gated"]["value_errors"] == 0 and rec["gated"]["phase_errors"] == 0
    assert rec["baseline"]["num_qubits"] == rec["gated"]["num_qubits"] == 1893
    assert 1.30 < rec["toffoli_ratio"] < 1.34, rec["toffoli_ratio"]


def test_the_r_clear_costs_what_the_mr_clear_costs_but_is_wrong():
    """Removing the reverse EEA is free in Toffoli; only `R` versus MR decides correctness."""
    rec = _instance("official")
    r_clear, mr_clear = rec["restore_gated"], rec["mr_clear"]
    # The MR clear adds 1,803 classically controlled X's: CX is not Toffoli-like, so
    # both clears are free in the metric that prices the run.
    assert r_clear["static_toffoli"] == mr_clear["static_toffoli"]
    assert abs(r_clear["toffoli_per_shot"] - mr_clear["toffoli_per_shot"]) < 1e-6
    assert r_clear["value_errors"] == 0 and r_clear["phase_errors"] > 0
    assert r_clear["phase_errors"] / rec["num_shots"] > 0.4, "R must be a coin flip on the phase"
    assert mr_clear["value_errors"] == 0 and mr_clear["phase_errors"] == 0


def test_mr_clear_halves_the_eea_and_reaches_2_65x():
    """Both levers together: 5,725,071.61 -> 2,158,651.45 Toffoli per shot."""
    rec = _instance("official")["mr_clear"]
    assert round(rec["toffoli_per_shot"], 2) == 2158651.45
    assert rec["num_qubits"] == 1893, "the clear must not cost qubits"
    assert 2.60 < rec["toffoli_ratio_vs_baseline"] < 2.70, rec["toffoli_ratio_vs_baseline"]
    assert 1_700 < rec["cleared_qubits"] < 1_900, rec["cleared_qubits"]
    # Still short of the 12.86x the Benchmark II target requires.
    assert 12.86 / rec["toffoli_ratio_vs_baseline"] > 4.5


def test_gating_the_dead_multiply_sweep_reaches_3_6x():
    """Two of every four EEA rounds emit a multiply-accumulate sweep that does nothing."""
    rec = _instance("official")["mr_clear_sweep_gated"]
    assert round(rec["toffoli_per_shot"], 2) == 1590938.28
    assert rec["num_qubits"] == 1893
    assert rec["value_errors"] == 0 and rec["phase_errors"] == 0
    assert 0.30 < rec["dead_sweep_fraction"] < 0.40, rec["dead_sweep_fraction"]
    assert 3.55 < rec["toffoli_ratio_vs_baseline"] < 3.65, rec["toffoli_ratio_vs_baseline"]
    # Still short of the 12.86x the Benchmark II target requires.
    assert 12.86 / rec["toffoli_ratio_vs_baseline"] > 3.5


def test_gating_the_dead_register_swap_reaches_3_93x():
    """`_swap_a_and_b` is live only when `c3` is set; it exchanges 1,752 qubits per round."""
    rec = _instance("official")["mr_clear_sweep_swap_gated"]
    assert round(rec["toffoli_per_shot"], 2) == 1457103.45
    assert rec["num_qubits"] == 1893
    assert rec["num_shots"] == 64
    assert rec["value_errors"] == 0 and rec["phase_errors"] == 0
    assert 0.60 < rec["dead_swap_fraction"] < 0.75, rec["dead_swap_fraction"]
    assert 3.90 < rec["toffoli_ratio_vs_baseline"] < 3.95, rec["toffoli_ratio_vs_baseline"]
    assert rec["round_bits_needed"] == 5 * (3 * (rec["n"] + 1) + 9)
    assert 12.86 / rec["toffoli_ratio_vs_baseline"] > 3.2


def test_dead_sweep_fraction_holds_across_widths():
    for name, low, high in (("tiny_m15", 0.40, 0.55), ("scaled_m255", 0.35, 0.45)):
        rec = _instance(name)["mr_clear_sweep_gated"]
        assert rec["value_errors"] == 0 and rec["phase_errors"] == 0, name
        assert low < rec["dead_sweep_fraction"] < high, (name, rec["dead_sweep_fraction"])
        assert rec["toffoli_ratio_vs_baseline"] > 1.5, name


def test_all_four_gates_reach_4_14x():
    """Sweep, shift and swap gated together, with the shift reading the post-update flag."""
    rec = _instance("official")["mr_clear_all_gated"]
    assert round(rec["toffoli_per_shot"], 2) == 1382207.08
    assert rec["num_qubits"] == 1893
    assert rec["num_shots"] == 64
    assert rec["value_errors"] == 0 and rec["phase_errors"] == 0
    assert rec["sweep_gated"] and rec["shift_gated"] and rec["swap_gated"]
    assert rec["round_bits_needed"] == 6 * (3 * (rec["n"] + 1) + 9)
    assert 4.10 < rec["toffoli_ratio_vs_baseline"] < 4.20, rec["toffoli_ratio_vs_baseline"]
    assert 12.86 / rec["toffoli_ratio_vs_baseline"] > 3.0


def test_the_shift_gate_needs_the_post_update_flag():
    """`flag` is toggled by `_update_flag` before the shift reads it.

    Reading it at the top of the round instead fails on every shot (measured: 512/512
    value errors at m=255), so the six-wide probe carries the post-update value and the
    liveness is `c1 ∨ c2 ∨ (¬flag ∧ (c0 ∨ c3))`, not `c1 ∨ c2 ∨ ¬flag`.
    """
    bank = _instance("scaled_m255")
    assert "mr_clear_shift_gated" not in bank, "the stale-flag arm must never be recorded as a result"
    assert bank["mr_clear_sweep_swap_gated"]["shift_gated"] is False
    all_gated = bank["mr_clear_all_gated"]
    assert all_gated["shift_gated"] is True
    assert all_gated["value_errors"] == 0 and all_gated["phase_errors"] == 0
    assert all_gated["round_bits_needed"] == 6 * (3 * (bank["n"] + 1) + 9)


def test_mr_clear_ratio_holds_across_widths():
    for name, low in (("tiny_m15", 1.9), ("scaled_m255", 2.2)):
        rec = _instance(name)["mr_clear"]
        assert rec["value_errors"] == 0 and rec["phase_errors"] == 0, name
        assert low < rec["toffoli_ratio_vs_baseline"] < 2.5, name


def test_remeasured_small_instance_matches_the_record_when_kickmix_is_importable():
    sys.path.insert(0, str(BUNDLE / "kickmix"))
    sys.path.insert(0, str(BUNDLE / "verifiers" / "dqi"))
    try:
        import kickmix  # noqa: F401
    except Exception as exc:  # built for a different interpreter
        pytest.skip(f"kickmix not importable here: {exc}")

    sys.path.insert(0, str(ROOT / "scripts"))
    import probe_dqi_predicate_gating as probe
    from verify_submission_dqi import ProblemSpecDQI, default_num_accepted_values

    spec = ProblemSpecDQI(m=15, n=6, degree=4, r=default_num_accepted_values(4), num_shots=64)
    fresh = probe.measure(spec)
    pinned = _instance("tiny_m15")
    for key in ("toffoli_ratio", "forney_body_share", "root_density", "semantics_preserved"):
        assert fresh[key] == pinned[key], key
    for arm in ("baseline", "gated", "restore", "restore_gated"):
        assert fresh[arm] == pinned[arm], arm

    mr = probe.measure_mr_clear(spec)
    assert mr["toffoli_per_shot"] == pinned["mr_clear"]["toffoli_per_shot"]
    assert mr["value_errors"] == 0 and mr["phase_errors"] == 0
    assert mr["cleared_qubits"] == pinned["mr_clear"]["cleared_qubits"]
