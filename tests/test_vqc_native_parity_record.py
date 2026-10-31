"""Pins for `benchmarks/quantum_compilation/vqc_native_parity.json`.

The record is the only place the Rust claim is quantified, so it has to be
checkable without re-running the timings -- a benchmark that cannot be checked is
a number with a story attached. What is pinned:

* the parity claims: every case agrees, and a case cannot claim a speedup while
  disagreeing about the answer;
* the reproducibility block: the script, the six Rust sources and the two Python
  implementations it measured are identified by SHA-256, so a later edit to any of
  them is visible rather than silent;
* the workloads: every case's circuit can be rebuilt from its description and
  must hash to the recorded `circuit_sha256`, so "identical frozen inputs" is
  reproducible instead of asserted;
* the limitations, including the one that matters most for trust: above the
  parity limit the table is *not* re-derived in Python.

Timings themselves are not pinned. A wall-clock number from another machine is
not evidence about this one, and a test that asserted one would fail for the
wrong reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "quantum_compilation" / "vqc_native_parity.json"
SCRIPT = ROOT / "scripts" / "measure_vqc_rust_parity.py"
RUST = ROOT / "rust" / "apedp-vqc" / "src"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_record_is_the_schema_it_claims_and_names_both_engines(record):
    assert record["schema"] == "apedp.vqc.native_parity.v1"
    assert record["engines"]["rust"]["engine"] == "apedp_vqc"
    assert record["engines"]["python"]["engine"] == "python"
    assert record["engines"]["python"]["role"].startswith("reference implementation")


def test_every_case_agrees_and_none_claims_speed_without_it(record):
    cases = {case["case"]: case for case in record["cases"]}
    assert set(cases) == {"images_100q", "images_400q", "sample_100q_256shots",
                          "distribution_10q", "adjudication_100q"}
    for name, case in cases.items():
        assert case["agrees"] is True, name
        for engine in ("python", "rust"):
            assert case[engine]["median_seconds"] > 0
            assert case[engine]["best_seconds"] <= case[engine]["median_seconds"]
        ratio = case["python"]["median_seconds"] / case["rust"]["median_seconds"]
        assert case["speedup_median"] == pytest.approx(ratio, rel=1e-9)
        # A record that reported a regression as a speedup would be the failure
        # this whole file exists to prevent.
        assert case["speedup_median"] > 1.0, f"{name} is reported as faster than Python"


def test_the_image_and_distribution_cases_compare_what_they_say(record):
    cases = {case["case"]: case for case in record["cases"]}
    assert cases["images_100q"]["images_compared"] == 200
    assert cases["images_400q"]["images_compared"] == 800
    assert cases["distribution_10q"]["outcomes_nonzero"] == 2
    assert cases["distribution_10q"]["workload"]["outcomes"] == 1024
    assert cases["sample_100q_256shots"]["outcomes_seen"] == ["0" * 100, "1" * 100]
    assert cases["adjudication_100q"]["verdict"] == "VALID"
    # Above PARITY_LIMIT the referee did not re-derive the table, and says so.
    assert cases["adjudication_100q"]["image_engine"] == "rust"
    assert cases["adjudication_100q"]["image_verified_by"] == "suite"
    assert cases["adjudication_100q"]["python_image_engine"] == "python"


def test_the_parity_corpus_is_exact_agreement_not_a_sample(record):
    corpus = record["parity_corpus"]
    assert corpus["circuits_compared"] == 7
    assert corpus["rows"] and [row["qubits"] for row in corpus["rows"]] == list(range(2, 9))
    for row in corpus["rows"]:
        assert row["engine"] == "apedp_vqc"
        assert row["images_compared"] == 2 * row["qubits"]
        assert row["generators_compared"] == row["qubits"]
        # Distribution agreement is only attempted where enumeration is readable.
        assert row["distribution_compared"] in (0, 2)


def test_each_case_can_be_rebuilt_from_its_own_description(record):
    """The `circuit_sha256` fields are reproducible, not decorative."""
    from apedp_dyn.vqc.circuit import Circuit, Gate
    from apedp_dyn.vqc.stabilizer import ghz_circuit

    for case in record["cases"]:
        workload = case["workload"]
        if case["case"].startswith("images") or case["case"].startswith("distribution"):
            rebuilt = ghz_circuit(workload["qubits"], measure=False)
        elif case["case"].startswith("sample"):
            rebuilt = ghz_circuit(workload["qubits"], measure=True)
        else:
            reference = ghz_circuit(workload["qubits"], measure=False)
            ops = [Gate("h", (0,))]
            ops += [Gate("cx", (qubit - 1, qubit)) for qubit in range(1, workload["qubits"])]
            ops += [Gate("z", (0,)), Gate("z", (0,))]
            candidate = Circuit(ops=tuple(ops), num_qubits=workload["qubits"], num_bits=0)
            assert candidate.sha256() == workload["candidate_sha256"], case["case"]
            assert reference.sha256() == workload["reference_sha256"], case["case"]
            continue
        assert rebuilt.sha256() == workload["circuit_sha256"], case["case"]


def test_the_reproducibility_block_pins_the_sources_it_measured(record):
    reproducibility = record["reproducibility"]
    assert reproducibility["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    for name, digest in reproducibility["native_source_sha256"].items():
        assert digest == hashlib.sha256((RUST / name).read_bytes()).hexdigest(), name
    assert set(reproducibility["native_source_sha256"]) == {
        "lib.rs", "pauli.rs", "stabilizer.rs", "equivalence.rs", "runtime.rs", "ir.rs",
        "bits.rs"}
    for key, relative in (("stabilizer_sha256", "apedp_dyn/vqc/stabilizer.py"),
                          ("equivalence_sha256", "apedp_dyn/vqc/equivalence.py"),
                          ("native_bridge_sha256", "apedp_dyn/vqc/native.py")):
        assert reproducibility[key] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_the_limitations_name_the_parity_limit_and_the_missing_tiers(record):
    limitations = " ".join(record["limitations"])
    assert "PARITY_LIMIT" in limitations
    assert "image_verified_by='suite'" in limitations or "image_verified_by" in limitations
    assert "statevector" in limitations and "decoders" in limitations
    assert record["method"]["reference_timings_do_not_touch_the_fast_path"]
    assert record["method"]["repeats"] >= 5


def test_the_methodology_is_recorded_not_just_the_medians(record):
    """A median with no spread and no machine is a number with a story attached."""
    method = record["method"]
    assert method["repeats"] == record["command"]["repeats"]
    assert "interquartile" in method["statistic"]
    assert method["compile_time"].startswith("excluded")
    assert "peak RSS" in method["memory"]
    for case in record["cases"]:
        for engine in ("python", "rust"):
            timing = case[engine]
            assert len(timing["samples_seconds"]) == method["repeats"], case["case"]
            assert timing["best_seconds"] <= timing["median_seconds"]
            assert timing["iqr_seconds"] >= 0.0
            assert timing["iqr_seconds"] < timing["median_seconds"], case["case"]


def test_the_hardware_the_numbers_came_from_is_named(record):
    hardware = record["hardware"]
    for key in ("platform", "machine", "cpu_count", "python", "numpy", "rustc",
                "cargo", "build", "peak_rss_kib"):
        assert hardware[key], key
    # The build is named and its compilation is not in any timing.
    assert "release" in hardware["build"] and "excluded" in hardware["build"]
    # Peak memory is process-wide and says so, rather than being attributed per case.
    assert "not per case" in hardware["peak_rss_note"]
    limitations = " ".join(record["limitations"])
    assert "IQR" in limitations and "not a confidence interval" in limitations
    assert "hardware block pins" in limitations
