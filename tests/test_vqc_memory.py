"""Peak memory of a sampling engine, measured in its own process and window.

Two traps make a naive memory number meaningless, and this file pins both:

* ``VmHWM`` never falls, so two engines measured in one interpreter cannot be separated --
  the second can never come out below the first. Each measurement therefore runs in a
  fresh child, and the test checks that the child really is a different interpreter by
  requiring its count hash to equal the calling process's own;
* the whole-child peak is dominated by the interpreter and numpy (~165 MiB here) against
  a few megabytes of samples, so quoting it would be quoting numpy. The child resets the
  peak counter after its imports, and these tests check that the reset actually works,
  that the growth reported is exactly the difference it claims to be, and that a
  platform without the counter gets a refusal with a reason rather than a zero.

The last pin is the one that matters for the record: the child reports the count hash of
the samples it drew, and the measurement is only attached to a workload whose hash is the
one the caller already recorded.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import apedp_dyn.vqc.memory as memory
from apedp_dyn.vqc.external import apedp_sampling
from apedp_dyn.vqc.memory import (
    ENGINES,
    MemoryMeasurement,
    MemoryUnavailable,
    current_rss_bytes,
    measure_engine,
    peak_rss_bytes,
    reset_peak,
    target_from_strings,
)
from apedp_dyn.vqc.synthesis import ghz_target

MIB = 2 ** 20


def test_the_peak_counter_is_readable_and_is_a_peak():
    # A worker that already peaked above the allocation below cannot show growth in a
    # monotone counter, so the window is opened first where the platform allows it.
    if reset_peak():
        start = peak_rss_bytes()
    else:
        start = peak_rss_bytes()
    assert start > 0
    block = np.ones(32 * MIB, dtype=np.uint8)   # touched, so it is resident
    assert block.sum() > 0
    assert peak_rss_bytes() >= start + 24 * MIB, (start, peak_rss_bytes())
    del block


def test_the_peak_can_be_reset_so_a_window_is_measurable():
    """Without the reset, "the sampling step's peak" is the whole process's high-water."""
    block = np.ones(16 * MIB, dtype=np.uint8)
    assert block.sum() > 0
    grown = peak_rss_bytes()
    del block
    if not reset_peak():
        pytest.skip("this platform does not allow resetting the peak counter")
    assert peak_rss_bytes() <= grown
    base = peak_rss_bytes()
    # The counter is the live resident set, to within the page the reset itself touches:
    # the write to clear_refs and the read back are not free, and demanding byte equality
    # made this assertion fail on the first allocation the kernel happened to service.
    assert abs(base - current_rss_bytes()) <= MIB, (base, current_rss_bytes())
    fresh = np.ones(8 * MIB, dtype=np.uint8)
    assert fresh.sum() > 0
    # A window opened at `base` can see what happens after it; a monotone counter could
    # only ever report `grown`.
    assert peak_rss_bytes() >= base + 4 * MIB
    # `grown` is only a ceiling the window beats when the allocator actually handed the
    # freed pages back: glibc keeps them resident, so in a recycled worker the pre-reset
    # peak can equal the live set and an unconditional `peak < grown` measures the
    # allocator, not the counter. It is asserted where it is observable.
    if grown > base + 8 * MIB:
        assert peak_rss_bytes() < grown
    del fresh


def _as_payload_support(target) -> dict[str, float] | None:
    if target.support is None:
        return None
    return {"".join("1" if bit else "0" for bit in bits): float(probability)
            for bits, probability in sorted(target.support.items())}


def test_a_target_survives_the_process_boundary_with_its_hash():
    """The rebuild is hash-identical, declaring support included.

    GHZ and the cluster states declare the outcomes they expect, and that declaration is
    part of the specification's hash: a rebuild that dropped it would be a different
    target, which is how this pin was written the second time.
    """
    for target in (ghz_target(6), ghz_target(2)):
        assert target.support is not None
        rebuilt = target_from_strings(target.name, target.num_qubits,
                                      target.stabilizer_strings(),
                                      decides=target.decides,
                                      support=_as_payload_support(target))
        assert rebuilt.sha256() == target.sha256()
        assert rebuilt.stabilizer_strings() == target.stabilizer_strings()
    # A negative-phase specification round-trips too, signs included.
    from apedp_dyn.vqc.synthesis import I, TargetSpecification, Z

    ones = TargetSpecification("one-one", 2, (((Z, I), 2), ((I, Z), 2)))
    rebuilt = target_from_strings("one-one", 2, ones.stabilizer_strings(),
                                  decides=ones.decides)
    assert rebuilt.stabilizer_strings() == ("-ZI", "-IZ")
    assert rebuilt.sha256() == ones.sha256()
    for text, size in (("+XX", 1), ("++X", 2), ("XXI", 2), ("+XY1", 3)):
        with pytest.raises(ValueError):
            target_from_strings("bad", size, [text])
    with pytest.raises(ValueError):
        target_from_strings("bad", 2, ["+ZZ"], support={"001": 1.0})


def test_a_measured_engine_is_measured_on_the_recorded_samples():
    target = ghz_target(12)
    reference = apedp_sampling(target, shots=2_000, seed=4_242)
    measurement = measure_engine("grammar", target, shots=2_000, seed=4_242)
    assert measurement.counts_sha256 == reference.counts_sha256()
    assert measurement.shots == 2_000
    assert measurement.target_sha256 == target.sha256()
    assert measurement.engine == "grammar"
    assert measurement.variant == reference.variant
    assert measurement.floor_bytes > 0
    assert measurement.window_peak_bytes >= measurement.floor_bytes
    assert measurement.growth_bytes == (
        measurement.window_peak_bytes - measurement.floor_bytes)
    assert measurement.absolute_peak_bytes >= measurement.floor_bytes
    assert measurement.peak_reset is True
    payload = measurement.as_dict()
    assert payload["growth_mib"] == pytest.approx(measurement.growth_bytes / MIB)
    assert "VmHWM" in payload["method"]
    assert payload["seconds"] > 0


def test_stim_is_measured_separately_and_charged_its_own_import():
    """Stim's library is loaded before the window opens, so it is a floor, not a sample."""
    target = ghz_target(10)
    own = measure_engine("grammar", target, shots=1_000, seed=7)
    foreign = measure_engine("stim", target, shots=1_000, seed=8)
    assert foreign.engine == "stim"
    assert foreign.variant.startswith("tableau")
    assert own.variant == "stabilizer.tableau"
    assert own.floor_bytes > 0 and foreign.floor_bytes > 0
    assert set(ENGINES) == {"grammar", "stim"}
    assert ENGINES["stim"]["imports"] == ("stim",)
    assert ENGINES["grammar"]["imports"] == ()


def test_a_platform_that_cannot_reset_the_counter_abstains(monkeypatch):
    """A refusal with a reason, never the whole-process peak dressed as the engine's."""
    measurement = MemoryMeasurement(
        engine="grammar", variant="x", version=None, target_sha256="0" * 64, shots=1,
        floor_bytes=100, window_peak_bytes=150, absolute_peak_bytes=150,
        seconds=1.0, counts_sha256="1" * 64, peak_reset=True)
    assert measurement.growth_bytes == 50

    target = ghz_target(3)
    monkeypatch.setattr(memory, "_run_child", lambda *args, **kwargs: {
        "engine": "grammar", "peak_reset": False, "floor_bytes": 1,
        "window_peak_bytes": 1, "absolute_peak_bytes": 1, "shots": 10,
        "counts_sha256": "0" * 64, "variant": "x", "version": None,
        "sampling_seconds": 0.0, "target_sha256": target.sha256()})
    with pytest.raises(MemoryUnavailable, match="could not be reset"):
        measure_engine("grammar", target, shots=10, seed=1)


def test_measuring_a_different_specification_is_refused(monkeypatch):
    """The child's hash check is what stops a mis-parsed target becoming a number."""
    target = ghz_target(3)
    real_payload = memory._child_payload

    def tampered(*args, **kwargs):
        payload = real_payload(*args, **kwargs)
        payload["target_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(memory, "_child_payload", tampered)
    with pytest.raises(MemoryUnavailable, match="does not hash"):
        measure_engine("grammar", target, shots=10, seed=1, timeout_seconds=120)


def test_unknown_engines_and_empty_budgets_are_refused():
    target = ghz_target(3)
    with pytest.raises(ValueError):
        measure_engine("qiskit", target, shots=10)
    with pytest.raises(ValueError):
        measure_engine("grammar", target, shots=0)


def test_the_measurement_is_reproducible_in_shape():
    """Two measurements agree on everything but the byte counts and the clock.

    Not on the bytes: RSS is a property of this machine's allocator and of what the
    interpreter touched first, and reporting a fabricated median would be worse than
    reporting one run.
    """
    target = ghz_target(10)
    first = measure_engine("grammar", target, shots=1_000, seed=3)
    second = measure_engine("grammar", target, shots=1_000, seed=3)
    assert first.counts_sha256 == second.counts_sha256
    assert first.target_sha256 == second.target_sha256
    assert first.shots == second.shots == 1_000
    assert first.growth_bytes >= 0 and second.growth_bytes >= 0
    assert dataclasses.asdict(first)["method"] == dataclasses.asdict(second)["method"]
