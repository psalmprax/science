"""Pins for `scripts/measure_dqi_scratch_pool.py`: a scratch pool has nothing to collect.

`docs/PHOTONIC_ACCELERATOR_BOUNDARY.md` §5.5 said a shared scratch pool was "the only
mechanism that could move" the transient qubits, and the −104 the V6 row claims is 77%
of them. Measured on the pinned Benchmark II instance, three things are true instead.

1. **The peak is not where the projection says.** 1,828 qubits are concurrently live
   against 1,893 distinct ids, and the peak sits in `_chien_search_and_forney_mbuc`,
   not in the EEA round scratch.
2. **A pool's ceiling is the innermost frame at the peak.** That frame holds 25
   qubits, so pooling can redistribute at most 25 — not 65, and certainly not 104.
   The remaining 1,803 are held by `build_syndrome_decoding_mbuc` because the decoder
   needs them at that instant.
3. **The allocator question is open, not answered.** Replaying the recorded
   alloc/free trace returns 1,821 ids against 1,893 shipped, and the proxy sees only
   1,873 ids: allocations made inside the extension are missing from the trace. A
   lowest-hole free list returns exactly the same 1,821 as the shipped LIFO stack, so
   no "the stranding is in alloc_qubit" claim is supported by these numbers.

Re-measurement needs the `kickmix` extension on the interpreter it was built for
(CPython 3.13; a 3.12 interpreter fails at import), so this test pins the recorded
artifact. Regenerate it with:

    python3.13 scripts/measure_dqi_scratch_pool.py --json \
        > benchmarks/quantum_compilation/dqi_scratch_pool.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "quantum_compilation" / "dqi_scratch_pool.json"


def _record() -> dict:
    assert ARTIFACT.is_file(), f"missing measured artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_builder_recycles_freed_ids_without_double_booking():
    recycling = _record()["recycling"]
    assert recycling["recycles"] is True
    assert recycling["no_double_booking"] is True
    assert sorted(recycling["after_free_realloc"]) == sorted(recycling["first_alloc"])


def test_peak_is_below_the_shipped_id_count():
    m = _record()["measurement"]
    assert m["total_qubits"] == 1893
    assert m["peak_live"] == 1828
    assert m["stranded_ids"] == m["total_qubits"] - m["peak_live"] == 65


def test_peak_is_reached_inside_chien_and_forney_not_the_eea_round():
    """The boundary doc attributed the peak to `_multiply_and_add_or_sub`; it is not."""
    m = _record()["measurement"]
    assert m["peak_at_frame"] == "_chien_search_and_forney_mbuc"
    frames = [f["frame"] for f in m["live_frames_at_peak"]]
    assert frames[-1] == "_chien_search_and_forney_mbuc"
    assert "_multiply_and_add_or_sub" not in frames


def test_pool_ceiling_is_the_innermost_frame_not_the_whole_gap():
    m = _record()["measurement"]
    innermost = m["live_frames_at_peak"][-1]
    assert m["pool_ceiling"] == innermost["own_peak"] == 25
    # The V6 row claims -104 qubits: 77% of every transient, and 4x this ceiling.
    assert m["pool_ceiling"] < 104


def test_free_list_claim_is_not_supported_by_the_trace():
    """Both allocator policies replay identically, and the replay misses shipped ids."""
    m = _record()["measurement"]
    assert m["replay"]["lifo"] == m["replay"]["lowest"] == 1821
    assert m["replay_covers_shipped_ids"] is False
    assert m["trace_max_id"] < m["total_qubits"]
