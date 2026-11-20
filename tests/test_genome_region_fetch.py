"""The genome-slice fetcher: validation, provenance, and a scan on a real slice.

The live slice is saved under ``tests/fixtures/genome`` (Ensembl REST, GRCh38
chr13), so the end-to-end path runs offline and the numbers are properties of
this code against that slice rather than of today's assembly.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from apedp.ingestion.genome_region import (
    MAX_REGION_BASES,
    GenomeFetchError,
    GenomeRegion,
    fetch_region,
    scan_region,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "genome" / "grch38_chr13_slice.json"
REGION = "13:32930000..32932000"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _region() -> GenomeRegion:
    data = _fixture()
    # The record also carries derived fields (length, region) and a note; the
    # dataclass takes only its own, so filter rather than splatting blindly.
    names = {field.name for field in dataclasses.fields(GenomeRegion)}
    provenance = {k: v for k, v in data["provenance"].items() if k in names}
    return GenomeRegion(sequence=data["sequence"], **provenance)


def _served(**overrides) -> tuple:
    """A fetch stub serving the saved slice's body, with fields overridable."""
    data = _fixture()
    provenance = data["provenance"]
    payload = {
        "id": "chromosome:{assembly}:{chromosome}:{start}:{end}:1".format(**provenance),
        "molecule": "dna",
        "seq": data["sequence"],
    }
    payload.update(overrides)
    calls: list[str] = []

    def fetch(url: str, timeout: float) -> str:
        calls.append(url)
        return json.dumps(payload)

    return fetch, calls, data


def test_saved_slice_is_the_sequence_its_record_claims():
    data = _fixture()
    sequence = data["sequence"]
    provenance = data["provenance"]

    assert len(sequence) == provenance["length"]
    assert hashlib.sha256(sequence.encode("utf-8")).hexdigest() == provenance["sha256"]
    assert provenance["assembly"] == "GRCh38"
    assert provenance["provider"] == "ensembl"
    assert set(sequence) <= set("ACGTN")
    assert provenance["end"] - provenance["start"] + 1 == len(sequence)


def test_fetch_records_url_assembly_and_checksum_without_a_socket():
    fetch, calls, data = _served()

    region = fetch_region(REGION, retrieved="2026-11-01", fetch=fetch)

    assert calls and calls[0].startswith("https://rest.ensembl.org/sequence/region/human/")
    assert REGION in calls[0]
    assert region.sequence == data["sequence"]
    assert region.assembly == "GRCh38"
    assert region.length == data["provenance"]["length"]
    assert region.sha256 == data["provenance"]["sha256"]
    assert region.retrieved == "2026-11-01"
    assert region.region == REGION


def test_fetch_refuses_a_truncated_slice():
    """A short body must not become "no off-targets found"."""
    sequence = _fixture()["sequence"]
    fetch, _, _ = _served(seq=sequence[:-1])
    with pytest.raises(GenomeFetchError, match="truncated or shifted"):
        fetch_region(REGION, fetch=fetch)


def test_fetch_refuses_non_dna_and_a_missing_seq():
    length = len(_fixture()["sequence"])
    # Same length as requested, so the alphabet check is what fires.
    bad = "ACGT" * (length // 4) + "Z" * (length - 4 * (length // 4))
    fetch, _, _ = _served(seq=bad)
    with pytest.raises(GenomeFetchError, match="non-DNA"):
        fetch_region(REGION, fetch=fetch)

    fetch, _, _ = _served(seq=None)
    with pytest.raises(GenomeFetchError, match="no 'seq' field"):
        fetch_region(REGION, fetch=fetch)


def test_fetch_rejects_requests_it_will_not_make():
    with pytest.raises(ValueError, match="must look like"):
        fetch_region("chr13")
    with pytest.raises(ValueError, match="before its start"):
        fetch_region("13:100..50")
    with pytest.raises(ValueError, match="cap is"):
        fetch_region(f"13:1..{MAX_REGION_BASES + 1}")
    with pytest.raises(ValueError, match="unknown provider"):
        fetch_region(REGION, provider="ncbi")
    with pytest.raises(ValueError, match="strand"):
        fetch_region(REGION, strand=0)


def test_scan_region_runs_end_to_end_on_the_saved_slice():
    data = _fixture()
    region = _region()
    guide = data["guide"]["protospacer"]

    result = scan_region(guide, region)

    # The guide was taken from this slice, so its own locus must be found.
    assert result["exact_matches"] >= 1
    assert result["reference_length"] == region.length
    assert result["total_off_targets"] == (result["predicted_off_targets"]
                                           + result["bulge_off_targets"]
                                           + result["indel_off_targets"])
    assert result["counts_by_kind"] == {"mismatch_only": result["predicted_off_targets"],
                                        "bulge": result["bulge_off_targets"],
                                        "indel": result["indel_off_targets"]}

    provenance = result["reference_provenance"]
    assert provenance["source"] == "fetched"
    assert provenance["assembly"] == "GRCh38"
    assert provenance["sha256"] == region.sha256
    assert "sequence" not in provenance, "provenance describes the slice, it does not carry it"

    frame = result["coordinate_frame"]
    assert frame["region"] == REGION and frame["start"] == region.start
    for key in ("exact_sites", "top_sites", "top_gapped_sites"):
        for site in result[key]:
            assert site["genomic_start"] == region.start + site["forward_start"]
    # The pinned guide sits at the recorded offset, in assembly coordinates.
    assert data["guide"]["genomic_start"] in {s["genomic_start"] for s in result["exact_sites"]}


def test_scan_region_labels_a_plain_string_without_inventing_provenance():
    data = _fixture()
    result = scan_region(data["guide"]["protospacer"], data["sequence"])
    assert result["reference_provenance"] == {"source": "caller_supplied_string"}
    assert result["coordinate_frame"]["region"] == "caller_supplied_string"
    assert "genomic_start" not in result["exact_sites"][0]


def test_scan_region_refuses_an_empty_reference():
    with pytest.raises(ValueError, match="empty"):
        scan_region("GAGTCCGAGCAGAAGAAGAA", "")
