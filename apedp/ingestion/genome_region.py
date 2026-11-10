"""Fetch a reference region from a public genome API, then search it for off-targets.

Status: experimental

``CRISPRDesigner.predict_off_targets`` takes a reference *string*; nothing in the
repository could produce a real one, so every run used a hand-written sequence.
This module closes that loop: :func:`fetch_region` retrieves a slice from the
Ensembl REST API and records where it came from, and :func:`scan_region` runs the
scan on it with that provenance carried into the result, so a reported site can be
traced to an assembly, a chromosome, a coordinate pair and a checksum.

Refusals are deliberate and loud. An HTTP failure, a body that is not DNA, a
slice whose length disagrees with the coordinates requested, or a region larger
than :data:`MAX_REGION_BASES` all raise :class:`GenomeFetchError` (or
``ValueError`` for a bad request). Returning a short or empty sequence would turn
a network or coordinate mistake into "no off-targets found", which is the one
answer this path must never invent.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from apedp.applied_synthetic_bio import CRISPRDesigner

#: Sent on every request, as the Ensembl terms ask.
USER_AGENT = "SciencePlatformAssurance/1.0 (contact@scienceplatform.org)"

#: Ensembl REST slice endpoint. Coordinates are 1-based inclusive, as the API expects.
ENSEMBL_REGION_URL = (
    "https://rest.ensembl.org/sequence/region/{species}/{region}:{strand}"
    "?content-type=application/json"
)

#: Refuse to pull unbounded ranges: this is a site search around a locus, not a
#: genome download. A chromosome would be ~250 Mb and would exhaust the caller's
#: memory before any biology happened.
MAX_REGION_BASES = 1_000_000

_DNA = frozenset("ACGTN")
_REGION_RE = re.compile(r"^(?P<chrom>[0-9A-Za-z_.]+):(?P<start>[0-9]+)\.\.(?P<end>[0-9]+)$")
_ASSEMBLY_RE = re.compile(r"^[a-z]+:(?P<assembly>[A-Za-z0-9_.]+):")


class GenomeFetchError(RuntimeError):
    """A region could not be fetched, or what came back was not the region asked for."""


@dataclass(frozen=True)
class GenomeRegion:
    """A fetched slice plus everything needed to find it again."""

    sequence: str
    species: str
    assembly: str
    chromosome: str
    start: int
    end: int
    strand: int
    provider: str
    url: str
    retrieved: str
    sha256: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def region(self) -> str:
        return f"{self.chromosome}:{self.start}..{self.end}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # ``sequence`` is not duplicated into the record: the checksum identifies
        # it, and a provenance block should describe the slice, not carry it.
        data.pop("sequence")
        data["length"] = self.length
        data["region"] = self.region
        return data


def _http_get(url: str, timeout: float) -> str:
    # The scheme/host are the ENSEMBL_REGION_URL constant; only the region varies.
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https Ensembl endpoint: {url}")
    request = urllib.request.Request(  # noqa: S310 - hardcoded https host
        url, headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310  # noqa: S310 - scheme checked immediately above
        return response.read().decode("utf-8")


def _assembly_from_id(identifier: Any) -> str:
    """``chromosome:GRCh38:13:...`` -> ``GRCh38``; empty when the API omits it."""
    match = _ASSEMBLY_RE.match(str(identifier or ""))
    return match["assembly"] if match else ""


def fetch_region(region: str, *, species: str = "human", provider: str = "ensembl",
                 strand: int = 1, timeout: float = 30.0, retrieved: str | None = None,
                 fetch: Callable[[str, float], str] | None = None) -> GenomeRegion:
    """Fetch ``"13:32930000..32932000"`` and validate it before anyone scans it.

    ``fetch`` is injectable so tests can serve a saved body without a socket; the
    validation below is the same for a stub and for the live service.
    """
    match = _REGION_RE.match(str(region).strip())
    if match is None:
        raise ValueError(f"region must look like '13:32930000..32932000', got {region!r}")
    chromosome = match["chrom"]
    start, end = int(match["start"]), int(match["end"])
    if end < start:
        raise ValueError(f"region end {end} is before its start {start}")
    span = end - start + 1
    if span > MAX_REGION_BASES:
        raise ValueError(
            f"region spans {span} bases; the cap is {MAX_REGION_BASES}. "
            "Fetch the locus you intend to search, not a whole chromosome.")
    if provider not in {"ensembl"}:
        raise ValueError(f"unknown provider {provider!r}; supported: 'ensembl'")
    if strand not in (1, -1):
        raise ValueError("strand must be 1 or -1")

    url = ENSEMBL_REGION_URL.format(species=species, region=f"{chromosome}:{start}..{end}",
                                    strand=strand)
    try:
        body = (fetch or _http_get)(url, timeout)
    except OSError as exc:
        raise GenomeFetchError(f"could not fetch {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GenomeFetchError(f"{url} did not return JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("seq"), str):
        raise GenomeFetchError(f"{url} returned no 'seq' field")

    sequence = payload["seq"].upper()
    unexpected = sorted(set(sequence) - _DNA)
    if unexpected:
        raise GenomeFetchError(f"{url} returned non-DNA characters: {unexpected}")
    if len(sequence) != span:
        raise GenomeFetchError(
            f"asked {url} for {span} bases and got {len(sequence)}: refusing to scan a "
            "truncated or shifted slice")

    return GenomeRegion(
        sequence=sequence,
        species=species,
        assembly=_assembly_from_id(payload.get("id")) or "unknown",
        chromosome=chromosome,
        start=start,
        end=end,
        strand=strand,
        provider=provider,
        url=url,
        retrieved=retrieved or datetime.now(timezone.utc).date().isoformat(),
        sha256=hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
    )


def scan_region(guide_rna: str, region: GenomeRegion | str, **scan_kwargs: Any) -> dict[str, Any]:
    """Run the off-target scan on a region and attach where the reference came from.

    Sites also gain ``genomic_start``: ``forward_start`` is 0-based within the
    slice, so a reported site can be read against the assembly rather than
    against a string offset.
    """
    if isinstance(region, GenomeRegion):
        sequence = region.sequence
        provenance = {"source": "fetched", **region.to_dict()}
        frame = {"region": region.region, "start": region.start, "strand": region.strand,
                 "note": "'forward_start' is 0-based within the slice; 'genomic_start' "
                         "adds the slice start"}
    else:
        sequence = str(region)
        provenance = {"source": "caller_supplied_string"}
        frame = {"region": "caller_supplied_string", "start": 0, "strand": 1,
                 "note": "'forward_start' is 0-based within the supplied string"}

    result = CRISPRDesigner.predict_off_targets(guide_rna, sequence, **scan_kwargs)
    result["reference_provenance"] = provenance
    result["coordinate_frame"] = frame
    if isinstance(region, GenomeRegion):
        for key in ("exact_sites", "top_sites", "top_gapped_sites"):
            for site in result.get(key, []):
                site["genomic_start"] = region.start + site["forward_start"]
    return result
