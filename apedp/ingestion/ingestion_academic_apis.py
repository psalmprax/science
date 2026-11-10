"""Academic literature discovery with retrieval provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

INSTITUTIONAL_TARGETS: dict[str, str] = {
    "Crossref": "https://api.crossref.org/works",
    "arXiv": "http://export.arxiv.org/api/query",
    "HAL Open Science": "https://api.archives-ouvertes.fr/search",
    "CERN Document Server": "https://cds.cern.ch/search",
    "NASA ADS": "https://api.adsabs.harvard.edu/v1/search",
    "NIST DLMF": "https://dlmf.nist.gov/",
    "Materials Project (LBNL)": "https://api.materialsproject.org/",
    "PTB Germany": "https://www.ptb.de/",
    "Wolfram Alpha": "https://api.wolframalpha.com/v2/query",
    "Wikidata": "https://query.wikidata.org/sparql",
    "zbMATH Open": "https://api.zbmath.org/v2/",
    "CODATA": "https://codata.org/",
    "PubChem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
    "MIT OCW": "https://ocw.mit.edu/",
    "Caltech JPL": "https://naif.jpl.nasa.gov/",
    "Princeton IAS": "https://www.ias.edu/",
    "Max Planck Society": "https://www.mpg.de/",
    "ETH Zurich": "https://ethz.ch/",
    "Stanford Encyclopedia": "https://plato.stanford.edu/",
    "Imperial College London": "https://www.imperial.ac.uk/",
    "University of Tokyo": "https://www.u-tokyo.ac.jp/",
}

CROSSREF_API_URL = INSTITUTIONAL_TARGETS["Crossref"]


@dataclass(frozen=True)
class LiteratureRecord:
    """Crossref work metadata with explicit source and retrieval provenance."""

    doi: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    publication_date: str | None
    landing_page: str
    work_type: str
    cited_by_count: int
    relevance_score: float | None
    source_url: str
    retrieved_at: str


@dataclass(frozen=True)
class LiteratureSearchResult:
    """A bounded, auditable metadata search; absence is not evidence of novelty."""

    query: str
    source: str
    source_url: str
    retrieved_at: str
    total_results: int
    records: tuple[LiteratureRecord, ...]

    def to_dict(self) -> dict:
        """Return JSON-compatible metadata and provenance."""
        return {
            "query": self.query,
            "source": self.source,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "total_results": self.total_results,
            "returned_results": len(self.records),
            "records": [asdict(record) for record in self.records],
            "scope_note": (
                "Crossref metadata search only; results are not a complete literature "
                "corpus, novelty determination, or legal opinion."
            ),
        }


def _publication_date(item: dict) -> str | None:
    for field in ("published-online", "published-print", "issued", "created"):
        date_parts = item.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return "-".join(f"{part:02d}" if index else str(part)
                            for index, part in enumerate(date_parts[0]))
    return None


def _author_name(author: dict) -> str:
    return " ".join(
        part.strip() for part in (author.get("given", ""), author.get("family", "")) if part.strip()
    )


def _to_literature_record(item: dict, retrieved_at: str) -> LiteratureRecord:
    doi = str(item.get("DOI", ""))
    work_url = f"{CROSSREF_API_URL}/{doi}" if doi else CROSSREF_API_URL
    score = item.get("score")
    return LiteratureRecord(
        doi=doi,
        title=next((str(title) for title in item.get("title", []) if title), ""),
        authors=tuple(name for author in item.get("author", [])
                      if (name := _author_name(author))),
        abstract=str(item.get("abstract", "")),
        publication_date=_publication_date(item),
        landing_page=str(item.get("URL", "")),
        work_type=str(item.get("type", "")),
        cited_by_count=int(item.get("is-referenced-by-count", 0)),
        relevance_score=float(score) if score is not None else None,
        source_url=work_url,
        retrieved_at=retrieved_at,
    )


def search_crossref(
    query: str,
    *,
    limit: int = 20,
    mailto: str | None = None,
    timeout: float = 10.0,
) -> LiteratureSearchResult:
    """Search Crossref works and return bounded metadata with retrieval provenance.

    This performs a metadata lookup, not full-text search. Keep the query and
    returned records with any downstream novelty assessment; a missing result
    only means Crossref did not return a match for this query.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    params = {"query.bibliographic": normalized_query, "rows": limit}
    if mailto:
        params["mailto"] = mailto
    request_url = f"{CROSSREF_API_URL}?{urlencode(params)}"
    # The scheme and authority are fixed by CROSSREF_API_URL; all parameters
    # are encoded as query data, so caller input cannot select a URL scheme.
    request = Request(  # noqa: S310  # nosec B310
        request_url,
        headers={"User-Agent": "APEDP literature metadata search/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310  # nosec B310
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Crossref literature search failed: {exc}") from exc

    try:
        message = payload["message"]
        items = message["items"]
        total_results = int(message["total-results"])
        if not isinstance(items, list):
            raise TypeError("items must be a list")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Crossref returned an invalid works response") from exc

    retrieved_at = datetime.now(UTC).isoformat()
    records = tuple(_to_literature_record(item, retrieved_at) for item in items[:limit])
    return LiteratureSearchResult(
        query=normalized_query,
        source="Crossref",
        source_url=request_url,
        retrieved_at=retrieved_at,
        total_results=total_results,
        records=records,
    )


def list_targets() -> dict[str, str]:
    """Return configured institutional targets."""
    return dict(INSTITUTIONAL_TARGETS)


__all__ = [
    "CROSSREF_API_URL",
    "INSTITUTIONAL_TARGETS",
    "LiteratureRecord",
    "LiteratureSearchResult",
    "list_targets",
    "search_crossref",
]
