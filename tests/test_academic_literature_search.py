"""Contract tests for the provenance-preserving Crossref literature search."""
from __future__ import annotations

import io
import json
from urllib.parse import parse_qs, urlparse

import pytest

from apedp.ingestion import LiteratureRecord, search_crossref
from apedp.ingestion import ingestion_academic_apis


def _crossref_response(items: list[dict], total_results: int = 1) -> io.BytesIO:
    payload = {"message": {"total-results": total_results, "items": items}}
    return io.BytesIO(json.dumps(payload).encode())


def test_search_crossref_returns_records_with_query_and_source_provenance(monkeypatch):
    observed: dict = {}

    def fake_urlopen(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return _crossref_response(
            [{
                "DOI": "10.1234/example",
                "title": ["A scientific result"],
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "abstract": "<jats:p>Abstract text</jats:p>",
                "published-print": {"date-parts": [[2024, 3, 5]]},
                "URL": "https://doi.org/10.1234/example",
                "type": "journal-article",
                "is-referenced-by-count": 8,
                "score": 12.5,
            }],
            total_results=30,
        )

    monkeypatch.setattr(ingestion_academic_apis, "urlopen", fake_urlopen)
    result = search_crossref("  scientific result  ", limit=5, mailto="research@example.org")

    assert result.query == "scientific result"
    assert result.source == "Crossref"
    assert result.total_results == 30
    assert len(result.records) == 1
    record = result.records[0]
    assert isinstance(record, LiteratureRecord)
    assert record.doi == "10.1234/example"
    assert record.title == "A scientific result"
    assert record.authors == ("Ada Lovelace",)
    assert record.publication_date == "2024-03-05"
    assert record.source_url.endswith("/10.1234/example")
    assert record.retrieved_at == result.retrieved_at
    assert observed["timeout"] == 10.0
    assert "APEDP" in observed["request"].get_header("User-agent")
    query = parse_qs(urlparse(observed["request"].full_url).query)
    assert query["query.bibliographic"] == ["scientific result"]
    assert query["rows"] == ["5"]
    assert query["mailto"] == ["research@example.org"]
    assert result.to_dict()["returned_results"] == 1
    assert "not a complete literature corpus" in result.to_dict()["scope_note"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": "   "}, "query must not be empty"),
        ({"query": "test", "limit": 0}, "limit must be between 1 and 100"),
        ({"query": "test", "limit": 101}, "limit must be between 1 and 100"),
        ({"query": "test", "timeout": 0}, "timeout must be positive"),
    ],
)
def test_search_crossref_rejects_invalid_request_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        search_crossref(**kwargs)


def test_search_crossref_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(
        ingestion_academic_apis, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"{}")
    )
    with pytest.raises(ValueError, match="invalid works response"):
        search_crossref("test")


def test_search_crossref_reports_transport_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ingestion_academic_apis.URLError("offline")

    monkeypatch.setattr(ingestion_academic_apis, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Crossref literature search failed"):
        search_crossref("test")
