"""The bulk-source paths, pinned offline.

Two things matter here and neither needs the network: a source that cannot be
queried must **refuse with the measured reason and exit 2** rather than return an
empty list that would read as "no patents found", and the EPO OPS claims parser must
keep claim numbering and normalise whitespace when given a response in the documented
shape.

The XML below is a **synthetic fixture** built to the documented OPS claims structure.
It is not a fetched record and never enters the corpus; it exists so the parser's
behaviour is testable without credentials.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "grow_patent_corpus.py"
    spec = importlib.util.spec_from_file_location("_grow_patent_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


grow = _module()

FIXTURES = ROOT / "data" / "patents" / "ops_fixtures"
OPS_PROVENANCE = FIXTURES / "PROVENANCE.json"
CLAIMS_RECORDING = FIXTURES / "claims_EP4334731.xml"


def _recorded(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()

_EPO_CREDENTIALS = (
    "EPO_OPS_KEY", "EPO_OPS_SECRET", "EPO_OPS_CONSUMER_KEY", "EPO_OPS_CONSUMER_SECRET",
)

_EPO_CLAIMS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC fixture: constructed to the documented OPS claims shape, not a fetched record. -->
<ops:world-patent-data xmlns:ops="http://ops.epo.org">
  <ops:exchange-documents>
    <ops:exchange-document>
      <claims>
        <claim num="1">
          <claim-text>A method comprising determining a <b>ratio</b> of an output
          power to an input power.</claim-text>
        </claim>
        <claim num="2"><claim-text>  The method  of claim 1,  wherein the ratio is filtered.  </claim-text></claim>
      </claims>
    </ops:exchange-document>
  </ops:exchange-documents>
</ops:world-patent-data>
"""


_EPO_SEARCH_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<!-- SYNTHETIC fixture: documented OPS search shape, not a fetched record. -->
<ops:world-patent-data xmlns:ops="http://ops.epo.org">
  <ops:search-result>
    <ops:exchange-documents>
      <ops:exchange-document>
        <bibliographic-data>
          <publication-reference>
            <document-id document-id-type="docdb">
              <country>EP</country><doc-number>4334731</doc-number><kind>B1</kind>
            </document-id>
          </publication-reference>
        </bibliographic-data>
      </ops:exchange-document>
    </ops:exchange-documents>
  </ops:search-result>
</ops:world-patent-data>
"""


def _clear_credentials(monkeypatch) -> None:
    for name in ("PATENTSVIEW_API_KEY", *_EPO_CREDENTIALS):
        monkeypatch.delenv(name, raising=False)


def test_patentsview_refuses_without_a_key(monkeypatch):
    _clear_credentials(monkeypatch)
    with pytest.raises(grow.SourceUnavailable) as excinfo:
        grow.patentsview_search("battery state of health")
    assert "PATENTSVIEW_API_KEY" in str(excinfo.value)
    assert excinfo.value.source == "patentsview"
    assert excinfo.value.status is None      # nothing was reached, nothing was 404


def test_epo_ops_refuses_without_credentials(monkeypatch):
    _clear_credentials(monkeypatch)
    with pytest.raises(grow.SourceUnavailable, match="EPO_OPS_KEY"):
        grow.epo_ops_token()


def test_epo_claims_parser_keeps_numbering_and_normalises_whitespace():
    claims = grow.parse_epo_claims(_EPO_CLAIMS_XML)
    assert [claim["num"] for claim in claims] == ["1", "2"]
    assert claims[0]["text"] == (
        "A method comprising determining a ratio of an output power to an input power.")
    assert claims[1]["text"] == (
        "The method of claim 1, wherein the ratio is filtered.")
    assert "\n" not in claims[0]["text"]


def test_epo_ops_claims_path_runs_end_to_end_through_a_replay(monkeypatch):
    """Token -> search -> claims -> raw artefact, with no network and no credentials.

    This is the path that would grow the corpus without Google. Every step goes
    through the real adapter functions, so the URLs, the Bearer headers, the numbering
    and the digest are the ones a live run would produce; only the transport is
    replayed. Live use still needs EPO_OPS_KEY/EPO_OPS_SECRET.
    """
    monkeypatch.setenv("EPO_OPS_KEY", "replay-key")
    monkeypatch.setenv("EPO_OPS_SECRET", "replay-secret")
    calls: list[dict] = []

    def transport(url, timeout=45, headers=None, data=None):
        calls.append({"url": url, "headers": headers or {}, "data": data})
        if "auth" in url:
            return 200, b'{"access_token": "replay-token"}'
        if "search" in url:
            return 200, _EPO_SEARCH_XML
        return 200, _EPO_CLAIMS_XML

    token = grow.epo_ops_token(transport=transport)
    assert token.startswith("replay-")
    assert calls[0]["headers"]["Authorization"].startswith("Basic ")
    assert calls[0]["data"] == b"grant_type=client_credentials"

    hits = grow.epo_ops_search(token, "battery state of health", 1, transport=transport)
    assert [hit["publication_number"] for hit in hits] == ["EP4334731B1"]
    assert hits[0]["claims_source"] == "epo_ops"
    assert calls[1]["headers"]["Authorization"] == "Bearer replay-token"

    artefact = grow.fetch_publication_epo("EP4334731B1", token, transport=transport)
    assert [claim["num"] for claim in artefact["claims"]] == ["1", "2"]
    assert artefact["http_status"] == 200
    assert artefact["bytes"] == len(_EPO_CLAIMS_XML)
    assert len(artefact["html_sha256"]) == 64
    assert artefact["source"].endswith("/EP4334731B1/claims")
    assert artefact["extractor"].endswith("fetch_publication_epo")
    assert calls[2]["headers"]["Accept"] == "application/xml"


def test_every_recorded_ops_fixture_is_attributed_and_unmodified():
    """A replayed fixture is only evidence if its bytes still match the recording."""
    provenance = json.loads(OPS_PROVENANCE.read_text(encoding="utf-8"))
    fixtures = provenance["fixtures"]
    assert fixtures, "no recorded OPS response is committed"
    for name, meta in fixtures.items():
        payload = _recorded(name)
        assert hashlib.sha256(payload).hexdigest() == meta["sha256"], name
        assert meta["url"].startswith("https://ops.epo.org/"), name
        assert isinstance(meta["http_status"], int), name
        assert meta["retrieved_utc"], name


def test_recorded_ops_claims_refusal_replays_through_the_adapter():
    """The recorded 403 body is replayed through the adapter, URL and header included."""
    body = _recorded("claims_EP4334731_403.xml")
    assert b"<code>403</code>" in body
    seen: dict[str, object] = {}

    def transport(url, timeout=45, headers=None, data=None):
        seen["url"] = url
        seen["headers"] = headers
        return 403, body

    with pytest.raises(grow.SourceUnavailable) as excinfo:
        grow.epo_ops_claims("EP4334731", "token-from-recording", transport=transport)
    assert excinfo.value.status == 403
    assert "claims HTTP 403" in str(excinfo.value)
    assert str(seen["url"]).endswith("publication/docdb/EP4334731/claims")
    assert seen["headers"]["Authorization"] == "Bearer token-from-recording"
    assert seen["headers"]["Accept"] == "application/xml"


def test_recorded_ops_token_refusal_names_the_missing_identifier():
    body = _recorded("token_401.xml")
    assert b"<code>401</code>" in body
    assert b"Client identifier is required" in body


@pytest.mark.skipif(
    not CLAIMS_RECORDING.exists(),
    reason=("no authenticated OPS claims response is recorded; run "
            "python3 scripts/record_epo_ops_fixture.py --number EP4334731 with "
            "EPO_OPS_KEY and EPO_OPS_SECRET to record one"))
def test_recorded_authenticated_claims_response_replays():
    """A recorded authenticated response is replayed through the adapter, not the parser.

    Skipped while no recording exists: the committed fixtures are the two refusals,
    and a synthetic body is not a recording. The skip reason names the command that
    produces one.
    """
    body = CLAIMS_RECORDING.read_bytes()

    def transport(url, timeout=45, headers=None, data=None):
        return 200, body

    claims = grow.epo_ops_claims("EP4334731", "token-from-recording", transport=transport)
    assert claims, "the recording carries no claims"
    assert claims[0]["num"].strip()
    assert all(claim["text"].strip() for claim in claims)
    numbers = [claim["num"] for claim in claims]
    assert len(numbers) == len(set(numbers)), "claim numbering repeats in the recording"


def test_epo_claims_parser_refuses_a_response_without_claim_text():
    empty = b"<ops:world-patent-data xmlns:ops=\"http://ops.epo.org\"><claims/></ops:world-patent-data>"
    with pytest.raises(grow.SourceUnavailable, match="no <claim-text>"):
        grow.parse_epo_claims(empty)


def test_epo_claims_parser_refuses_non_xml():
    with pytest.raises(grow.SourceUnavailable, match="not XML"):
        grow.parse_epo_claims(b"<html>blocked</html")


def test_every_harvest_attempt_is_logged_with_its_measured_status():
    """A blocked retry is recorded, so "we tried again later" is auditable."""
    attempts = ROOT / "data" / "patents" / "harvest_attempts.jsonl"
    lines = [line for line in attempts.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "no harvest attempt is logged"
    entries = [json.loads(line) for line in lines]
    for entry in entries:
        assert entry["attempted_utc"]
        assert entry["endpoint"].startswith("https://patents.google.com/")
        assert entry["outcome"] in {"open", "blocked", "unreachable"}
        assert isinstance(entry["records_added"], int)
    blocked = [entry for entry in entries if entry["outcome"] != "open"]
    assert blocked, "the log should record at least one blocked attempt"
    assert all(entry["records_added"] == 0 for entry in blocked)


def test_each_new_attempt_records_the_decisive_split_it_left_unchanged():
    """A blocked attempt states what the corpus still is, not only what it did not get."""
    attempts = ROOT / "data" / "patents" / "harvest_attempts.jsonl"
    entries = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    with_split = [entry for entry in entries if "corpus_split" in entry]
    assert with_split, "no attempt recorded the corpus split"
    current_keys = {"EP4334731B1", "US12021452B2", "US20250321289A1", "US8965721B2"}
    for entry in with_split:
        split = entry["corpus_split"]
        assert split["decisive_claims"] + split["operand_only_claims"] == split["searchable_claims"]
        # Earlier splits predate the records that grew the corpus, so they are a
        # subset; every split must still agree on the record it already held.
        assert set(split["refused_claims"]) <= current_keys
        assert split["refused_claims"]["US8965721B2"] == ["1", "8"]
    populations = [entry["corpus_split"]["records"] for entry in with_split]
    assert populations == sorted(populations), "a recorded split should never shrink"
    # The split stamped at harvest time is the population the attempt started from, so
    # the current corpus is read from disk rather than from the last log line.
    assert grow.corpus_split() == {
        "records": 29,
        "searchable_claims": 15,
        "decisive_claims": 8,
        "operand_only_claims": 7,
        "refused_claims": {"EP4334731B1": ["2", "7", "8"],
                           "US12021452B2": ["1"],
                           "US20250321289A1": ["11"],
                           "US8965721B2": ["1", "8"]},
    }


def test_source_probe_records_every_endpoint_tried():
    """A refusal is only auditable if the attempt is written down.

    The table must name every endpoint tried and say what each one is, so a reader
    can tell "needs a key" from "would not serve claims even with a key".
    """
    text = (ROOT / "data" / "patents" / "SOURCES.md").read_text(encoding="utf-8")
    for name in ("patentsview_api", "epo_ops", "google_patents_query",
                 "uspto_ds_api", "uspto_open_data"):
        assert f"`{name}`" in text, name
    assert "needs OAuth" in text
    assert "no claim text" in text
    assert "needs a key" in text
    assert "portal page, not an API response" in text


def test_named_bulk_source_exits_two_and_says_why(monkeypatch, capsys):
    """An unqueried source must not look like a searched one."""
    _clear_credentials(monkeypatch)
    code = grow.main(["--source", "patentsview", "--query", "battery", "--limit", "2"])
    assert code == 2
    err = capsys.readouterr().err
    assert "source unavailable" in err
    assert "absence of records is not absence of patents" in err


def test_epo_ops_source_exits_two_without_credentials(monkeypatch, capsys):
    _clear_credentials(monkeypatch)
    code = grow.main(["--source", "epo-ops", "--query", "battery", "--limit", "2"])
    assert code == 2
    assert "EPO_OPS_KEY" in capsys.readouterr().err
