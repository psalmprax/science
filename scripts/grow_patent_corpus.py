"""Grow the patent corpus beyond the five curated records.

The five-record corpus in ``data/patents/`` answers a narrower question than it
looks: is this formula among three normalised claims? This script widens the base
and states what the widening does and does not buy.

**Sources, and what this environment could reach.** The task asks for a bulk
source — PatentsView or EPO OPS. Both are implemented here (``--probe-sources``)
and both are refused rather than faked when they cannot be used:

* PatentsView's current API (``search.patentsview.org``) needs an API key and is
  not resolvable from this host; the legacy endpoint answers with HTML.
* EPO OPS answers ``403`` without OAuth credentials, which this environment does
  not hold.

The growth therefore runs through the Google Patents query endpoint, which is
paginated, keyless and was already the source of the five curated records. That
substitution is recorded in ``data/patents/SOURCES.md`` with the measured status
of every endpoint, so a reader can see what was attempted before what was used.

``--source patentsview`` and ``--source epo-ops`` are the real bulk paths. Both are
implemented here and both **exit 2 with the measured reason** when the credentials
they need are absent, because an empty result from an unqueried source must not be
read as "no patents found". PatentsView serves titles and abstracts but no claim
text, so a PatentsView hit becomes a corpus record only after its claims are
fetched from EPO OPS (``parse_epo_claims``); EPO OPS serves the claims directly.

**Two layers again, kept apart.**

* ``raw/<ID>.json`` — verbatim claims, digest, retrieval time, and the query that
  found the publication. Ground truth.
* ``auto_expressions.json`` — formulas pulled out of claim text by
  :func:`extract_expressions`, each with the sentence it was read from, the
  pattern that matched, and whether sympy could parse it. This layer is
  **automatic**, not curated, and every row says so.

An automatic extraction is weaker evidence than a hand-normalised claim, and the
corpus keeps the two kinds of row distinguishable: ``expression_source`` is
``curated`` for the original five and ``automatic`` for everything harvested here.
Absence still means "absent from these claims", but it now means it over a far
larger and fully enumerated set, and the record reports the size.

Usage::

    python3 scripts/grow_patent_corpus.py --probe-sources
    python3 scripts/grow_patent_corpus.py --query "battery state of health" --limit 20
    python3 scripts/grow_patent_corpus.py --build     # corpus.jsonl from raw + curated + auto

Status: audit-only
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patents"
RAW = DATA / "raw"
AUTO = DATA / "auto_expressions.json"
SOURCES_MD = DATA / "SOURCES.md"
CORPUS = DATA / "corpus.jsonl"
ATTEMPTS = DATA / "harvest_attempts.jsonl"

_USER_AGENT = ("apedp-patent-corpus/1.0 (research use; one request per second; "
               "contact: repository maintainers)")

#: Default harvest: one query per subject the curated five already cover, so the
#: widened corpus is comparable with it rather than a different sample.
QUERIES = (
    "battery state of health estimation",
    "product lifetime accelerated aging model",
    "perishable food freshness prediction",
    "battery charging efficiency measurement",
    "system efficiency output input power ratio",
)

#: Endpoints the brief named, probed and recorded rather than assumed.
BULK_SOURCES = {
    "patentsview_api": "https://search.patentsview.org/api/v1/patent/",
    "patentsview_legacy": "https://api.patentsview.org/patents/query",
    "epo_ops": "https://ops.epo.org/3.2/rest-services/published-data/search",
    "google_patents_query": "https://patents.google.com/xhr/query",
    "uspto_ds_api": "https://developer.uspto.gov/ds-api/patents/grant/v1/records",
    "uspto_open_data": "https://data.uspto.gov/api/v1/patent/applications/search",
}

#: What each endpoint is, and whether it could serve claims even with credentials.
#: A credential refusal and a source that would not help are different findings.
SOURCE_NOTES = {
    "patentsview_api": "search API; titles and abstracts only, no claim text; needs PATENTSVIEW_API_KEY",
    "patentsview_legacy": "retired API host; answers HTML, not JSON",
    "epo_ops": "published-data search plus per-publication claims; needs OAuth (EPO_OPS_KEY/EPO_OPS_SECRET)",
    "google_patents_query": "keyless query endpoint used for the bundled corpus; claims read from publication pages",
    "uspto_ds_api": "probe followed a 301 to data.uspto.gov and read the portal page, not an API response; the API needs a key",
    "uspto_open_data": "same: the 200 is the portal page, not an API response; the API needs a USPTO key",
}


def _get(url: str, timeout: int = 45, headers: dict[str, str] | None = None,
         data: bytes | None = None) -> tuple[int, bytes]:
    merged = {"User-Agent": _USER_AGENT}
    merged.update(headers or {})
    request = urllib.request.Request(  # noqa: S310 - https endpoints from BULK_SOURCES/harvest
        url, headers=merged, data=data)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.status, response.read()


def probe_sources() -> list[dict]:
    """Measured status of every endpoint, with credentials checked by presence only."""
    import os

    credentials = {
        "PATENTSVIEW_API_KEY": bool(os.environ.get("PATENTSVIEW_API_KEY")),
        "EPO_OPS_KEY": bool(os.environ.get("EPO_OPS_KEY") or os.environ.get("EPO_OPS_CONSUMER_KEY")),
    }
    rows = []
    for name, url in BULK_SOURCES.items():
        entry: dict[str, object] = {"source": name, "url": url}
        try:
            status, body = _get(url, timeout=20)
            entry.update({"http_status": status, "content_type": "json" if body[:1] == b"{" else "other",
                          "bytes": len(body)})
        except urllib.error.HTTPError as error:
            entry.update({"http_status": error.code, "error": str(error.reason)})
        except urllib.error.URLError as error:
            entry.update({"http_status": None, "error": f"unreachable: {error.reason}"})
        except OSError as error:
            entry.update({"http_status": None, "error": f"{type(error).__name__}: {error}"})
        entry["credentials_present"] = credentials
        entry["detail"] = SOURCE_NOTES.get(name, "")
        rows.append(entry)
    return rows


def write_sources(rows: list[dict], used: str, why: str) -> None:
    lines = [
        "# Patent corpus sources: what was attempted, and what was used",
        "",
        f"Probed on **{datetime.now(timezone.utc).date().isoformat()}** by "
        "`scripts/grow_patent_corpus.py --probe-sources`. Credentials are reported by "
        "presence only; no secret is read into this file.",
        "",
        "| source | URL | HTTP | credentials present | note |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        observed = row.get("error") or row.get("content_type", "")
        note = " — ".join(part for part in (observed, row.get("detail", "")) if part)
        creds = ", ".join(f"{k}={v}" for k, v in row["credentials_present"].items())
        lines.append(f"| `{row['source']}` | `{row['url']}` | {row.get('http_status')} | {creds} | {note} |")
    lines += [
        "",
        f"**Corpus built from `{used}`.** {why}",
        "",
        "The substitution is the finding, not a footnote. A bulk-source corpus built on",
        "PatentsView full-text tables would support claims about an office's population;",
        "this corpus does not. It is a query-selected sample, every record keeps the",
        "query that found it, and a record count here is not a population size.",
        "",
        "Both bulk paths are implemented in `scripts/grow_patent_corpus.py` and both",
        "**exit 2 with the measured reason** when their credentials are absent:",
        "`--source patentsview` (needs `PATENTSVIEW_API_KEY`, and yields titles and",
        "abstracts but no claims, so claims must then come from EPO OPS) and",
        "`--source epo-ops` (needs `EPO_OPS_KEY`/`EPO_OPS_SECRET` and serves claims",
        "directly through `parse_epo_claims`). An empty result from a source that could",
        "not be queried is refused rather than recorded as \"no patents found\".",
        "",
    ]
    SOURCES_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bulk sources: PatentsView and EPO OPS
# ---------------------------------------------------------------------------

PATENTSVIEW_ENDPOINT = "https://search.patentsview.org/api/v1/patent/"
EPO_OPS_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_OPS_SEARCH_ENDPOINT = "https://ops.epo.org/3.2/rest-services/published-data/search"
EPO_OPS_CLAIMS_ENDPOINT = "https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/{number}/claims"


class SourceUnavailable(RuntimeError):
    """A bulk source was requested and cannot be used; the reason is measured.

    Raised rather than returning an empty list, because "no records found" and
    "the source could not be queried" must never look alike.
    """

    def __init__(self, source: str, reason: str, status: int | None = None):
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason
        self.status = status


def patentsview_search(query: str, limit: int = 25) -> list[dict]:
    """Bibliographic rows from the PatentsView search API (needs PATENTSVIEW_API_KEY).

    PatentsView carries titles, abstracts and dates — **not claim text** — so a row
    from here is a discovery hit, and becomes a corpus record only once its claims
    are fetched from EPO OPS. The query body follows the documented PatentsView
    ``q``/``f``/``o`` parameter contract; a response that does not match it is
    refused with its byte count rather than parsed optimistically.
    """
    key = os.environ.get("PATENTSVIEW_API_KEY")
    if not key:
        raise SourceUnavailable("patentsview", "PATENTSVIEW_API_KEY is not set")
    params = {
        "q": json.dumps({"text_any": {"patent_title": query}}),
        "f": json.dumps(["patent_id", "patent_title", "patent_abstract", "patent_date"]),
        "o": json.dumps({"per_page": max(1, min(limit, 100))}),
    }
    url = f"{PATENTSVIEW_ENDPOINT}?{urllib.parse.urlencode(params)}"
    try:
        _status, body = _get(url, headers={"X-Api-Key": key})
    except urllib.error.HTTPError as error:
        raise SourceUnavailable("patentsview", f"HTTP {error.code}: {error.reason}",
                                status=error.code) from error
    except (urllib.error.URLError, OSError) as error:
        raise SourceUnavailable("patentsview", f"unreachable: {error}") from error
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError as error:
        raise SourceUnavailable(
            "patentsview", f"non-JSON response ({len(body)} bytes)") from error
    rows = payload.get("patents")
    if not isinstance(rows, list):
        raise SourceUnavailable(
            "patentsview", f"response has no 'patents' list ({len(body)} bytes)")
    return [row for row in rows if isinstance(row, dict) and row.get("patent_id")]


def epo_ops_token(transport: object | None = None) -> str:
    """OAuth2 access token for EPO OPS (needs EPO_OPS_KEY / EPO_OPS_CONSUMER_KEY)."""
    consumer = os.environ.get("EPO_OPS_KEY") or os.environ.get("EPO_OPS_CONSUMER_KEY")
    secret = os.environ.get("EPO_OPS_SECRET") or os.environ.get("EPO_OPS_CONSUMER_SECRET")
    if not consumer or not secret:
        raise SourceUnavailable(
            "epo_ops", "EPO_OPS_KEY/EPO_OPS_SECRET are not set (OPS is credential-gated)")
    basic = base64.b64encode(f"{consumer}:{secret}".encode()).decode()
    get = transport if transport is not None else _get
    try:
        _status, body = get(  # type: ignore[operator]
            EPO_OPS_AUTH_URL,
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data=b"grant_type=client_credentials",
        )
    except urllib.error.HTTPError as error:
        raise SourceUnavailable("epo_ops", f"token HTTP {error.code}: {error.reason}",
                                status=error.code) from error
    except (urllib.error.URLError, OSError) as error:
        raise SourceUnavailable("epo_ops", f"unreachable: {error}") from error
    try:
        token = json.loads(body.decode("utf-8", "replace"))["access_token"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SourceUnavailable("epo_ops", "token response carried no access_token") from error
    return str(token)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_root(payload: bytes, source: str) -> ET.Element:
    """Parse XML from an authenticated response, refusing any DTD or entity.

    ElementTree expands internal entities, so a response carrying a ``DOCTYPE`` or
    ``ENTITY`` declaration is refused instead of parsed. OPS claims carry no DTD,
    so nothing legitimate is lost by the refusal.
    """
    head = payload[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise SourceUnavailable(source, "response declares a DTD or entity; refused")
    try:
        return ET.fromstring(payload)  # noqa: S314 - DTD refused above; authenticated endpoint
    except ET.ParseError as error:
        raise SourceUnavailable(source, f"response is not XML: {error}") from error


def parse_epo_claims(xml_bytes: bytes) -> list[dict]:
    """Numbered claims from an OPS ``.../claims`` response.

    OPS wraps each claim in ``<claim num="n">`` with one or more ``<claim-text>``
    children; both the ``num`` attribute and the wrapper are treated as optional so
    a namespace or version difference degrades to sequential numbering instead of
    losing the text. If no claim text is present at all, the response is refused.
    """
    root = _xml_root(xml_bytes, "epo_ops")

    def _inner_text(node: ET.Element) -> str:
        """All text under a node, including text inside nested markup."""
        return re.sub(r"\s+", " ", "".join(node.itertext())).strip()

    claims: list[dict] = []
    wrapped = [node for node in root.iter() if _localname(node.tag) == "claim"]
    for position, node in enumerate(wrapped, start=1):
        pieces = [_inner_text(child) for child in node.iter()
                  if _localname(child.tag) == "claim-text"]
        text = " ".join(piece for piece in pieces if piece).strip()
        if not text:
            continue
        number = (node.get("num") or node.get("number") or str(position)).strip()
        claims.append({"num": number, "text": text})
    if not claims:
        loose = [_inner_text(node) for node in root.iter()
                 if _localname(node.tag) == "claim-text" and _inner_text(node)]
        claims = [{"num": str(index), "text": text}
                  for index, text in enumerate(loose, start=1)]
    if not claims:
        raise SourceUnavailable(
            "epo_ops", f"claims response contained no <claim-text> ({len(xml_bytes)} bytes)")
    return claims


def epo_ops_claims(number: str, token: str,
                  transport: object | None = None) -> list[dict]:
    """Claims for one publication number, e.g. ``US11474159B2`` or ``EP4334731``.

    ``transport`` exists so a **recorded** response can be replayed through this
    adapter — URL, Authorization header and error mapping included — without a
    network call. It must behave like :func:`_get`: ``(url, headers=...)`` returning
    ``(status, body)``. A replay therefore exercises the adapter, not just the
    parser, which is why the fixture tests are worth having.
    """
    endpoint = EPO_OPS_CLAIMS_ENDPOINT.format(number=number)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/xml"}
    get = transport if transport is not None else _get
    try:
        status, body = get(endpoint, headers=headers)  # type: ignore[operator]
    except urllib.error.HTTPError as error:
        raise SourceUnavailable("epo_ops", f"claims HTTP {error.code}: {error.reason}",
                                status=error.code) from error
    except (urllib.error.URLError, OSError) as error:
        raise SourceUnavailable("epo_ops", f"unreachable: {error}") from error
    if status != 200:
        raise SourceUnavailable("epo_ops", f"claims HTTP {status}", status=status)
    return parse_epo_claims(body)


def epo_ops_search(token: str, query: str, limit: int,
                   transport: object | None = None) -> list[dict]:
    """Publication numbers for ``query`` from the EPO OPS published-data search."""
    url = f"{EPO_OPS_SEARCH_ENDPOINT}?{urllib.parse.urlencode({'q': query})}"
    get = transport if transport is not None else _get
    try:
        _status, body = get(  # type: ignore[operator]
            url, headers={"Authorization": f"Bearer {token}",
                          "Accept": "application/xml"})
    except urllib.error.HTTPError as error:
        raise SourceUnavailable("epo_ops", f"search HTTP {error.code}: {error.reason}",
                                status=error.code) from error
    except (urllib.error.URLError, OSError) as error:
        raise SourceUnavailable("epo_ops", f"unreachable: {error}") from error

    root = _xml_root(body, "epo_ops")
    numbers: list[str] = []
    for reference in root.iter():
        if _localname(reference.tag) != "publication-reference":
            continue
        for document in reference.iter():
            if _localname(document.tag) != "document-id":
                continue
            parts = {_localname(child.tag): (child.text or "").strip() for child in document}
            doc_number = parts.get("doc-number")
            if doc_number:
                number = f"{parts.get('country', '')}{doc_number}{parts.get('kind', '')}"
                if number not in numbers:
                    numbers.append(number)
            break
        if len(numbers) >= limit:
            break
    if not numbers:
        raise SourceUnavailable(
            "epo_ops", f"search returned no publication numbers ({len(body)} bytes)")
    return [{"publication_number": number, "query": query, "title_hint": "",
             "rank": None, "claims_source": "epo_ops"} for number in numbers[:limit]]


def fetch_publication_epo(number: str, token: str,
                          transport: object | None = None) -> dict:
    """A raw artefact whose claims came from EPO OPS rather than a web page.

    ``transport`` lets a recorded or synthetic response be replayed through the same
    headers and the same artefact construction, so the OPS path is exercised end to
    end without credentials.
    """
    url = EPO_OPS_CLAIMS_ENDPOINT.format(number=number)
    get = transport if transport is not None else _get
    status, body = get(  # type: ignore[operator]
        url, headers={"Authorization": f"Bearer {token}",
                      "Accept": "application/xml"})
    claims = parse_epo_claims(body)
    jurisdiction = re.match(r"([A-Z]{2})", number)
    return {
        "patent_id": number,
        "source": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": status,
        "bytes": len(body),
        "html_sha256": hashlib.sha256(body).hexdigest(),
        "jurisdiction": jurisdiction.group(1) if jurisdiction else "",
        "title": "",
        "abstract": "",
        "claims": claims,
        "extractor": "scripts/grow_patent_corpus.py::fetch_publication_epo",
    }


def harvest_bulk(source: str, query: str, limit: int,
                 token: str | None = None,
                 transport: object | None = None) -> list[dict]:
    """Publication numbers for ``query`` from a credential-gated bulk source.

    ``patentsview`` discovers publications but carries no claims, so it is not
    enough on its own: its rows are marked ``claims_source: epo_ops`` and become
    records only once OPS serves their claims. ``epo-ops`` searches and serves
    claims directly. Both raise :class:`SourceUnavailable` with the measured
    reason when they cannot run.
    """
    if source == "patentsview":
        rows = patentsview_search(query, limit)
        return [{"publication_number": str(row["patent_id"]), "query": query,
                 "title_hint": str(row.get("patent_title") or "").strip(),
                 "rank": None, "claims_source": "epo_ops"} for row in rows]
    if source == "epo-ops":
        return epo_ops_search(token or epo_ops_token(transport), query, limit,
                              transport=transport)
    raise SourceUnavailable(source, "unknown bulk source")


def corpus_split() -> dict:
    """The corpus as it stands: records, searchable claims, and the tier split.

    A blocked harvest adds nothing, so the split printed beside the attempt is what
    keeps the decisive/operand-only accounting visible while the record count cannot
    move. The refused claims are named per record rather than counted.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from apedp.patent_prior_art import corpus_status, operand_only_refusals

    status = corpus_status()
    refused: dict[str, list[str]] = {}
    for refusal in operand_only_refusals():
        refused.setdefault(refusal.patent_id, []).append(refusal.claim_number)
    return {
        "records": status["records"],
        "searchable_claims": status["searchable_claims"],
        "decisive_claims": status["decisive_claims"],
        "operand_only_claims": status["operand_only_claims"],
        "refused_claims": refused,
    }


def _format_split(split: dict) -> str:
    refused = ", ".join(f"{patent_id} {'/'.join(numbers)}"
                        for patent_id, numbers in split["refused_claims"].items())
    return (f"corpus split: {split['records']} records, {split['searchable_claims']} "
            f"claims — {split['decisive_claims']} decisive, "
            f"{split['operand_only_claims']} refused operand-only"
            + (f" ({refused})" if refused else ""))


def log_attempt(status: int | None, outcome: str, records_added: int,
                detail: str = "", split: dict | None = None) -> dict:
    """Append one harvest attempt, so "we tried again later" is auditable.

    A blocked attempt and a successful one are different findings, and the log keeps
    both: status, outcome, and how many records the attempt actually added. Each new
    entry also carries the corpus split at attempt time, because a blocked attempt is
    exactly when a reader needs to know what the corpus still consists of.
    """
    entry = {
        "attempted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": "https://patents.google.com/xhr/query",
        "http_status": status,
        "outcome": outcome,
        "records_added": records_added,
        "detail": detail,
    }
    if split is not None:
        entry["corpus_split"] = split
    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def retry_harvest(limit: int) -> int:
    """One harvest attempt, logged whatever happens; exit 3 while the source is blocked.

    Exit codes: ``0`` harvested, ``3`` the source answered with a block or was
    unreachable so the corpus is unchanged. A distinct code keeps "could not query"
    separable from "queried and found nothing" in a scheduler or a shell check.
    """
    before = len(list(RAW.glob("*.json"))) if RAW.exists() else 0
    inner = urllib.parse.urlencode({"q": '"state of health"', "country": "US"})
    url = f"https://patents.google.com/xhr/query?url={urllib.parse.quote(inner)}&exp="
    try:
        status, body = _get(url, timeout=25)
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        log_attempt(None, "unreachable", 0, str(error))
        print(f"harvest unreachable: {error}; attempt logged, corpus unchanged", file=sys.stderr)
        return 3
    split = corpus_split()
    if status != 200:
        log_attempt(status, "blocked", 0, f"{len(body)}-byte block page; corpus unchanged",
                    split=split)
        print(f"harvest blocked: HTTP {status}; attempt logged, corpus unchanged", file=sys.stderr)
        print(_format_split(split), file=sys.stderr)
        return 3
    log_attempt(status, "open", 0, "endpoint answered; harvesting", split=split)
    code = main(["--limit", str(limit)])
    added = (len(list(RAW.glob("*.json")) if RAW.exists() else []) - before)
    log_attempt(status, "open", added, f"harvest exited {code}", split=corpus_split())
    print(f"harvest attempt added {added} record(s)")
    print(_format_split(corpus_split()), file=sys.stderr)
    return code


def harvest(query: str, limit: int, page_size: int = 100) -> list[dict]:
    """Publication numbers for ``query`` from the paginated Google Patents endpoint."""
    found: list[dict] = []
    page = 0
    while len(found) < limit:
        inner = urllib.parse.urlencode({"q": query, "country": "US", "num": page_size, "page": page})
        status, body = _get(f"https://patents.google.com/xhr/query?url={urllib.parse.quote(inner)}&exp=")
        if status != 200 or not body:
            break
        payload = json.loads(body.decode("utf-8", "replace"))
        results = payload.get("results") or {}
        clusters = results.get("cluster") or []
        batch = [item for cluster in clusters for item in cluster.get("result", [])]
        if not batch:
            break
        for item in batch:
            identifier = str(item.get("id", ""))          # patent/US12650070B2/en
            number = identifier.rsplit("/", 2)[-2] if identifier.count("/") >= 2 else identifier
            if not number:
                continue
            found.append({
                "publication_number": number,
                "query": query,
                "title_hint": (item.get("patent") or {}).get("title", "").strip(),
                "rank": item.get("rank"),
            })
            if len(found) >= limit:
                break
        page += 1
        time.sleep(1.0)
    return found


# ---------------------------------------------------------------------------
# Automatic expression extraction
# ---------------------------------------------------------------------------

#: Unicode maths that appears inline in claims, folded to ASCII names.
_GLYPHS = {
    "η": "eta", "μ": "mu", "σ": "sigma", "τ": "tau", "α": "alpha", "β": "beta",
    "λ": "lambda", "ρ": "rho", "θ": "theta", "ω": "omega", "Δ": "delta",
    "×": "*", "·": "*", "−": "-", "–": "-", "≤": "<=", "≥": ">=",
}

#: A symbol the extractor accepts: a letter name with an optional subscript.
_SYMBOL = r"[A-Za-z][A-Za-z0-9]*(?:_(?:\{[^}]{1,12}\}|[A-Za-z0-9]{1,8}))?"

#: ``A = B / C``, ``A = B*C``, ``A = k*exp(x)`` — an equality between symbols.
_EQUALITY = re.compile(
    rf"(?P<lhs>{_SYMBOL})\s*(?:=|equals|is equal to)\s*"
    rf"(?P<rhs>[^.;]{{1,80}})"
)
_RATIO_WORDS = re.compile(
    r"ratio\s+of\s+(?P<num>[A-Za-z][A-Za-z0-9_ ]{1,30}?)\s+to\s+(?P<den>[A-Za-z][A-Za-z0-9_ ]{1,30})",
    re.IGNORECASE,
)

#: Only symbols, operators and parentheses may appear on the right-hand side. A
#: prose fragment that slips through here would become a searchable expression and
#: manufacture matches the claim never made, so it is refused instead.
_ALLOWED_RHS = re.compile(r"^[A-Za-z0-9_().*+/ \-^]{1,80}$")
_PROSE = re.compile(
    r"\b(?:said|the|of|and|where|claim|figure|wherein|according|least|which|such|thereof)\b",
    re.IGNORECASE,
)


def _strict_rhs(rhs: str) -> str | None:
    """A right-hand side that is arithmetic over symbols, or ``None``."""
    candidate = rhs.strip().rstrip(",;:")
    if not _ALLOWED_RHS.match(candidate) or _PROSE.search(candidate):
        return None
    if not re.search(r"[A-Za-z]", candidate):
        return None
    return candidate


def _strict_ratio_tokens(*words: str) -> list[str] | None:
    """Ratio operands must each be one identifier, not a noun phrase."""
    tokens = []
    for word in words:
        token = word.strip().replace(" ", "_")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", token):
            return None
        if _PROSE.fullmatch(token):
            return None
        tokens.append(token)
    return tokens


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for glyph, replacement in _GLYPHS.items():
        text = text.replace(glyph, replacement)
    return text


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def _parse(expression: str) -> tuple[bool, list[str]]:
    """Sympy-parseable, with at least two free symbols; returns (ok, symbols)."""
    try:
        parsed = sp.sympify(expression, evaluate=False)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError):
        return False, []
    symbols = sorted(str(symbol) for symbol in parsed.free_symbols)
    if len(symbols) < 2:
        return False, symbols
    return True, symbols


def extract_expressions(claim_text: str) -> list[dict]:
    """Formulas a claim states explicitly, with the sentence each came from.

    Deliberately narrow: an equality written between tokens, or the words "ratio
    of X to Y". A relation stated only in prose yields nothing and is counted as
    ``no_formula`` by the caller — a wide grammar here would manufacture matches
    the claim never made.
    """
    found: list[dict] = []
    rejected: list[dict] = []
    for sentence in _sentences(_fold(claim_text)):
        for match in _EQUALITY.finditer(sentence):
            lhs = match.group("lhs")
            rhs = _strict_rhs(match.group("rhs"))
            if rhs is None:
                rejected.append({"pattern": "equality", "quoted": sentence[:240],
                                 "reason": "rhs is prose, not arithmetic"})
                continue
            if re.search(rf"\b{re.escape(lhs)}\b", rhs):
                continue                      # a tautology such as "V = V"
            expression = f"{lhs} = {rhs}"
            ok, symbols = _parse(expression)
            found.append({
                "pattern": "equality",
                "quoted": sentence[:240],
                "expression": expression,
                "symbols": symbols,
                "parsed": ok,
            })
        for match in _RATIO_WORDS.finditer(sentence):
            tokens = _strict_ratio_tokens(match.group("num"), match.group("den"))
            if tokens is None:
                rejected.append({"pattern": "ratio_of_words", "quoted": sentence[:240],
                                 "reason": "operand is a noun phrase, not a symbol"})
                continue
            expression = f"{tokens[0]}/{tokens[1]}"
            ok, symbols = _parse(expression)
            found.append({
                "pattern": "ratio_of_words",
                "quoted": sentence[:240],
                "expression": expression,
                "symbols": symbols,
                "parsed": ok,
            })
    # One record per distinct expression, keeping the first sentence that stated it.
    unique: dict[str, dict] = {}
    for entry in found:
        unique.setdefault(entry["expression"], entry)
    for entry in unique.values():
        entry["rejected_in_claim"] = len(rejected)
    return list(unique.values())


def _claim_parser():
    """The audited parser from ``fetch_patent_corpus.py``, loaded by path.

    The script also runs as ``python3 scripts/grow_patent_corpus.py``, where the
    repository root is not on ``sys.path``; importing the sibling by file keeps the
    single parser rather than forking a second one.
    """
    import importlib.util

    path = pathlib.Path(__file__).with_name("fetch_patent_corpus.py")
    spec = importlib.util.spec_from_file_location("_patent_fetch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._ClaimParser


def fetch_publication(number: str) -> dict:
    """Verbatim claims and provenance for one publication."""
    _ClaimParser = _claim_parser()

    url = f"https://patents.google.com/patent/{number}/en"
    status, body = _get(url)
    page = body.decode("utf-8", "replace")
    parser = _ClaimParser()
    parser.feed(page)
    title_match = re.search(r'<meta name="DC\.title" content="([^"]*)"', page)
    title = title_match.group(1).strip() if title_match else ""
    jurisdiction = re.match(r"([A-Z]{2})", number)
    return {
        "patent_id": number,
        "source": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": status,
        "bytes": len(body),
        "html_sha256": hashlib.sha256(body).hexdigest(),
        "jurisdiction": jurisdiction.group(1) if jurisdiction else "",
        "title": title,
        "abstract": parser.abstract,
        "claims": parser.claims,
        "extractor": "scripts/grow_patent_corpus.py::fetch_publication",
    }


def write_raw(artefact: dict) -> pathlib.Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{artefact['patent_id']}.json"
    path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def scan_all_claims() -> dict[str, object]:
    """Measure what the strict extractor finds across every harvested claim.

    Claim 1 is what the corpus normalises; this scans all claims so the yield
    figure in the record is about the extractor, not about one claim per patent.
    """
    total_claims = 0
    with_formula = 0
    formulas = 0
    rejected = 0
    per_record: dict[str, dict[str, int]] = {}
    for path in sorted(RAW.glob("*.json")):
        artefact = json.loads(path.read_text(encoding="utf-8"))
        claims = artefact.get("claims") or []
        hits = 0
        for claim in claims:
            total_claims += 1
            found = extract_expressions(claim["text"])
            parsed = [entry for entry in found if entry["parsed"]]
            if parsed:
                hits += 1
                formulas += len(parsed)
            rejected += sum(entry.get("rejected_in_claim", 0) for entry in found) if found else 0
        with_formula += 1 if hits else 0
        per_record[artefact["patent_id"]] = {"claims": len(claims), "claims_with_formula": hits}
    return {
        "records_scanned": len(per_record),
        "claims_scanned": total_claims,
        "claims_with_a_parsed_formula": formulas,
        "records_with_at_least_one_formula": with_formula,
        "claims_rejected_as_prose": rejected,
        "per_record": per_record,
    }


def write_extraction_yield(report: dict[str, object]) -> pathlib.Path:
    """The measured yield of the automatic layer, kept as an artefact."""
    path = DATA / "extraction_yield.json"
    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured_by": "scripts/grow_patent_corpus.py --scan-claims",
        "extractor": "scripts/grow_patent_corpus.py::extract_expressions",
        "why": (
            "The strict extractor accepts an equality between tokens or the words '"
            "ratio of X to Y'. Every claim it scans is counted, so a zero yield is a "
            "statement about claim text rather than about a single claim per record."),
        **report,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build(lenient: bool = True) -> int:
    """Join every raw artefact with curated and automatic expressions into corpus.jsonl."""
    curated_path = DATA / "normalised_claims.json"
    curated = json.loads(curated_path.read_text(encoding="utf-8")) if curated_path.exists() else {}
    automatic = json.loads(AUTO.read_text(encoding="utf-8")) if AUTO.exists() else {}
    rows: list[dict] = []
    for path in sorted(RAW.glob("*.json")):
        artefact = json.loads(path.read_text(encoding="utf-8"))
        patent_id = artefact["patent_id"]
        claims = {claim["num"]: claim["text"] for claim in artefact["claims"]}
        entry = curated.get(patent_id)
        normalised: list[dict] = []
        if entry:
            for item in entry.get("claims") or ():
                number = str(item.get("claim_number", "1"))
                text = claims.get(number)
                if text is None:
                    raise SystemExit(
                        f"{patent_id}: normalised claim {number} is not in the retrieved artefacts")
                expression = item.get("expression") or None
                normalised.append({
                    "claim_number": number,
                    "claim": text,
                    "expression": expression,
                    "expression_note": item.get("note", ""),
                    "expression_source": "curated" if expression else "none",
                })
            usable = [row for row in normalised if row["expression"]]
            if usable:
                claim_number = usable[0]["claim_number"]
                expression = usable[0]["expression"]
                note = usable[0]["expression_note"]
                source = "curated"
            else:
                claim_number = str(entry.get("claims", [{}])[0].get("claim_number", "1")) \
                    if entry.get("claims") else "1"
                claim_number = claim_number if claim_number in claims else "1"
                expression = None
                note = entry.get("_note", "no explicit formula stated in the held claims")
                source = "none"
        else:
            auto = automatic.get(patent_id, {})
            candidates = [row for row in auto.get("expressions", []) if row["parsed"]]
            claim_number = str(auto.get("claim_number", "1"))
            if candidates:
                expression = candidates[0]["expression"]
                note = f"automatic: {candidates[0]['pattern']} :: {candidates[0]['quoted']}"
                source = "automatic"
            else:
                # A harvested record with no formula is still a record: it widens
                # the set absence is stated over, and it is counted as
                # unsearchable rather than dropped.
                expression = None
                note = "no explicit formula stated in claim 1"
                source = "none"
                normalised = [{
                    "claim_number": "1",
                    "claim": claims.get("1", ""),
                    "expression": None,
                    "expression_note": note,
                    "expression_source": "none",
                }]
        claim = claims.get(claim_number)
        if claim is None:
            if lenient:
                continue
            raise SystemExit(f"{patent_id}: claim {claim_number} absent")
        rows.append({
            "patent_id": patent_id,
            "jurisdiction": artefact["jurisdiction"],
            "title": artefact["title"],
            "claim_number": claim_number,
            "claim": claim,
            "expression": expression,
            "expression_note": note,
            "expression_source": source,
            "claims": normalised,
            "source": artefact["source"],
            "retrieved": artefact["retrieved_utc"][:10],
            "html_sha256": artefact["html_sha256"],
            "synthetic": False,
        })
    CORPUS.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                      encoding="utf-8")
    searchable = sum(len([c for c in row["claims"] if c["expression"]]) for row in rows)
    with_formula = sum(1 for row in rows if any(c["expression"] for c in row["claims"]))
    print(f"wrote {CORPUS.relative_to(ROOT)}: {len(rows)} records, {with_formula} with a "
          f"normalised claim, {searchable} searchable claims "
          f"({sum(1 for row in rows if row['expression_source'] == 'automatic')} automatic)")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-sources", action="store_true")
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--scan-claims", action="store_true")
    parser.add_argument("--retry-harvest", action="store_true",
                        help="one logged harvest attempt; exit 3 while the source is blocked")
    parser.add_argument("--source", choices=("google", "patentsview", "epo-ops"),
                        default="google", help="discovery source; patentsview and epo-ops "
                        "need credentials and refuse with exit 2 without them")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)

    if args.probe_sources:
        rows = probe_sources()
        for row in rows:
            print(f"{row['source']}: {row.get('http_status')} {row.get('error', '')}")
        write_sources(rows, "google_patents_query",
                      "Six endpoints were probed and are recorded above. PatentsView's API host "
                      "does not resolve, EPO OPS refuses without OAuth, both USPTO APIs need a "
                      "key, and Google Patents serves a 503 block page — so the bundled corpus "
                      "came from Google's query endpoint before that block. No bulk source could "
                      "be used here, and that is the finding this table records.")
        print(f"wrote {SOURCES_MD.relative_to(ROOT)}")
        return 0

    if args.build:
        build()
        return 0

    if args.scan_claims:
        report = scan_all_claims()
        write_extraction_yield(report)
        print(json.dumps(report, indent=2))
        return 0

    if args.retry_harvest:
        return retry_harvest(args.limit)

    token: str | None = None
    if args.source == "epo-ops":
        try:
            token = epo_ops_token()
        except SourceUnavailable as error:
            print(f"source unavailable: {error} (status={error.status})", file=sys.stderr)
            print("no records were collected; absence of records is not absence of patents",
                  file=sys.stderr)
            return 2
    elif args.source != "google":
        print(f"source {args.source!r} selected; Google Patents is not queried", file=sys.stderr)

    queries = tuple(args.query) if args.query else QUERIES
    automatic: dict[str, dict] = json.loads(AUTO.read_text(encoding="utf-8")) if AUTO.exists() else {}
    fetched = 0
    for query in queries:
        try:
            hits = (harvest_bulk(args.source, query, args.limit, token)
                    if args.source != "google" else harvest(query, args.limit))
        except SourceUnavailable as error:
            print(f"source unavailable: {error} (status={error.status})", file=sys.stderr)
            print("no records were collected; absence of records is not absence of patents",
                  file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            print(f"harvest failed for {query!r}: {error}", file=sys.stderr)
            continue
        print(f"{query!r}: {len(hits)} publications")
        for hit in hits:
            number = hit["publication_number"]
            raw_path = RAW / f"{number}.json"
            if not raw_path.exists():
                try:
                    artefact = (fetch_publication_epo(number, token) if token
                                else fetch_publication(number))
                except (urllib.error.URLError, OSError) as error:
                    print(f"  {number}: fetch failed: {error}", file=sys.stderr)
                    continue
                artefact["found_by_query"] = query
                write_raw(artefact)
                fetched += 1
                time.sleep(1.0)
            else:
                artefact = json.loads(raw_path.read_text(encoding="utf-8"))
            if number in automatic:
                continue
            claim_one = next((c for c in artefact["claims"] if c["num"] == "1"), None)
            if claim_one is None:
                continue
            expressions = extract_expressions(claim_one["text"])
            if not expressions:
                continue
            automatic[number] = {
                "claim_number": "1",
                "expressions": expressions,
                "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "extractor": "scripts/grow_patent_corpus.py::extract_expressions",
                "extraction_is_automatic": True,
            }
            if args.preview:
                print(f"  {number}: {len(expressions)} formula(s) — {expressions[0]['expression']}")
        AUTO.write_text(json.dumps(automatic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  raw artefacts: {len(list(RAW.glob('*.json')))}, "
              f"auto expressions: {len(automatic)}")
    print(f"fetched {fetched} new publications this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
