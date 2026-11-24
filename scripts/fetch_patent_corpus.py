"""Build a small, real patent corpus from Google Patents, with per-record provenance.

This script exists because of a specific rule in ``apedp/patent_prior_art.py``: a
record is either a real patent with an http(s) source and a retrieval date, or it
is marked ``synthetic``. A corpus assembled by hand from memory cannot satisfy
that rule honestly, so every record here descends from bytes actually fetched.

What is stored, and why:

* ``data/patents/raw/<ID>.json`` — the fetched artefact: the source URL, the UTC
  retrieval timestamp, the HTTP status, the byte count, the SHA-256 of the HTML,
  and the **verbatim** title, abstract and claims extracted from it. This is the
  ground truth; everything else is derived from it.
* ``data/patents/normalised_claims.json`` — hand-authored. For each record, the
  claim number that was normalised, the normalised ``expression``, and a note
  quoting the sentence it was read from. **The formula is the author's reading,
  not the patent's text**, and keeping it in a separate file with a note makes the
  derivation visible instead of blending it into the claim.
* ``data/patents/corpus.jsonl`` — the engine's input: one line per patent, the
  verbatim claim plus the normalised expression (``null`` where the claim states
  no formula). Extra provenance keys are carried through for audit; the loader
  ignores them.

Claims whose formula is a rendered image rather than text extract as prose, and
are recorded with ``expression: null`` and reported by the engine as
not-searchable. That count is the point: "no claim carried a formula" and "no
claim matched" are different findings and the engine distinguishes them.

Known limits of the extractor, stated rather than hidden: a claim is split across
several ``div.claim-text`` blocks by the source, so blocks are joined under their
enclosing ``div.claim``; publications that render their claims only as page images
(older US patents, some CN publications) have no claim markup at all and are not
usable as records. Selection therefore favours publications whose claims the
source exposes as text, which is a sampling bias of this corpus and is recorded
as such in ``data/patents/PROVENANCE.md``.

Usage::

    python3 scripts/fetch_patent_corpus.py --ids US6285999B1 US8965721B2
    python3 scripts/fetch_patent_corpus.py            # the recorded set
    python3 scripts/fetch_patent_corpus.py --build    # join with curated -> corpus.jsonl
    python3 scripts/fetch_patent_corpus.py --verify   # re-fetch and compare digests
    python3 scripts/fetch_patent_corpus.py --preview  # print claim 1 of each

Status: audit-only (builds a data artefact; produces no scientific claim)
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patents"
RAW = DATA / "raw"
CURATED = DATA / "normalised_claims.json"
CORPUS = DATA / "corpus.jsonl"

#: The recorded set: publications whose claims the source exposes as text, across
#: two offices (US, EP) and several subjects. Three candidates were fetched and
#: rejected because their claims are page images with no extractable text —
#: US2237713A (1941), CN102403767B and DE102011079469A1. That rejection is a
#: selection bias towards text-extractable claims and is recorded in
#: ``data/patents/PROVENANCE.md`` rather than glossed over.
RECORDED_IDS = (
    "US6285999B1",        # linked-database node ranking
    "US8965721B2",        # battery DC impedance / degradation factor
    "US7120566B1",        # product life, exponential acceleration model
    "US11488017B2",       # perishable-food freshness, CNN+RNN pipeline
    "EP4334731B1",        # battery one-way efficiency
)

_USER_AGENT = (
    "apedp-patent-corpus/1.0 (research use; one request per second; "
    "contact: repository maintainers)"
)


class _ClaimParser(HTMLParser):
    """Join the source's ``div.claim-text`` fragments under their ``div.claim``.

    Google splits a single claim into several ``claim-text`` blocks (each
    paragraph is its own block), so flushing on every block yields one claim per
    paragraph — wrong, and wrong in a way that silently truncates claim 1. The
    parser therefore accumulates fragments until the enclosing claim div closes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.claims: list[dict[str, str]] = []
        self.abstract: str = ""
        self._stack: list[dict[str, str]] = []
        self._text_depth: int = 0
        self._buf: list[str] = []
        self._current: dict[str, object] | None = None
        self._abstract_depth: int = 0
        self._abstract_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        classes = attr.get("class", "").split()
        if tag == "div":
            if "claim-text" in classes:
                kind = "text"
            elif "claim" in classes:
                kind = "claim"
            else:
                kind = "other"
            self._stack.append({"kind": kind, "num": attr.get("num", "")})
            if kind == "claim":
                self._current = {"num": attr.get("num", ""), "parts": []}
            elif kind == "text":
                self._text_depth += 1
                if self._text_depth == 1:
                    self._buf = []
        if "abstract" in attr.get("itemprop", "").split():
            self._abstract_depth += 1
            if self._abstract_depth == 1:
                self._abstract_buf = []
        if self._text_depth:
            self._buf.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag == "img" and attr.get("alt") and self._text_depth:
            self._buf.append(f" {attr['alt']} ")

    def handle_data(self, data: str) -> None:
        if self._text_depth:
            self._buf.append(data)
        if self._abstract_depth:
            self._abstract_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._stack:
            frame = self._stack.pop()
            if frame["kind"] == "text":
                self._text_depth = max(0, self._text_depth - 1)
                if self._text_depth == 0:
                    text = " ".join("".join(self._buf).split())
                    if text and self._current is not None:
                        self._current["parts"].append(text)
            elif frame["kind"] == "claim":
                if self._current and self._current["parts"]:
                    num = str(self._current["num"] or "").lstrip("0") or str(
                        len(self.claims) + 1)
                    self.claims.append({"num": num, "text": " ".join(self._current["parts"])})
                self._current = None
        if self._text_depth:
            self._buf.append(" ")
        if tag == "section" and self._abstract_depth:
            self._abstract_depth = max(0, self._abstract_depth - 1)
            text = " ".join("".join(self._abstract_buf).split())
            if len(text) > len(self.abstract):
                self.abstract = text


def _text_of(fragment: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(fragment)).split())


def fetch_one(patent_id: str) -> dict:
    """Fetch one publication and return its raw artefact (not yet written)."""
    url = f"https://patents.google.com/patent/{patent_id}/en"
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
        status = response.status
        body = response.read()
    page = body.decode("utf-8", errors="replace")

    parser = _ClaimParser()
    parser.feed(page)

    title_match = re.search(r'<meta name="DC\.title" content="([^"]*)"', page)
    title = html.unescape(title_match.group(1)).strip() if title_match else ""
    if not title:
        stripped = re.search(r"<title>(.*?)</title>", page, re.S)
        title = _text_of(stripped.group(1)) if stripped else ""

    jurisdiction = re.match(r"([A-Z]{2})", patent_id)
    return {
        "patent_id": patent_id,
        "source": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": status,
        "bytes": len(body),
        "html_sha256": hashlib.sha256(body).hexdigest(),
        "jurisdiction": jurisdiction.group(1) if jurisdiction else "",
        "title": title,
        "abstract": parser.abstract,
        "claims": parser.claims,
        "extractor": "scripts/fetch_patent_corpus.py::_ClaimParser",
    }


def write_raw(artefact: dict) -> pathlib.Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{artefact['patent_id']}.json"
    path.write_text(json.dumps(artefact, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_curated() -> dict[str, dict]:
    if not CURATED.exists():
        return {}
    return json.loads(CURATED.read_text(encoding="utf-8"))


def build_corpus(ids: tuple[str, ...]) -> list[dict]:
    """Join raw artefacts with the curated expressions and write corpus.jsonl."""
    curated = load_curated()
    rows: list[dict] = []
    for patent_id in ids:
        raw_path = RAW / f"{patent_id}.json"
        if not raw_path.exists():
            raise SystemExit(f"missing raw artefact for {patent_id}: run the fetch first")
        artefact = json.loads(raw_path.read_text(encoding="utf-8"))
        entry = curated.get(patent_id, {})
        claim_number = str(entry.get("claim_number", "1"))
        claim = next((c["text"] for c in artefact["claims"] if c["num"] == claim_number), None)
        if claim is None:
            available = ", ".join(c["num"] for c in artefact["claims"]) or "none"
            raise SystemExit(f"{patent_id}: claim {claim_number} absent (have: {available})")
        rows.append({
            "patent_id": patent_id,
            "jurisdiction": artefact["jurisdiction"],
            "title": artefact["title"],
            "claim_number": claim_number,
            "claim": claim,
            "expression": entry.get("expression") or None,
            "expression_note": entry.get("note", ""),
            "source": artefact["source"],
            "retrieved": artefact["retrieved_utc"][:10],
            "html_sha256": artefact["html_sha256"],
            "synthetic": False,
        })
    CORPUS.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    return rows


def verify() -> int:
    """Re-fetch each recorded patent and compare the stored digest."""
    problems = 0
    for patent_id in RECORDED_IDS:
        raw_path = RAW / f"{patent_id}.json"
        recorded = json.loads(raw_path.read_text(encoding="utf-8"))
        fresh = fetch_one(patent_id)
        same = fresh["html_sha256"] == recorded["html_sha256"]
        print(f"{patent_id}: {'digest matches' if same else 'DIGEST CHANGED'} "
              f"(recorded {recorded['html_sha256'][:12]}…, now {fresh['html_sha256'][:12]}…)")
        problems += 0 if same else 1
        time.sleep(1.0)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)

    ids = tuple(args.ids) if args.ids else RECORDED_IDS
    if args.verify:
        return 1 if verify() else 0
    if args.build:
        rows = build_corpus(ids)
        searchable = sum(1 for r in rows if r["expression"])
        print(f"wrote {CORPUS.relative_to(ROOT)}: {len(rows)} records, "
              f"{searchable} with a normalised expression")
        return 0

    for patent_id in ids:
        try:
            artefact = fetch_one(patent_id)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"{patent_id}: FETCH FAILED: {exc}", file=sys.stderr)
            return 1
        write_raw(artefact)
        first = artefact["claims"][0]["text"] if artefact["claims"] else "(no claims parsed)"
        print(f"{patent_id}: {artefact['http_status']} {artefact['bytes']}B "
              f"{len(artefact['claims'])} claims  {artefact['title'][:70]}")
        if args.preview:
            print(f"   claim 1: {first[:400]}")
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
