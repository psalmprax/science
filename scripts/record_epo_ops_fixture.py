"""Record an EPO OPS response as a replayable fixture.

The adapter tests replay a **recorded** response rather than synthetic XML, so the
recording has to be an actual response from ``ops.epo.org``. This script is how it is
made, and it refuses with exit ``2`` while the OPS credentials are absent — an empty
fixture directory must never look like a recorded one.

``--probe`` records the two refusals an unauthenticated request receives (the token
endpoint's ``401`` and the claims endpoint's ``403``); those are already committed, so
the replay tests assert on them today. With credentials, ``--number`` records the
claims response itself, which turns the authenticated replay test from a skip into a
run.

Usage::

    python3 scripts/record_epo_ops_fixture.py --probe
    export EPO_OPS_KEY=... EPO_OPS_SECRET=...
    python3 scripts/record_epo_ops_fixture.py --number EP4334731

Status: audit-only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import urllib.error
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "patents" / "ops_fixtures"
PROVENANCE = FIXTURES / "PROVENANCE.json"


def _grow():
    """The adapter module, loaded by path so this script runs without the package."""
    path = pathlib.Path(__file__).with_name("grow_patent_corpus.py")
    spec = importlib.util.spec_from_file_location("_grow_patent_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entry(url: str, method: str, status: int, body: bytes, authenticated: bool,
           note: str) -> dict:
    return {
        "url": url,
        "method": method,
        "http_status": status,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "authenticated": authenticated,
        "note": note,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _save(name: str, body: bytes, entry: dict) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / name).write_bytes(body)
    recorded = json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.exists() else {
        "why": "Recorded EPO OPS responses, replayed by tests/test_patent_bulk_sources.py.",
        "recorded_by": "scripts/record_epo_ops_fixture.py",
        "fixtures": {},
    }
    recorded.setdefault("fixtures", {})[name] = entry
    PROVENANCE.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {name}: HTTP {entry['http_status']}, {entry['bytes']} bytes")
    print(f"  {FIXTURES.relative_to(ROOT)}/{name}")


def record_probe() -> int:
    """Record the refusals an unauthenticated request receives."""
    grow = _grow()
    for name, url, method, data in (
        ("claims_EP4334731_403.xml", grow.EPO_OPS_CLAIMS_ENDPOINT.format(number="EP4334731"), "GET", None),
        ("token_401.xml", grow.EPO_OPS_AUTH_URL, "POST", b"grant_type=client_credentials"),
    ):
        try:
            status, body = grow._get(url, data=data)
        except urllib.error.HTTPError as error:
            status, body = error.code, error.read()
        _save(name, body, _entry(url, method, status, body, False,
                                 "refusal recorded from the live endpoint with no OAuth token"))
    return 0


def record_claims(number: str) -> int:
    """Record one authenticated claims response; refuses without credentials."""
    grow = _grow()
    try:
        token = grow.epo_ops_token()
    except grow.SourceUnavailable as error:
        print(f"source unavailable: {error} (status={error.status})", file=sys.stderr)
        print("no fixture was written; an absent recording is not a recorded one", file=sys.stderr)
        return 2
    url = grow.EPO_OPS_CLAIMS_ENDPOINT.format(number=number)
    try:
        status, body = grow._get(url, headers={"Authorization": f"Bearer {token}",
                                              "Accept": "application/xml"})
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    claims = grow.parse_epo_claims(body) if status == 200 else []
    _save(f"claims_{number}.xml", body,
          _entry(url, "GET", status, body, status == 200,
                 f"authenticated claims response, {len(claims)} claims parsed"))
    return 0 if status == 200 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true",
                        help="record the unauthenticated refusals")
    parser.add_argument("--number", action="append", default=None,
                        help="DOCDB publication number to record claims for, e.g. EP4334731")
    args = parser.parse_args(argv)
    if args.probe:
        return record_probe()
    if args.number:
        codes = [record_claims(number) for number in args.number]
        return max(codes)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
