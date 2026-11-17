"""Build a survivorship-free daily panel of every Binance USDT spot pair.

Why this exists: every market file in ``data/market`` was built from a list of
names chosen years after its first bar (``PROVENANCE.json`` records
``hindsight_selected: true`` and ``includes_assets_that_stopped_trading:
false``). Ranking such a list point-in-time removes selection *within* it; it
cannot put back the assets the list never contained
(``apedp_dyn.market.universe.point_in_time_membership``). This builds the
widest candidate pool the public archive allows: every ``*USDT`` spot pair
Binance has listed, including the ones it later delisted, each present up to
its last real bar.

Source: ``https://data.binance.vision`` (the public kline archive, which keeps
delisted pairs). Every monthly zip is checked against its published SHA-256
before it is read, and the manifest records each file's digest.

Output (``--out``):
  - ``panel_development.csv.gz`` -- bars before ``--holdout-start``.
  - ``panel_holdout.csv.gz`` -- bars from ``--holdout-start`` on. Written to a
    separate file so it can be kept unread until a preregistered analysis runs.
  - ``manifest.json`` -- sources, digests, exclusions, symbols that stopped
    trading, and the digests of both panels.

Columns follow ``data/market/crypto_top15_orderflow.csv``: ``observation_date``
then ``{ASSET}_CLOSE``, ``{ASSET}_VOL``, ``{ASSET}_TAKER_BUY``, ``{ASSET}_OFI``.
A pair with no bar on a date has empty cells there.

The archive host is not reachable from every network; the workflow
``.github/workflows/market-data.yml`` runs this on GitHub's runners.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE = "https://data.binance.vision"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines/"
QUOTE = "USDT"

#: Bases that are a dollar, a fiat currency or a gold token: their "returns" are
#: a peg, not a market. Listed, not inferred, so the exclusion is reviewable.
PEGGED_BASES = frozenset({
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "UST", "SUSD",
    "AEUR", "EUR", "EURI", "GBP", "AUD", "BRL", "TRY", "RUB", "IDRT", "BIDR", "BKRW",
    "NGN", "UAH", "ZAR", "PAXG", "USD1", "USDE", "XUSD", "BFUSD", "RLUSD",
})
#: Leveraged tokens (BTCUP, ETHDOWN, BNBBULL...): a rebalanced derivative of
#: another listed asset, not an asset. A base is one only if what precedes the
#: suffix is itself a listed base, so JUP or SYRUP keep their columns.
LEVERAGED = re.compile(r"^(?P<under>[A-Z0-9]+?)(UP|DOWN|BULL|BEAR)$")

Fetch = Callable[[str], bytes]


def archive_url(key: str) -> str:
    """The URL of an object in the archive, path percent-encoded.

    Pair symbols are not all ASCII (``币安人生USDT`` is listed), and an HTTP
    request line is. Every ASCII key the archive uses -- letters, digits,
    ``-``, ``_``, ``.``, ``/`` -- comes back unchanged.
    """
    return f"{ARCHIVE}/{urllib.parse.quote(key, safe='/')}"


def http_get(url: str, attempts: int = 4) -> bytes:
    """GET with retries; raises after the last attempt."""
    if not url.startswith((ARCHIVE + "/", LISTING + "?")):
        raise ValueError(f"refusing a URL outside the archive: {url}")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "apedp-market-data/1"})  # noqa: S310 - checked above: the two fixed https hosts only
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - checked above: the two fixed https hosts only
                return bytes(response.read())
        except Exception as exc:  # noqa: BLE001 - network faults of every kind are retried, then re-raised
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {last}") from last


def list_prefixes(prefix: str, fetch: Fetch = http_get) -> list[str]:
    """Every key prefix one level under ``prefix`` in the archive bucket."""
    out: list[str] = []
    marker = ""
    while True:
        query = {"delimiter": "/", "prefix": prefix}
        if marker:
            query["marker"] = marker
        root = ET.fromstring(fetch(f"{LISTING}?{urllib.parse.urlencode(query)}"))  # noqa: S314 - the archive's own S3 listing over https; expat >= 2.4.1 bounds entity expansion
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        tag = (lambda name: f"s3:{name}") if ns else (lambda name: name)
        page = [el.text or "" for el in root.findall(f"{tag('CommonPrefixes')}/{tag('Prefix')}", ns)]
        keys = [el.text or "" for el in root.findall(f"{tag('Contents')}/{tag('Key')}", ns)]
        out.extend(page)
        truncated = (root.findtext(tag("IsTruncated"), default="false", namespaces=ns) or "").lower() == "true"
        if not truncated:
            return out
        marker = root.findtext(tag("NextMarker"), default="", namespaces=ns) or (page + keys)[-1]


def list_keys(prefix: str, fetch: Fetch = http_get) -> list[str]:
    """Every object key under ``prefix``."""
    out: list[str] = []
    marker = ""
    while True:
        query = {"prefix": prefix}
        if marker:
            query["marker"] = marker
        root = ET.fromstring(fetch(f"{LISTING}?{urllib.parse.urlencode(query)}"))  # noqa: S314 - the archive's own S3 listing over https; expat >= 2.4.1 bounds entity expansion
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        tag = (lambda name: f"s3:{name}") if ns else (lambda name: name)
        keys = [el.text or "" for el in root.findall(f"{tag('Contents')}/{tag('Key')}", ns)]
        out.extend(keys)
        truncated = (root.findtext(tag("IsTruncated"), default="false", namespaces=ns) or "").lower() == "true"
        if not truncated or not keys:
            return out
        marker = keys[-1]


def eligible(symbol: str, listed_bases: frozenset[str] = frozenset()) -> str | None:
    """The base asset of a USDT pair worth a column, or None with no column."""
    if not symbol.endswith(QUOTE) or symbol == QUOTE:
        return None
    base = symbol[: -len(QUOTE)]
    if base in PEGGED_BASES:
        return None
    leveraged = LEVERAGED.match(base)
    if leveraged and leveraged.group("under") in listed_bases:
        return None
    return base


def months(start: dt.date, end: dt.date) -> list[str]:
    """``YYYY-MM`` for every month that starts on or before ``end``, from ``start``'s."""
    out, year, month = [], start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _bar_date(open_time: str) -> dt.date:
    """Binance spot timestamps are milliseconds, and microseconds from 2025."""
    stamp = int(open_time)
    seconds = stamp / 1e6 if stamp >= 10**14 else stamp / 1e3
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).date()


def read_month(blob: bytes, checksum_text: str, name: str) -> list[tuple[dt.date, float, float, float]]:
    """Verify a monthly zip against its published digest; return (date, close, volume, taker buy)."""
    expected = checksum_text.split()[0].strip().lower()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise ValueError(f"{name}: sha256 {actual} does not match the published {expected}")
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            text = archive.read(member).decode("utf-8")
            for record in csv.reader(io.StringIO(text)):
                if not record or not record[0].strip().isdigit():
                    continue  # a header row, which some months carry
                rows.append((_bar_date(record[0]), float(record[4]), float(record[5]), float(record[9])))
    return rows


def fetch_symbol(symbol: str, wanted: Iterable[str], fetch: Fetch = http_get) -> tuple[dict, list[dict]]:
    """All daily bars of one pair in the wanted months, and the digest of each file read."""
    keys = set(list_keys(f"{KLINE_PREFIX}{symbol}/1d/", fetch))
    bars: dict[dt.date, tuple[float, float, float]] = {}
    files = []
    for month in wanted:
        name = f"{symbol}-1d-{month}.zip"
        key = f"{KLINE_PREFIX}{symbol}/1d/{name}"
        if key not in keys:
            continue
        blob = fetch(archive_url(key))
        checksum = fetch(archive_url(f"{key}.CHECKSUM")).decode("utf-8")
        for date, close, volume, taker in read_month(blob, checksum, name):
            bars[date] = (close, volume, taker)
        files.append({"file": key, "sha256": hashlib.sha256(blob).hexdigest()})
    return bars, files


def assemble(per_base: dict[str, dict], start: dt.date, end: dt.date) -> list[list[str]]:
    """Wide rows, one per calendar day, every base's four columns."""
    bases = sorted(b for b, bars in per_base.items() if bars)
    header = ["observation_date"]
    for base in bases:
        header += [f"{base}_CLOSE", f"{base}_VOL", f"{base}_TAKER_BUY", f"{base}_OFI"]
    rows = [header]
    day = start
    while day <= end:
        row = [day.isoformat()]
        for base in bases:
            bar = per_base[base].get(day)
            if bar is None:
                row += ["", "", "", ""]
                continue
            close, volume, taker = bar
            ofi = (2.0 * taker - volume) / volume if volume > 0 else 0.0
            row += [repr(close), repr(volume), repr(taker), repr(ofi)]
        rows.append(row)
        day += dt.timedelta(days=1)
    return rows


def write_gz_csv(rows: list[list[str]], path: Path) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    data = gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build(out: Path, start: dt.date, end: dt.date, holdout_start: dt.date,
          fetch: Fetch = http_get, workers: int = 16) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    symbols = sorted(p[len(KLINE_PREFIX):].strip("/") for p in list_prefixes(KLINE_PREFIX, fetch))
    bases = frozenset(s[: -len(QUOTE)] for s in symbols if s.endswith(QUOTE))
    chosen = {s: eligible(s, bases) for s in symbols}
    excluded = sorted(s for s, b in chosen.items() if b is None and s.endswith(QUOTE))
    wanted = months(start, end)
    jobs = {s: b for s, b in chosen.items() if b is not None}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(jobs, pool.map(lambda s: fetch_symbol(s, wanted, fetch), jobs), strict=True))
    per_base: dict[str, dict] = {}
    sources: list[dict] = []
    for symbol, base in jobs.items():
        bars, files = results[symbol]
        bars = {d: v for d, v in bars.items() if start <= d <= end}
        if bars:
            per_base[base] = bars
            sources += files
    last_bar = {b: max(bars).isoformat() for b, bars in per_base.items()}
    stopped = sorted(b for b, d in last_bar.items() if d < (end - dt.timedelta(days=7)).isoformat())

    rows = assemble(per_base, start, end)
    header, body = rows[0], rows[1:]
    split = holdout_start.isoformat()
    development = [header] + [r for r in body if r[0] < split]
    holdout = [header] + [r for r in body if r[0] >= split]
    manifest = {
        "source": ARCHIVE,
        "quote": QUOTE,
        "coverage": [start.isoformat(), end.isoformat()],
        "holdout_start": split,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbols_listed_in_archive": len(symbols),
        "excluded_pairs": excluded,
        "exclusion_rule": "pegged bases (listed in PEGGED_BASES) and leveraged tokens (UP/DOWN/BULL/BEAR)",
        "assets": sorted(per_base),
        "first_bar": {b: min(bars).isoformat() for b, bars in per_base.items()},
        "last_bar": last_bar,
        "stopped_trading_before_end": stopped,
        "universe_selection": {
            "basis": "every_listed_pair",
            "selected_as_of": None,
            "hindsight_selected": False,
            "includes_assets_that_stopped_trading": True,
            "note": "Every eligible USDT pair the archive holds, delisted ones included, each up to its "
                    "last bar. Apply apedp_dyn.market.universe.point_in_time_membership for a tradable universe.",
        },
        "source_files": sorted(sources, key=lambda f: f["file"]),
        "panel_development.csv.gz": write_gz_csv(development, out / "panel_development.csv.gz"),
        "panel_holdout.csv.gz": write_gz_csv(holdout, out / "panel_holdout.csv.gz"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    manifest = build(args.out, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end),
                     dt.date.fromisoformat(args.holdout_start), workers=args.workers)
    print(f"{len(manifest['assets'])} assets, {len(manifest['stopped_trading_before_end'])} stopped trading, "
          f"{len(manifest['source_files'])} source files verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
