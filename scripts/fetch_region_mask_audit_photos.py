"""Fetch a small public-domain photograph set for the region-mask audit.

The region masks in ``apedp_dyn.video.region_masks`` were only ever exercised
against synthetic latents. Auditing them against *photographs* needs
photographs, and the repository has none, so this script fetches a small set
from Wikimedia Commons together with per-file provenance (page title, licence,
attribution, sizes and a SHA-256 of the bytes actually written).

Selection is deliberate rather than random. Each query targets an archetype
that the row-texture heuristic has to get right or be shown to get wrong:

    sea sunset horizon   the intended case (smooth sky over textured water)
    breaking waves       the intended case with a mid-frame subject
    forest path          textured above *and* below: no clean step exists
    snow field           near-uniform texture: nothing to cut at
    aerial coastline     a diagonal boundary, which a row split cannot follow
    portrait             a subject whose rows do not match its content
    city at night        extreme contrast, smooth above
    sand dunes           a smooth monotone gradient

Only images under a public-domain, CC0, CC BY or CC BY-SA licence are accepted;
each record carries the licence it was accepted under. Nothing is written for a
query that yields no admissible file, and the absence is recorded.

Usage::

    python3 scripts/fetch_region_mask_audit_photos.py \
        --out data/region_mask_audit

Then::

    python3 scripts/audit_region_masks_on_photos.py \
        --photos data/region_mask_audit/photos
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "APEDP-region-mask-audit/1.0 (scientific audit harness; "
    "https://github.com/apedp/cpmd)"
)

# Licence labels that permit redistribution with attribution. Matched by prefix
# so "CC BY-SA 4.0" and "CC BY-SA 2.0 de" are both accepted.
ADMISSIBLE_LICENCES = ("cc0", "cc by", "cc-by", "public domain", "pd-", "pd ")

# A first pass returned a Kustodiev oil painting for "portrait" and a spice
# collage for "sand dunes". The audit is about photographs, so anything whose
# title or description says it is a made image is refused and the refusal is
# recorded rather than silently dropped.
# Whole words only: an earlier substring match threw out a Manhattan skyline
# photo because its credit line contained "map".
EXCLUDED_TITLE = re.compile(
    r"\b(paintings?|collages?|drawings?|engravings?|etchings?|lithographs?|"
    r"illustrations?|posters?|stamps?|diagrams?|cartoons?|logos?|icons?|"
    r"seals?|coat of arms)\b",
    re.IGNORECASE,
)


QUERIES: tuple[tuple[str, str], ...] = (
    ("sea_sunset_horizon", "sunset over the sea horizon photograph"),
    ("breaking_waves", "breaking waves beach photograph"),
    ("forest_path", "forest path trees photograph"),
    ("snow_field", "snow covered village mountains winter photograph"),
    ("aerial_coastline", "aerial view coastline photograph"),
    ("portrait_outdoors", "hiker walking on a mountain trail photograph"),
    ("city_skyline_night", "city skyline at night photograph"),
    ("sand_dunes", "Namib desert sand dunes photograph"),
    ("mountain_lake_reflection", "mountain lake reflection photograph"),
    ("beach_palm_trees", "beach palm trees sea photograph"),
)


def _api(params: dict[str, Any]) -> dict[str, Any]:
    query = dict(params)
    query["format"] = "json"
    query["formatversion"] = "2"
    url = f"{COMMONS_API}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(  # noqa: S310 - https endpoint
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _plain(value: Any) -> str:
    """Strip the HTML that Commons embeds in attribution fields."""
    text = str(value or "")
    out: list[str] = []
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)
    return " ".join("".join(out).split())


def _licence_ok(short_name: str) -> bool:
    lowered = short_name.strip().lower()
    return any(lowered.startswith(prefix) for prefix in ADMISSIBLE_LICENCES)


def _made_image(candidate: dict[str, Any]) -> str | None:
    """Return the matched word if this file is not a photograph.

    Titles lie: a Commons file called "Photograph of Native American Woman"
    is an etching, and "Kuindzhi Winter 1890 1895" is an oil painting. The
    category list is where the truth lives ("Paintings by Arkhip Kuindzhi",
    "Etchings"), so it is checked alongside the title and description.
    """
    # Title and categories only. Free-text descriptions carry prose that trips
    # the pattern without describing the file (credits, related media).
    haystack = [str(candidate.get("page_title") or ""),
                " ".join(candidate.get("categories") or [])]
    for field in haystack:
        match = EXCLUDED_TITLE.search(field)
        if match:
            return match.group(0)
    return None


def _search(query: str, limit: int) -> list[str]:
    data = _api({
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": 6,          # File: namespace
        "srlimit": limit,
    })
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def _imageinfo(titles: list[str]) -> list[dict[str, Any]]:
    if not titles:
        return []
    data = _api({
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo|categories",
        "iiprop": "url|size|extmetadata|sha1",
        "cllimit": "max",
        "iiurlwidth": 640,         # keep the fetch small; audit resizes anyway
    })
    out = []
    for page in data.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        if not info:
            continue
        meta = info.get("extmetadata", {})
        categories = [c.get("title", "")
                      for c in page.get("categories") or []]
        out.append({
            "categories": categories,
            "page_title": page.get("title"),
            "pageid": page.get("pageid"),
            "description_url": info.get("descriptionurl"),
            "download_url": info.get("thumburl") or info.get("url"),
            "source_width": info.get("width"),
            "source_height": info.get("height"),
            "commons_sha1": info.get("sha1"),
            "licence": _plain(meta.get("LicenseShortName", {}).get("value")),
            "licence_url": _plain(meta.get("LicenseUrl", {}).get("value")),
            "artist": _plain(meta.get("Artist", {}).get("value")),
            "credit": _plain(meta.get("Credit", {}).get("value")),
            "description": _plain(
                meta.get("ImageDescription", {}).get("value"))[:400],
        })
    return out


def _download(url: str) -> bytes:
    request = urllib.request.Request(  # noqa: S310 - https endpoint
        url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
        return response.read()


def fetch(out_dir: Path, per_query: int, limit: int) -> dict[str, Any]:
    photos_dir = out_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []

    for slug, query in QUERIES:
        titles = _search(query, limit)
        time.sleep(0.4)
        candidates = _imageinfo(titles[:limit])
        time.sleep(0.4)
        chosen = None
        for candidate in candidates:
            if not _licence_ok(candidate["licence"]):
                refusals.append({
                    "query": slug,
                    "page_title": candidate["page_title"],
                    "reason": f"licence not admissible: {candidate['licence']!r}",
                })
                continue
            made = _made_image(candidate)
            if made:
                refusals.append({
                    "query": slug,
                    "page_title": candidate["page_title"],
                    "reason": f"not a photograph: title/description says {made!r}",
                })
                continue
            if candidate["download_url"] is None:
                continue
            chosen = candidate
            break
        if chosen is None:
            refusals.append({
                "query": slug,
                "page_title": None,
                "reason": "no admissible photograph returned for this query",
            })
            continue
        try:
            payload = _download(chosen["download_url"])
        except Exception as exc:                       # noqa: BLE001 - reported
            refusals.append({
                "query": slug,
                "page_title": chosen["page_title"],
                "reason": f"download failed: {exc}",
            })
            continue
        if len(payload) < 4096:
            refusals.append({
                "query": slug,
                "page_title": chosen["page_title"],
                "reason": f"download too small to be a photograph: {len(payload)} bytes",
            })
            continue
        target = photos_dir / f"{slug}.jpg"
        target.write_bytes(payload)
        record = dict(chosen)
        record["query"] = slug
        record["search_query"] = query
        record["file"] = str(target.relative_to(out_dir))
        record["bytes"] = len(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["retrieved_from"] = "Wikimedia Commons API"
        records.append(record)
        print(f"  {slug:20s} {record['licence']:18s} {len(payload):>8d} B  "
              f"{record['page_title']}")

    manifest = {
        "source": "Wikimedia Commons",
        "api": COMMONS_API,
        "purpose": (
            "photographs for auditing apedp_dyn.video.region_masks against real "
            "images; the repository contains none"
        ),
        "admissible_licences": list(ADMISSIBLE_LICENCES),
        "excluded_title_words": EXCLUDED_TITLE.pattern,
        "queries": [{"slug": s, "query": q} for s, q in QUERIES],
        "photos": records,
        "refusals": refusals,
        "counts": {
            "queries": len(QUERIES),
            "photos": len(records),
            "refusals": len(refusals),
        },
    }
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(records)}/{len(QUERIES)} queries produced an admissible photo; "
          f"{len(refusals)} refusals recorded.")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/region_mask_audit")
    parser.add_argument("--per-query", type=int, default=1,
                        help="how many photos to keep per query")
    parser.add_argument("--limit", type=int, default=8,
                        help="how many search hits to inspect per query")
    args = parser.parse_args()
    manifest = fetch(Path(args.out), args.per_query, args.limit)
    return 0 if manifest["counts"]["photos"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
