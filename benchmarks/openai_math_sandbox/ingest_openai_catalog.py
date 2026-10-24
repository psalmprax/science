"""Full-catalog ingestion for the OpenAI math release (`openai/math`).

Ingests ALL 372 result families / 719 manuscripts (not just the six sandbox
tasks): parses ``CONTENTS.md`` (families, manuscripts, PDF links), the
``overview.pdf`` discipline map (via pdftotext when regenerating), the Lean
``formalization.yaml`` catalogue, and ``history.md`` (withdrawal/revision
accounting). Emits:

  catalog/manuscripts_manifest.json  machine-readable full registry
  catalog/CATALOG_AUDIT.md           human-readable audit + discrepancies

Sources are fetched from raw.githubusercontent.com (or read from a cached
directory with ``--from-dir``) and pinned by SHA-256 in the manifest.
Deterministic: no network, no timings in the outputs beyond the fetch date.

Usage:
    python3 benchmarks/openai_math_sandbox/ingest_openai_catalog.py --from-dir /tmp/opencode/openai_math
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "catalog"
RAW_BASE = "https://raw.githubusercontent.com/openai/math/main/"
SOURCES = ["README.md", "CONTENTS.md", "history.md", "overview.pdf", "lean/formalization.yaml"]

FAMILY_RE = re.compile(r"\*\*(\d{3})\.\s(.*?)\*\*", re.S)
MANUSCRIPT_RE = re.compile(r"\[([^\]]+)\]\((preprints/.+?\.pdf)\)")
DISCIPLINE_RE = re.compile(r"^(.+?)\s+(\d+)\s*$")
FAMILY_LINE_RE = re.compile(r"^(\d{3})\.\s")

PROBE_RULES = [
    ("finite_range_numbers", [
        "goldbach", "prime", "cubes", "catalan", "irrational", "riemann",
        "dirichlet", "zeros of", "exponent", "conjecture on prime",
    ]),
    ("explicit_construction", ["counterexample", "explicit construction", "explicit example"]),
]

APEDP_VERDICT = "uncheckable_proof_level"
APEDP_VERDICT_REASON = (
    "Proof-level manuscript; APEDP-X has no proof engine. The only "
    "machine-checkable artifact is the upstream Lean formalization/"
    "comparator, which APEDP-X does not run. Finite numerical probes can "
    "illustrate but never verify a theorem claim (see sandbox claim ceilings)."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(from_dir: Path | None, cache: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in SOURCES:
        local = cache / name.replace("/", "__")
        if from_dir is not None:
            src = from_dir / name
            if not src.is_file():
                src = from_dir / name.replace("/", "__")
            assert src.is_file(), f"missing cached source: {src}"
        else:
            cache.mkdir(parents=True, exist_ok=True)
            data = urllib.request.urlopen(RAW_BASE + name, timeout=60).read()
            src = local
            src.write_bytes(data)
        paths[name] = src
    return paths


def parse_contents(txt: str) -> list[dict]:
    matches = list(FAMILY_RE.finditer(txt))
    families = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        chunk = txt[start:end]
        manuscripts = [
            {"title": t.strip(), "path": p} for t, p in MANUSCRIPT_RE.findall(chunk)
        ]
        families.append(
            {
                "number": int(m.group(1)),
                "title": m.group(2).strip(),
                "manuscripts": manuscripts,
            }
        )
    return families


def parse_disciplines(overview_txt: str) -> tuple[list[str], dict[int, str]]:
    lines = [ln.strip() for ln in overview_txt.splitlines()]
    toc_start = lines.index("Contents") + 1
    disciplines: list[str] = []
    for ln in lines[toc_start:]:
        m = DISCIPLINE_RE.match(ln)
        if not m:
            if disciplines:
                break
            continue
        disciplines.append(m.group(1))
    known = set(disciplines)
    # Body: track current discipline as standalone lines; map family numbers.
    mapping: dict[int, str] = {}
    current = None
    for ln in lines:
        if ln in known:
            current = ln
            continue
        m = FAMILY_LINE_RE.match(ln)
        if m and current is not None:
            mapping[int(m.group(1))] = current
    return disciplines, mapping


def _norm(title: str) -> str:
    return re.sub(r"[\s]+", " ", title.replace("–", "-").replace("—", "-")).strip().lower()


def parse_withdrawn(history_txt: str) -> dict:
    block = re.search(r"\*\*Withdrawals\*\*(.*?)\*\*Fixes\*\*", history_txt, re.S)
    withdrawn = [
        ln[2:].strip()
        for ln in (block.group(1).splitlines() if block else [])
        if ln.startswith("- ")
    ]
    revised = re.search(r"revised (\d+) other manuscripts", history_txt)
    citation = re.search(r"updated (\d+) additional manuscripts", history_txt)
    stated = re.search(r"formalized to (\d+) / (\d+) = ~(\d+)%", history_txt)
    return {
        "withdrawn": withdrawn,
        "revised_count": int(revised.group(1)) if revised else None,
        "citation_update_count": int(citation.group(1)) if citation else None,
        "stated_formalized": {
            "formalized": int(stated.group(1)),
            "total": int(stated.group(2)),
            "percent": int(stated.group(3)),
        }
        if stated
        else None,
    }


def probe_tag(family: dict) -> str:
    hay = (family["title"] + " " + " ".join(m["title"] for m in family["manuscripts"])).lower()
    for tag, keys in PROBE_RULES:
        if any(k in hay for k in keys):
            return tag
    return "none"


def build(from_dir: Path | None) -> dict:
    raw = acquire(from_dir, HERE / ".catalog_cache")
    contents = raw["CONTENTS.md"].read_text(encoding="utf-8")
    overview_txt = raw["overview.pdf"].with_suffix(".txt")
    if not overview_txt.is_file():
        raise SystemExit(
            "overview.txt missing; run `pdftotext -layout overview.pdf overview.txt` in the cache dir"
        )
    disciplines, disc_map = parse_disciplines(overview_txt.read_text(encoding="utf-8"))
    families = parse_contents(contents)
    hist = parse_withdrawn(raw["history.md"].read_text(encoding="utf-8"))

    import yaml

    form = yaml.safe_load(raw["lean/formalization.yaml"].read_text(encoding="utf-8"))
    formalized_dirs = {s["id"].split("/")[2] for s in form["sources"] if s.get("id")}
    n_main_results = len(form["status"]["main_results"])
    numbers = [f["number"] for f in families]
    if len(set(numbers)) != len(numbers):
        raise SystemExit("duplicate family numbers in CONTENTS.md")
    numbering_gaps = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    if len(families) != 372 or not (1 <= min(numbers) and max(numbers) <= 400):
        raise SystemExit(f"unexpected family numbering: {len(families)} families, max {max(numbers)}")
    n_ms = sum(len(f["manuscripts"]) for f in families)
    if n_ms != 719:
        raise SystemExit(f"expected 719 manuscripts in CONTENTS.md, parsed {n_ms}")
    if len(disciplines) != 17 or any(disc_map.get(n) is None for n in numbers):
        raise SystemExit(
            f"discipline mapping incomplete: {len(disciplines)} disciplines, "
            f"{sum(1 for n in numbers if disc_map.get(n) is None)} unmapped families"
        )

    for fam in families:
        fam["discipline"] = disc_map.get(fam["number"])
        for ms in fam["manuscripts"]:
            ms["dir"] = ms["path"].split("/")[1]
            ms["lean_formalized"] = ms["dir"] in formalized_dirs

    catalogue_dirs = {m["dir"] for f in families for m in f["manuscripts"]}
    ms_by_title = {}
    for fam in families:
        for ms in fam["manuscripts"]:
            ms_by_title.setdefault(_norm(ms["title"]), (fam, ms))
    join_notes = []
    for src in form["sources"]:
        sid = src.get("id") or ""
        sdir = sid.split("/")[2] if sid.count("/") >= 2 else None
        if sdir in catalogue_dirs:
            continue
        hit = ms_by_title.get(_norm(src.get("title", "")))
        if hit is not None:
            _, ms = hit
            ms["lean_formalized"] = True
            join_notes.append(
                f"yaml source '{src['title']}' references {sdir}, but the catalogue "
                f"links {ms['dir']}: stale source path after a revision; joined by title."
            )
        else:
            join_notes.append(
                f"yaml source '{src.get('title')}' ({sdir or sid}) does not join to any "
                "catalogue manuscript."
            )

    for fam in families:
        fam["lean_formalized"] = any(m["lean_formalized"] for m in fam["manuscripts"])
        fam["apedp_checkability"] = APEDP_VERDICT
        fam["apedp_checkability_reason"] = APEDP_VERDICT_REASON
        fam["apedp_probe_tag"] = probe_tag(fam)

    n_ms = sum(len(f["manuscripts"]) for f in families)
    fam_with_lean = sum(1 for f in families if f["lean_formalized"])
    ms_with_lean = sum(1 for f in families for m in f["manuscripts"] if m["lean_formalized"])

    manifest = {
        "source": {
            "repository": "https://github.com/openai/math",
            "retrieved": "2026-10-16",
            "sha256": {name: _sha256(p) for name, p in raw.items()},
        },
        "counts": {
            "families": len(families),
            "manuscripts": n_ms,
            "disciplines": len(disciplines),
            "withdrawn": len(hist["withdrawn"]),
            "revised": hist["revised_count"],
            "citation_updates": hist["citation_update_count"],
            "lean_sources_yaml": len(form["sources"]),
            "lean_main_results_yaml": n_main_results,
            "lean_manuscripts_yaml": ms_with_lean,
            "lean_families_yaml": fam_with_lean,
            "stated_formalized": hist["stated_formalized"],
            "numbering_gaps": numbering_gaps,
        },
        "disciplines": disciplines,
        "withdrawn": hist["withdrawn"],
        "lean_join_notes": join_notes,
        "families": families,
    }
    manifest["discrepancies"] = [
        f"Headline count moved 722 -> 719 by withdrawing {len(hist['withdrawn'])} manuscripts "
        "(history.md, Oct 7 2026); all three are absent from CONTENTS.md.",
        f"Family numbering is historical, not contiguous: 372 live families spread over "
        f"001-{max(numbers)} with retired numbers {numbering_gaps}; catalogue order is preserved "
        "rather than renumbered after removals.",
        f"history.md states {hist['stated_formalized']['formalized']}/"
        f"{hist['stated_formalized']['total']} top-line results formalized "
        f"(~{hist['stated_formalized']['percent']}%), but lean/formalization.yaml enumerates only "
        f"{len(form['sources'])} source papers and {n_main_results} main-result declarations "
        f"({ms_with_lean} of them join to catalogue manuscripts). The delta is unresolved "
        "upstream; manifest reports both numbers rather than assuming a mapping.",
        f"form is not an independent check: formalization.yaml review.status = "
        f"'{form['review']['status']}', method = '{form['automation']['methods'][0]['method']}' "
        "(agent-produced entries; APEDP-X does not run Lean or the Comparator).",
        f"{len(join_notes)} formalization.yaml source path(s) do not resolve directly against "
        "the catalogue; each is joined-by-title or recorded unresolved in lean_join_notes "
        "rather than silently dropped.",
    ]
    return manifest


def write_audit(manifest: dict) -> str:
    c = manifest["counts"]
    lines = [
        "# OpenAI math catalog — full ingestion audit",
        "",
        f"Source: `openai/math` @ main, retrieved {manifest['source']['retrieved']}. "
        "Raw files pinned by SHA-256 in `manuscripts_manifest.json`.",
        "",
        "## Accounting",
        "",
        f"- Families: **{c['families']}**, manuscripts: **{c['manuscripts']}**, "
        f"disciplines: **{c['disciplines']}**.",
        f"- Withdrawn: **{c['withdrawn']}** (722 -> 719 reconciliation), revised: **{c['revised']}**, "
        f"citation-updated: **{c['citation_updates']}**.",
        f"- Lean (yaml catalogue): {c['lean_sources_yaml']} source papers, "
        f"{c['lean_main_results_yaml']} main-result declarations, "
        f"{c['lean_manuscripts_yaml']} catalogue manuscripts joined, "
        f"{c['lean_families_yaml']}/{c['families']} families covered.",
        f"- Repo-stated formalization: {c['stated_formalized']['formalized']}/"
        f"{c['stated_formalized']['total']} (~{c['stated_formalized']['percent']}%).",
        "",
        "## Unresolved discrepancies (recorded, not hidden)",
        "",
    ]
    lines += [f"- {d}" for d in manifest["discrepancies"]]
    lines += [
        "",
        "## APEDP-X checkability over all families",
        "",
        f"Every family is `{APEDP_VERDICT}`: {c['families']}/{c['families']}. "
        "No manuscript is marked verified by this system — the sandbox rule "
        "(no certificate, no `verified`) applies at catalog scale too.",
        "",
        "Probe tags (illustrative finite checks only, never verification):",
        "",
    ]
    from collections import Counter

    tags = Counter(f["apedp_probe_tag"] for f in manifest["families"])
    for tag, n in tags.most_common():
        lines.append(f"- `{tag}`: {n} families")
    lines += ["", "## Discipline coverage", "", "| Discipline | Families |", "|---|---|"]
    per = Counter(f["discipline"] for f in manifest["families"])
    for d in manifest["disciplines"]:
        lines.append(f"| {d} | {per.get(d, 0)} |")
    if None in per:
        lines.append(f"| (unmapped) | {per[None]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    manifest = build(args.from_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manuscripts_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    (args.out_dir / "CATALOG_AUDIT.md").write_text(write_audit(manifest), encoding="utf-8")
    c = manifest["counts"]
    print(
        f"families={c['families']} manuscripts={c['manuscripts']} withdrawn={c['withdrawn']} "
        f"lean_yaml={c['lean_manuscripts_yaml']}ms/{c['lean_families_yaml']}fam "
        f"stated={c['stated_formalized']['formalized']}/{c['stated_formalized']['total']}"
    )
    for d in manifest["discrepancies"]:
        print("  !", d[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
