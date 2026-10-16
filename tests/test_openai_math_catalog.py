"""Pins for the full openai/math catalog ingestion (372 families / 719 manuscripts).

Gates the accounting (`manuscripts_manifest.json`) against a committed source
snapshot (`catalog/source_snapshot/`, SHA-256-pinned): counts, discipline
coverage, the three withdrawals, the recorded discrepancies, and the
catalog-scale honesty invariant — no family is ever marked verified by
APEDP-X (no proof engine; the sandbox rule applies at catalog scale).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[1] / "benchmarks" / "openai_math_sandbox" / "catalog"
SNAPSHOT = CATALOG / "source_snapshot"

WITHDRAWN = {
    "Algebraicity of Weil classes on split abelian eightfolds",
    "Algebraicity of Kuga–Satake Correspondences for K3 Surfaces",
    "The rational Hodge conjecture for products of K3 surfaces",
}


def _manifest() -> dict:
    path = CATALOG / "manuscripts_manifest.json"
    assert path.is_file(), f"missing manifest: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("–", "-").replace("—", "-")).strip().lower()


def test_counts_and_accounting():
    m = _manifest()
    c = m["counts"]
    assert c["families"] == 372
    assert c["manuscripts"] == 719
    assert c["disciplines"] == 17
    assert c["withdrawn"] == 3
    assert c["numbering_gaps"] == [45, 61, 70, 123, 163]
    assert c["stated_formalized"] == {"formalized": 300, "total": 719, "percent": 42}
    assert c["lean_sources_yaml"] == 173
    assert c["lean_main_results_yaml"] == 200


def test_withdrawals_are_the_named_three_and_absent_from_catalog():
    m = _manifest()
    assert set(m["withdrawn"]) == WITHDRAWN
    titles = {_norm(ms["title"]) for f in m["families"] for ms in f["manuscripts"]}
    for w in WITHDRAWN:
        assert _norm(w) not in titles, f"withdrawn manuscript still catalogued: {w}"


def test_every_family_is_classified_and_never_verified():
    m = _manifest()
    known = set(m["disciplines"])
    tags = {"none", "finite_range_numbers", "explicit_construction"}
    assert len(m["families"]) == 372
    for fam in m["families"]:
        assert fam["discipline"] in known, fam["number"]
        assert fam["manuscripts"], fam["number"]
        assert fam["apedp_checkability"] == "uncheckable_proof_level", fam["number"]
        assert "proof engine" in fam["apedp_checkability_reason"]
        assert fam["apedp_probe_tag"] in tags, fam["number"]
        assert fam["apedp_checkability"] != "verified"


def test_manuscript_records_complete():
    m = _manifest()
    n = 0
    for fam in m["families"]:
        for ms in fam["manuscripts"]:
            assert ms["title"].strip()
            assert ms["path"].startswith("preprints/") and ms["path"].endswith(".pdf")
            assert isinstance(ms["lean_formalized"], bool)
            n += 1
    assert n == 719


def test_lean_join_is_lossless_with_the_stale_path_recorded():
    m = _manifest()
    notes = m["lean_join_notes"]
    assert len(notes) == 1, notes
    assert "Taming implies compatibility" in notes[0]
    assert "stale source path" in notes[0] and "joined by title" in notes[0]
    joined = m["counts"]["lean_manuscripts_yaml"]
    assert joined == m["counts"]["lean_sources_yaml"], (
        "every yaml source must join to exactly one catalogue manuscript"
    )


def test_discrepancies_recorded_not_hidden():
    m = _manifest()
    assert len(m["discrepancies"]) >= 5
    for d in m["discrepancies"]:
        assert len(d) > 80, d


def test_snapshot_hashes_match_manifest():
    m = _manifest()
    for name, sha in m["source"]["sha256"].items():
        f = SNAPSHOT / name.replace("/", "__")
        assert f.is_file(), f"missing snapshot: {f.name}"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f.name
