"""Fetch the 200 Lean comparator-challenge STATEMENT pairs (not proof trees).

The formalization.yaml `main_results` section gives, per entry: declaration,
solution file (OAI/...), and `comparator_config` (ComparatorChallenges/<Name>.json).
The comparator challenge pair pins the exact formal statement:
  - <Name>.json  -> challenge metadata (solution module, theorem names, axioms)
  - <Name>.lean  -> the formal statement ("challenge_module", proof = sorry)

These are statement-level artifacts. APEDP-X does NOT run Lean: the point is to
extract exact claim specs (constants, quantifiers, explicit constructions) for
the system's own recompute/regenerate checkers, and to pin them by SHA-256.

Usage: python3 fetch_comparator_statements.py [--from-dir CACHE]
Outputs: catalog/comparator_snapshot/*.{json,lean} + catalog/comparator_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "catalog" / "source_snapshot"
OUT_DIR = HERE / "catalog" / "comparator_snapshot"
MANIFEST = HERE / "catalog" / "comparator_manifest.json"
RAW_BASE = "https://raw.githubusercontent.com/openai/math/main/lean/"


def load_yaml():
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml required")
    return yaml.safe_load((SNAPSHOT / "lean__formalization.yaml").read_text())


def main_results_entries(doc: dict) -> list[dict]:
    for key, value in doc.items():
        if isinstance(value, dict) and "main_results" in value:
            return value["main_results"]
    raise SystemExit("main_results not found in yaml")


def fetch(url: str, cache_dir: Path | None, rel: str) -> bytes:
    if cache_dir is not None:
        cached = cache_dir / rel
        if cached.is_file():
            return cached.read_bytes()
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError("unreachable")


def main() -> int:
    cache_dir = None
    if "--from-dir" in sys.argv:
        cache_dir = Path(sys.argv[sys.argv.index("--from-dir") + 1])
    doc = load_yaml()
    entries = main_results_entries(doc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}
    records: dict[str, dict] = {}
    failures = []
    for e in entries:
        cfg = e.get("comparator_config")
        declaration = e.get("declaration")
        solution_file = e.get("file")
        if not cfg:
            failures.append({"declaration": declaration, "error": "no comparator_config"})
            continue
        stem = Path(cfg).stem.lstrip("/").split("/")[-1]
        rec = seen.setdefault(
            stem, {"challenge": stem, "comparator_config": cfg, "entries": []}
        )
        rec["entries"].append({"declaration": declaration, "solution_file": solution_file})
        for ext in ("json", "lean"):
            rel = f"ComparatorChallenges/{stem}.{ext}"
            try:
                body = fetch(RAW_BASE + rel, cache_dir, rel)
            except Exception as exc:
                failures.append({"challenge": stem, "file": rel, "error": repr(exc)})
                continue
            (OUT_DIR / f"{stem}.{ext}").write_bytes(body)
            records[rel] = {
                "challenge": stem,
                "file": rel,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }

    manifest = {
        "source": RAW_BASE,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "yaml_main_results": len(entries),
        "unique_challenges": len(seen),
        "files": sorted(records.values(), key=lambda r: r["file"]),
        "challenges": [
            {
                "challenge": c["challenge"],
                "comparator_config": c["comparator_config"],
                "declarations": sorted({x["declaration"] for x in c["entries"]}),
                "solution_files": sorted({x["solution_file"] for x in c["entries"]}),
            }
            for c in sorted(seen.values(), key=lambda c: c["challenge"])
        ],
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "yaml_main_results": len(entries),
        "unique_challenges": len(seen),
        "files_fetched": len(records),
        "failures": len(failures),
    }, indent=1))
    for f in failures[:5]:
        print("  FAIL:", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
