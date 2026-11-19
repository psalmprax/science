"""Fetch SRBench's ground-truth problems (Feynman, Strogatz) from PMLB, verified by checksum.

    python scripts/fetch_srbench_ground_truth.py manifest      # rebuild benchmarks/srbench/manifest.json
    python scripts/fetch_srbench_ground_truth.py data          # download every dataset into the cache

SRBench's ground-truth track is 119 Feynman equations and 14 Strogatz ODE right-hand
sides, distributed through PMLB (github.com/EpistasisLab/pmlb). Each dataset's
``metadata.yaml`` states the formula, the sampling ranges and, for Feynman, the
units of every variable; the data file is stored with git LFS, so the raw URL
serves a pointer holding the file's SHA-256 and size, and the media URL serves
the file. The manifest records the pointer; :func:`fetch_dataset` refuses a
download whose SHA-256 differs from it.

Units: Feynman states them in kg, m, s, T (temperature) and V (volt), the basis
AI Feynman uses. They map onto :class:`Dimension` as m, l, t, th and, for the
volt, the ``i`` slot: {kg, m, s, K, V} is an invertible change of basis from SI
(A = kg m^2 s^-3 V^-1), so dimensional consistency is unchanged. Strogatz
variables carry no units and are treated as dimensionless.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "srbench" / "manifest.json"
CACHE = ROOT / "data" / "srbench"
PMLB_RAW = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master"
PMLB_MEDIA = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master"
#: Commit of PMLB the manifest was built from is not pinned by the raw URL; the
#: per-file SHA-256 is what pins the data.
UNIT_TO_FIELD = {"kg": "m", "m": "l", "s": "t", "T": "th", "V": "i"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https hosts above
        return bytes(response.read())


def parse_units(description: str) -> dict[str, str]:
    """``"Electric field, m^(-1) * V^1"`` -> ``{"l": "-1", "i": "1"}``; no units -> ``{}``."""
    exps: dict[str, str] = {}
    for unit, exp in re.findall(r"\b(kg|m|s|T|V)\^\(?(-?\d+(?:/\d+)?)\)?", description.split(",")[-1]):
        exps[UNIT_TO_FIELD[unit]] = exp
    return exps


def parse_metadata(name: str, text: str) -> dict[str, Any]:
    import yaml

    meta = yaml.safe_load(text)
    lines = [line.strip() for line in meta["description"].splitlines()]
    features = [f["name"] for f in meta["features"]]
    family = "strogatz" if name.startswith("strogatz") else "feynman"
    equation = next(line for line in lines if "=" in line and " in [" not in line)
    lhs, rhs = (part.strip() for part in equation.split("=", 1))
    ranges = {}
    for line in lines:
        match = re.fullmatch(r"(\S+) in \[([-\d.eE+]+),\s*([-\d.eE+]+)\]", line)
        if match:
            ranges[match.group(1)] = [float(match.group(2)), float(match.group(3))]
    entry: dict[str, Any] = {
        "name": name, "family": family, "target": lhs, "formula": rhs, "features": features,
        "ranges": ranges,
        "units": ({f["name"]: parse_units(f["description"]) for f in meta["features"]}
                  if family == "feynman" else {f: {} for f in features}),
        "target_units": parse_units(meta["target"]["description"]) if family == "feynman" else {},
    }
    return entry


def parse_lfs_pointer(text: str) -> dict[str, Any]:
    oid = re.search(r"oid sha256:([0-9a-f]{64})", text)
    size = re.search(r"size (\d+)", text)
    if not oid or not size:
        raise ValueError("not a git-lfs pointer")
    return {"sha256": oid.group(1), "size": int(size.group(1))}


def dataset_names() -> list[str]:
    index = _get(f"{PMLB_RAW}/pmlb/all_summary_stats.tsv").decode()
    return sorted(line.split("\t", 1)[0] for line in index.splitlines()
                  if line.startswith(("feynman_", "strogatz_")))


def build_manifest() -> dict[str, Any]:
    names = dataset_names()

    def one(name: str) -> dict[str, Any]:
        entry = parse_metadata(name, _get(f"{PMLB_RAW}/datasets/{name}/metadata.yaml").decode())
        entry["data"] = parse_lfs_pointer(_get(f"{PMLB_RAW}/datasets/{name}/{name}.tsv.gz").decode())
        entry["data"]["url"] = f"{PMLB_MEDIA}/datasets/{name}/{name}.tsv.gz"
        return entry

    with ThreadPoolExecutor(8) as pool:
        problems = list(pool.map(one, names))
    return {"source": "PMLB (github.com/EpistasisLab/pmlb), SRBench ground-truth track",
            "unit_basis": "kg->m, m->l, s->t, T->th, V->i (see module docstring)",
            "problems": problems}


def fetch_dataset(entry: dict[str, Any], cache: Path = CACHE) -> Path:
    """The dataset's ``.tsv.gz`` in the cache, downloaded if absent; refused on checksum mismatch."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{entry['name']}.tsv.gz"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["data"]["sha256"]:
        return path
    blob = _get(entry["data"]["url"])
    digest = hashlib.sha256(blob).hexdigest()
    if digest != entry["data"]["sha256"] or len(blob) != entry["data"]["size"]:
        raise ValueError(f"{entry['name']}: downloaded {digest} ({len(blob)} B), manifest says "
                         f"{entry['data']['sha256']} ({entry['data']['size']} B)")
    path.write_bytes(blob)
    return path


def load_dataset(entry: dict[str, Any], cache: Path = CACHE) -> Any:
    import pandas as pd

    raw = gzip.decompress(fetch_dataset(entry, cache).read_bytes())
    return pd.read_csv(io.BytesIO(raw), sep="\t")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("what", choices=("manifest", "data"))
    args = parser.parse_args(argv)
    if args.what == "manifest":
        manifest = build_manifest()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {len(manifest['problems'])} problems to {MANIFEST.relative_to(ROOT)}")
    else:
        problems = json.loads(MANIFEST.read_text(encoding="utf-8"))["problems"]
        with ThreadPoolExecutor(8) as pool:
            for path in pool.map(fetch_dataset, problems):
                print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
