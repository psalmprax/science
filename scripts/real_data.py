#!/usr/bin/env python3
"""Tables of real measurements: the registry, the development tables, and the lock on the held-back ones.

Everything the route has been measured on so far was generated: textbook laws evaluated on drawn rows, invented
laws, simulated systems. ``benchmarks/real_data/registry.json`` lists tables of real measurements, each pinned by
its source repository, commit and SHA-256:

* **known law**: the measurements a law was found in or is checked against (Kepler, Langmuir, Leavitt, Hubble,
  Tully and Fisher, time dilation). The check is whether the route arrives at the law's form;
* **open**: measurements with no accepted closed formula (pipe friction, the expansion rate). The check is
  prediction on rows left out, against the published formulas.

**The split.** Three tables are for development and are copied under ``benchmarks/real_data/development/``. The
others are held back for a preregistered run: they are not in this repository, ``load`` refuses them, and
``fetch`` downloads one only when called with the name of the preregistration that allows it. Real tables are
few, so one read in development cannot be used again to test anything.

    python scripts/real_data.py list
    python scripts/real_data.py check        # the development files are the pinned bytes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "real_data" / "registry.json"
DEVELOPMENT = ROOT / "benchmarks" / "real_data" / "development"
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
SPLITS = ("development", "held_back")


class HeldBack(PermissionError):
    """A held-back table was asked for outside a preregistered run."""


def registry() -> dict[str, dict[str, Any]]:
    record = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {table["key"]: table for table in record["tables"]}


def development_path(key: str) -> Path:
    table = registry()[key]
    return DEVELOPMENT / f"{key}.{'csv' if table['format'] == 'csv' else 'dat'}"


def _number(cell: str) -> float:
    """A cell as a number; a cell that is not one (a name, a blank, a flag) is a missing value."""
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def parse(table: dict[str, Any], text: str) -> dict[str, np.ndarray]:
    """The table's columns by name. Comments (``# ...`` after a row) and a byte-order mark are dropped.

    A cell that is not a number becomes a missing value (NaN), so that a column of names or flags the table's law
    does not use cannot make the table unreadable; the caller drops the rows missing a value it needs.
    """
    lines = [line.split("#", 1)[0].strip() if not line.lstrip("﻿").lstrip().startswith("#") else line.lstrip("﻿").strip()
             for line in text.lstrip("﻿").splitlines()]
    lines = [line for line in lines if line]
    separator = "," if table["format"] == "csv" else None
    if table["format"] == "whitespace_no_header":
        names, rows = list(table["columns"]), lines
    else:
        names, rows = [n.strip() for n in lines[0].lstrip("#").strip().split(separator)], lines[1:]
    cells = [[_number(cell) for cell in row.split(separator)] for row in rows if not row.startswith("#")]
    ragged = [len(row) for row in cells if len(row) != len(names)]
    if not cells or ragged:
        raise ValueError(f"{table['key']}: {len(cells)} rows, {len(ragged)} of them without {len(names)} values for the columns {names}")
    values = np.array(cells, dtype=float)
    columns = {name: values[:, i] for i, name in enumerate(names)}
    for name, factor in (table.get("scale") or {}).items():
        columns[name] = columns[name] * float(factor)
    return columns


def load(key: str) -> dict[str, Any]:
    """A development table: its inputs, its target and its registry entry. A held-back table is refused."""
    table = registry()[key]
    if table["split"] != "development":
        raise HeldBack(f"{key} is held back for a preregistered run; development code does not read it")
    data = development_path(key).read_bytes()
    if hashlib.sha256(data).hexdigest() != table["sha256"]:
        raise ValueError(f"{key}: the development file is not the pinned bytes")
    columns = parse(table, data.decode("utf-8"))
    return {"table": table, "X": {name: columns[name] for name in table["inputs"]}, "y": columns[table["target"]],
            "columns": columns}


def fetch(key: str, *, preregistration: str, into: Path) -> Path:
    """Download a held-back table from its pinned commit, for the preregistered run named. Checks the pin."""
    if not preregistration.startswith("PREREG-") or not (ROOT / "preregistration" / f"{preregistration}.md").exists():
        raise HeldBack(f"{preregistration!r} is not a preregistration in this repository; {key} stays unread")
    table = registry()[key]
    source = table["source"]
    url = RAW.format(repo=source["repo"], commit=source["commit"], path=source["path"])
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise ValueError(f"{key}: not a pinned GitHub source")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - the scheme and host are checked on the line above  # nosec B310
        data = response.read()
    if hashlib.sha256(data).hexdigest() != table["sha256"]:
        raise ValueError(f"{key}: the downloaded file is not the pinned bytes")
    into.mkdir(parents=True, exist_ok=True)
    path = into / f"{key}.{'csv' if table['format'] == 'csv' else 'dat'}"
    path.write_bytes(data)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("list", "check"))
    args = parser.parse_args(argv)
    tables = registry()
    if args.command == "list":
        for key, table in tables.items():
            print(f"{key:22} {table['split']:12} {table['kind']:10} {table['lines']:4} lines  {table['source']['repo']}")
        return 0
    bad = []
    for key, table in tables.items():
        path = development_path(key)
        if table["split"] == "development":
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != table["sha256"]:
                bad.append(f"{key}: missing or not the pinned bytes")
        elif path.exists():
            bad.append(f"{key}: a held-back table is in the repository")
    print("\n".join(bad) if bad else f"{sum(t['split'] == 'development' for t in tables.values())} development tables match their pins; "
          f"{sum(t['split'] == 'held_back' for t in tables.values())} held back and absent")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
