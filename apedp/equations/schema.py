"""Canonical equation record schema with provenance and license."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class EquationRecord:
    equation_id: str
    canonical: str
    aliases: list = field(default_factory=list)
    representations: list = field(default_factory=list)
    occurrences: list = field(default_factory=list)
    classification: str = "CANDIDATE"
    kind: str = "CANDIDATE"
    domain: str = ""
    license: str = ""
    units: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EquationRecord:
        import dataclasses
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore
                kwargs[f.name] = f.default_factory()  # type: ignore
        return cls(**kwargs)


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as fh:
        for r in records:
            d = r.to_dict() if isinstance(r, EquationRecord) else r
            fh.write(json.dumps(d) + "\n")


def read_jsonl(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
