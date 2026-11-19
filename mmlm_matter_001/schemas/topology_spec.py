"""Catalogue entries as the proposers see them.

Blueprint §10.2 lets P1 inspect the same public topology catalogue and metadata
as P0. This record carries only that public metadata; the enumeration,
canonicalisation and counting rules of Phase 0 stay in the (future)
`topology/` package, which is what produces these entries.
"""

from __future__ import annotations

from dataclasses import dataclass

_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class TopologyEntry:
    topology_id: str
    link_count: int
    joint_count: int
    inversion: str
    ground_link: str
    input_link: str
    output_link: str
    parent_chain_id: str
    graph_hash: str

    def __post_init__(self) -> None:
        for name in ("topology_id", "inversion", "ground_link", "input_link",
                     "output_link", "parent_chain_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.link_count < 4:
            raise ValueError(
                f"a closed 1-DOF chain needs at least 4 links, got {self.link_count}")
        if self.joint_count < 4:
            raise ValueError(f"joint_count must be at least 4, got {self.joint_count}")
        if len(self.graph_hash) != 64 or set(self.graph_hash.lower()) - _HEX_DIGITS:
            raise ValueError(
                f"graph_hash must be a 64-character SHA-256 hex digest, got {self.graph_hash!r}")

    def describe(self) -> str:
        """One catalogue line, used verbatim in the P1/P2 prompts."""
        return (
            f"- id={self.topology_id} links={self.link_count} joints={self.joint_count} "
            f"inversion={self.inversion} ground={self.ground_link} "
            f"input={self.input_link} output={self.output_link} "
            f"parent={self.parent_chain_id}")
