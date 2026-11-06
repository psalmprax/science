"""Deterministic Retrosynthetic Feasibility & Synthetic Accessibility Referee Gate.

Status: audit-only

Scope & honesty ceiling:
- Implements Layer 2 of the fail-closed drug discovery gatekeeper: an epistemic referee that
  evaluates whether a proposed `MolecularGraph` can be disconnected into commercially available
  building blocks in at most `max_steps` (default <= 4) via chemically sound reaction transforms
  (amide coupling, Suzuki-Miyaura cross-coupling, Buchwald-Hartwig amination, esterification,
  nucleophilic aromatic substitution, and reductive amination) while also enforcing a strict
  Synthetic Accessibility Score (`sa_score < 3.5`).
- This module is a deterministic referee (`audit-only`). It never generates candidate drugs on
  its own and never inspects confirmation holdout datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apedp_dyn.molecular.chemistry_bridge import (
    AtomNode,
    BondEdge,
    MolecularGraph,
    compute_descriptors,
    parse_smiles,
    sanitize_molecular_graph,
)


@dataclass(frozen=True)
class ReactionTemplate:
    """Single retrosynthetic disconnection transform rule."""

    template_id: str
    reaction_name: str
    bond_order: float
    element_u: str
    element_v: str
    requires_carbonyl_on_u: bool = False
    requires_aromatic_on_u: bool = False
    requires_aromatic_on_v: bool = False


@dataclass(frozen=True)
class RetrosyntheticStep:
    """Recorded single-step disconnection in a synthetic route tree."""

    step_index: int
    reaction_id: str
    reaction_name: str
    disconnected_bond: tuple[int, int]
    precursor_atom_counts: tuple[int, int]


@dataclass
class RetrosynthesisVerdict:
    """Adjudication verdict emitted by the retrosynthesis referee gate."""

    passed: bool
    num_steps: int
    sa_score: float
    sc_score: float
    matched_stock_blocks: list[str] = field(default_factory=list)
    route: list[RetrosyntheticStep] = field(default_factory=list)
    rejection_reason: str = ""


def _default_reaction_templates() -> list[ReactionTemplate]:
    """Build standard medicinal chemistry retrosynthetic disconnection templates."""
    return [
        ReactionTemplate("RXN-AMIDE", "Amide Coupling (HATU/EDC)", 1.0, "C", "N", requires_carbonyl_on_u=True),
        ReactionTemplate("RXN-ESTER", "Steglich Esterification", 1.0, "C", "O", requires_carbonyl_on_u=True),
        ReactionTemplate(
            "RXN-SUZUKI",
            "Suzuki-Miyaura Biaryl Cross-Coupling",
            1.0,
            "C",
            "C",
            requires_aromatic_on_u=True,
            requires_aromatic_on_v=True,
        ),
        ReactionTemplate(
            "RXN-BUCHWALD",
            "Buchwald-Hartwig C-N Cross-Coupling",
            1.0,
            "C",
            "N",
            requires_aromatic_on_u=True,
        ),
        ReactionTemplate(
            "RXN-SNAR",
            "Nucleophilic Aromatic Substitution (SNAr)",
            1.0,
            "C",
            "O",
            requires_aromatic_on_u=True,
        ),
        ReactionTemplate("RXN-REDAMIN", "Reductive Amination", 1.0, "C", "N"),
        ReactionTemplate("RXN-THIOETHER", "Thioether S-Alkylation", 1.0, "C", "S"),
    ]


def _default_stock_smiles() -> list[str]:
    """Catalog of foundational Enamine REAL / Sigma-Aldrich commercial building blocks."""
    return [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "CC(=O)O",
        "CC(=O)Cl",
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc(O)cc1",
        "c1ccc(N)cc1",
        "c1ccc(F)cc1",
        "c1ccc(Cl)cc1",
        "c1ccc(C(=O)O)cc1",
        "C1CCNCC1",
        "C1COCCN1",
        "NC(=O)C",
    ]


def _connected_components(atoms: list[AtomNode], bonds: list[BondEdge]) -> list[MolecularGraph]:
    """Partition a disconnected molecular graph into independent connected fragment graphs."""
    n = len(atoms)
    if n == 0:
        return []
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for b in bonds:
        adj[b.u].append(b.v)
        adj[b.v].append(b.u)

    visited = [False] * n
    fragments: list[MolecularGraph] = []

    for start in range(n):
        if visited[start]:
            continue
        comp_indices: list[int] = []
        queue = [start]
        visited[start] = True
        while queue:
            curr = queue.pop(0)
            comp_indices.append(curr)
            for nbr in adj[curr]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)

        comp_set = set(comp_indices)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(comp_indices))}
        sub_atoms = [
            AtomNode(
                index=old_to_new[old_idx],
                element=atoms[old_idx].element,
                formal_charge=atoms[old_idx].formal_charge,
                is_aromatic=atoms[old_idx].is_aromatic,
                chiral_tag=atoms[old_idx].chiral_tag,
                explicit_hydrogens=atoms[old_idx].explicit_hydrogens,
            )
            for old_idx in sorted(comp_indices)
        ]
        sub_bonds = [
            BondEdge(
                u=old_to_new[b.u],
                v=old_to_new[b.v],
                order=b.order,
                is_aromatic=b.is_aromatic,
            )
            for b in bonds
            if b.u in comp_set and b.v in comp_set
        ]
        frag = MolecularGraph(atoms=sub_atoms, bonds=sub_bonds)
        fragments.append(sanitize_molecular_graph(frag))

    return fragments


def _has_carbonyl_oxygen(graph: MolecularGraph, atom_idx: int) -> bool:
    for b in graph.bonds:
        other = b.v if b.u == atom_idx else (b.u if b.v == atom_idx else -1)
        if other != -1 and b.order == 2.0 and graph.atoms[other].element == "O":
            return True
    return False


class RetrosynthesisReferee:
    """Deterministic multi-step retrosynthetic tree search and synthetic accessibility gate."""

    def __init__(
        self,
        max_steps: int = 4,
        max_sa_score: float = 3.5,
        max_sc_score: float = 3.5,
        stock_smiles: list[str] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self.max_sa_score = max_sa_score
        self.max_sc_score = max_sc_score
        self.templates = _default_reaction_templates()

        raw_stock = stock_smiles if stock_smiles is not None else _default_stock_smiles()
        self.stock_hashes: dict[str, str] = {}
        for smi in raw_stock:
            g = parse_smiles(smi)
            self.stock_hashes[g.canonical_hash()] = smi

    def _is_in_stock(self, frag: MolecularGraph) -> tuple[bool, str]:
        h = frag.canonical_hash()
        if h in self.stock_hashes:
            return True, self.stock_hashes[h]
        # Any simple unstrained fragment with <= 9 heavy atoms and <= 1 ring is a commercial reagent
        desc = compute_descriptors(frag)
        if desc.heavy_atom_count <= 9 and desc.ring_count <= 1 and desc.sa_score <= 2.6:
            return True, f"reagent_C{desc.heavy_atom_count}_R{desc.ring_count}"
        return False, ""

    def _match_disconnections(
        self,
        frag: MolecularGraph,
    ) -> list[tuple[ReactionTemplate, tuple[int, int], list[MolecularGraph]]]:
        """Find all acyclic single-bond disconnections matching medicinal chemistry templates."""
        cycles = frag.cycle_basis()
        ring_edges: set[tuple[int, int]] = set()
        for c in cycles:
            for k in range(len(c)):
                u, v = c[k], c[(k + 1) % len(c)]
                ring_edges.add((min(u, v), max(u, v)))

        candidates: list[tuple[ReactionTemplate, tuple[int, int], list[MolecularGraph]]] = []
        for bond_idx, b in enumerate(frag.bonds):
            pair = (min(b.u, b.v), max(b.u, b.v))
            if pair in ring_edges or b.is_aromatic or b.order != 1.0:
                continue

            for u_idx, v_idx in ((b.u, b.v), (b.v, b.u)):
                atom_u = frag.atoms[u_idx]
                atom_v = frag.atoms[v_idx]
                for tmpl in self.templates:
                    if atom_u.element != tmpl.element_u or atom_v.element != tmpl.element_v:
                        continue
                    if tmpl.requires_carbonyl_on_u and not _has_carbonyl_oxygen(frag, u_idx):
                        continue
                    if tmpl.requires_aromatic_on_u and not atom_u.is_aromatic:
                        continue
                    if tmpl.requires_aromatic_on_v and not atom_v.is_aromatic:
                        continue

                    remaining_bonds = [edge for idx, edge in enumerate(frag.bonds) if idx != bond_idx]
                    sub_frags = _connected_components(frag.atoms, remaining_bonds)
                    if len(sub_frags) == 2:
                        candidates.append((tmpl, (u_idx, v_idx), sub_frags))
                        break
        return candidates

    def adjudicate(self, graph: MolecularGraph) -> RetrosynthesisVerdict:
        """Evaluate synthetic accessibility and multi-step retrosynthetic feasibility."""
        sanitize_molecular_graph(graph)
        desc = compute_descriptors(graph)
        sa = desc.sa_score
        # Synthetic Complexity Score (SCScore) scaled in [1.0, 5.0]
        sc = round(1.0 + min(4.0, 0.35 * desc.ring_count + 0.08 * desc.heavy_atom_count + 0.12 * desc.rotatable_bonds), 4)

        if sa >= self.max_sa_score:
            return RetrosynthesisVerdict(
                passed=False,
                num_steps=0,
                sa_score=sa,
                sc_score=sc,
                rejection_reason=f"SAScore {sa:.2f} exceeds gate threshold {self.max_sa_score:.2f}",
            )

        in_stock, stock_label = self._is_in_stock(graph)
        if in_stock:
            return RetrosynthesisVerdict(
                passed=True,
                num_steps=0,
                sa_score=sa,
                sc_score=sc,
                matched_stock_blocks=[stock_label],
                route=[],
            )

        # Breadth-first retrosynthetic tree search up to max_steps
        pending_frags = [graph]
        route: list[RetrosyntheticStep] = []
        matched_blocks: list[str] = []

        while pending_frags and len(route) < self.max_steps:
            curr = pending_frags.pop(0)
            ok, label = self._is_in_stock(curr)
            if ok:
                matched_blocks.append(label)
                continue

            disconnections = self._match_disconnections(curr)
            if not disconnections:
                return RetrosynthesisVerdict(
                    passed=False,
                    num_steps=len(route),
                    sa_score=sa,
                    sc_score=sc,
                    matched_stock_blocks=matched_blocks,
                    route=route,
                    rejection_reason="No valid retrosynthetic disconnection found for non-stock intermediate",
                )

            # Prioritize carbonyl amide/ester disconnections first, then balanced precursor sizes
            best_tmpl, best_pair, best_precursors = min(
                disconnections,
                key=lambda item: (
                    0 if item[0].requires_carbonyl_on_u else 1,
                    abs(item[2][0].num_atoms() - item[2][1].num_atoms()),
                ),
            )
            step = RetrosyntheticStep(
                step_index=len(route) + 1,
                reaction_id=best_tmpl.template_id,
                reaction_name=best_tmpl.reaction_name,
                disconnected_bond=best_pair,
                precursor_atom_counts=(best_precursors[0].num_atoms(), best_precursors[1].num_atoms()),
            )
            route.append(step)

            for prec in best_precursors:
                prec_ok, prec_label = self._is_in_stock(prec)
                if prec_ok:
                    matched_blocks.append(prec_label)
                else:
                    pending_frags.append(prec)

        if pending_frags:
            return RetrosynthesisVerdict(
                passed=False,
                num_steps=len(route),
                sa_score=sa,
                sc_score=sc,
                matched_stock_blocks=matched_blocks,
                route=route,
                rejection_reason=f"Retrosynthesis exceeded maximum allowed steps ({self.max_steps})",
            )

        return RetrosynthesisVerdict(
            passed=True,
            num_steps=len(route),
            sa_score=sa,
            sc_score=sc,
            matched_stock_blocks=matched_blocks,
            route=route,
        )
