"""APEDP Applied Synthetic Biology & Genomics Module.

Biological system design and optimization:
1. CRISPR/Cas design and off-target prediction
2. Protein engineering and design
3. Metabolic pathway optimization
4. Synthetic gene circuits
5. Genome-scale modeling

Scope is declared in :data:`SCOPE` and recorded in
``docs/SCAFFOLD_INVENTORY.md``. The off-target search is a bounded,
PAM-adjacent Hamming scan over a **caller-supplied** reference sequence: it is
not a genome-wide search, not a CFD model, and it does not search structural
variants or bulges longer than one base.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SCOPE",
    "CRISPRDesigner",
    "MetabolicPathwayOptimizer",
    "ProteinDesigner",
    "position_weighted_score",
    "position_weights",
    "reverse_complement",
]

#: What each component actually computes, and what its return value does not
#: establish. Tiers follow ``docs/SCAFFOLD_INVENTORY.md``: ``rule_screen`` (a
#: named threshold rule), ``literature_lookup`` (a published or typical value),
#: ``planning_template`` (a fixed answer that does not depend on the input),
#: ``computed`` (derived from the inputs by arithmetic or a solver).
#: ``not_predicted`` lists return keys that read like predictions but are not
#: computed from the scientific input, so a caller cannot mistake the tier.
SCOPE: dict[str, dict[str, Any]] = {
    "CRISPRDesigner.design_guide_rna": {
        "tier": "rule_screen",
        "computes": "GC content by exact arithmetic, plus guide and PAM slicing",
        "evidence": "gc_heuristic",
        "not_predicted": (
            "on_target_efficiency",
            "off_target_risk",
            "specificity_score",
        ),
    },
    "CRISPRDesigner.predict_off_targets": {
        "tier": "computed",
        "computes": (
            "a PAM-adjacent Hamming scan of a caller-supplied reference on both "
            "strands, with single-base bulge/indel and position-weighted ranking"
        ),
        "evidence": "seeded_hamming_search",
        "not_predicted": (
            "genome_wide_scan",
            "cfd_matrix",
            "bulges_longer_than_one_base",
            "structural_variants",
        ),
    },
    "ProteinDesigner.design_enzyme": {
        "tier": "planning_template",
        "computes": "a fixed design template naming external tools; no model is run",
        "evidence": "planning_template",
        "not_predicted": ("predicted_activity", "success_probability"),
    },
    "MetabolicPathwayOptimizer.optimize_pathway": {
        "tier": "literature_lookup",
        "computes": "a published theoretical yield times an assumed 0.85 recovery",
        "evidence": "literature_lookup",
        "not_predicted": ("titer",),
    },
}

_DNA = frozenset("ACGT")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_SEED_LENGTH = 8
_DEFAULT_MAX_MISMATCHES = 4
_DISTAL_WEIGHT = 0.25


def reverse_complement(sequence: str) -> str:
    """Reverse complement of a DNA sequence (5'->3')."""
    return sequence.translate(_COMPLEMENT)[::-1]


def position_weights(length: int) -> list[float]:
    """Per-position weights: 0.25 at the 5' end, 1.0 adjacent to the PAM.

    The ramp is linear, so the sum over 20 positions is exactly 12.5 and a
    single PAM-proximal mismatch weighs four distal ones.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if length == 1:
        return [1.0]
    span = length - 1
    return [
        (_DISTAL_WEIGHT * (span - index) + index) / span
        for index in range(length)
    ]


def position_weighted_score(guide_rna: str, protospacer: str) -> float:
    """PAM-weighted mismatch distance in mismatch-equivalents (0 = identical).

    Lower is a closer match. The normaliser is the closed form
    ``0.625 * length`` rather than ``sum(position_weights(...))``, so the pinned
    ``0.08`` / ``0.02`` identities are exact instead of float-accumulated.
    """
    if len(guide_rna) != len(protospacer):
        raise ValueError("guide and protospacer must be the same length")
    weights = position_weights(len(guide_rna))
    total = sum(
        weight
        for weight, guide_base, target_base in zip(
            weights, guide_rna, protospacer, strict=True
        )
        if guide_base != target_base
    )
    return total / (0.625 * len(guide_rna))


def _matches(pattern: str, text: str) -> bool:
    """IUPAC-free match where ``N`` in ``pattern`` matches any base."""
    return len(pattern) == len(text) and all(
        expected == "N" or expected == actual
        for expected, actual in zip(pattern, text, strict=True)
    )


def _hamming(guide: str, other: str) -> int:
    return sum(
        1
        for guide_base, target_base in zip(guide, other, strict=False)
        if guide_base != target_base
    )


def _best_gapped(guide: str, target: str) -> tuple[int, int]:
    """Best single-gap alignment of ``guide`` to ``target`` as (mismatches, gap).

    ``target`` must be one base longer (the reference carries the unpaired base)
    or one base shorter (the guide carries it). Ties break to the smallest gap
    index, so the result is deterministic.
    """
    candidates: list[tuple[int, int]] = []
    if len(target) == len(guide) + 1:
        for gap in range(len(guide) + 1):
            mismatches = _hamming(guide[:gap], target[:gap]) + _hamming(
                guide[gap:], target[gap + 1:]
            )
            candidates.append((mismatches, gap))
    elif len(target) == len(guide) - 1:
        for gap in range(len(target) + 1):
            mismatches = _hamming(guide[:gap], target[:gap]) + _hamming(
                guide[gap + 1:], target[gap:]
            )
            candidates.append((mismatches, gap))
    else:
        return len(guide) + 1, 0
    return min(candidates, key=lambda item: (item[0], item[1]))


def _validated_sequence(name: str, sequence: str, *, minimum_length: int) -> str:
    if not isinstance(sequence, str):
        raise ValueError(f"{name} must be a DNA string")
    text = sequence.upper()
    if not text:
        raise ValueError(f"{name} must not be empty")
    invalid = sorted(set(text) - _DNA)
    if invalid:
        raise ValueError(f"{name} contains non-DNA characters: {''.join(invalid)}")
    if len(text) < minimum_length:
        raise ValueError(f"{name} must be at least {minimum_length} nt")
    return text


def _ranking_comparison(
    top_sites: list[dict[str, Any]], weighted_top: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the Hamming order with the position-weighted order.

    Hamming ranks by (mismatches, seed mismatches); the weighted order ranks by
    ascending weighted distance, so a PAM-proximal mismatch is treated as more
    disruptive than two distal ones. A discordant pair is ``cross_band`` when
    the two sites differ in mismatch count and ``within_band`` when they do not.
    """
    weighted_rank = {id(site): rank for rank, site in enumerate(weighted_top)}
    concordant = discordant = within_band = cross_band = 0
    disagreements: list[dict[str, Any]] = []
    for left in range(len(top_sites)):
        for right in range(left + 1, len(top_sites)):
            first, second = top_sites[left], top_sites[right]
            if weighted_rank[id(first)] < weighted_rank[id(second)]:
                concordant += 1
                continue
            discordant += 1
            if first["mismatches"] == second["mismatches"]:
                within_band += 1
            else:
                cross_band += 1
            disagreements.append(
                {"hamming_prefers": first, "weighted_prefers": second}
            )
    pairs = concordant + discordant
    return {
        "basis": "pairwise_order_comparison",
        "sites_compared": len(top_sites),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "within_band_discordant_pairs": within_band,
        "cross_band_discordant_pairs": cross_band,
        "kendall_tau": (concordant - discordant) / pairs if pairs else None,
        "disagreements": disagreements,
    }


def _scan(
    guide: str,
    reference: str,
    *,
    pam: str,
    max_mismatches: int,
    tolerate_bulges: bool,
    tolerate_indels: bool,
) -> dict[str, Any]:
    span = len(guide) + len(pam)
    forward_pam = pam
    reverse_pam = reverse_complement(pam)
    counts_by_mismatch = dict.fromkeys(range(max_mismatches + 1), 0)
    exact_sites: list[dict[str, Any]] = []
    mismatch_sites: list[dict[str, Any]] = []
    gapped_sites: list[dict[str, Any]] = []
    sites_scanned = 0

    for start in range(len(reference) - span + 1):
        end = start + span
        for strand in ("+", "-"):
            if strand == "+":
                if not _matches(forward_pam, reference[end - len(pam):end]):
                    continue
                protospacer = reference[start:start + len(guide)]
                pam_text = reference[end - len(pam):end]
                forward_start = start
                bulge_target = (
                    reference[start - 1:start + len(guide)] if start >= 1 else None
                )
                indel_target = (
                    reference[start + 1:start + len(guide)]
                    if start + len(guide) <= len(reference) else None
                )
            else:
                if not _matches(reverse_pam, reference[start:start + len(pam)]):
                    continue
                protospacer = reverse_complement(reference[start + len(pam):end])
                pam_text = reverse_complement(reference[start:start + len(pam)])
                forward_start = start + len(pam)
                bulge_end = start + span + 1
                bulge_target = (
                    reverse_complement(reference[start + len(pam):bulge_end])
                    if bulge_end <= len(reference) else None
                )
                indel_end = start + span - 1
                indel_target = (
                    reverse_complement(reference[start + len(pam):indel_end])
                    if indel_end <= len(reference) else None
                )

            sites_scanned += 1
            mismatches = _hamming(guide, protospacer)
            record = {
                "strand": strand,
                "mismatches": mismatches,
                "seed_mismatches": _hamming(
                    guide[-_SEED_LENGTH:], protospacer[-_SEED_LENGTH:]
                ),
                "protospacer": protospacer,
                "pam": pam_text,
                "forward_start": forward_start,
                "position_weighted_score": position_weighted_score(
                    guide, protospacer
                ),
            }
            if mismatches == 0:
                exact_sites.append(record)
                counts_by_mismatch[0] += 1
                continue
            if mismatches <= max_mismatches:
                mismatch_sites.append(record)
                counts_by_mismatch[mismatches] += 1
                continue

            # A gapped alignment is reported only where the plain scan could not
            # explain the window. Letting a gap reclassify a site the Hamming
            # scan already counted would make `predicted_off_targets` depend on
            # the tolerance flags, which it must not.
            if (
                tolerate_bulges
                and bulge_target is not None
                and len(bulge_target) == len(guide) + 1
            ):
                gap_mismatches, gap_position = _best_gapped(guide, bulge_target)
                if gap_mismatches <= max_mismatches:
                    gapped_sites.append({
                        "strand": strand,
                        "kind": "bulge",
                        "unpaired_base": "target",
                        "mismatches": gap_mismatches,
                        "gap_position": gap_position,
                        "protospacer": protospacer,
                        "pam": pam_text,
                        "forward_start": forward_start,
                    })
                    continue
            if (
                tolerate_indels
                and indel_target is not None
                and len(indel_target) == len(guide) - 1
            ):
                gap_mismatches, gap_position = _best_gapped(guide, indel_target)
                if gap_mismatches <= max_mismatches:
                    gapped_sites.append({
                        "strand": strand,
                        "kind": "indel",
                        "unpaired_base": "guide",
                        "mismatches": gap_mismatches,
                        "gap_position": gap_position,
                        "protospacer": protospacer,
                        "pam": pam_text,
                        "forward_start": forward_start,
                    })

    top_sites = sorted(
        mismatch_sites,
        key=lambda site: (site["mismatches"], site["seed_mismatches"]),
    )
    weighted_top = sorted(
        mismatch_sites, key=lambda site: site["position_weighted_score"]
    )
    banded = exact_sites + mismatch_sites
    bulge_count = sum(1 for site in gapped_sites if site["kind"] == "bulge")
    indel_count = sum(1 for site in gapped_sites if site["kind"] == "indel")

    return {
        "guide_rna": guide,
        "guide_length": len(guide),
        "reference_length": len(reference),
        "pam": pam,
        "max_mismatches": max_mismatches,
        "sites_scanned": sites_scanned,
        "exact_matches": len(exact_sites),
        "predicted_off_targets": len(mismatch_sites),
        "counts_by_mismatch": counts_by_mismatch,
        "high_risk_sites": sum(
            1 for site in banded if site["mismatches"] <= 1
        ),
        "medium_risk_sites": sum(
            1 for site in banded if 2 <= site["mismatches"] <= 3
        ),
        "low_risk_sites": sum(
            1 for site in banded if site["mismatches"] >= 4
        ),
        "exact_sites": exact_sites,
        "top_sites": top_sites,
        "weighted_top_sites": weighted_top,
        "top_gapped_sites": sorted(
            gapped_sites,
            key=lambda site: (site["kind"], site["mismatches"], site["gap_position"]),
        ),
        "bulge_off_targets": bulge_count,
        "indel_off_targets": indel_count,
        "total_off_targets": len(mismatch_sites) + bulge_count + indel_count,
        "counts_by_kind": {
            "mismatch_only": len(mismatch_sites),
            "bulge": bulge_count,
            "indel": indel_count,
        },
        "bulge_tolerance": tolerate_bulges,
        "indel_tolerance": tolerate_indels,
        "ranking_comparison": _ranking_comparison(top_sites, weighted_top),
        "score_basis": "hamming_mismatch_count",
        "evidence": "seeded_hamming_search",
        "not_modelled": [
            "genome_wide_scan",
            "cfd_matrix",
            "bulges_longer_than_one_base",
            "structural_variants",
        ],
    }


class CRISPRDesigner:
    """Design and optimize CRISPR gene editing experiments."""

    @staticmethod
    def design_guide_rna(target_sequence: str, pam: str = "NGG") -> dict[str, Any]:
        """Design a guide RNA.

        The efficiency figure is the named linear GC rule
        ``85 + (GC% - 50) * 0.5`` clipped to 0..100, not a model. Off-target
        assessment is a separate call; this function does not perform it.
        """
        if not target_sequence:
            raise ValueError("target_sequence must not be empty")
        gc_content = (
            (target_sequence.count("G") + target_sequence.count("C"))
            / len(target_sequence)
            * 100
        )
        on_target_score = 85 + (gc_content - 50) * 0.5
        off_target_score = 15 - gc_content * 0.1
        return {
            "target_sequence": target_sequence,
            "pam_site": pam,
            "guide_rna_sequence": target_sequence[:20],
            "gc_content_percent": round(gc_content, 1),
            "on_target_efficiency": min(100, max(0, on_target_score)),
            "off_target_risk": max(0, off_target_score),
            "specificity_score": max(0, 100 - off_target_score * 5),
            "recommendations": [
                "Use high-fidelity Cas9"
                if off_target_score > 10
                else "Standard Cas9 suitable"
            ],
            "applications": ["Knockout", "Knockin", "Base editing"],
            "score_basis": "linear_gc_rule",
            "off_target_scan": "not_performed",
            "not_modelled": [
                "off_target_search",
                "on_target_efficiency_model",
                "cfd_matrix",
            ],
            "evidence": "gc_heuristic",
        }

    @staticmethod
    def predict_off_targets(
        guide_rna: str,
        reference_sequence: str | None = None,
        *,
        pam: str = "NGG",
        max_mismatches: int = _DEFAULT_MAX_MISMATCHES,
        tolerate_bulges: bool = True,
        tolerate_indels: bool = True,
    ) -> dict[str, Any]:
        """Search a caller-supplied reference for PAM-adjacent off-target sites.

        Two refusals are deliberate. A genome *size* cannot produce an
        off-target count, so the call raises rather than returning a plausible
        number; and a sequence that is not DNA raises rather than scanning. All
        coordinate fields are 0-based offsets into the supplied string (for a
        minus-strand site, ``forward_start`` is the forward-strand segment whose
        reverse complement is the protospacer).
        """
        if reference_sequence is None:
            raise NotImplementedError(
                "off-target search requires a reference sequence; a genome size "
                "cannot produce an off-target count"
            )
        if not isinstance(reference_sequence, str):
            raise TypeError(
                "genome_size_mb was removed; pass the reference sequence as the "
                "second argument"
            )
        if len(pam) != 3:
            raise ValueError("pam must be a 3-nt pattern such as 'NGG'")
        guide = _validated_sequence("guide_rna", guide_rna, minimum_length=8)
        reference = _validated_sequence(
            "reference_sequence",
            reference_sequence,
            minimum_length=len(guide) + len(pam),
        )
        if not isinstance(max_mismatches, int) or max_mismatches < 0:
            raise ValueError("max_mismatches must be a non-negative integer")
        return _scan(
            guide,
            reference,
            pam=pam.upper(),
            max_mismatches=max_mismatches,
            tolerate_bulges=tolerate_bulges,
            tolerate_indels=tolerate_indels,
        )


class ProteinDesigner:
    """Design and optimize proteins."""

    @staticmethod
    def design_enzyme(substrate: str, reaction_type: str) -> dict[str, Any]:
        """Return a planning template for enzyme design.

        ``predicted_activity`` and ``success_probability`` are fixed planning
        figures, not model outputs; :data:`SCOPE` records them as not predicted.
        """
        return {
            "target_reaction": reaction_type,
            "substrate": substrate,
            "design_approach": "de_novo_or_directed_evolution",
            "predicted_activity": 0.75,
            "stability_tm_c": 65,
            "solubility": "good",
            "expression_system": "E. coli",
            "design_iterations": 1000,
            "success_probability": 0.3,
            "tools_referenced": ["Rosetta", "AlphaFold", "ProteinMPNN"],
            "executed_tools": [],
            "evidence": "planning_template",
            "not_modelled": ["activity_model", "structure_prediction", "assay"],
        }


class MetabolicPathwayOptimizer:
    """Optimize metabolic pathways for production."""

    @staticmethod
    def optimize_pathway(
        product: str, host_organism: str = "E. coli"
    ) -> dict[str, Any]:
        """Return published theoretical yields, with the titer left as a reference.

        The optimized yield is the literature theoretical yield times an
        explicit assumed recovery fraction. No titer is computed, so
        ``titer_computed_g_l`` is ``None`` and the published reference sits in
        ``titer_reference_g_l``.
        """
        yields = {
            "ethanol": 0.9,
            "butanol": 0.7,
            "isoprene": 0.6,
            "artemisinin_precursor": 0.5,
        }
        theoretical = yields.get(product, 0.5)
        recovery = 0.85
        return {
            "target_product": product,
            "host_organism": host_organism,
            "theoretical_yield_g_g": theoretical,
            "assumed_recovery_fraction": recovery,
            "optimized_yield_g_g": theoretical * recovery,
            "titer_computed_g_l": None,
            "titer_reference_g_l": 50,
            "knockout_targets": ["ldhA", "pta", "ackA"],
            "overexpression_targets": ["gapA", "pgk"],
            "cofactors": ["NADPH balancing required"],
            "fermentation_conditions": {"temp_C": 37, "pH": 7.0, "aerobic": False},
            "evidence": "literature_lookup",
            "not_modelled": ["titer", "kinetics", "regulation", "scale_up"],
        }
