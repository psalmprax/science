
# APEDP Classification Ontology
# 8-state system replacing the old 3-state (REJECTED/SPECULATION/APPROVED)

from enum import Enum, auto


class Classification(Enum):
    """Eight-class classification for discovered expressions, replacing the
    old three-tier system (REJECTED/SPECULATION/APPROVED)."""
    
    # === High-confidence classifications ===
    ESTABLISHED = auto()
    """A known physical law or equation. The expression matches a 
    canonical reference from the prior art corpus or has been 
    formally derived from first principles."""
    
    DERIVED = auto()
    """A valid expression derived from established laws through 
    legitimate mathematical manipulation. Not a new law, but a 
    correct derived relationship (e.g., E_alpha + E_n from D-T fusion)."""
    
    PHYSICALLY_PLAUSIBLE = auto()
    """Dimensionally consistent expression that is physically 
    plausible but lacks formal derivation or corpus match. Requires 
    further investigation."""
    
    # === Medium-confidence classifications ===
    CANDIDATE = auto()
    """A promising expression that shows structure consistent with 
    physical law but needs additional validation (dimensional check, 
    limiting cases, or empirical comparison)."""
    
    HEURISTIC = auto()
    """A useful composite metric or heuristic that captures a 
    meaningful physical relationship but is not claimed as a 
    fundamental law (e.g., nτσT⁴ as radiative confinement metric)."""
    
    # === Low-confidence / negative classifications ===
    IDENTITY = auto()
    """A mathematical identity, not a physical discovery. Automatically 
    detected patterns include: x/x=1, x-x=0, x⁰=1, a·a†=|a|² etc."""
    
    DIMENSIONALLY_INVALID = auto()
    """Expression fails basic dimensional consistency. The dimensions 
    of the result do not correspond to any physically meaningful quantity."""
    
    PHYSICALLY_UNSUPPORTED = auto()
    """Expression passes structural checks but has no known physical 
    derivation, no corpus match, and no empirical support. Requires 
    extraordinary evidence for law-like status."""
    
    # Sentinel for unclassified
    UNASSIGNED = auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Classification.{self.name}"


# Mapping from old 3-state system to new 8-state (module-level, not an enum member)
OLD_TO_NEW = {
    "APPROVED": {
        "ESTABLISHED": "nTtau (Lawson), E_alpha+E_n (D-T energy)",
        "DERIVED": "E_alpha+E_n from known reaction",
        "PHYSICALLY_PLAUSIBLE": "Promising but unproven expression",
        "CANDIDATE": "Shows promise, needs validation",
        "HEURISTIC": "Useful composite metric",
    },
    "SPECULATION": {
        "PHYSICALLY_PLAUSIBLE": "Plausible but unproven",
        "CANDIDATE": "Needs more validation",
        "HEURISTIC": "Useful heuristic",
    },
    "REJECTED": {
        "DIMENSIONALLY_INVALID": "Fails dimensional analysis",
        "PHYSICALLY_UNSUPPORTED": "No physical derivation",
        "IDENTITY": "Mathematical identity, not discovery",
    },
}

# ESTABLISHED requires evidence, never score alone.
ESTABLISHED_REQUIRES = ("corpus_match", "formal_derivation", "replicated_empirical_evidence")

# Reviewer-facing taxonomy: precise scientific terms mapped onto the 8-state
# Classification plus RelationshipKind (see scientific_layers.py). Documents
# intent; the enforced enum values above are unchanged.
REVIEWER_TAXONOMY = {
    "CANONICAL_LAW": ("ESTABLISHED", "LAW"),  # e.g. F=ma
    "CANONICAL_RELATION": ("ESTABLISHED", "LAW"),  # e.g. j*=sigma T^4 with context
    "ALGEBRAIC_REARRANGEMENT": ("DERIVED", "DERIVATION"),  # e.g. a=F/m
    "APPROXIMATE_DERIVED_RELATION": ("DERIVED", "MODEL"),  # e.g. mc^2+mv^2/2, v<<c
    "KNOWN_APPROXIMATION": ("DERIVED", "MODEL"),
    "DIMENSIONALLY_VALID_COMPOSITE": ("CANDIDATE", "MODEL"),  # e.g. mc^2/sigma T^4
    "UNINTERPRETED_COMPOSITE": ("CANDIDATE", "CANDIDATE"),
    "ENGINEERING_METRIC_CANDIDATE": ("HEURISTIC", "MODEL"),
    "FIGURE_OF_MERIT": ("HEURISTIC", "MODEL"),
    "EMPIRICAL_CANDIDATE": ("CANDIDATE", "MODEL"),
    "SPECULATIVE_CANDIDATE": ("CANDIDATE", "CANDIDATE"),
    "NOT_FOUND_IN_CURRENT_CORPUS": ("CANDIDATE", "CANDIDATE"),  # never "novel physics"
}
