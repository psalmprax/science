"""Proof-claim tier, independent of empirical confidence (C-classes).

This enum lives in ``apedp_dyn`` because the Lean bridge -- which is owned by
``apedp_dyn`` -- needs it, and importing it from ``apedp.mathematical.mathematical_layer``
was one of the six import-time edges from the canonical package back into the legacy
root. ``apedp.mathematical.mathematical_layer`` re-exports it so existing
``from apedp.mathematical.mathematical_layer import ProofClaimStatus`` keeps working.
"""
from __future__ import annotations

from enum import Enum


class ProofClaimStatus(str, Enum):
    """Proof-claim tier, independent of empirical confidence (C-classes).

    C-classes (C1..C5) report empirical support within the tested domain.
    P-tiers report proof standing: P0 = no proof, P1 = proof candidate
    (deductive argument proposed, unverified), P2 = formally verified proof.
    No finite-sample pipeline in this repo emits above P0; P1/P2 are reserved
    for a future proof-candidate/verification pipeline. RH therefore reports
    (C1,P0): empirically supported on N tested, proof not established.
    """

    P0_NO_PROOF = "P0"
    P1_CANDIDATE = "P1"
    P2_VERIFIED = "P2"


__all__ = ["ProofClaimStatus"]
