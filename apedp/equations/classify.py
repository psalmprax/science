"""Offline-safe classification of canonical groups.

Maps to Classification + RelationshipKind without network. ESTABLISHED only
on corpus match; score alone never grants it.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def classify_group(canonical: str, members: list, corpus: dict | None = None) -> dict:
    from apedp.gated_assurance import safe_parse_expr
    from apedp.identity_detection import detect_identity

    # Identity check first (with pipeline current-symbol override)
    try:
        from apedp.equations.normalize import physics_symbols
        is_id, _, _ = detect_identity(
            safe_parse_expr(canonical, extra_symbols=physics_symbols()))
        if is_id:
            return {"classification": "IDENTITY", "kind": "IDENTITY"}
    except (ImportError, TypeError, ValueError, ArithmeticError, KeyError) as exc:
        logger.debug(
            "Identity pre-check failed for %r; continuing to corpus match: %s", canonical, exc
        )

    # Corpus match -> ESTABLISHED (evidence-gated)
    try:
        if corpus:
            from apedp.epistemic import PriorArtAuditor
            res = PriorArtAuditor.audit_novelty(canonical, custom_corpus=corpus)
            status = str(res.status)
            if "KNOWN" in status:
                return {"classification": "ESTABLISHED", "kind": "LAW",
                        "reference": res.closest_reference_id}
    except (ImportError, TypeError, ValueError, AttributeError, KeyError) as exc:
        logger.debug(
            "Corpus novelty audit unavailable for %r; defaulting to CANDIDATE: %s", canonical, exc
        )

    return {"classification": "CANDIDATE", "kind": "CANDIDATE"}
