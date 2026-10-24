"""Canonical grouping: collapse rearrangements/duplicates via residual norm."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def group_representations(exprs: list) -> list:
    """Group expression strings into canonical buckets.

    Returns list of dicts: {canonical, members}.
    Uses physics_validity_engine.same_relationship when available.
    """
    try:
        from apedp.physics_validity_engine import canonical_form as _cf
        from apedp.physics_validity_engine import same_relationship as _sr
        canonical_form, same_relationship = _cf, _sr
    except ImportError as exc:
        logger.warning(
            "physics_validity_engine unavailable; falling back to identity grouping: %s", exc
        )
        canonical_form = None  # type: ignore
        same_relationship = None  # type: ignore

    try:
        from apedp.equations.normalize import physics_symbols
        extra = physics_symbols()
    except ImportError as exc:
        logger.debug("physics_symbols unavailable; grouping without symbol overrides: %s", exc)
        extra = None
    groups: list = []
    for e in exprs:
        placed = False
        for g in groups:
            try:
                if same_relationship is not None and same_relationship(
                        e, g["members"][0], extra):
                    g["members"].append(e)
                    placed = True
                    break
            except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
                logger.debug("same_relationship failed for %r; treating as distinct: %s", e, exc)
                continue
        if not placed:
            canon = e
            try:
                if canonical_form is not None:
                    c = canonical_form(e, extra)
                    canon = c or e
            except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
                logger.debug("canonical_form failed for %r; using raw form: %s", e, exc)
            groups.append({"canonical": canon, "members": [e]})
    return groups
