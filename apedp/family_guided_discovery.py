# Databricks notebook source
"""Automatic routing of generic family proposals into symbolic discovery."""
from __future__ import annotations

from typing import Any

import numpy as np

from apedp.discovery import MultiVariableSRE
from apedp.family_proposal import (
    select_univariate_family,
    select_univariate_family_validated,
)


class FamilyGuidedMultiVariableSRE(MultiVariableSRE):
    """Seed compatible discovery runs with a training-only family proposal.

    One-variable dimensionless runs with monotonically ordered input reserve
    the final 20% of their supplied training rows for family selection. The
    selected expression is then seeded into a normal search over the complete
    supplied training partition. Other runs retain the existing grammar.
    """

    def run_discovery(
        self,
        x_data: dict[str, np.ndarray],
        y_data: np.ndarray,
        generations: int = 30,
        pop_size: int = 100,
        seed: int | None = 42,
    ) -> tuple[Any, float, list[dict[str, Any]]]:
        proposal: dict[str, Any] | None = None
        if len(x_data) == 1 and self.target_unit_str == "dimensionless":
            name, values = next(iter(x_data.items()))
            values = np.asarray(values, dtype=float).ravel()
            targets = np.asarray(y_data, dtype=float).ravel()
            try:
                ordered = bool(np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))
                validation_rows = max(3, int(np.ceil(0.2 * len(values))))
                if ordered and len(values) - validation_rows >= 8:
                    candidate = select_univariate_family_validated(
                        values[:-validation_rows], targets[:-validation_rows],
                        values[-validation_rows:], targets[-validation_rows:],
                        variable=name,
                    )
                    selection = "trailing_validation"
                else:
                    candidate = select_univariate_family(values, targets, variable=name)
                    selection = "training_bic"
                if candidate is not None:
                    if candidate.expression not in self.template_exprs:
                        self.template_exprs.insert(0, candidate.expression)
                    proposal = {
                        "proposed_family": candidate.family,
                        "proposal_bic": candidate.bic,
                        "proposal_expression": str(candidate.expression),
                        "proposal_selection": selection,
                        "proposal_validation_rows": validation_rows if ordered else 0,
                    }
                else:
                    proposal = {"proposed_family": "unavailable", "proposal_selection": selection}
            except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError):
                proposal = {"proposed_family": "unavailable"}
        expression, score, history = super().run_discovery(
            x_data, y_data, generations=generations, pop_size=pop_size, seed=seed
        )
        if proposal is not None:
            history.insert(0, proposal)
        return expression, score, history


__all__ = ["FamilyGuidedMultiVariableSRE"]
