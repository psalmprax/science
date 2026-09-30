"""Unified compatibility façade for APEDP-DYN.

``apedp_dyn`` is the canonical home of the executable dynamical systems
architecture (mechanical grammar, candidate IR, dimensional Pint gates,
DAO validity, forward LSODA simulation, intervention holdout, and
epistemic adjudication).

This module re-exports the complete public API of ``apedp_dyn`` to ensure
unbroken backward compatibility with existing workflows and imports under
``apedp.dyn``.
"""
from __future__ import annotations

import sys
import apedp_dyn
from apedp_dyn import *

# Alias subpackages if requested
from apedp_dyn import (
    epistemic,
    estimation,
    fitting,
    gates,
    grammar,
    model,
    problems,
    simulation,
    validation,
)

__all__ = list(apedp_dyn.__all__) + [
    "epistemic",
    "estimation",
    "fitting",
    "gates",
    "grammar",
    "model",
    "problems",
    "simulation",
    "validation",
]
