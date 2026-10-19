"""APEDP Independent Adversarial Falsification Engine.

Provides an isolated, non-generative falsification prober that actively searches for:
1. Mathematical counterexamples: Denominator roots, complex branch cuts, non-analytic points.
2. Boundary counterexamples: Divergences as x -> 0, x -> inf, directional limit mismatch.
3. Numerical counterexamples: NaN, Inf, catastrophic floating-point cancellation, condition collapse.
4. Structural & Physical Invariants: Positivity, monotonicity, probability ceilings, conservation laws.
5. Empirical counterexamples: Residual minimization against reference observations.
6. Decisive Refusal: Formally issues REFUSED_CANNOT_REPAIR when physical invariants cannot be
   satisfied without unphysical distortion, preventing unprincipled ad-hoc regularizations.
"""
# ruff: noqa: F401 - this module is the public facade; every name it
# imports from its implementation modules is part of the old module's API.

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.evidence import EvidenceAssessment, EvidenceLedger, EvidenceTier
from apedp.safety import LimitTimeout, limit_with_timeout, quiet_float_warnings, safe_parse_expr

from .blind import (
    BlindAdversarialAuditor,
    BlindAuditResult,
)
from .constants import (
    ENVELOPE_DIVERGENCE_RATIO,
    INVARIANT_SAMPLES,
)
from .constraints import (
    ContradictoryConstraintEngine,
)
from .counterexamples import (
    Counterexample,
    InfeasibilityCertificate,
    unrepairable_counterexamples,
)
from .falsifier import (
    IndependentAdversarialFalsifier,
)
from .probe_support import (
    _ProbeSupportMixin,
)
from .probes_boundary import (
    _BoundaryProbesMixin,
)
from .probes_empirical import (
    _EmpiricalProbesMixin,
)
from .probes_mathematical import (
    _MathematicalProbesMixin,
)
from .probes_numerical import (
    _NumericalProbesMixin,
)
from .probes_structural import (
    _EnvelopeScan,
    _StructuralContext,
    _StructuralProbesMixin,
)
from .report import (
    FalsificationReport,
)
from .taxonomy import (
    AXIS_EVIDENCE_TIER,
    PROBE_EVIDENCE_TIER,
    UNREPAIRABLE_FAILURE_CLASSES,
    FailureClass,
    FalsificationAxis,
)

logger = logging.getLogger(__name__)
