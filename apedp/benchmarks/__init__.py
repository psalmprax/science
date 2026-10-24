"""APEDP Pathological Equation Benchmark Suite & Empirical Refusal Harness.

Provides an objective 100-case cross-disciplinary benchmark suite testing:
1. Solvable physical breakdown regimes across 10 distinct categories.
2. Intentionally unrepairable / ill-posed pathological models (Negative Kelvin, Superluminal acausality,
   Negative entropy, Imaginary sound speed, Naked singularities, Unitarity violations, Mutually contradictory constraints).
3. Objective metrics: Defect-detection rate, correct diagnosis rate, solvable repair success rate,
   pathological refusal accuracy, false-repair rate, infeasibility refusal accuracy, and overall system trust score.
4. Stratification across Novelty Tiers:
   - Tier A: Known physical/mathematical repairs (90 cases)
   - Tier B: Hidden known repairs (5 cases)
   - Tier C: Novel / open synthesis challenges (5 cases)
"""
# ruff: noqa: F401 - this module is the public facade; every name it
# imports from its implementation modules is part of the old module's API.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from apedp.evidence import EvidenceTier
from apedp.falsification import FalsificationReport, IndependentAdversarialFalsifier

from .cases import (
    _get_25_case_suite,
    _get_100_case_suite,
)
from .harness import (
    BenchmarkHarness,
)
from .pathological import (
    PathologicalEquationBenchmark,
)
from .results import (
    BenchmarkCaseResult,
    BenchmarkSummary,
    PathologicalCase,
)
from .types import (
    TRUST_WEIGHT_DEFECT_DETECTION,
    TRUST_WEIGHT_NO_FALSE_REPAIR,
    TRUST_WEIGHT_PATHOLOGICAL_DETECTION,
    TRUST_WEIGHT_SOLVABLE_REPAIR,
    BenchmarkCategory,
    NoveltyTier,
    UnmeasuredMetricError,
    _rate,
    _require,
)

