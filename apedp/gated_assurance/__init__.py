"""APEDP Gated Scientific Assurance & Blind Falsification Harness.

Transforms model verification from simple regression testing into an executable,
gated scientific assurance control plane. Enforces strict isolation between development
and locked challenge partitions, evaluates multidimensional scientific improvement metrics,
benchmarks diagnostic reasoning, and classifies failure severity to govern lifecycle decisions.
"""
# ruff: noqa: F401 - this module is the public facade; every name it
# imports from its implementation modules is part of the old module's API.

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from apedp.core import DimensionalAssuranceEngine
from apedp.epistemic import (
    DataPartitionVault,
    LifecycleManager,
    ModelEvidenceRecord,
    TestableModelPackage,
)
from apedp.evidence import EvidenceTier
from apedp.falsification import (
    ContradictoryConstraintEngine,
    FalsificationAxis,
    IndependentAdversarialFalsifier,
)
from apedp.safety import safe_parse_expr

from .benchmark_contracts import (
    create_butler_volmer_benchmark_contract,
    create_casimir_benchmark_contract,
    create_gibbs_thomson_benchmark_contract,
    create_hall_petch_benchmark_contract,
    create_prandtl_glauert_benchmark_contract,
)
from .checks import (
    _ASYMPTOTE_GROWTH_FACTOR,
    _ASYMPTOTE_SAMPLE_COUNT,
    _BOUNDARY_TOLERANCE,
    _FINITE_TARGET_WORDS,
    _INFINITY_WORDS,
    _LIMIT_KEY_PATTERN,
    _NUMERIC_LIMIT_TOLERANCE,
    _PROBE_ERRORS,
    _RELATION_PATTERN,
    AsymptoteCheck,
    ConstraintCheck,
    _asymptote_probe_points,
    _asymptote_target,
    _check_asymptote,
    _check_constraint,
    _clamp01,
    _column_index,
    _compare,
    _declared_limit_series,
    _declared_symbol_names,
    _declared_units,
    _input_columns,
    _measure_dimensional_consistency,
    _measure_generality,
    _measure_stability,
    _observable_name,
    _relation,
    _resolve_threshold,
    _series_trend,
)
from .contracts import (
    ADJUDICATION_EVIDENCE_BASIS,
    ADJUDICATION_EVIDENCE_TIER,
    AdjudicationReport,
    AdjudicationVerdict,
    BlindDiagnosticBenchmark,
    DiagnosticReasoningBenchmark,
    FailureSeverity,
    FalsificationBenchResult,
    KnownDiagnosticBenchmark,
    ParetoFrontierTracker,
    ReplicationIndependenceVector,
    ScientificBenchmarkContract,
    ScientificImprovementMetric,
)
from .engine import (
    GatedScientificAssuranceEngine,
)
from .feasibility import (
    EpistemicActionStance,
    ScientificFeasibilityConfusionMatrix,
    SpecificationFeasibility,
    _has_singularity_inside_declared_domain,
    classify_specification,
    map_feasibility_to_stance,
    validate_three_gates,
)
from .mechanistic import (
    MechanisticAdjudicationEngine,
)
from .operational import (
    BlindChallengeHarness,
    OperationalGateAuditor,
    OperationalReadinessReport,
)

