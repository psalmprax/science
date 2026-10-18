"""The split packages must stay transparent facades.

`apedp/falsification`, `apedp/gated_assurance` and `apedp/benchmarks` used to be
single modules; they are now packages whose ``__init__`` re-exports every name the
old module exposed. That property is what keeps existing imports working, and it
is invisible to ordinary use of the submodules -- so it is asserted here, against
the implementation modules themselves rather than against a hand-written list:

* every class/function defined in a submodule is the *same object* on the facade
  (no copy, no wrapper, no missing name);
* the public entry points that callers import from the old module path resolve.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

PACKAGES = ["apedp.falsification", "apedp.gated_assurance", "apedp.benchmarks"]

#: Names imported from these three module paths elsewhere in the repository.
#: If the facade stops exporting one of them, those imports break at call time,
#: not at import time -- so they are checked explicitly.
PUBLIC_ENTRY_POINTS = {
    "apedp.falsification": [
        "AXIS_EVIDENCE_TIER", "PROBE_EVIDENCE_TIER", "BlindAdversarialAuditor",
        "BlindAuditResult", "ContradictoryConstraintEngine", "Counterexample",
        "FailureClass", "FalsificationAxis", "FalsificationReport", "InfeasibilityCertificate",
        "IndependentAdversarialFalsifier", "LimitTimeout", "safe_parse_expr",
        "unrepairable_counterexamples",
    ],
    "apedp.gated_assurance": [
        "AdjudicationReport", "AdjudicationVerdict", "BlindChallengeHarness",
        "BlindDiagnosticBenchmark", "DiagnosticReasoningBenchmark", "FailureSeverity",
        "FalsificationBenchResult", "GatedScientificAssuranceEngine",
        "KnownDiagnosticBenchmark", "MechanisticAdjudicationEngine",
        "OperationalGateAuditor", "OperationalReadinessReport", "ParetoFrontierTracker",
        "ReplicationIndependenceVector", "ScientificBenchmarkContract",
        "ScientificFeasibilityConfusionMatrix", "ScientificImprovementMetric",
        "classify_specification", "create_casimir_benchmark_contract",
        "create_hall_petch_benchmark_contract", "map_feasibility_to_stance",
        "validate_three_gates",
    ],
    "apedp.benchmarks": [
        "BenchmarkCaseResult", "BenchmarkCategory", "BenchmarkHarness", "BenchmarkSummary",
        "FalsificationReport", "IndependentAdversarialFalsifier", "NoveltyTier",
        "PathologicalCase", "PathologicalEquationBenchmark", "UnmeasuredMetricError",
    ],
}


def _implementation_modules(package_name: str):
    package = importlib.import_module(package_name)
    for info in pkgutil.iter_modules(package.__path__, package_name + "."):
        yield package, importlib.import_module(info.name)


@pytest.mark.parametrize("package_name", PACKAGES)
def test_facade_reexports_every_defined_symbol(package_name):
    checked = 0
    for package, module in _implementation_modules(package_name):
        for name, obj in vars(module).items():
            if name.startswith("__"):
                continue
            owner = getattr(obj, "__module__", None)
            if owner != module.__name__:
                continue  # imported here; the facade is not the source of truth
            assert hasattr(package, name), f"{package_name} does not re-export {module.__name__}.{name}"
            assert getattr(package, name) is obj, (
                f"{package_name}.{name} is not {module.__name__}.{name}"
            )
            checked += 1
    assert checked >= 10, f"{package_name}: only {checked} symbols checked; the walk is vacuous"


@pytest.mark.parametrize("package_name", PACKAGES)
def test_public_entry_points_still_resolve(package_name):
    package = importlib.import_module(package_name)
    missing = [n for n in PUBLIC_ENTRY_POINTS[package_name] if not hasattr(package, n)]
    assert not missing, f"{package_name} no longer exports: {missing}"
