"""Mutation-sensitivity harness for the benchmark suite.

A benchmark only measures something if its metrics *move* when the system under
test changes. These tests inject defects into the falsifier (and into the case
data) and require the headline metrics to react, and they pin the property that
matters most: the case's own label must never reach the falsifier.

If a test here fails, the benchmark has stopped measuring and started confirming.
"""
from __future__ import annotations

from dataclasses import replace

from apedp.benchmarks import (
    BenchmarkHarness,
    BenchmarkSummary,
    PathologicalCase,
    PathologicalEquationBenchmark,
)
from apedp.falsification import (
    FalsificationReport,
    IndependentAdversarialFalsifier,
)

# Small fixed subsets: enough cases for a rate to be meaningful, small enough that
# each mutated run stays fast.
N_SOLVABLE = 4
N_PATHOLOGICAL = 4


def _suite() -> list[PathologicalCase]:
    """A small mixed suite, drawn from the frozen 25-case benchmark."""
    cases = PathologicalEquationBenchmark._get_25_case_suite()
    solvable = [c for c in cases if c.expected_repairable][:N_SOLVABLE]
    pathological = [c for c in cases if not c.expected_repairable][:N_PATHOLOGICAL]
    return solvable + pathological


class _BlindFalsifier(IndependentAdversarialFalsifier):
    """Mutation: the falsifier never finds anything."""

    def falsify(self, *args, **kwargs) -> FalsificationReport:  # type: ignore[override]
        return FalsificationReport(
            formula_str=str(kwargs.get("formula", "")),
            verdict="SURVIVED_PROVISIONAL",
            survived=True,
            cannot_repair=False,
        )


class _AlwaysRefusingFalsifier(IndependentAdversarialFalsifier):
    """Mutation: the falsifier refuses everything, repairable or not."""

    def falsify(self, *args, **kwargs) -> FalsificationReport:  # type: ignore[override]
        return FalsificationReport(
            formula_str=str(kwargs.get("formula", "")),
            verdict="REFUSED_CANNOT_REPAIR",
            survived=False,
            cannot_repair=True,
            refusal_reason="mutation",
        )


class _RecordingFalsifier(IndependentAdversarialFalsifier):
    """Records every call's keyword arguments, then delegates."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def falsify(self, *args, **kwargs) -> FalsificationReport:  # type: ignore[override]
        self.calls.append(dict(kwargs))
        return super().falsify(*args, **kwargs)


def test_probe_receives_no_case_label() -> None:
    """P1: the probe must be identical for two cases that differ only by label.

    This is the regression guard for ``allow_repair_attempt=case.expected_repairable``.
    Feeding that flag in handed the falsifier the answer it was about to be graded
    on: the same formula scored REFUSED_CANNOT_REPAIR when the flag was False and
    FALSIFIED when it was True.
    """
    cases = _suite()
    target = cases[0]
    filler_pathological = next(c for c in cases if not c.expected_repairable)
    filler_solvable = next(c for c in cases if c.expected_repairable and c is not target)

    kwargs_by_label: dict[bool, dict] = {}
    for label in (True, False):
        case = replace(target, expected_repairable=label)
        # The suite must stay measurable in both configurations -- the trust score
        # refuses to be computed from a suite that is missing a subgroup -- so pair
        # the case under test with a filler of the opposite kind. Only the case
        # under test differs between the two runs.
        suite = [case, filler_pathological] if label else [case, filler_solvable]
        recorder = _RecordingFalsifier()
        BenchmarkHarness(falsifier=recorder).run_benchmark(cases=suite)
        probe = next(
            (c for c in recorder.calls if c["formula"] == target.formula_str), None
        )
        assert probe is not None, "the case under test was never probed"
        kwargs_by_label[label] = probe

    as_solvable, as_pathological = kwargs_by_label[True], kwargs_by_label[False]
    assert as_solvable == as_pathological, (
        "the falsifier received different arguments for the same case under "
        f"different labels: {as_solvable} vs {as_pathological}"
    )
    # ...and the policy it did receive is a constant, not a per-case choice.
    assert as_solvable["allow_repair_attempt"] is True


def test_metrics_collapse_when_the_falsifier_is_blinded() -> None:
    """A falsifier that finds nothing must not score well."""
    baseline = BenchmarkHarness().run_benchmark(cases=_suite())
    blinded = BenchmarkHarness(falsifier=_BlindFalsifier()).run_benchmark(cases=_suite())

    assert baseline.defect_detection_rate > 0.0
    assert blinded.defect_detection_rate == 0.0, (
        "a falsifier that never finds a defect still scored detection; the metric "
        "is not reading the falsifier"
    )
    # Unrepairable laws now sail through, which the suite must report as false repairs.
    assert blinded.false_repair_rate == 1.0
    assert blinded.pathological_detection_rate == 0.0
    assert blinded.system_trust_score < baseline.system_trust_score


def test_metrics_collapse_when_the_falsifier_refuses_everything() -> None:
    """A falsifier that refuses everything must not score well either.

    Detection alone cannot catch this mutation -- an all-refusing falsifier sets
    ``defect_detected`` on every case -- so the repair metric is the one that has
    to move. If this ever stops moving, "repair success" has become decorative.
    """
    baseline = BenchmarkHarness().run_benchmark(cases=_suite())
    refusing = BenchmarkHarness(falsifier=_AlwaysRefusingFalsifier()).run_benchmark(cases=_suite())

    assert baseline.solvable_repair_success_rate > 0.0
    assert refusing.solvable_repair_success_rate == 0.0, (
        "every repair candidate was refused, yet repair success stayed non-zero"
    )
    assert refusing.system_trust_score < baseline.system_trust_score


def test_repair_metric_responds_to_a_sabotaged_repair_factory() -> None:
    """Data mutation: a 'repair' that returns the original defective law is no repair."""
    cases = _suite()
    baseline = BenchmarkHarness().run_benchmark(cases=cases)
    assert baseline.solvable_repair_success_rate > 0.0

    sabotaged = [
        replace(c, regularized_candidate_factory=(lambda _eps, _f=c.formula_str: _f))
        if c.expected_repairable
        else c
        for c in cases
    ]
    mutated = BenchmarkHarness().run_benchmark(cases=sabotaged)

    assert mutated.solvable_repair_success_rate < baseline.solvable_repair_success_rate, (
        "replacing the repaired law with the defective original did not change the "
        "repair metric"
    )


def test_summary_separates_detection_from_refusal() -> None:
    """The two pathological metrics are distinct measurements, not aliases."""
    summary: BenchmarkSummary = BenchmarkHarness().run_benchmark(cases=_suite())
    assert summary.to_dict()["pathological_detection_rate"] == summary.pathological_detection_rate
    assert summary.pathological_detection_rate >= summary.pathological_refusal_accuracy, (
        "a case can be detected without being refused, never the reverse"
    )
