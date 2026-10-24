"""Benchmark harness orchestrating cases and summaries."""
from __future__ import annotations

from apedp.falsification import IndependentAdversarialFalsifier

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
    _rate,
    _require,
)


class BenchmarkHarness:
    """Executes the benchmark suite through the Independent Falsifier and evaluates system rigor."""

    def __init__(self, falsifier: IndependentAdversarialFalsifier | None = None):
        self.falsifier = falsifier or IndependentAdversarialFalsifier()

    def run_benchmark(
        self,
        cases: list[PathologicalCase] | None = None,
        suite_size: int = 25
    ) -> BenchmarkSummary:
        """Runs the benchmark cases and computes comprehensive trust and refusal metrics."""
        if cases is None:
            cases = PathologicalEquationBenchmark.get_benchmark_cases(suite_size=suite_size)

        results: list[BenchmarkCaseResult] = []

        for case in cases:
            # 1. Probe the raw equation with the Independent Falsifier.
            #
            # The probe is deliberately told nothing about the case: no label, no
            # per-case policy. ``case.expected_repairable`` is used only *after* the
            # run, to group results. Passing it in (as
            # ``allow_repair_attempt=case.expected_repairable``) handed the system
            # the answer it was about to be graded on: an identical input scored
            # REFUSED_CANNOT_REPAIR when the flag was False and FALSIFIED when it
            # was True, so refusal accuracy measured the label rather than the
            # engine. Every case now gets the same fixed policy.
            raw_report = self.falsifier.falsify(
                formula=case.formula_str,
                variable_domains=case.variable_domains,
                physical_invariants=case.invariants,
                critical_boundaries=case.critical_boundaries,
                constraints=case.constraints,
                allow_repair_attempt=True,
            )

            defect_detected = not raw_report.survived
            correct_diagnosis = False
            refusal_issued = raw_report.cannot_repair or raw_report.verdict in (
                "REFUSED_CANNOT_REPAIR",
                "REFUSED_INCOMPATIBLE_CONSTRAINTS"
            )
            regularization_attempted = False
            regularization_survived = False
            false_repair_flagged = False

            # NOTE: the label is consumed here, after the probe has run.
            if case.expected_repairable:
                # Solvable case: expect defect to be detected, then candidate regularized
                if defect_detected:
                    correct_diagnosis = True
                    if case.regularized_candidate_factory:
                        regularization_attempted = True
                        reg_formula = case.regularized_candidate_factory(0.04)
                        reg_report = self.falsifier.falsify(
                            formula=reg_formula,
                            variable_domains=case.variable_domains,
                            physical_invariants=case.invariants,
                            critical_boundaries=case.critical_boundaries,
                            allow_repair_attempt=True
                        )
                        regularization_survived = reg_report.survived
            else:
                # Pathological case: expect defect to be detected and refusal to be issued
                if defect_detected or refusal_issued:
                    correct_diagnosis = True
                if not refusal_issued and not defect_detected:
                    # System failed to refuse an unrepairable law!
                    false_repair_flagged = True

            results.append(BenchmarkCaseResult(
                case_id=case.case_id,
                name=case.name,
                domain=case.domain,
                expected_repairable=case.expected_repairable,
                defect_detected=defect_detected,
                correct_diagnosis=correct_diagnosis,
                refusal_issued=refusal_issued,
                regularization_attempted=regularization_attempted,
                regularization_survived_falsification=regularization_survived,
                false_repair_flagged=false_repair_flagged,
                category=case.category,
                novelty_tier=case.novelty_tier,
                falsification_report=raw_report
            ))

        # Compute summary metrics
        total = len(results)
        solvable = [r for r in results if r.expected_repairable]
        pathological = [r for r in results if not r.expected_repairable]
        incompat_cases = [r for r in results if r.category == BenchmarkCategory.DELIBERATELY_INCOMPATIBLE_CONSTRAINTS]

        defect_detection_rate = _rate(sum(1 for r in results if r.defect_detected), total)
        correct_diagnosis_rate = _rate(sum(1 for r in results if r.correct_diagnosis), total)

        solvable_success = _rate(
            sum(1 for r in solvable if r.regularization_survived_falsification), len(solvable)
        )
        # Two distinct measured quantities, kept apart on purpose:
        #   * ``refusal`` is the rate at which the engine issues an outright
        #     refusal. It is a real measurement, but it is no longer the headline:
        #     a refusal can also be the correct-but-uninformative response, and it
        #     used to be forced by the case label.
        #   * ``detection`` is the rate at which the pathology was found at all --
        #     which is the property an unrepairable-law suite is supposed to
        #     measure now that the probe is label-free.
        pathological_refusal_acc = _rate(
            sum(1 for r in pathological if r.refusal_issued), len(pathological)
        )
        pathological_detection_rate = _rate(
            sum(1 for r in pathological if r.defect_detected), len(pathological)
        )
        infeasibility_refusal_acc = _rate(
            sum(1 for r in incompat_cases if r.refusal_issued), len(incompat_cases)
        )
        false_repair_rate = _rate(
            sum(1 for r in pathological if r.false_repair_flagged), len(pathological)
        )

        # Trust score: a weighted combination of the rates measured above. Weights
        # are module constants (see TRUST_WEIGHT_*), and the score is 0.0 for an
        # empty suite rather than a defaulted pass.
        # Every weighted factor must have been measured. A missing factor is
        # refused rather than silently substituted, which is what the
        # ``if solvable else 0.0`` fallbacks used to do -- deflating the score for
        # a suite that simply could not measure that property.
        trust_score: float | None = None
        if total > 0:
            trust_score = (
                TRUST_WEIGHT_DEFECT_DETECTION * _require(defect_detection_rate, "defect_detection_rate")
                + TRUST_WEIGHT_SOLVABLE_REPAIR * _require(solvable_success, "solvable_repair_success_rate")
                + TRUST_WEIGHT_PATHOLOGICAL_DETECTION
                * _require(pathological_detection_rate, "pathological_detection_rate")
                + TRUST_WEIGHT_NO_FALSE_REPAIR * (1.0 - _require(false_repair_rate, "false_repair_rate"))
            ) * 100.0

        # Compute breakdown by novelty tier
        tier_metrics: dict[str, dict[str, float | None]] = {}
        for tier in NoveltyTier:
            t_cases = [r for r in results if r.novelty_tier == tier]
            if t_cases:
                t_solvable = [r for r in t_cases if r.expected_repairable]
                t_patho = [r for r in t_cases if not r.expected_repairable]
                tier_metrics[tier.value] = {
                    "count": float(len(t_cases)),
                    "defect_detection_rate": sum(1 for r in t_cases if r.defect_detected) / len(t_cases),
                    "solvable_repair_success_rate": _rate(
                        sum(1 for r in t_solvable if r.regularization_survived_falsification),
                        len(t_solvable),
                    ),
                    "refusal_accuracy": _rate(
                        sum(1 for r in t_patho if r.refusal_issued), len(t_patho)
                    ),
                    "pathological_detection_rate": _rate(
                        sum(1 for r in t_patho if r.defect_detected), len(t_patho)
                    ),
                }

        # Compute breakdown by category
        cat_metrics: dict[str, dict[str, float | None]] = {}
        for cat in BenchmarkCategory:
            c_cases = [r for r in results if r.category == cat]
            if c_cases:
                c_solvable = [r for r in c_cases if r.expected_repairable]
                c_patho = [r for r in c_cases if not r.expected_repairable]
                cat_metrics[cat.value] = {
                    "count": float(len(c_cases)),
                    "defect_detection_rate": sum(1 for r in c_cases if r.defect_detected) / len(c_cases),
                    "solvable_repair_success_rate": _rate(
                        sum(1 for r in c_solvable if r.regularization_survived_falsification),
                        len(c_solvable),
                    ),
                    "refusal_accuracy": _rate(
                        sum(1 for r in c_patho if r.refusal_issued), len(c_patho)
                    ),
                    "pathological_detection_rate": _rate(
                        sum(1 for r in c_patho if r.defect_detected), len(c_patho)
                    ),
                }

        conformance_note = (
            f"Benchmark conformance: {round(trust_score, 2)}% on {total} predefined cases. "
            if trust_score is not None
            else "Benchmark conformance: not measurable -- the suite does not cover every "
            "subgroup the score depends on. "
        )

        return BenchmarkSummary(
            total_cases=total,
            solvable_cases=len(solvable),
            pathological_cases=len(pathological),
            defect_detection_rate=defect_detection_rate,
            correct_diagnosis_rate=correct_diagnosis_rate,
            solvable_repair_success_rate=solvable_success,
            pathological_refusal_accuracy=pathological_refusal_acc,
            pathological_detection_rate=pathological_detection_rate,
            false_repair_rate=false_repair_rate,
            system_trust_score=trust_score,
            benchmark_conformance_score=trust_score,
            out_of_sample_validity_established=False,
            universal_correctness_claimed=False,
            epistemic_qualification=(
                conformance_note
                + "Out-of-sample scientific validity is not established; "
                "universal correctness is explicitly not claimed."
            ),
            infeasibility_refusal_accuracy=infeasibility_refusal_acc,
            tier_metrics=tier_metrics,
            category_metrics=cat_metrics,
            detailed_results=results
        )
