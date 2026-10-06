"""APEDP-RH: Riemann Hypothesis Mathematical Discovery Benchmark Tournament.

Executes all 16 pre-registered benchmark cases across structure discovery, coordinate
transformation, derived invariants, adversarial alternatives, noise limits, fake-RH worlds,
and mathematical proof boundary verification.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from apedp.decision.stage_c import (
    EpistemicState,
)
from apedp.mathematical.mathematical_layer import MathematicalAssuranceEngine
from apedp.mathematical.rh_adversarial import RiemannAdversarialEngine
from apedp.mathematical.rh_dataset import RiemannZetaBenchmarkDataset
from apedp.mathematical.rh_evaluator import RHCaseResult, RHEvaluator, RHScorecard
from apedp.mathematical.rh_precision import (
    IdentifiabilityFrontierEngine,
)
from apedp.pipeline import DiscoveryPipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("apedp.rh_benchmark")


class RHBenchmarkRunner:
    """Orchestrates the 16-case APEDP-RH Mathematical Discovery Tournament."""

    def __init__(self, output_dir: Path | str = "tournament_results/mathematical/rh_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = RiemannZetaBenchmarkDataset(n_total_zeros=200, train_n=100, val_n=50)
        self.evaluator = RHEvaluator()
        self.math_engine = MathematicalAssuranceEngine()

        # Build adversarial engine from base dataset
        self.adversarial = RiemannAdversarialEngine(
            base_n=self.dataset.n_arr,
            base_gamma=self.dataset.gamma_arr,
        )

    def run_all_cases(self) -> RHScorecard:
        """Executes all 16 benchmark cases and outputs the comprehensive RH scorecard."""
        results: list[RHCaseResult] = []

        logger.info("=================================================================")
        logger.info("STARTING APEDP-RH MATHEMATICAL DISCOVERY BENCHMARK (16 CASES)")
        logger.info("=================================================================")

        # ----------------------------------------------------------------------
        # Stage RH-1: Discovery (Variants A, B, C)
        # ----------------------------------------------------------------------
        results.append(self._run_rh_01_discovery_variant_a())
        results.append(self._run_rh_02_discovery_variant_b())
        results.append(self._run_rh_03_discovery_variant_c())

        # ----------------------------------------------------------------------
        # Stage RH-2: Identifiability & Adversarial Alternatives
        # ----------------------------------------------------------------------
        results.append(self._run_rh_04_alt_a())
        results.append(self._run_rh_05_alt_b())
        results.append(self._run_rh_06_alt_c())
        results.append(self._run_rh_07_alt_d())
        results.append(self._run_rh_08_alt_e())

        # ----------------------------------------------------------------------
        # Precision & Noise Threshold Variants
        # ----------------------------------------------------------------------
        results.append(self._run_rh_09_precision_exact())
        results.append(self._run_rh_10_precision_1e12())
        results.append(self._run_rh_11_precision_1e6())
        results.append(self._run_rh_12_precision_1e3())
        results.append(self._run_rh_13_precision_high_noise())

        # ----------------------------------------------------------------------
        # Fake-RH Counter-Benchmarks (Anti-Overfitting Controls)
        # ----------------------------------------------------------------------
        results.append(self._run_rh_14_fake_rh_1())
        results.append(self._run_rh_15_fake_rh_2())
        results.append(self._run_rh_16_fake_rh_3())

        # Core 16-Case Preregistered Scorecard
        scorecard = self.evaluator.compile_scorecard(results)
        out_file = self.output_dir / "rh_scorecard.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(scorecard.to_dict(), f, indent=2)

        # ----------------------------------------------------------------------
        # Expanded Adversarial Stress Tests (Cases RH-17 to RH-21)
        # ----------------------------------------------------------------------
        expanded_results = list(results)
        expanded_results.append(self._run_rh_17_local_agreement())
        expanded_results.append(self._run_rh_18_delayed_divergence())
        expanded_results.append(self._run_rh_19_oscillatory_cancellation())
        expanded_results.append(self._run_rh_20_slowly_vanishing())
        expanded_results.append(self._run_rh_21_tiny_deviation())

        scorecard_expanded = self.evaluator.compile_scorecard(expanded_results)
        out_file_exp = self.output_dir / "rh_scorecard_expanded.json"
        with open(out_file_exp, "w", encoding="utf-8") as f:
            json.dump(scorecard_expanded.to_dict(), f, indent=2)

        # ----------------------------------------------------------------------
        # Identifiability Frontier Protocol (Epsilon-Sweep)
        # ----------------------------------------------------------------------
        logger.info("Executing Identifiability Frontier Epsilon-Sweep...")
        frontier_engine = IdentifiabilityFrontierEngine()
        frontier_report = frontier_engine.map_frontier(base_n=self.dataset.n_arr, noise_sigma=0.0)
        with open(self.output_dir / "rh_identifiability_frontier.json", "w", encoding="utf-8") as f:
            json.dump(frontier_report.to_dict(), f, indent=2)

        logger.info("=================================================================")
        logger.info("APEDP-RH TOURNAMENT COMPLETE")
        logger.info(f"Core 16-Case Structure Discovery: {scorecard.structure_discovery_rate * 100:.1f}%")
        logger.info(f"Proof Boundary Compliance: {scorecard.proof_boundary_compliance_rate * 100:.1f}%")
        logger.info(f"Ambiguity Detection Rate: {scorecard.ambiguity_detection_rate * 100:.1f}%")
        logger.info(f"False Discovery Rate: {scorecard.false_discovery_rate * 100:.1f}%")
        logger.info(f"Median OOD Error: {scorecard.median_ood_error:.2e}")
        logger.info(f"Identifiability Frontier Resolution: {frontier_report.critical_resolution_boundary:.1e}")
        logger.info("=================================================================")

        return scorecard

    # --------------------------------------------------------------------------
    # Case Implementations
    # --------------------------------------------------------------------------
    def _run_rh_01_discovery_variant_a(self) -> RHCaseResult:
        """RH-01: Full Zeros (n, gamma -> beta = 0.5)."""
        logger.info("Running Case RH-01: Structure Discovery (Full Zeros)...")
        data = self.dataset.get_variant_a_full_zeros()
        full_train = {k: np.concatenate([data["train"][k], data["val"][k]]) for k in data["train"]}

        hyps = [
            {"candidate_id": "critical_line", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "decay_harmonic", "expression": "0.5 + 1.0 / n", "complexity": 3.0},
            {"candidate_id": "linear_gamma", "expression": "0.48 + 0.0001 * gamma", "complexity": 2.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_01"))
        res = pipe.run_scientific_investigation(
            data=full_train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=data["ood"],
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-01",
            case_name="Discovery: Full Zeros (n, gamma -> beta)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5",
            investigation_result=res,
            ood_x={"n": data["ood"]["n"], "gamma": data["ood"]["gamma"]},
            ood_y=data["ood"]["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_02_discovery_variant_b(self) -> RHCaseResult:
        """RH-02: Transformed Coordinates u = 2*beta - 1 -> u = 0.0."""
        logger.info("Running Case RH-02: Transformed Coordinates (u = 2*beta - 1)...")
        data = self.dataset.get_variant_b_transformed()
        full_train = {k: np.concatenate([data["train"][k], data["val"][k]]) for k in data["train"]}

        hyps = [
            {"candidate_id": "zero_line", "expression": "0.0", "complexity": 1.0},
            {"candidate_id": "oscillatory", "expression": "0.01 * sin(gamma)", "complexity": 3.0},
            {"candidate_id": "decay", "expression": "0.05 / gamma", "complexity": 2.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_02"))
        res = pipe.run_scientific_investigation(
            data=full_train,
            target_column="u",
            hypotheses_to_evaluate=hyps,
            near_ood_data=data["ood"],
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-02",
            case_name="Discovery: Transformed Coordinates (u = 2*beta - 1 -> 0)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.0",
            investigation_result=res,
            ood_x={"gamma": data["ood"]["gamma"]},
            ood_y=data["ood"]["u"],
            claimed_universal_proof=False,
        )

    def _run_rh_03_discovery_variant_c(self) -> RHCaseResult:
        """RH-03: Derived Invariants (norm, gamma -> beta = sqrt(norm^2 - gamma^2) = 0.5)."""
        logger.info("Running Case RH-03: Derived Invariants (|rho|, gamma -> beta)...")
        data = self.dataset.get_variant_c_invariants()
        full_train = {k: np.concatenate([data["train"][k], data["val"][k]]) for k in data["train"]}

        hyps = [
            {"candidate_id": "critical_line", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "spacing_model", "expression": "0.25 * spacing", "complexity": 2.0},
            {"candidate_id": "ratio_model", "expression": "norm / gamma - 1.0", "complexity": 3.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_03"))
        res = pipe.run_scientific_investigation(
            data=full_train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=data["ood"],
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-03",
            case_name="Discovery: Derived Invariants (|rho|, gamma -> beta)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5",
            investigation_result=res,
            ood_x={"norm": data["ood"]["norm"], "gamma": data["ood"]["gamma"], "spacing": data["ood"]["spacing"]},
            ood_y=data["ood"]["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_04_alt_a(self) -> RHCaseResult:
        """RH-04: Identifiability Alt A (beta = 0.5 + 1e-3, distinguishable)."""
        logger.info("Running Case RH-04: Identifiability Alt A (+1e-3)...")
        world = self.adversarial.get_alternative_a()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "alt_a", "expression": "0.501", "complexity": 1.0},
            {"candidate_id": "true_rh", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_04"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-04",
            case_name="Identifiability: Alt A (beta = 0.5 + 1e-3)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.501",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_05_alt_b(self) -> RHCaseResult:
        """RH-05: Identifiability Alt B (beta = 0.5 + 1e-6, distinguishable)."""
        logger.info("Running Case RH-05: Identifiability Alt B (+1e-6)...")
        world = self.adversarial.get_alternative_b()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "alt_b", "expression": "0.500001", "complexity": 1.0},
            {"candidate_id": "true_rh", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_05"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-05",
            case_name="Identifiability: Alt B (beta = 0.5 + 1e-6)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.500001",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_06_alt_c(self) -> RHCaseResult:
        """RH-06: Identifiability Alt C (beta = 0.5 + 1 / n^2)."""
        logger.info("Running Case RH-06: Identifiability Alt C (+1/n^2)...")
        world = self.adversarial.get_alternative_c()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "alt_c", "expression": "0.5 + 1.0 / n**2", "complexity": 3.0},
            {"candidate_id": "true_rh", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_06"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-06",
            case_name="Identifiability: Alt C (beta = 0.5 + 1/n^2)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 1.0/n**2",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_07_alt_d(self) -> RHCaseResult:
        """RH-07: Identifiability Alt D (beta = 0.5 + 1e-12*sin(n), below noise resolution)."""
        logger.info("Running Case RH-07: Identifiability Alt D (+1e-12 sin(n) -> MULTI_HYPOTHESIS)...")
        world = self.adversarial.get_alternative_d()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "true_rh", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "alt_d", "expression": "0.5 + 1e-12 * sin(n)", "complexity": 4.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_07"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-07",
            case_name="Identifiability: Alt D (Sub-Resolution Noise -> MULTI_HYPOTHESIS)",
            expected_state=EpistemicState.MULTI_HYPOTHESIS.value,
            ground_truth_form="0.5",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_08_alt_e(self) -> RHCaseResult:
        """RH-08: Identifiability Alt E (Delayed Regime Shift at N=80 -> INSUFFICIENT_EVIDENCE / MULTI)."""
        logger.info("Running Case RH-08: Identifiability Alt E (Delayed Regime Shift)...")
        world = self.adversarial.get_alternative_e(split_n=80, shift=0.05)
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "constant_half", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "linear_drift", "expression": "0.45 + 0.001 * n", "complexity": 2.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_08"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-08",
            case_name="Identifiability: Alt E (Regime Shift at N=80)",
            expected_state=EpistemicState.INSUFFICIENT_EVIDENCE.value,
            ground_truth_form="0.55",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_09_precision_exact(self) -> RHCaseResult:
        """RH-09: Precision Exact (sigma = 0)."""
        logger.info("Running Case RH-09: Precision Variant (Exact sigma = 0)...")
        world = self.adversarial.get_precision_variant(noise_scale=0.0)
        return self._evaluate_precision_case("RH-09", "Precision: Exact (sigma = 0)", world, 0.0)

    def _run_rh_10_precision_1e12(self) -> RHCaseResult:
        """RH-10: Precision Ultra-Low (sigma = 1e-12)."""
        logger.info("Running Case RH-10: Precision Variant (sigma = 1e-12)...")
        world = self.adversarial.get_precision_variant(noise_scale=1e-12)
        return self._evaluate_precision_case("RH-10", "Precision: Ultra-Low (sigma = 1e-12)", world, 1e-12)

    def _run_rh_11_precision_1e6(self) -> RHCaseResult:
        """RH-11: Precision Standard (sigma = 1e-6)."""
        logger.info("Running Case RH-11: Precision Variant (sigma = 1e-6)...")
        world = self.adversarial.get_precision_variant(noise_scale=1e-6)
        return self._evaluate_precision_case("RH-11", "Precision: Standard (sigma = 1e-6)", world, 1e-6)

    def _run_rh_12_precision_1e3(self) -> RHCaseResult:
        """RH-12: Precision Moderate (sigma = 1e-3)."""
        logger.info("Running Case RH-12: Precision Variant (sigma = 1e-3)...")
        world = self.adversarial.get_precision_variant(noise_scale=1e-3)
        return self._evaluate_precision_case("RH-12", "Precision: Moderate (sigma = 1e-3)", world, 1e-3)

    def _run_rh_13_precision_high_noise(self) -> RHCaseResult:
        """RH-13: Precision High Noise (sigma = 0.05)."""
        logger.info("Running Case RH-13: Precision Variant (High Noise sigma = 0.05)...")
        world = self.adversarial.get_precision_variant(noise_scale=0.05)
        return self._evaluate_precision_case("RH-13", "Precision: High Noise (sigma = 0.05)", world, 0.05)

    def _evaluate_precision_case(self, cid: str, cname: str, world: dict[str, np.ndarray], noise: float) -> RHCaseResult:
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "critical_line", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "polynomial_fit", "expression": "0.49 + 0.0001 * gamma", "complexity": 2.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / cid.replace("-", "_")))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id=cid,
            case_name=cname,
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_14_fake_rh_1(self) -> RHCaseResult:
        """RH-14: Fake-RH-1 (beta = 0.5 + 1e-3 * sin(n))."""
        logger.info("Running Case RH-14: Fake-RH-1 (beta = 0.5 + 1e-3 sin(n))...")
        world = self.adversarial.get_fake_rh_1()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "true_generative", "expression": "0.5 + 0.001 * sin(n)", "complexity": 3.0},
            {"candidate_id": "fake_critical_line", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_14"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-14",
            case_name="Fake-RH-1: Sinusoidal (0.5 + 1e-3 sin(n))",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 0.001 * sin(n)",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_15_fake_rh_2(self) -> RHCaseResult:
        """RH-15: Fake-RH-2 (beta = 0.5 + 1 / n)."""
        logger.info("Running Case RH-15: Fake-RH-2 (beta = 0.5 + 1/n)...")
        world = self.adversarial.get_fake_rh_2()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "true_generative", "expression": "0.5 + 1.0 / n", "complexity": 2.0},
            {"candidate_id": "fake_critical_line", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_15"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-15",
            case_name="Fake-RH-2: Harmonic Decay (0.5 + 1/n)",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 1.0/n",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_16_fake_rh_3(self) -> RHCaseResult:
        """RH-16: Fake-RH-3 (beta = 0.5 for n <= 80, 0.51 for n > 80)."""
        logger.info("Running Case RH-16: Fake-RH-3 (Regime shift at N=80 to 0.51)...")
        world = self.adversarial.get_fake_rh_3(split_n=80, ood_val=0.51)
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        # In-sample data has regime shift, so simple 0.5 fails on holdout/OOD
        hyps = [
            {"candidate_id": "naive_half", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "shifted_val", "expression": "0.51", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_16"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-16",
            case_name="Fake-RH-3: Step Shift (0.5 -> 0.51 OOD)",
            expected_state=EpistemicState.MULTI_HYPOTHESIS.value,
            ground_truth_form="0.51",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_17_local_agreement(self) -> RHCaseResult:
        """RH-17: Local Agreement (diverges only far outside observed range)."""
        logger.info("Running Case RH-17: Local Agreement Adversary...")
        world = self.adversarial.get_adversary_local_agreement(epsilon=1e-4, n_obs_max=100)
        cut = 100
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "naive_constant", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "quartic_departure", "expression": "0.5 + 1e-4 * (n / 100.0)**4", "complexity": 4.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_17"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-17",
            case_name="Expanded Adversary: Local Agreement with OOD Divergence",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 1e-4 * (n / 100.0)**4",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_18_delayed_divergence(self) -> RHCaseResult:
        """RH-18: Delayed Divergence (beta = 0.5 for n < 100, 0.55 for n >= 100)."""
        logger.info("Running Case RH-18: Delayed Divergence Adversary...")
        world = self.adversarial.get_adversary_delayed_divergence(split_n=100, shift=0.05)
        cut = 140
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "naive_half", "expression": "0.5", "complexity": 1.0},
            {"candidate_id": "shifted_half", "expression": "0.55", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_18"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-18",
            case_name="Expanded Adversary: Delayed Divergence",
            expected_state=EpistemicState.MULTI_HYPOTHESIS.value,
            ground_truth_form="0.55",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_19_oscillatory_cancellation(self) -> RHCaseResult:
        """RH-19: Oscillatory Cancellation (beta = 0.5 + 1e-3 * sin(log(n+1)))."""
        logger.info("Running Case RH-19: Oscillatory Cancellation Adversary...")
        world = self.adversarial.get_adversary_oscillatory_cancellation(epsilon=1e-3)
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "true_log_osc", "expression": "0.5 + 1e-3 * sin(log(n + 1.0))", "complexity": 4.0},
            {"candidate_id": "constant_half", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_19"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-19",
            case_name="Expanded Adversary: Oscillatory Cancellation",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 1e-3 * sin(log(n + 1.0))",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_20_slowly_vanishing(self) -> RHCaseResult:
        """RH-20: Slowly Vanishing Deviation (beta = 0.5 + 1 / log(n+2))."""
        logger.info("Running Case RH-20: Slowly Vanishing Deviation Adversary...")
        world = self.adversarial.get_adversary_slowly_vanishing_deviation()
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "true_slow_decay", "expression": "0.5 + 1.0 / log(n + 2.0)", "complexity": 3.0},
            {"candidate_id": "constant_half", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_20"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-20",
            case_name="Expanded Adversary: Slowly Vanishing Logarithmic Deviation",
            expected_state=EpistemicState.UNIQUE_SUPPORT.value,
            ground_truth_form="0.5 + 1.0 / log(n + 2.0)",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )

    def _run_rh_21_tiny_deviation(self) -> RHCaseResult:
        """RH-21: Adversarially Tiny Deviation (beta = 0.5 + 10^-8 * cos(n))."""
        logger.info("Running Case RH-21: Adversarially Tiny Deviation (Below Finite Variance)...")
        world = self.adversarial.get_adversary_tiny_deviation(k=8)
        cut = 150
        train = {k: v[:cut] for k, v in world.items()}
        ood = {k: v[cut:] for k, v in world.items()}

        hyps = [
            {"candidate_id": "sub_resolution_dev", "expression": "0.5 + 1e-8 * cos(n)", "complexity": 4.0},
            {"candidate_id": "constant_half", "expression": "0.5", "complexity": 1.0},
        ]

        pipe = DiscoveryPipeline(str(self.output_dir / "RH_21"))
        res = pipe.run_scientific_investigation(
            data=train,
            target_column="beta",
            hypotheses_to_evaluate=hyps,
            near_ood_data=ood,
            include_neural=False,
            include_symbolic=False,
        )

        return self.evaluator.evaluate_case(
            case_id="RH-21",
            case_name="Expanded Adversary: Tiny Deviation (Below Finite Dynamic Variance)",
            expected_state=EpistemicState.MULTI_HYPOTHESIS.value,
            ground_truth_form="0.5",
            investigation_result=res,
            ood_x={"n": ood["n"], "gamma": ood["gamma"]},
            ood_y=ood["beta"],
            claimed_universal_proof=False,
        )


if __name__ == "__main__":
    runner = RHBenchmarkRunner()
    card = runner.run_all_cases()
    print("\n=======================================================")
    print("RH BENCHMARK SCORECARD SUMMARY:")
    print(f"Total Cases: {card.total_cases}")
    print(f"Structure Discovery Rate: {card.structure_discovery_rate * 100:.1f}%")
    print(f"Algebraic Recovery Rate: {card.algebraic_recovery_rate * 100:.1f}%")
    print(f"Ambiguity Detection Rate: {card.ambiguity_detection_rate * 100:.1f}%")
    print(f"Proof Guard Compliance: {card.proof_boundary_compliance_rate * 100:.1f}%")
    print(f"False Discovery Rate: {card.false_discovery_rate * 100:.1f}%")
    print(f"Median OOD MSE: {card.median_ood_error:.2e}")
    print("=======================================================")
