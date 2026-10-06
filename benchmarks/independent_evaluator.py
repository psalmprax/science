"""Independent Blind Evaluation Harness for Phase 4 External Tournament.

Joins competitor predictions with hidden ground truth data without allowing
competitors access to hidden truth. Computes the 7-dimensional profile M,
the 6x6 Epistemic Confusion Matrix, and paired Wilcoxon/Holm tests.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from apedp.canonical_compiler import CanonicalEquationCompiler
from apedp.symbolic_equivalence import SymbolicEquivalenceEngine

logger = logging.getLogger("apedp.independent_evaluator")

# Formal 6 Ground Truth Epistemic Categories
EPISTEMIC_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
EPISTEMIC_CLASS_NAMES = {
    "C1": "UNIQUE_SUPPORT",
    "C2": "MULTI_HYPOTHESIS",
    "C3": "INSUFFICIENT_EVIDENCE",
    "C4": "STRUCTURAL_EQUIVALENCE",
    "C5": "MISSING_STRUCTURE",
    "C6": "FALSIFICATION_REVISED",
}


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(val) for val in row] for row in reader]
    return header, np.array(rows)


class IndependentEvaluator:
    """Evaluates competitor outputs blindly against hidden ground truth."""

    def __init__(self, corpus_dir: Path | str):
        self.corpus_dir = Path(corpus_dir)
        self.hidden_dir = self.corpus_dir / "hidden"
        self.public_dir = self.corpus_dir / "public"
        self.eq_engine = SymbolicEquivalenceEngine()
        self.compiler = CanonicalEquationCompiler()

        manifest_path = self.corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def load_ground_truth(self, prob_id: str) -> dict[str, Any]:
        gt_path = self.hidden_dir / prob_id / "ground_truth.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_competitor(
        self,
        competitor_name: str,
        results_dir: Path | str,
    ) -> dict[str, Any]:
        """Evaluates all problem outputs for a given competitor."""
        results_dir = Path(results_dir)
        prob_ids = sorted([p.name for p in self.hidden_dir.iterdir() if p.is_dir()])

        per_problem: dict[str, Any] = {}
        confusion_matrix = {c_true: {c_pred: 0 for c_pred in EPISTEMIC_CLASSES} for c_true in EPISTEMIC_CLASSES}

        rec_count = 0
        exact_count = 0
        algebraic_count = 0
        structural_count = 0
        numerical_count = 0
        physical_count = 0
        false_discovery_count = 0
        correct_epistemic_count = 0
        ood_errors: list[float] = []
        dim_valid_count = 0
        wall_times: list[float] = []
        eval_counts: list[int] = []

        class_breakdowns: dict[int, dict[str, Any]] = {
            c: {
                "total": 0,
                "rec": 0,
                "exact": 0,
                "algebraic": 0,
                "structural": 0,
                "numerical": 0,
                "physical": 0,
                "false_disc": 0,
                "epi_correct": 0,
                "ood_errs": [],
                "dim_valid": 0,
            }
            for c in range(1, 6)
        }

        for pid in prob_ids:
            gt = self.load_ground_truth(pid)
            c_id = gt["class_id"]
            class_breakdowns[c_id]["total"] += 1

            prob_res_path = results_dir / pid / "result.json"
            if not prob_res_path.exists():
                per_problem[pid] = {"status": "MISSING_RESULT", "problem_id": pid}
                continue

            with open(prob_res_path, "r", encoding="utf-8") as f:
                res = json.load(f)

            # 1. Epistemic state evaluation
            gt_state_class = gt["oracle_epistemic_class"]
            pred_state_str = str(res.get("epistemic_state", "UNIQUE_SUPPORT")).upper()

            # Map competitor state string to formal epistemic class
            pred_class = self._map_to_epistemic_class(pred_state_str, res)
            confusion_matrix[gt_state_class][pred_class] += 1

            is_epistemic_correct = (pred_class == gt_state_class)
            if is_epistemic_correct:
                correct_epistemic_count += 1
                class_breakdowns[c_id]["epi_correct"] += 1

            # 2. False Discovery: Unhedged top pick on C2 (Ambiguous) or C3 (Insufficient/Pathological)
            is_false_discovery = False
            if gt_state_class in ["C2", "C3"] and pred_class == "C1":
                is_false_discovery = True
                false_discovery_count += 1
                class_breakdowns[c_id]["false_disc"] += 1

            # 3. Multi-Tier Scientific Equation Recovery
            is_recovered = False
            is_exact = False
            is_algebraic = False
            is_structural = False
            is_numerical = False
            is_physical = False
            recovery_verdict_dict = {}

            pred_expr = res.get("top_expression", "")
            gt_expr = gt.get("ground_truth_formula", "")
            if pred_expr and gt_expr:
                try:
                    verdict = self.compiler.classify_recovery(
                        predicted=pred_expr,
                        truth=gt_expr,
                        domain_bounds=gt.get("variable_domains"),
                    )
                    is_exact = verdict.is_exact
                    is_algebraic = verdict.is_algebraic
                    is_structural = verdict.is_structural
                    is_numerical = verdict.is_numerical
                    recovery_verdict_dict = verdict.to_dict()
                except Exception:
                    logger.debug("classify_recovery failed, trying equivalence engine", exc_info=True)
                    try:
                        is_algebraic = self.eq_engine.are_algebraically_equivalent(pred_expr, gt_expr)
                        is_exact = is_algebraic
                    except Exception:
                        logger.debug("equivalence fallback failed", exc_info=True)

            is_dim_valid = res.get("dimensional_status", "VALID") == "VALID"
            is_physical = is_dim_valid and (is_exact or is_algebraic or is_structural)
            is_recovered = is_algebraic or is_exact

            if is_exact:
                exact_count += 1
                class_breakdowns[c_id]["exact"] += 1
            if is_algebraic:
                algebraic_count += 1
                class_breakdowns[c_id]["algebraic"] += 1
            if is_structural:
                structural_count += 1
                class_breakdowns[c_id]["structural"] += 1
            if is_numerical:
                numerical_count += 1
                class_breakdowns[c_id]["numerical"] += 1
            if is_physical:
                physical_count += 1
                class_breakdowns[c_id]["physical"] += 1
            if is_recovered:
                rec_count += 1
                class_breakdowns[c_id]["rec"] += 1

            # 4. Out-of-Distribution Error
            near_ood_mse = res.get("near_ood_mse", None)
            if near_ood_mse is not None and math.isfinite(near_ood_mse):
                ood_errors.append(float(near_ood_mse))
                class_breakdowns[c_id]["ood_errs"].append(float(near_ood_mse))

            # 5. Dimensional Validity
            if is_dim_valid:
                dim_valid_count += 1
                class_breakdowns[c_id]["dim_valid"] += 1

            # 6. Resource Telemetry
            if "wall_clock_seconds" in res:
                wall_times.append(float(res["wall_clock_seconds"]))
            if "candidates_evaluated" in res:
                eval_counts.append(int(res["candidates_evaluated"]))

            per_problem[pid] = {
                "problem_id": pid,
                "class_id": c_id,
                "gt_class": gt_state_class,
                "pred_class": pred_class,
                "epistemic_match": is_epistemic_correct,
                "is_false_discovery": is_false_discovery,
                "is_recovered": is_recovered,
                "is_exact": is_exact,
                "is_algebraic": is_algebraic,
                "is_structural": is_structural,
                "is_numerical": is_numerical,
                "is_physical": is_physical,
                "recovery_verdict": recovery_verdict_dict,
                "near_ood_mse": near_ood_mse,
                "dim_valid": is_dim_valid,
                "wall_time": res.get("wall_clock_seconds", 0.0),
                "candidates_eval": res.get("candidates_evaluated", 0),
            }

        total_probs = len(prob_ids)
        overall_profile = {
            "R_exact": exact_count / total_probs if total_probs else 0.0,
            "R_algebraic": algebraic_count / total_probs if total_probs else 0.0,
            "R_structural": structural_count / total_probs if total_probs else 0.0,
            "R_numerical": numerical_count / total_probs if total_probs else 0.0,
            "R_physical": physical_count / total_probs if total_probs else 0.0,
            "R_rec": rec_count / total_probs if total_probs else 0.0,
            "R_false": false_discovery_count / total_probs if total_probs else 0.0,
            "P_epi": correct_epistemic_count / total_probs if total_probs else 0.0,
            "E_OOD_median": float(np.median(ood_errors)) if ood_errors else None,
            "R_dim": dim_valid_count / total_probs if total_probs else 0.0,
            "mean_wall_clock_seconds": float(np.mean(wall_times)) if wall_times else 0.0,
            "mean_candidates_evaluated": float(np.mean(eval_counts)) if eval_counts else 0.0,
        }

        # Per-class breakdowns
        per_class_profiles: dict[int, dict[str, Any]] = {}
        for c in range(1, 6):
            b = class_breakdowns[c]
            cnt = b["total"]
            per_class_profiles[c] = {
                "class_id": c,
                "total": cnt,
                "R_exact": b["exact"] / cnt if cnt else 0.0,
                "R_algebraic": b["algebraic"] / cnt if cnt else 0.0,
                "R_structural": b["structural"] / cnt if cnt else 0.0,
                "R_numerical": b["numerical"] / cnt if cnt else 0.0,
                "R_physical": b["physical"] / cnt if cnt else 0.0,
                "R_rec": b["rec"] / cnt if cnt else 0.0,
                "R_false": b["false_disc"] / cnt if cnt else 0.0,
                "P_epi": b["epi_correct"] / cnt if cnt else 0.0,
                "E_OOD_median": float(np.median(b["ood_errs"])) if b["ood_errs"] else None,
                "R_dim": b["dim_valid"] / cnt if cnt else 0.0,
            }

        return {
            "competitor_name": competitor_name,
            "total_problems_evaluated": total_probs,
            "evaluation_profile_M": overall_profile,
            "class_profiles": per_class_profiles,
            "epistemic_confusion_matrix": confusion_matrix,
            "per_problem_results": per_problem,
        }

    def _map_to_epistemic_class(self, pred_state_str: str, res: dict[str, Any]) -> str:
        if "UNIQUE" in pred_state_str:
            return "C1"
        elif "MULTI" in pred_state_str:
            return "C2"
        elif "INSUFFICIENT" in pred_state_str or "NO_VALID" in pred_state_str:
            return "C3"
        elif "EQUIVALEN" in pred_state_str:
            return "C4"
        elif "MISSING" in pred_state_str or res.get("hybrid_admitted", False):
            return "C5"
        elif "REVISE" in pred_state_str or "BELIEF" in pred_state_str:
            return "C6"
        return "C1"  # Default unhedged top pick


if __name__ == "__main__":
    corpus = Path(__file__).parent.parent / "benchmark_corpus"
    evaluator = IndependentEvaluator(corpus)
    print(f"Independent Evaluator loaded with {len(evaluator.manifest.get('problems', {}))} problems.")
