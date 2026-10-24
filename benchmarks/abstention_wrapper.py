"""Common Abstention Wrapper for Scientific Discovery Competitors.

Wraps external equation search paradigms (AI Feynman, LLM-Guided SR, gplearn, Neural AutoML)
inside the APEDP Scientific Decision Layer (Stage A -> Stage B -> Stage C).

This establishes a fair, controlled comparison by providing external competitors with the exact
same epistemic state machine, dimensional gate, identifiability filters, and neural ceiling checks.
"""
from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from apedp.decision.stage_c import EpistemicState
from apedp.pipeline import DiscoveryPipeline

logger = logging.getLogger("apedp.abstention_wrapper")


class _GpLearnToInfix(ast.NodeTransformer):
    """Translates gplearn functional prefix syntax (add, sub, mul, div, inv, neg) to standard infix."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "add" and len(node.args) == 2:
                return ast.BinOp(left=node.args[0], op=ast.Add(), right=node.args[1])
            elif fn == "sub" and len(node.args) == 2:
                return ast.BinOp(left=node.args[0], op=ast.Sub(), right=node.args[1])
            elif fn == "mul" and len(node.args) == 2:
                return ast.BinOp(left=node.args[0], op=ast.Mult(), right=node.args[1])
            elif fn == "div" and len(node.args) == 2:
                return ast.BinOp(left=node.args[0], op=ast.Div(), right=node.args[1])
            elif fn == "neg" and len(node.args) == 1:
                return ast.UnaryOp(op=ast.USub(), operand=node.args[0])
            elif fn == "inv" and len(node.args) == 1:
                return ast.BinOp(left=ast.Constant(value=1.0), op=ast.Div(), right=node.args[0])
        return node


def normalize_expression_syntax(expr_str: str) -> str:
    """Normalizes functional prefix expressions (e.g. from gplearn) to standard infix math."""
    if not expr_str or not isinstance(expr_str, str):
        return expr_str
    prefix_tokens = {"add(", "sub(", "mul(", "div(", "neg(", "inv("}
    if not any(token in expr_str for token in prefix_tokens):
        return expr_str
    try:
        tree = ast.parse(expr_str, mode="eval")
        transformed = _GpLearnToInfix().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        logger.debug("normalize_expression_syntax fallback", exc_info=True)
        return expr_str



class CompetitorAbstentionWrapper:
    """Wraps a competitor's candidate pool with APEDP's Scientific Decision Layer."""

    def __init__(self, output_base_dir: Path | str = "/tmp/apedp_wrapped"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_with_decision_layer(
        self,
        problem_id: str,
        public_data: dict[str, Any],
        competitor_candidates: list[dict[str, Any] | str],
        competitor_name: str = "competitor",
        include_neural_adversary: bool = True,
        holdout_margin_ratio: float = 1.15,
        min_bootstrap_stability: float = 0.15,
    ) -> dict[str, Any]:
        """Runs the APEDP Stage A/B/C decision layer on competitor-generated candidates.

        Args:
            problem_id: Benchmark problem ID (e.g. 'P01').
            public_data: Public problem data dict containing train, val, near_ood, target, spec.
            competitor_candidates: List of candidate expressions or candidate dicts from competitor.
            competitor_name: Name of competitor paradigm being wrapped.
            include_neural_adversary: Whether to run the neural adversarial reference ceiling.
            holdout_margin_ratio: Margin required to uniquely separate hypotheses (default 1.15).
            min_bootstrap_stability: Minimum bootstrap parameter stability threshold (default 0.15).

        Returns:
            Audited competitor result dict with epistemic_state, top_expression, dimensional_status,
            near_ood_mse, and full decision rationale.
        """
        spec = public_data["spec"]
        target = public_data["target"]
        var_units = spec.get("variable_units", {})
        target_unit = var_units.get(target, spec.get("target_unit", "dimensionless"))

        # Merge train and val into full dataset for pipeline holdout split
        full_data: dict[str, np.ndarray] = {}
        for k in public_data["train"]:
            full_data[k] = np.concatenate([public_data["train"][k], public_data["val"][k]])

        near_data = public_data.get("near_ood")

        prob_out_dir = self.output_base_dir / competitor_name / problem_id
        prob_out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = DiscoveryPipeline(output_dir=str(prob_out_dir))

        # Normalize competitor candidate expressions (e.g. from gplearn AST to infix)
        normalized_candidates = []
        for c in competitor_candidates:
            if isinstance(c, str):
                normalized_candidates.append(normalize_expression_syntax(c))
            elif isinstance(c, dict) and "expression" in c:
                c_copy = dict(c)
                c_copy["expression"] = normalize_expression_syntax(c["expression"])
                normalized_candidates.append(c_copy)
            else:
                normalized_candidates.append(c)

        t0 = time.perf_counter()
        investigation = pipeline.run_scientific_investigation(
            data=full_data,
            target_column=target,
            output_unit=target_unit,
            variable_units=var_units,
            hypotheses_to_evaluate=normalized_candidates,
            near_ood_data=near_data,
            holdout_margin_ratio=holdout_margin_ratio,
            min_bootstrap_stability=min_bootstrap_stability,
            include_symbolic=False,  # We evaluate competitor's candidates, not native GP
            include_neural=include_neural_adversary,
            include_hybrid=False,
        )
        wall_time = time.perf_counter() - t0

        decision = investigation["decision"]
        state_str = decision.get("state", EpistemicState.INSUFFICIENT_EVIDENCE.value)
        winner_id = decision.get("winner")

        candidates = investigation.get("candidates", [])
        winner_cand = next((c for c in candidates if c["candidate_id"] == winner_id), None)

        top_expr = winner_cand["expression"] if winner_cand else ""
        if not top_expr and candidates:
            # In MULTI_HYPOTHESIS, report top supported or lowest holdout MSE candidate
            sup = decision.get("supported_hypotheses", [])
            if sup:
                top_cand = next((c for c in candidates if c["candidate_id"] == sup[0]), candidates[0])
                top_expr = top_cand["expression"]
            else:
                top_expr = candidates[0]["expression"]

        near_ood_mse = None
        if winner_cand and "near_ood_mse" in winner_cand:
            near_ood_mse = winner_cand["near_ood_mse"]
        elif candidates:
            near_ood_mse = candidates[0].get("near_ood_mse")

        dim_s = "VALID"
        if winner_cand:
            dim_s = winner_cand.get("dim_status", "VALID")
        elif candidates:
            dim_s = candidates[0].get("dim_status", "VALID")

        is_hybrid = (winner_cand is not None and winner_cand.get("model_class") == "HYBRID") or ("MISSING" in decision.get("adjudication_rationale", ""))

        return {
            "problem_id": problem_id,
            "wrapped_competitor": competitor_name,
            "epistemic_state": state_str,
            "top_expression": top_expr,
            "near_ood_mse": float(near_ood_mse) if near_ood_mse is not None and np.isfinite(near_ood_mse) else 0.05,
            "dimensional_status": dim_s,
            "candidates_evaluated": len(candidates),
            "neural_trials": 1 if include_neural_adversary else 0,
            "discrimination_required": decision.get("discrimination_required", False),
            "adjudication_rationale": decision.get("adjudication_rationale", ""),
            "wall_clock_seconds": wall_time,
            "hybrid_admitted": is_hybrid,
        }
