"""Wrapped Competitor Benchmark Tournament.

Executes the head-to-head tournament where ALL external competitor paradigms
(AI Feynman 2.0, LLM-Guided SR, gplearn, Neural AutoML) are equipped with the
APEDP Scientific Decision Layer (Stage A -> Stage B -> Stage C).

Computes the post-wrapping multi-dimensional evaluation profile M^wrapped and
compares directly with the unwrapped baseline results.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from benchmarks.abstention_wrapper import CompetitorAbstentionWrapper
from benchmarks.independent_evaluator import IndependentEvaluator
from benchmarks.tournament_runner import TournamentRunner, load_problem_public_data

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("apedp.wrapped_tournament")


class WrappedTournamentRunner:
    """Orchestrates the wrapped tournament across all 30 benchmark problems."""

    def __init__(
        self,
        corpus_dir: Path | str = "benchmark_corpus",
        unwrapped_results_dir: Path | str = "tournament_results",
        output_dir: Path | str = "tournament_results/wrapped",
    ):
        self.corpus_dir = Path(corpus_dir)
        self.unwrapped_dir = Path(unwrapped_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = IndependentEvaluator(self.corpus_dir)
        self.wrapper = CompetitorAbstentionWrapper(output_base_dir=self.output_dir / "scratch")
        self.base_runner = TournamentRunner(corpus_dir=self.corpus_dir, results_dir=self.unwrapped_dir)
        self.prob_ids = [f"P{i:02d}" for i in range(1, 31)]

    def run_wrapped_tournament(self) -> dict[str, Any]:
        """Runs wrapped evaluation for all competitor paradigms."""
        competitors = ["gplearn", "ai_feynman", "llm_sr", "apedp"]
        summary: dict[str, Any] = {}

        for comp in competitors:
            logger.info("==========================================")
            logger.info("Executing Wrapped Competitor: %s", comp.upper())
            logger.info("==========================================")
            t0 = time.perf_counter()
            self._run_wrapped_competitor(comp)
            dt = time.perf_counter() - t0
            logger.info("Completed wrapped %s in %.2fs", comp, dt)

            # Blind Independent Evaluation on wrapped outputs
            comp_eval = self.evaluator.evaluate_competitor(
                f"wrapped_{comp}" if comp != "apedp" else "apedp",
                self.output_dir / (f"wrapped_{comp}" if comp != "apedp" else "apedp"),
            )
            summary[f"wrapped_{comp}" if comp != "apedp" else "apedp"] = comp_eval

        # Save summary
        summary_path = self.output_dir / "wrapped_tournament_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _run_wrapped_competitor(self, comp: str) -> None:
        target_name = f"wrapped_{comp}" if comp != "apedp" else "apedp"
        comp_dir = self.output_dir / target_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        for pid in self.prob_ids:
            prob_dir = comp_dir / pid
            prob_dir.mkdir(parents=True, exist_ok=True)
            res_file = prob_dir / "result.json"

            p_data = load_problem_public_data(pid)

            if comp == "apedp":
                # Native APEDP 3.0 with Stage A constant synthesis
                res = self.base_runner._run_apedp_on_problem(pid, p_data)
            else:
                # Retrieve competitor's candidate pool / proposed expressions
                candidates = self._get_competitor_candidates(comp, pid, p_data)
                res = self.wrapper.evaluate_with_decision_layer(
                    problem_id=pid,
                    public_data=p_data,
                    competitor_candidates=candidates,
                    competitor_name=comp,
                )

            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    def _get_competitor_candidates(self, comp: str, pid: str, p_data: dict[str, Any]) -> list[str]:
        """Extracts candidate expression(s) produced by competitor for given problem."""
        # Check if pre-computed unwrapped result exists
        unwrapped_res_path = self.unwrapped_dir / comp / pid / "result.json"
        if unwrapped_res_path.exists():
            try:
                with open(unwrapped_res_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                expr = cached.get("top_expression", "")
                if expr:
                    return [expr]
            except Exception:
                logger.debug("cached unwrapped result unreadable for %s", pid, exc_info=True)

        # Fallback to competitor runner logic
        if comp == "gplearn":
            r = self.base_runner._run_gplearn_on_problem(pid, p_data)
            return [r.get("top_expression", "1.0")]
        elif comp == "ai_feynman":
            r = self.base_runner._run_ai_feynman_on_problem(pid, p_data)
            return [r.get("top_expression", "1.0")]
        elif comp == "llm_sr":
            r = self.base_runner._run_llm_sr_on_problem(pid, p_data)
            return [r.get("top_expression", "1.0")]
        return ["1.0"]


if __name__ == "__main__":
    runner = WrappedTournamentRunner()
    summary = runner.run_wrapped_tournament()
    print("\nWrapped Tournament Complete! Results summary saved to tournament_results/wrapped/wrapped_tournament_summary.json")
