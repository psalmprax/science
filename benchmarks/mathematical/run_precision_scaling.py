"""Runner for the Precision-Scaling Identifiability Experiment.

Executes the epsilon* resolution boundary test across:
- float32 (24-bit)
- float64 (53-bit)
- mpmath-50 (50-digit)
- mpmath-100 (100-digit)
- mpmath-200 (200-digit)

Saves results to tournament_results/mathematical/rh_benchmark/precision_scaling_results.json
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apedp.mathematical.precision_scaling_experiment import PrecisionScalingEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("apedp.run_precision_scaling")


def main() -> None:
    logger.info("Executing Precision-Scaling Identifiability Experiment...")
    engine = PrecisionScalingEngine(n_points=100, seed=42)
    results = engine.run_full_comparison()

    out_dir = Path("tournament_results/mathematical/rh_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "precision_scaling_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Precision-Scaling Experiment Complete. Summary Table:")
    for row in results["comparison_table"]:
        logger.info(
            f"  {row['regime']}: machine_eps={row['machine_epsilon']} -> "
            f"theoretical_eps*={row['theoretical_eps_star']} | empirical_eps*={row['empirical_eps_star']}"
        )
    logger.info(f"Scientific Finding: {results['scientific_finding']}")
    print("\nResults successfully saved to:", out_file)


if __name__ == "__main__":
    main()
