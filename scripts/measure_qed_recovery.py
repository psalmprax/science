#!/usr/bin/env python3
"""QED-001: can a quantum-assisted proposer recover a governing equation?

The research question this record answers is the one that can actually be
falsified, and it is deliberately not "can a quantum circuit discover physics":

    Given noisy observations of a damped harmonic oscillator and a fixed
    ten-term library, does a QAOA-mediated search recover the generating
    equation as well as four classical proposers and a RIME variant, when every
    arm is held to the same number of objective fits?

The design, in the order the record reports it:

* **The task is fair, and that is measured.** An exhaustive sweep of all
  ``2**10 = 1024`` selections -- run by the harness, outside every search budget
  -- confirms that the penalised optimum *is* the generating support
  ``{x, v}``, and reports the margin to the runner-up. Without that number a
  missed optimum would be indistinguishable from a badly posed problem.
* **Every arm pays the same currency.** The budget is a fixed number of fits of
  ``J``, and the space is 1024 selections, so no arm can enumerate it and search
  is required. A repeated selection is free because it is already known; that is
  the only way an arm may spend fewer *distinct* fits than its budget.
* **Two budgets are measured.** At 256 fits a majority of arms is expected to
  reach the optimum, which ranks nothing; at 64 fits the search has to be
  selective, so a method that only looks good at one budget cannot hide. Both
  conditions use the same data, library and referee; only the budget differs.
* **The quantum arm's surrogate costs money too.** Its quadratic approximation
  needs ``1 + m + m(m-1)/2 = 56`` exact fits, which come out of its own budget.
  It is then scored by the *real* objective, so the approximation is measured
  rather than assumed: the record keeps both the surrogate's minimiser and the
  selection the arm actually scored.
* **An abstention is not a loss.** An arm that cannot run says so with a reason,
  and its row is not counted as a failure of the search idea.

Example::

    python scripts/measure_qed_recovery.py
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apedp_dyn.qed.arms import ARMS  # noqa: E402
from apedp_dyn.qed.records import build_condition, build_record, write_record  # noqa: E402
from apedp_dyn.qed.tasks import damped_oscillator_task  # noqa: E402

DEFAULT_OUTPUT = ROOT / "benchmarks" / "equation_discovery" / "qed_001_recovery.json"
BUDGETS = (256, 64)
MODULES = (
    "apedp_dyn/qed/__init__.py",
    "apedp_dyn/qed/library.py",
    "apedp_dyn/qed/objective.py",
    "apedp_dyn/qed/tasks.py",
    "apedp_dyn/qed/arms.py",
    "apedp_dyn/qed/qaoa.py",
    "apedp_dyn/qed/referee.py",
    "apedp_dyn/qed/records.py",
    "apedp_dyn/qed/proposer.py",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=11, help="data-generation seed")
    parser.add_argument("--noise", type=float, default=0.01, help="observation noise, as a fraction")
    parser.add_argument("--penalty", type=float, default=0.002, help="parsimony penalty per term")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13], help="per-arm seeds")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = parser.parse_args(argv)

    conditions = []
    for budget in args.budgets:
        task = damped_oscillator_task(
            seed=args.seed, noise_level=args.noise, penalty=args.penalty, budget=budget
        )
        conditions.append(build_condition(task, arms=args.arms, seeds=args.seeds))

    record = build_record(
        conditions,
        root=ROOT,
        script="scripts/measure_qed_recovery.py",
        modules=MODULES,
    )
    record["reproducibility"]["python"] = platform.python_version()
    record["reproducibility"]["platform"] = platform.platform()
    record["reproducibility"]["command"] = (
        f"python scripts/measure_qed_recovery.py --seed {args.seed} --noise {args.noise} "
        f"--penalty {args.penalty} --budgets {' '.join(map(str, args.budgets))} "
        f"--seeds {' '.join(map(str, args.seeds))}"
    )
    write_record(record, args.output)

    for condition in conditions:
        task_block = condition["task"]
        oracle = condition["oracle"]
        print(f"task: {task_block['name']}, library {len(task_block['library'])} terms, "
              f"space {oracle['space']}, budget {condition['budget']} fits per arm")
        print(f"oracle: best {oracle['best']['support']} "
              f"({'the truth' if oracle['agrees_with_truth'] else 'NOT the truth'}), "
              f"margin {oracle['margin']:.4%}, runner-up {oracle['runner_up']['support']}")
        for row in condition["arms"]:
            best = row["best_objective"]
            print(f"  {row['arm']:<20} statuses {row['statuses']} "
                  f"recovered {row['recovered_runs']}/{len(row['runs'])} "
                  f"best J {best if best is None else f'{best:.6f}'} "
                  f"fits {row['evaluations_per_run']}")
        hits = condition["comparison"]["arms_matching_the_oracle"]
        print(f"  arms matching the oracle: {hits if hits else 'none'}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
