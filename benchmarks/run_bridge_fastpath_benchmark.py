"""Before/after latency benchmark for the Rust kernel boundary.

Protocol (fixed before any run; identical driver for both stacks):

- **Stack:** whatever `import apedp.rust_bridge` resolves in the running
  interpreter. The "before" stack = `apedp_kernel` built from git HEAD
  (per-call owned `Vec<f64>` copies + `kernel.compile` at every crossing)
  plus the HEAD bridge (no compile cache, no lambdify cache). The "after"
  stack = the zero-copy-value kernel + cached bridge. The driver itself is
  byte-identical for both: it only calls the public bridge API.
- **Machine-fairness:** both stacks run sequentially on the same machine,
  same dataset (600 rows x 4 columns, float64), same iteration counts,
  second pass reported (first pass discarded as warmup).
- **Candidate-evaluation call** = one `evaluate_expression` crossing
  (compile-or-cache + evaluate + metadata transfer back), the unit of the
  referee/generation loop. Batch round = one `score_candidates_batch` over
  15 candidates (the per-generation reload unit).

Output: JSON on stdout with µs/call medians over the second pass.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available  # noqa: E402

VARS = ["x0", "x1", "x2", "x3"]
EXPR = "x0 + 2*x1 + sin(x2)*x3 + 0.5*x0**2 - 1e-3*x3"

N_ROWS = 600
N_REPEATED = 4000
N_UNIQUE = 300
N_BATCH_ROUNDS = 80
BATCH_SIZE = 15
N_METRICS = 3000
N_FALLBACK = 120


def _pass(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6  # µs/call


def main() -> None:
    np.random.seed(0)
    xs = [np.random.uniform(-3.0, 3.0, N_ROWS) for _ in VARS]
    target = xs[0] + 2.0 * xs[1] + np.sin(xs[2]) * xs[3]

    bridge = RustKernelBridge()
    bridge.evaluate_expression(EXPR, VARS, xs)  # warmup

    unique_exprs = [
        f"x0 + {i}*x1 + {i}*x2**2 + {i}.5*x3" for i in range(N_UNIQUE)
    ]
    batch_cands = [f"x0 + {i}*x1 + {i}*x2**2 + {i}.5*x3" for i in range(BATCH_SIZE)]

    print(f"# stack: rust_available={is_rust_kernel_available()} "
          f"kernel_path_probe={bridge.evaluate_expression(EXPR, VARS, xs) is not None}")

    results: dict[str, float] = {}

    # 1. Repeated identical candidate evaluation (compile cache + zero-copy).
    fn = lambda: bridge.evaluate_expression(EXPR, VARS, xs)
    _pass(fn, N_REPEATED)
    results["evaluate_expression_us_per_call"] = round(_pass(fn, N_REPEATED), 2)

    # 2. Unique-expression compilation (raw kernel compile+eval cost).
    unique = [f"x0 + {i} * x1 * x2 * x3" for i in range(N_UNIQUE)]
    idx = {"k": 0}

    def unique_fn() -> None:
        e = unique[idx["k"] % N_UNIQUE]
        idx["k"] += 1
        bridge.evaluate_expression(e, VARS, xs)

    _pass(unique_fn, N_UNIQUE)
    idx["k"] = 0
    results["evaluate_expression_unique_us_per_call"] = round(
        _pass(unique_fn, N_UNIQUE), 2)

    # 3. Batch scoring rounds (candidate reload per generation).
    fn = lambda: bridge.score_candidates_batch(
        batch_cands, VARS, xs, target)
    _pass(fn, N_BATCH_ROUNDS)
    results["score_candidates_batch_us_per_round"] = round(
        _pass(fn, N_BATCH_ROUNDS), 2)
    results["score_candidates_batch_us_per_candidate"] = round(
        results["score_candidates_batch_us_per_round"] / BATCH_SIZE, 2)

    # 4. Metrics.
    fn = lambda: bridge.compute_metrics(target, target + 0.1 * xs[0])
    _pass(fn, N_METRICS)
    results["compute_metrics_us_per_call"] = round(_pass(fn, N_METRICS), 2)

    # 5. Python reference engine repeated (lambdify cache).
    py_bridge = RustKernelBridge(force_fallback=True)
    fn = lambda: py_bridge._fallback_eval(EXPR, VARS, xs)  # noqa: SLF001
    _pass(fn, N_FALLBACK)
    results["fallback_eval_us_per_call"] = round(_pass(fn, N_FALLBACK), 2)

    print(json.dumps({
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        **results,
    }, indent=2))


if __name__ == "__main__":
    main()
