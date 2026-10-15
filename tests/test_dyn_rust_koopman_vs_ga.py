"""Koopman spectral discovery vs native GA under matched wall-clock.

Task: harmonic-oscillator trajectory x(t) = cos(2t), uniform dt.
- RIME-Koopman (Rust): fit delay-embedded linear spectrum on train, roll out
  the recurrence on the untouched holdout.
- Native GA (Rust kernel evolutionary search): discover an expression on
  train, evaluate the train-selected best on the same holdout.

Both methods see identical train/holdout splits; wall times are recorded.
gplearn/PySR are not installed in this environment, so the GP side is the
native genetic-programming kernel path.
"""
from __future__ import annotations

import json
import time

import numpy as np

import apedp_kernel as ak
from apedp.evolutionary_search import SearchConfig, evolve
from apedp.rust_bridge import RustKernelBridge, is_rust_kernel_available


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    var = float(np.var(yt))
    if var < 1e-15:
        return 1.0 if float(np.mean((yt - yp) ** 2)) < 1e-12 else 0.0
    return 1.0 - float(np.mean((yt - yp) ** 2)) / var


def _koopman_rollout(model: dict, z0: float, z1: float, steps: int) -> np.ndarray:
    a1, a2, b, mean = model["a1"], model["a2"], model["intercept"], model["mean"]
    p0, p1 = z0, z1
    out = np.empty(steps)
    for i in range(steps):
        v = a1 * p1 + a2 * p0 + b
        out[i] = v + mean
        p0, p1 = p1, v
    return out


def test_koopman_harmonic_frequency_and_holdout_rollout():
    dt = 0.05
    t_tr = np.arange(0.0, 4.0, dt)
    t_ho = np.arange(4.0, 8.0, dt)
    x_tr = np.cos(2.0 * t_tr)
    x_ho = np.cos(2.0 * t_ho)

    t0 = time.perf_counter()
    model = json.loads(ak.rime_koopman_discover_json(
        json.dumps(x_tr.tolist()), dt, json.dumps({})))
    pred_ho = _koopman_rollout(
        model, x_tr[-2] - model["mean"], x_tr[-1] - model["mean"], len(t_ho))
    koopman_ms = (time.perf_counter() - t0) * 1000.0

    assert abs(model["frequency"] - 2.0) < 0.02
    assert abs(model["spectral_radius"] - 1.0) < 0.02
    r2 = _r2(x_ho, pred_ho)
    assert r2 > 0.999, f"Koopman holdout R2 {r2}"
    print(f"\nKoopman: freq={model['frequency']:.4f} R2_ho={r2:.6f} wall={koopman_ms:.2f}ms")


def test_koopman_vs_native_ga_matched_split():
    assert is_rust_kernel_available(), "native kernel required for this comparison"
    dt = 0.05
    t_tr = np.arange(0.0, 4.0, dt)
    t_ho = np.arange(4.0, 8.0, dt)
    x_tr = np.cos(2.0 * t_tr)
    x_ho = np.cos(2.0 * t_ho)

    t0 = time.perf_counter()
    model = json.loads(ak.rime_koopman_discover_json(
        json.dumps(x_tr.tolist()), dt, json.dumps({})))
    pred_k = _koopman_rollout(
        model, x_tr[-2] - model["mean"], x_tr[-1] - model["mean"], len(t_ho))
    koopman_ms = (time.perf_counter() - t0) * 1000.0
    koopman_r2 = _r2(x_ho, pred_k)

    t0 = time.perf_counter()
    ga = evolve({"t": t_tr}, x_tr, config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    ga_ms = (time.perf_counter() - t0) * 1000.0
    assert ga.best_expression, "GA returned no expression"
    bridge = RustKernelBridge()
    pred_ga = bridge.evaluate_expression(ga.best_expression, ["t"], [t_ho])
    ga_r2 = _r2(x_ho, pred_ga)

    print(f"\nKoopman: R2_ho={koopman_r2:.6f} wall={koopman_ms:.2f}ms "
          f"freq={model['frequency']:.4f}")
    print(f"GA: best={ga.best_expression} R2_ho={ga_r2:.6f} "
          f"wall={ga_ms:.2f}ms evals={ga.n_evaluations}")

    assert koopman_r2 > 0.999
    assert ga_r2 > 0.99


def test_koopman_damped_oscillator_spectrum():
    dt = 0.02
    t = np.arange(0.0, 8.0, dt)
    x = np.exp(-0.5 * t) * np.cos(3.0 * t)
    model = json.loads(ak.rime_koopman_discover_json(
        json.dumps(x.tolist()), dt, json.dumps({})))
    assert abs(model["frequency"] - 3.0) < 0.05
    assert abs(model["decay_rate"] + 0.5) < 0.05
