"""Development study: weak form against the derivative route on the Lorenz system.

    python scripts/study_weak_form_lorenz.py

NOT a preregistered result. The world is the one PREREG-LORENZ-PUB-V1 froze
(sigma 10, rho 28, beta 8/3, 0.005 sampling over [0, 10], start (1, 1, 1), noise
as a fraction of each state's standard deviation) and the library is its twenty
polynomial terms. Both arms use the same sparse regression and threshold; only
how the derivative enters differs:

- derivative: Savitzky-Golay (21, 3) then ``np.gradient``, as that benchmark does;
- weak: ``apedp_dyn.ode.weak_form`` on the raw noisy states.

The window (201 samples), threshold (0.5) and test-function order were chosen
once. Two things were tried afterwards and are not in the table: a learned,
scale-free threshold (my implementation of Messenger & Bortz's MSTLS), which
left the weak arm with about ten spurious terms and was dropped; and PySINDy
2.1.0's own ``WeakPDELibrary`` on the same data as a reference (exact terms in 5
of 10 seeds at 1 % noise, 1 of 10 at 5 %, 0 of 10 at 10 %), which is no better
than this module, so the limit is the setting and not the implementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_dyn.ode import weak_form  # noqa: E402

OUT = ROOT / "benchmarks" / "weak_form_lorenz_dev_study.json"
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT, T_END = 0.005, 10.0
NOISE = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
SEEDS = tuple(range(101, 121))
THRESHOLD = 0.5
TERMS = ("1", "x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz",
         "x3", "y3", "z3", "xyz", "x2y", "x2z", "xy2", "xz2", "y2z", "yz2")
TRUTH = {("x", 0): -SIGMA, ("y", 0): SIGMA, ("x", 1): RHO, ("y", 1): -1.0, ("xz", 1): -1.0,
         ("xy", 2): 1.0, ("z", 2): -BETA}


def library(S: np.ndarray) -> np.ndarray:
    x, y, z = S.T
    return np.column_stack([np.ones(len(S)), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z,
                            x ** 3, y ** 3, z ** 3, x * y * z, x * x * y, x * x * z, x * y * y,
                            x * z * z, y * y * z, y * z * z])


def score(coef: np.ndarray) -> dict[str, float | bool]:
    truth = np.zeros_like(coef)
    for (term, eq), value in TRUTH.items():
        truth[TERMS.index(term), eq] = value
    support = bool(np.array_equal(coef != 0, truth != 0))
    on = truth != 0
    return {"exact_support": support, "spurious_terms": int(((coef != 0) & ~on).sum()),
            "missing_terms": int(((coef == 0) & on).sum()),
            "max_rel_coef_error": float(np.max(np.abs(coef[on] - truth[on]) / np.abs(truth[on])))}


def main() -> int:
    t = np.arange(0.0, T_END + 1e-12, DT)
    clean = solve_ivp(lambda _t, s: [SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]],
                      (0.0, T_END), [1.0, 1.0, 1.0], t_eval=t, rtol=1e-10, atol=1e-10).y.T
    runs = []
    for noise in NOISE:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noisy = clean + rng.normal(0.0, 1.0, clean.shape) * noise * clean.std(axis=0)
            smooth = savgol_filter(noisy, 21, 3, axis=0)
            strong = weak_form.stlsq(library(smooth), np.gradient(smooth, DT, axis=0), THRESHOLD)
            weak = weak_form.identify(t, noisy, library, THRESHOLD)
            runs.append({"noise": noise, "seed": seed, "derivative": score(strong), "weak": score(weak)})
    summary = {}
    for noise in NOISE:
        sel = [r for r in runs if r["noise"] == noise]
        summary[str(noise)] = {arm: {
            "exact_support": sum(r[arm]["exact_support"] for r in sel), "of": len(sel),
            "median_max_rel_coef_error": float(np.median([r[arm]["max_rel_coef_error"] for r in sel])),
            "mean_spurious_terms": float(np.mean([r[arm]["spurious_terms"] for r in sel])),
        } for arm in ("derivative", "weak")}
    OUT.write_text(json.dumps({"status": "development study, not preregistered", "threshold": THRESHOLD,
                               "summary": summary, "runs": runs}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for noise, cell in summary.items():
        print(noise, {arm: (v["exact_support"], round(v["median_max_rel_coef_error"], 4), round(v["mean_spurious_terms"], 1))
                      for arm, v in cell.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
