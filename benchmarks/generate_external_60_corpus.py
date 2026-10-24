"""Frozen external-benchmark corpus generator (PREREG-APEDP-EXTBENCH-60-2026-09-26).

Generates benchmark_corpus_external_60/ with 60 problems across 7 blocks:
  P (10): polynomials / power laws (on-grid + off-grid exponents)
  E (10): exponential decays (on/off-grid rates) + logarithmic boundary layers
  R (8):  saturation rationals + 808-class off-grid rational x5 noise seeds
  O (8):  damped / undamped oscillations (on/off-grid frequencies)
  C (8):  composite rational x exponential envelopes
  X (8):  extrapolation re-splits of P/E/R configs with extended far range
  A (8):  abstention adversaries (out-of-grammar, underspecified, unidentifiable)

Layout mirrors the existing tournament corpus so harness loaders read it:
  public/{pid}/train.csv, validation.csv, near_ood.csv, far_ood.csv, problem_spec.json
  hidden/{pid}/ground_truth.json
  manifest.json, generation_seed.txt, checksums.sha256

Ground truth NEVER appears outside hidden/. Noise/sizes cycle per protocol:
sigma_frac in {0.0, 0.015, 0.05}, N in {20, 40, 80} (A-block underspecified: N<=8).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).resolve().parent.parent / "benchmark_corpus_external_60"
GENERATION_SEED = 20260926

SIGMAS = [0.0, 0.015, 0.05]
SIZES = [20, 40, 80]

# Ground-truth formula strings (parseable expressions, exactly matching the
# generating lambdas below). The initial revision stored short names here,
# which are not parseable formulas and broke canonical family comparison.
FORMULAS = {
    "poly-deg1": "2.0*x + 1.0",
    "poly-deg2": "2.0*x**2 + 1.0",
    "poly-deg3": "0.5*x**3 - 2.0*x + 1.0",
    "poly-deg4": "0.1*x**4 + 0.5*x**2 + 1.0",
    "power-1.5": "2.5*x**1.5",
    "power-2.0": "1.5*x**2.0",
    "power-0.5": "3.0*x**0.5",
    "power-1.7": "3.0*x**1.7",
    "power-1.3": "2.0*x**1.3",
    "power-2.6": "0.8*x**2.6",
    "exp-0.5": "4.0*exp(-0.5*x)",
    "exp-1.0": "2.0*exp(-1.0*x)",
    "exp-0.37": "4.0*exp(-0.37*x)",
    "exp-0.73": "3.0*exp(-0.73*x)",
    "exp-1.3": "1.5*exp(-1.3*x)",
    "exp-0.2": "5.0*exp(-0.2*x)",
    "log-1": "3.0*log(1 + 2.0*x)",
    "log-2": "2.0*log(1 + 0.5*x)",
    "log-3": "1.0*log(1 + 3.0*x)",
    "log-pow": "2.0*x**1.5*log(1 + 1.0*x)",
    "sat-1": "3.0*x**1.5/(1.0 + 1.2*x**1.5)",
    "sat-2": "5.0*x/(2.0 + x)",
    "sat-3": "2.0*x**2.0/(1.0 + 0.5*x**2.0)",
    "damped-2.8": "2.0*exp(-0.2*x)*cos(2.8*x)",
    "damped-2.0": "1.5*exp(-0.3*x)*cos(2.0*x)",
    "damped-3.5": "3.0*exp(-0.1*x)*cos(3.5*x)",
    "damped-2.7": "2.0*exp(-0.2*x)*cos(2.7*x)",
    "damped-1.7": "2.5*exp(-0.4*x)*cos(1.7*x)",
    "damped-4.2": "1.0*exp(-0.25*x)*cos(4.2*x)",
    "undamped-2.0": "2.0*cos(2.0*x)",
    "undamped-3.0": "1.5*cos(3.0*x)",
    "env-1": "2.45*x**1.5/(1 + 0.85*x**2)*exp(-0.3*x)",
    "env-2": "1.8*x**1.0/(1 + 0.5*x**1.5)*exp(-0.2*x)",
    "env-3": "3.0*x**2.0/(1 + 1.0*x**2.0)*exp(-0.4*x)",
    "env-4": "2.0*x**0.75/(1 + 0.6*x**1.25)*exp(-0.15*x)",
    "expenv-1": "2.0*x**1.5*exp(-0.3*x)",
    "expenv-2": "1.0*x**1.0*exp(-0.5*x**1.5)",
    "expenv-3": "3.0*x**2.0*exp(-0.2*x**0.5)",
    "expenv-4": "1.5*x**0.5*exp(-0.4*x**2.0)",
    "x-poly2": "2.0*x**2 + 1.0",
    "x-pow15": "2.5*x**1.5",
    "x-exp05": "4.0*exp(-0.5*x)",
    "x-log1": "3.0*log(1 + 2.0*x)",
    "x-sat1": "3.0*x**1.5/(1.0 + 1.2*x**1.5)",
    "x-damp28": "2.0*exp(-0.2*x)*cos(2.8*x)",
    "x-pow17": "3.0*x**1.7",
    "x-exp037": "4.0*exp(-0.37*x)",
    "adv-gaussian": "3.0*exp(-(x - 2.0)**2/(2*0.7**2))",
    "adv-trig-identity": "sin(x)**2+cos(x)**2",
    "adv-logistic": "10.0/(1 + 9.0*exp(-1.2*x))",
}


def _write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{v:.12e}" for v in row] for row in rows])


def _splits(rng: np.random.Generator, x: np.ndarray, y_clean: np.ndarray,
            sigma_frac: float, far_extend: float = 1.0):
    """Interleaved train/val split + adjacent near-OOD + extended far-OOD."""
    n = len(x)
    noise = rng.normal(0.0, sigma_frac * max(float(np.std(y_clean)), 1e-12), n)
    y = y_clean + noise
    perm = rng.permutation(n)
    ntr = max(4, int(0.75 * n))
    tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    x_near = np.linspace(hi, hi + 0.3 * span, 30)
    x_far = np.linspace(hi + 0.5 * span * far_extend, hi + 1.5 * span * far_extend, 30)
    return (x[tr], y[tr]), (x[va], y[va]), x_near, x_far


def _emit(problems: list, pid: str, block: str, name: str, family: str,
          formula: str, params: dict, oracle: str, expected: str,
          x_tr, y_tr, x_va, y_va, x_near, gt_near_fn, x_far, gt_far_fn):
    problems.append({
        "pid": pid, "block": block, "name": name, "family": family,
        "formula": formula, "params": params, "oracle": oracle, "expected": expected,
        "train": (np.asarray(x_tr), np.asarray(y_tr)),
        "val": (np.asarray(x_va), np.asarray(y_va)),
        "near": (np.asarray(x_near), np.asarray(gt_near_fn(np.asarray(x_near)))),
        "far": (np.asarray(x_far), np.asarray(gt_far_fn(np.asarray(x_far)))),
    })


def build_problems(rng: np.random.Generator) -> list[dict]:
    ps: list[dict] = []
    k = 0

    def nxt(block: str) -> tuple[str, float, int]:
        nonlocal k
        k += 1
        i = k
        return f"{block}{i:02d}", SIGMAS[i % 3], SIZES[i % 3]

    # ---- Block P: polynomials / power laws ----
    poly_cfgs = [
        ("poly-deg1", lambda x: 2.0 * x + 1.0, (0.0, 5.0), {"a": 2.0, "b": 1.0}),
        ("poly-deg2", lambda x: 2.0 * x**2 + 1.0, (-2.0, 2.0), {"a": 2.0, "b": 1.0}),
        ("poly-deg3", lambda x: 0.5 * x**3 - 2.0 * x + 1.0, (-2.0, 2.0), {"a": 0.5, "b": -2.0, "c": 1.0}),
        ("poly-deg4", lambda x: 0.1 * x**4 + 0.5 * x**2 + 1.0, (-2.0, 2.0), {"a": 0.1, "b": 0.5, "c": 1.0}),
        ("power-1.5", lambda x: 2.5 * x**1.5, (0.5, 4.0), {"A": 2.5, "p": 1.5}),
        ("power-2.0", lambda x: 1.5 * x**2.0, (0.5, 4.0), {"A": 1.5, "p": 2.0}),
        ("power-0.5", lambda x: 3.0 * x**0.5, (0.5, 4.0), {"A": 3.0, "p": 0.5}),
        ("power-1.7", lambda x: 3.0 * x**1.7, (0.5, 4.0), {"A": 3.0, "p": 1.7}),
        ("power-1.3", lambda x: 2.0 * x**1.3, (0.5, 4.0), {"A": 2.0, "p": 1.3}),
        ("power-2.6", lambda x: 0.8 * x**2.6, (0.5, 4.0), {"A": 0.8, "p": 2.6}),
    ]
    for name, fn, (a, b), prm in poly_cfgs:
        pid, sig, n = nxt("P")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig)
        _emit(ps, pid, "P", name, "power_law" if "power" in name else "polynomial",
              FORMULAS[name], prm, "C1", "RECOVER",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)

    # ---- Block E: exponentials / logs ----
    exp_cfgs = [
        ("exp-0.5", lambda x: 4.0 * np.exp(-0.5 * x), (0.0, 6.0), {"A": 4.0, "k": 0.5}),
        ("exp-1.0", lambda x: 2.0 * np.exp(-1.0 * x), (0.0, 6.0), {"A": 2.0, "k": 1.0}),
        ("exp-0.37", lambda x: 4.0 * np.exp(-0.37 * x), (0.0, 6.0), {"A": 4.0, "k": 0.37}),
        ("exp-0.73", lambda x: 3.0 * np.exp(-0.73 * x), (0.0, 6.0), {"A": 3.0, "k": 0.73}),
        ("exp-1.3", lambda x: 1.5 * np.exp(-1.3 * x), (0.0, 6.0), {"A": 1.5, "k": 1.3}),
        ("exp-0.2", lambda x: 5.0 * np.exp(-0.2 * x), (0.0, 6.0), {"A": 5.0, "k": 0.2}),
        ("log-1", lambda x: 3.0 * np.log1p(2.0 * x), (0.1, 5.0), {"A": 3.0, "B": 2.0}),
        ("log-2", lambda x: 2.0 * np.log1p(0.5 * x), (0.1, 5.0), {"A": 2.0, "B": 0.5}),
        ("log-3", lambda x: 1.0 * np.log1p(3.0 * x), (0.1, 5.0), {"A": 1.0, "B": 3.0}),
        ("log-pow", lambda x: 2.0 * x**1.5 * np.log1p(1.0 * x), (0.5, 4.0), {"A": 2.0, "p": 1.5, "B": 1.0}),
    ]
    for name, fn, (a, b), prm in exp_cfgs:
        pid, sig, n = nxt("E")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig)
        fam = "logarithmic_boundary_layer" if name.startswith("log") else "exponential_decay"
        _emit(ps, pid, "E", name, fam, FORMULAS[name], prm, "C1", "RECOVER",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)

    # ---- Block R: saturation rationals + 808-class x5 seeds ----
    rat_cfgs = [
        ("sat-1", lambda x: 3.0 * x**1.5 / (1.0 + 1.2 * x**1.5), (0.5, 4.0), {"A": 3.0, "p": 1.5, "B": 1.2, "q": 1.5}),
        ("sat-2", lambda x: 5.0 * x / (2.0 + x), (0.0, 10.0), {"Vmax": 5.0, "Km": 2.0}),
        ("sat-3", lambda x: 2.0 * x**2.0 / (1.0 + 0.5 * x**2.0), (0.5, 4.0), {"A": 2.0, "p": 2.0, "B": 0.5, "q": 2.0}),
    ]
    for name, fn, (a, b), prm in rat_cfgs:
        pid, sig, n = nxt("R")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig)
        _emit(ps, pid, "R", name, "rational_transition", FORMULAS[name], prm, "C1", "RECOVER",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)
    fn808 = lambda x: 2.0 * x**1.3 / (1.0 + 0.7 * x**2.1)
    for s in range(5):
        pid, _, n = nxt("R")
        x = np.linspace(0.5, 4.5, 40)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn808(x), 0.015)
        _emit(ps, pid, "R", f"rational-808-seed{s}", "rational_transition",
              "2.0*x**1.3/(1.0+0.7*x**2.1)", {"A": 2.0, "p": 1.3, "B": 0.7, "q": 2.1},
              "C1/C2", "STABILITY_PROBE", x_tr, y_tr, x_va, y_va, x_near, fn808, x_far, fn808)

    # ---- Block O: oscillations ----
    osc_cfgs = [
        ("damped-2.8", lambda x: 2.0 * np.exp(-0.2 * x) * np.cos(2.8 * x), (0.0, 6.0), {"A": 2.0, "k": 0.2, "w": 2.8}),
        ("damped-2.0", lambda x: 1.5 * np.exp(-0.3 * x) * np.cos(2.0 * x), (0.0, 6.0), {"A": 1.5, "k": 0.3, "w": 2.0}),
        ("damped-3.5", lambda x: 3.0 * np.exp(-0.1 * x) * np.cos(3.5 * x), (0.0, 6.0), {"A": 3.0, "k": 0.1, "w": 3.5}),
        ("damped-2.7", lambda x: 2.0 * np.exp(-0.2 * x) * np.cos(2.7 * x), (0.0, 6.0), {"A": 2.0, "k": 0.2, "w": 2.7}),
        ("damped-1.7", lambda x: 2.5 * np.exp(-0.4 * x) * np.cos(1.7 * x), (0.0, 6.0), {"A": 2.5, "k": 0.4, "w": 1.7}),
        ("damped-4.2", lambda x: 1.0 * np.exp(-0.25 * x) * np.cos(4.2 * x), (0.0, 6.0), {"A": 1.0, "k": 0.25, "w": 4.2}),
        ("undamped-2.0", lambda x: 2.0 * np.cos(2.0 * x), (0.0, 6.0), {"A": 2.0, "w": 2.0}),
        ("undamped-3.0", lambda x: 1.5 * np.cos(3.0 * x), (0.0, 6.0), {"A": 1.5, "w": 3.0}),
    ]
    for name, fn, (a, b), prm in osc_cfgs:
        pid, sig, n = nxt("O")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig)
        fam = "undamped_oscillation" if name.startswith("undamped") else "damped_oscillation"
        _emit(ps, pid, "O", name, fam, FORMULAS[name], prm, "C1", "RECOVER",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)

    # ---- Block C: composite envelopes ----
    env_cfgs = [
        ("env-1", lambda x: 2.45 * x**1.5 / (1 + 0.85 * x**2) * np.exp(-0.3 * x), (0.5, 4.0), {"A": 2.45, "p": 1.5, "B": 0.85, "q": 2.0, "C": 0.3}),
        ("env-2", lambda x: 1.8 * x**1.0 / (1 + 0.5 * x**1.5) * np.exp(-0.2 * x), (0.5, 4.0), {"A": 1.8, "p": 1.0, "B": 0.5, "q": 1.5, "C": 0.2}),
        ("env-3", lambda x: 3.0 * x**2.0 / (1 + 1.0 * x**2.0) * np.exp(-0.4 * x), (0.5, 4.0), {"A": 3.0, "p": 2.0, "B": 1.0, "q": 2.0, "C": 0.4}),
        ("env-4", lambda x: 2.0 * x**0.75 / (1 + 0.6 * x**1.25) * np.exp(-0.15 * x), (0.5, 4.0), {"A": 2.0, "p": 0.75, "B": 0.6, "q": 1.25, "C": 0.15}),
        ("expenv-1", lambda x: 2.0 * x**1.5 * np.exp(-0.3 * x), (0.5, 4.0), {"A": 2.0, "p": 1.5, "C": 0.3}),
        ("expenv-2", lambda x: 1.0 * x**1.0 * np.exp(-0.5 * x**1.5), (0.5, 4.0), {"A": 1.0, "p": 1.0, "C": 0.5, "r": 1.5}),
        ("expenv-3", lambda x: 3.0 * x**2.0 * np.exp(-0.2 * x**0.5), (0.5, 4.0), {"A": 3.0, "p": 2.0, "C": 0.2, "r": 0.5}),
        ("expenv-4", lambda x: 1.5 * x**0.5 * np.exp(-0.4 * x**2.0), (0.5, 4.0), {"A": 1.5, "p": 0.5, "C": 0.4, "r": 2.0}),
    ]
    for name, fn, (a, b), prm in env_cfgs:
        pid, sig, n = nxt("C")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig)
        fam = "rational_exponential_envelope" if name.startswith("env-") else "exponential_envelope"
        _emit(ps, pid, "C", name, fam, FORMULAS[name], prm, "C1", "RECOVER",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)

    # ---- Block X: extrapolation re-splits (extended far range) ----
    xcfgs = [
        ("x-poly2", lambda x: 2.0 * x**2 + 1.0, (-2.0, 2.0), "polynomial", {"a": 2.0, "b": 1.0}),
        ("x-pow15", lambda x: 2.5 * x**1.5, (0.5, 4.0), "power_law", {"A": 2.5, "p": 1.5}),
        ("x-exp05", lambda x: 4.0 * np.exp(-0.5 * x), (0.0, 6.0), "exponential_decay", {"A": 4.0, "k": 0.5}),
        ("x-log1", lambda x: 3.0 * np.log1p(2.0 * x), (0.1, 5.0), "logarithmic_boundary_layer", {"A": 3.0, "B": 2.0}),
        ("x-sat1", lambda x: 3.0 * x**1.5 / (1.0 + 1.2 * x**1.5), (0.5, 4.0), "rational_transition", {"A": 3.0, "p": 1.5, "B": 1.2, "q": 1.5}),
        ("x-damp28", lambda x: 2.0 * np.exp(-0.2 * x) * np.cos(2.8 * x), (0.0, 6.0), "damped_oscillation", {"A": 2.0, "k": 0.2, "w": 2.8}),
        ("x-pow17", lambda x: 3.0 * x**1.7, (0.5, 4.0), "power_law", {"A": 3.0, "p": 1.7}),
        ("x-exp037", lambda x: 4.0 * np.exp(-0.37 * x), (0.0, 6.0), "exponential_decay", {"A": 4.0, "k": 0.37}),
    ]
    for name, fn, (a, b), fam, prm in xcfgs:
        pid, sig, n = nxt("X")
        x = np.linspace(a, b, n)
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fn(x), sig, far_extend=2.0)
        _emit(ps, pid, "X", name, fam, FORMULAS[name], prm, "C1", "EXTRAPOLATE",
              x_tr, y_tr, x_va, y_va, x_near, fn, x_far, fn)

    # ---- Block A: abstention adversaries ----
    # A1-A2: out-of-grammar (Gaussian peak with translation; sine-squared identity data)
    pid, _, _ = nxt("A")
    x = np.linspace(0, 4, 40)
    fg = lambda x: 3.0 * np.exp(-((x - 2.0) ** 2) / (2 * 0.7**2))
    (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, x, fg(x), 0.01)
    _emit(ps, pid, "A", "adv-gaussian", "translated_peak", FORMULAS["adv-gaussian"],
          {"A": 3.0, "mu": 2.0, "sigma": 0.7}, "C5", "ABSTAIN",
          x_tr, y_tr, x_va, y_va, x_near, fg, x_far, fg)
    pid, _, _ = nxt("A")
    xs2 = np.linspace(0, 2 * np.pi, 40)
    fs = lambda x: np.sin(x) ** 2 + np.cos(x) ** 2
    (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, xs2, fs(xs2), 0.001)
    _emit(ps, pid, "A", "adv-trig-identity", "constant_identity", FORMULAS["adv-trig-identity"],
          {}, "C4", "CERTIFY", x_tr, y_tr, x_va, y_va, x_near, fs, x_far, fs)
    # A3-A5: underspecified (N<=8)
    for nn in [5, 6, 8]:
        pid, _, _ = nxt("A")
        xu = np.linspace(0.5, 4.0, nn)
        fu = lambda x: 2.5 * x**1.5
        (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, xu, fu(xu), 0.01)
        _emit(ps, pid, "A", f"adv-sparse-{nn}", "power_law", "2.5*x**1.5",
              {"A": 2.5, "p": 1.5}, "C3", "ABSTAIN",
              x_tr, y_tr, x_va, y_va, x_near, fu, x_far, fu)
    # A6-A7: unidentifiable at noise floor (narrow exponential; high-noise rational)
    pid, _, _ = nxt("A")
    xn = np.linspace(0, 0.5, 30)
    fe = lambda x: np.exp(0.2 * x)
    (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, xn, fe(xn), 0.005)
    _emit(ps, pid, "A", "adv-narrow-exp", "exponential_decay", "exp(0.2*x)",
          {"k": -0.2}, "C2", "AMBIGUOUS", x_tr, y_tr, x_va, y_va, x_near, fe, x_far, fe)
    pid, _, _ = nxt("A")
    xr = np.linspace(0.5, 4.5, 40)
    fr = lambda x: 2.0 * x**1.3 / (1.0 + 0.7 * x**2.1)
    (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, xr, fr(xr), 0.05)
    _emit(ps, pid, "A", "adv-noisy-rational", "rational_transition",
          "2.0*x**1.3/(1.0+0.7*x**2.1)", {"A": 2.0, "p": 1.3, "B": 0.7, "q": 2.1},
          "C2", "AMBIGUOUS", x_tr, y_tr, x_va, y_va, x_near, fr, x_far, fr)
    # A8: logistic saturation (grammar gap probe)
    pid, _, _ = nxt("A")
    xl = np.linspace(0, 6, 40)
    fl = lambda x: 10.0 / (1 + 9.0 * np.exp(-1.2 * x))
    (x_tr, y_tr), (x_va, y_va), x_near, x_far = _splits(rng, xl, fl(xl), 0.01)
    _emit(ps, pid, "A", "adv-logistic", "saturating_exponential", FORMULAS["adv-logistic"],
          {"K": 10.0, "A": 9.0, "r": 1.2}, "C5", "GRAMMAR_GAP",
          x_tr, y_tr, x_va, y_va, x_near, fl, x_far, fl)

    assert len(ps) == 60, f"expected 60 problems, got {len(ps)}"
    return ps


def write_corpus(problems: list[dict], root: Path) -> None:
    manifest = {"prereg_id": "PREREG-APEDP-EXTBENCH-60-2026-09-26",
                "generation_seed": GENERATION_SEED,
                "revisions": ["r2: formula strings fixed (data-identical regen)"],
                "problems": []}
    for p in problems:
        pub = root / "public" / p["pid"]
        hid = root / "hidden" / p["pid"]
        tr_x, tr_y = p["train"]
        va_x, va_y = p["val"]
        ne_x, ne_y = p["near"]
        fa_x, fa_y = p["far"]
        _write_csv(pub / "train.csv", ["x", "y"], np.column_stack([tr_x, tr_y]))
        _write_csv(pub / "validation.csv", ["x", "y"], np.column_stack([va_x, va_y]))
        _write_csv(pub / "near_ood.csv", ["x", "y"], np.column_stack([ne_x, ne_y]))
        _write_csv(pub / "far_ood.csv", ["x", "y"], np.column_stack([fa_x, fa_y]))
        with open(pub / "problem_spec.json", "w", encoding="utf-8") as f:
            json.dump({"problem_id": p["pid"], "block": p["block"], "name": p["name"],
                       "input_variables": ["x"], "target_variable": "y",
                       "dataset_sizes": {"train": len(tr_x), "validation": len(va_x),
                                         "near_ood": len(ne_x), "far_ood": len(fa_x)}}, f, indent=2)
        hid.mkdir(parents=True, exist_ok=True)
        with open(hid / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump({"problem_id": p["pid"], "block": p["block"],
                       "canonical_structural_family": p["family"],
                       "ground_truth_formula": p["formula"], "true_parameters": p["params"],
                       "oracle_epistemic_class": p["oracle"], "expected_outcome": p["expected"]}, f, indent=2)
        manifest["problems"].append({"problem_id": p["pid"], "block": p["block"], "name": p["name"]})
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(root / "generation_seed.txt", "w", encoding="utf-8") as f:
        f.write(str(GENERATION_SEED) + "\n")
    sha = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            sha.update(path.relative_to(root).as_posix().encode())
            sha.update(path.read_bytes())
    with open(root / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write(sha.hexdigest() + "  corpus-tree\n")


def main() -> None:
    rng = np.random.default_rng(GENERATION_SEED)
    problems = build_problems(rng)
    write_corpus(problems, CORPUS_DIR)
    print(f"wrote {len(problems)} problems to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
