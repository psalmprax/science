"""APEDP-X mirror of the OpenAI math-release pipeline, as a sandbox.

OpenAI's 2026-10-06 release ("Sharing AI progress in mathematics") attempted
~4000 open problems, published 722 manuscripts, and later retracted 3 (the
repo now lists 719 across 372 result families). Its pipeline was:

    attempt -> publish claim -> independent verification -> retraction accounting

This sandbox runs the same loop for the subset of problems APEDP-X can
machine-check, and reports everything else as UNSUPPORTED with a reason taken
from ``problems.json``. Hard rule enforced in code: a task is only reported
VERIFIED when a checker produced a non-empty certificate; a checker that runs
but misses its threshold is FAILED (the sandbox's retraction count).

Usage:
    python3 benchmarks/openai_math_sandbox/run_sandbox.py [--fast] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import apedp_kernel as ak  # noqa: E402

GOLDBACH_BIN = ROOT / "rust" / "goldbach_verifier" / "target" / "release" / "goldbach_verifier"
RIEMANN_BIN = ROOT / "rust" / "riemann_verifier" / "target" / "release" / "riemann_verifier"

STATUSES = ("verified", "failed", "unavailable", "unsupported")


def _run_bin(path: Path, args: list[str], timeout: int = 300) -> str:
    proc = subprocess.run(
        [str(path), *args], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{path.name} exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout


def check_goldbach(fast: bool = False) -> dict:
    limit = 200_000 if fast else 2_000_000
    if not GOLDBACH_BIN.is_file():
        return {
            "status": "unavailable",
            "reason": f"binary not built: {GOLDBACH_BIN.relative_to(ROOT)} "
            "(run: cargo build --release in rust/goldbach_verifier)",
        }
    out = _run_bin(GOLDBACH_BIN, [str(limit), "--json"])
    res = json.loads(out)
    expected_evens = limit // 2 - 1
    ok = res["violations"] == 0 and res["evens_tested"] == expected_evens
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "evens_tested": res["evens_tested"],
            "expected_evens": expected_evens,
            "violations": res["violations"],
            "total_time_ms": res["total_time_ms"],
            "claim_ceiling": "C2_PREDICTIVE; universal claim abstains (finite range only)",
        },
    }


def check_riemann(fast: bool = False) -> dict:
    max_t = 30.0 if fast else 60.0
    if not RIEMANN_BIN.is_file():
        return {
            "status": "unavailable",
            "reason": f"binary not built: {RIEMANN_BIN.relative_to(ROOT)} "
            "(run: cargo build --release in rust/riemann_verifier)",
        }
    out = _run_bin(RIEMANN_BIN, [str(max_t), "3000"])
    roots = re.search(r"Dynamically isolated (\d+) roots", out)
    gate = re.search(r"Numerical Accuracy Gate: (PASS|FAIL).*max discrepancy: ([0-9.eE+-]+)", out)
    tally = re.search(r"Status: (\w+)", out)
    n_approx = re.search(r"N_approx\(T\).*: (\d+) zeros", out)
    r2 = re.search(r"Critical Line \(beta = 0\.5\): RMSE = ([0-9.]+) \| R\^2 = ([0-9.]+)", out)
    if not (roots and gate and n_approx):
        return {"status": "failed", "reason": "could not parse verifier output"}
    n_roots, gate_status, max_delta = int(roots.group(1)), gate.group(1), float(gate.group(2))
    expected = int(n_approx.group(1))
    # The approximate tally and the isolated roots may differ by the S(T)
    # remainder the verifier explicitly leaves uncertified; allow <= 2 wiggle.
    ok = gate_status == "PASS" and n_roots >= 3 and abs(n_roots - expected) <= 2
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "roots_discovered_from_scratch": n_roots,
            "approximate_tally": expected,
            "reference_calibration": gate_status,
            "max_reference_delta": max_delta,
            "tally_status": tally.group(1) if tally else "UNPARSED",
            "explicit_formula_r2": float(r2.group(2)) if r2 else None,
            "claim_ceiling": "finite-range numerics; certified zero count deliberately unresolved (S(T) remainder)",
        },
    }


def check_planck_riccati(fast: bool = False) -> dict:
    v = np.linspace(0.3, 5.0, 80)
    u = (v ** 3) / (np.exp(v) - 1.0)
    cfg = {
        "max_x_degree": 3,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.02,
        "rcond_threshold": 1e-12,
        "include_riccati": True,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(v.tolist()), json.dumps(u.tolist()), json.dumps(cfg)
        )
    )
    ok = (
        bool(res.get("is_exact_recovery"))
        and float(res["jet_rmse"]) < 1e-5
        and float(res["integrated_trajectory_r2"]) > 0.9999
    )
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "is_exact_recovery": res.get("is_exact_recovery"),
            "jet_rmse": res["jet_rmse"],
            "integrated_trajectory_r2": res["integrated_trajectory_r2"],
            "claim_ceiling": "exact recovery of a rational ODE on the fitted family",
        },
    }


def check_harmonic(fast: bool = False) -> dict:
    t = np.linspace(0.1, 3.0, 70)
    y = np.sin(2.0 * t)
    cfg = {
        "max_x_degree": 0,
        "max_diff_order": 2,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
        "include_riccati": False,
    }
    res = json.loads(
        ak.rime_diff_pade_discover_json(
            json.dumps(t.tolist()), json.dumps(y.tolist()), json.dumps(cfg)
        )
    )
    ok = (
        bool(res.get("is_exact_recovery"))
        and float(res["jet_rmse"]) < 1e-12
        and float(res["integrated_trajectory_r2"]) > 0.99999
    )
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "is_exact_recovery": res.get("is_exact_recovery"),
            "jet_rmse": res["jet_rmse"],
            "integrated_trajectory_r2": res["integrated_trajectory_r2"],
            "claim_ceiling": "exact recovery of y'' = -4y on the fitted family",
        },
    }


def _ga_best_mse(y: np.ndarray) -> tuple[float, str]:
    x = np.linspace(0.05, 3.0, 60)
    raw, _ = ak.evolve_native(
        [np.ascontiguousarray(x)], ["x"], np.ascontiguousarray(y),
        100, 25, 2, 5, 4, 4, 0.35, 0.70, True, True, True, True, True, 3, 42,
    )
    best = min(raw, key=lambda m: m[1])
    return best[1], best[0]


def check_ga_trig(fast: bool = False) -> dict:
    x = np.linspace(0.05, 3.0, 60)
    targets = {"sin(2x)": np.sin(2.0 * x)}
    if not fast:
        targets["2cos(1.3x)"] = 2.0 * np.cos(1.3 * x)
    cert = {}
    ok = True
    for name, y in targets.items():
        mse, expr = _ga_best_mse(y)
        cert[name] = {"mse": mse, "expression": expr}
        ok = ok and mse < 1e-12
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "targets": cert,
            "claim_ceiling": "exact recovery within the seeded trig grammar",
        },
    }


def check_pde_burgers(fast: bool = False) -> dict:
    if not hasattr(ak, "rime_pde_discover_json"):
        return {
            "status": "unavailable",
            "reason": "rime_pde_discover_json binding pending concurrent-session "
            "commit of rime/pde.rs wiring; frozen-transfer path pinned in "
            "tests/test_dyn_pde_transfer.py",
        }
    from apedp_dyn.epistemic.pde_transfer import evaluate_frozen_transfer

    dt, dx = 0.002, 0.1
    nx, nt, nu = 60, 50, 0.1
    u = np.zeros((nt, nx))
    x = np.arange(nx) * dx
    u[0] = np.sin(x)
    for i in range(nt - 1):
        adv = -u[i, 1:-1] * (u[i, 2:] - u[i, :-2]) / (2.0 * dx)
        diff = nu * (u[i, 2:] - 2.0 * u[i, 1:-1] + u[i, :-2]) / (dx * dx)
        u[i + 1, 1:-1] = u[i, 1:-1] + dt * (adv + diff)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]

    n_dev = max(5, int(nt * 0.75))
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": 0.01,
        "ridge_alpha": 1e-6,
    }
    dev = json.loads(
        ak.rime_pde_discover_json(json.dumps(u[:n_dev].tolist()), dt, dx, json.dumps(cfg))
    )
    frozen = evaluate_frozen_transfer(
        u[n_dev - 1:], dt, dx, dev["active_terms"], dev["coefficients"]
    )
    terms = dev["active_terms"]
    coeffs = dev["coefficients"]
    adv_ok = "u*u_x" in terms and abs(coeffs[terms.index("u*u_x")] - (-1.0)) < 0.15
    diff_ok = "u_xx" in terms and abs(coeffs[terms.index("u_xx")] - 0.1) < 0.15
    ok = frozen.r2 > 0.9 and adv_ok and diff_ok
    return {
        "status": "verified" if ok else "failed",
        "certificate": {
            "active_terms": terms,
            "coefficients": coeffs,
            "frozen_holdout_r2": frozen.r2,
            "holdout_samples": frozen.n_samples,
            "claim_ceiling": "frozen-coefficient transfer; re-fit metric intentionally not reported",
        },
    }


CHECKS = {
    "NT-1": check_goldbach,
    "NT-2": check_riemann,
    "LAW-1": check_planck_riccati,
    "LAW-2": check_harmonic,
    "GA-1": check_ga_trig,
    "PDE-1": check_pde_burgers,
}


def load_registry() -> dict:
    return json.loads((HERE / "problems.json").read_text(encoding="utf-8"))


def run_all(fast: bool = False) -> dict:
    registry = load_registry()
    results = []
    for entry in registry["problems"]:
        pid = entry["id"]
        started = time.perf_counter()
        if entry["status"] == "unsupported":
            res = {"status": "unsupported", "reason": entry["reason"]}
        else:
            checker = CHECKS.get(pid)
            if checker is None:
                res = {"status": "unavailable", "reason": "no checker dispatch registered"}
            else:
                try:
                    res = checker(fast=fast)
                except Exception as exc:  # checker crash is a failed attempt, not a pass
                    res = {"status": "failed", "reason": f"checker error: {exc!r}"}
        duration_ms = (time.perf_counter() - started) * 1000.0
        if res["status"] == "verified" and not res.get("certificate"):
            raise AssertionError(
                f"{pid}: runner must never report verified without a certificate"
            )
        results.append(
            {
                "id": pid,
                "category": entry["category"],
                "name": entry["name"],
                "registry_status": entry["status"],
                "status": res["status"],
                "duration_ms": round(duration_ms, 2),
                "certificate": res.get("certificate"),
                "reason": res.get("reason") or entry.get("reason"),
                "claim_ceiling": entry.get("claim_ceiling")
                or (res.get("certificate") or {}).get("claim_ceiling"),
            }
        )
    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUSES}
    report = {
        "provenance": registry["provenance"],
        "counts": counts,
        "results": results,
        "honesty_note": (
            "verified = checker produced a certificate in this run; failed = "
            "checker ran and missed threshold (would-be retraction); "
            "unavailable = engine present in registry but not runnable here; "
            "unsupported = no engine path (gap)."
        ),
    }
    return report


def _print_report(report: dict) -> None:
    print("=" * 79)
    print("  APEDP-X OPENAI-MATH SANDBOX  (attempt -> certificate -> verification)")
    print("=" * 79)
    print(f"{'ID':<7} {'STATUS':<12} {'CATEGORY':<28} {'MS':>8}  NAME")
    print("-" * 79)
    for r in report["results"]:
        print(
            f"{r['id']:<7} {r['status'].upper():<12} "
            f"{r['category'][:26]:<28} {r['duration_ms']:>8.1f}  {r['name'][:40]}"
        )
    c = report["counts"]
    print("-" * 79)
    attempted = c["verified"] + c["failed"]
    print(
        f"attempted={attempted}  verified={c['verified']}  failed={c['failed']}  "
        f"unavailable={c['unavailable']}  unsupported={c['unsupported']}"
    )
    for r in report["results"]:
        if r["status"] == "unsupported":
            print(f"  GAP {r['id']:<12} {r['reason'][:64]}")
        elif r["status"] == "unavailable":
            print(f"  N/A {r['id']:<12} {r['reason'][:64]}")
    del by_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="smaller ranges for CI")
    parser.add_argument(
        "--json", default=str(HERE / "sandbox_report.json"), help="report output path"
    )
    args = parser.parse_args()
    report = run_all(fast=args.fast)
    _print_report(report)
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {args.json}")
    return 0 if report["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
