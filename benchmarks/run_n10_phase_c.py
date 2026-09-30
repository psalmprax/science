"""N10 Phase-C benchmark runner with incremental save.

Design principles:
    1. Each completed case is serialized immediately after discovery.
    2. A timeout or exception on case N never overwrites earlier valid results.
    3. Explicit termination states: SUCCESS, TIMEOUT, CLEAN_EMPTY,
       SUBPROCESS_ERROR, NUMERICAL_ERROR.
    4. Final schema validation + artifact freeze.

Usage::

    python benchmarks/run_n10_phase_c.py           # run all 10
    python benchmarks/run_n10_phase_c.py N01 N05    # run specific cases
    python benchmarks/run_n10_phase_c.py --validate # validate existing results
"""
from __future__ import annotations

import hashlib
import json
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "benchmark_n10"
OUTDIR = ROOT / "n10_phase_c_results"
TIMEOUT_SECONDS = 300  # 5 minutes per case


class TerminationState(str, Enum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    CLEAN_EMPTY = "CLEAN_EMPTY"
    SUBPROCESS_ERROR = "SUBPROCESS_ERROR"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"


@dataclass
class CaseResult:
    """Frozen per-case result record."""

    problem_id: str
    termination: str
    expression: str = ""
    abstained: bool = False
    abstain_reason: str = ""
    epistemic_state: str = ""
    structural_family: str = ""
    val_mse: float | None = None
    near_ood_mse: float | None = None
    far_ood_mse: float | None = None
    wall_seconds: float = 0.0
    predictions: dict[str, list[float]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure JSON-safe
        for k, v in d.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                d[k] = None
        return d


class TimeoutError(Exception):
    """Raised when a case exceeds the time budget."""


def _timeout_handler(signum, frame):
    raise TimeoutError("Case exceeded time budget")


def _load_public_grids(problem_id: str) -> dict[str, np.ndarray]:
    """Read public CSVs for one N10 problem. Never touches hidden/."""
    import csv

    root = CORPUS / "public" / problem_id
    out = {}
    for name in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
        with open(root / name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        out[name] = np.array([[float(v) for v in r] for r in rows[1:]])
    return out


def _run_one_case(problem_id: str) -> CaseResult:
    """Run APEDP discovery on a single N10 case with timeout protection."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()

    try:
        from apedp.model_synthesis import AutonomousModelSynthesizer
    except ImportError as e:
        return CaseResult(
            problem_id=problem_id,
            termination=TerminationState.IMPORT_ERROR.value,
            error_message=f"Import error: {e}",
            timestamp=ts,
        )

    # Set timeout (Unix only; on Windows this is a no-op)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
    except (AttributeError, ValueError):
        pass  # Windows or non-main thread

    try:
        grids = _load_public_grids(problem_id)
        tr, va = grids["train.csv"], grids["validation.csv"]

        models = AutonomousModelSynthesizer().synthesize(
            tr[:, 0], tr[:, 1], va[:, 0], va[:, 1]
        )

        if not models:
            return CaseResult(
                problem_id=problem_id,
                termination=TerminationState.CLEAN_EMPTY.value,
                abstained=True,
                abstain_reason="No models synthesized",
                wall_seconds=time.time() - t0,
                timestamp=ts,
            )

        top = models[0]
        epi = top.epistemic_state.value if top.epistemic_state else ""
        abstained = epi in {"INSUFFICIENT_EVIDENCE", "NO_VALID_CANDIDATE"}

        # Evaluate predictions on all grids
        preds: dict[str, list[float]] = {}
        try:
            import sympy as sp

            f = sp.lambdify(sp.Symbol("x"), top.sympy_expr, modules=["numpy"])
            for key in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
                with np.errstate(all="ignore"):
                    p = np.asarray(f(grids[key][:, 0]), dtype=float).ravel()
                preds[key] = [
                    float(v) if np.isfinite(v) else float("nan") for v in p
                ]
        except (ArithmeticError, TypeError, ValueError):
            pass

        # Compute MSE against public grids
        val_mse = _grid_mse(grids["validation.csv"][:, 1], preds.get("validation.csv"))
        near_mse = _grid_mse(grids["near_ood.csv"][:, 1], preds.get("near_ood.csv"))
        far_mse = _grid_mse(grids["far_ood.csv"][:, 1], preds.get("far_ood.csv"))

        return CaseResult(
            problem_id=problem_id,
            termination=TerminationState.SUCCESS.value,
            expression=top.expression_str,
            abstained=abstained,
            abstain_reason="" if not abstained else f"epistemic={epi}",
            epistemic_state=epi,
            structural_family=top.structural_family,
            val_mse=val_mse,
            near_ood_mse=near_mse,
            far_ood_mse=far_mse,
            wall_seconds=time.time() - t0,
            predictions=preds,
            diagnostics={
                "train_mse": top.train_mse,
                "proof_claim": top.proof_claim,
                "cp_label": top.cp_label,
            },
            timestamp=ts,
        )

    except TimeoutError:
        return CaseResult(
            problem_id=problem_id,
            termination=TerminationState.TIMEOUT.value,
            error_message=f"Exceeded {TIMEOUT_SECONDS}s budget",
            wall_seconds=time.time() - t0,
            timestamp=ts,
        )
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as e:
        return CaseResult(
            problem_id=problem_id,
            termination=TerminationState.NUMERICAL_ERROR.value,
            error_message=f"{type(e).__name__}: {e}",
            wall_seconds=time.time() - t0,
            timestamp=ts,
        )
    except Exception as e:
        traceback.print_exc()
        return CaseResult(
            problem_id=problem_id,
            termination=TerminationState.SUBPROCESS_ERROR.value,
            error_message=f"{type(e).__name__}: {e}",
            wall_seconds=time.time() - t0,
            timestamp=ts,
        )
    finally:
        # Cancel alarm
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass


def _grid_mse(
    y_true: np.ndarray, y_pred: list[float] | None
) -> float | None:
    """MSE of predicted grid values vs public grid y."""
    if y_pred is None:
        return None
    try:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if yp.shape != yt.shape or not np.all(np.isfinite(yp)):
            return None
        return float(np.mean((yt - yp) ** 2))
    except (ArithmeticError, TypeError, ValueError):
        return None


def _save_case(result: CaseResult) -> None:
    """Incrementally save a single case result. Atomic write."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{result.problem_id}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    tmp.rename(path)  # atomic on POSIX


def _validate_results() -> dict[str, Any]:
    """Validate all existing results. Returns summary."""
    if not OUTDIR.exists():
        return {"valid": False, "error": "No results directory"}

    expected = [f"N{i:02d}" for i in range(1, 11)]
    found = []
    missing = []
    errors = []
    summary: dict[str, Any] = {}

    for pid in expected:
        path = OUTDIR / f"{pid}.json"
        if not path.exists():
            missing.append(pid)
            continue
        try:
            data = json.loads(path.read_text())
            # Schema validation
            required_fields = {
                "problem_id", "termination", "expression",
                "wall_seconds", "timestamp",
            }
            actual_fields = set(data.keys())
            missing_fields = required_fields - actual_fields
            if missing_fields:
                errors.append(f"{pid}: missing fields {missing_fields}")
            elif data["problem_id"] != pid:
                errors.append(f"{pid}: problem_id mismatch ({data['problem_id']})")
            else:
                found.append(pid)
                summary[pid] = {
                    "termination": data["termination"],
                    "expression": data.get("expression", "")[:60],
                    "epistemic_state": data.get("epistemic_state", ""),
                    "val_mse": data.get("val_mse"),
                    "wall_seconds": data.get("wall_seconds"),
                }
        except (json.JSONDecodeError, KeyError) as e:
            errors.append(f"{pid}: {type(e).__name__}: {e}")

    return {
        "valid": len(found) == 10 and not errors,
        "found": found,
        "missing": missing,
        "errors": errors,
        "summary": summary,
        "total": len(found),
    }


def _freeze_artifacts() -> dict[str, str]:
    """Compute SHA-256 checksums for all result files."""
    checksums = {}
    for path in sorted(OUTDIR.glob("*.json")):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[path.name] = h
    # Write checksum file
    cs_path = OUTDIR / "checksums.sha256"
    lines = [f"{h}  {name}" for name, h in checksums.items()]
    cs_path.write_text("\n".join(lines) + "\n")
    return checksums


def main(args: list[str]) -> None:
    if "--validate" in args:
        result = _validate_results()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["valid"] else 1)

    if "--freeze" in args:
        checksums = _freeze_artifacts()
        print(f"Frozen {len(checksums)} artifacts:")
        for name, h in checksums.items():
            print(f"  {h}  {name}")
        sys.exit(0)

    # Determine which cases to run
    all_pids = [f"N{i:02d}" for i in range(1, 11)]
    if args:
        pids = [p for p in args if p in all_pids]
        if not pids:
            print(f"No valid problem IDs in {args}. Valid: {all_pids}")
            sys.exit(1)
    else:
        pids = all_pids

    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "APEDP-N10-Phase-C",
        "runner_version": "1.0",
        "timeout_seconds": TIMEOUT_SECONDS,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": [],
    }

    total = len(pids)
    succeeded = 0
    for i, pid in enumerate(pids, 1):
        # Check if result already exists (resume support)
        existing = OUTDIR / f"{pid}.json"
        if existing.exists():
            try:
                data = json.loads(existing.read_text())
                if data.get("termination") == TerminationState.SUCCESS.value:
                    print(
                        f"[{i}/{total}] {pid} SKIP (already SUCCESS)",
                        flush=True,
                    )
                    manifest["cases"].append({
                        "problem_id": pid,
                        "status": "SKIPPED_EXISTING",
                    })
                    succeeded += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # Re-run if corrupt

        print(f"[{i}/{total}] {pid} RUNNING...", end=" ", flush=True)
        result = _run_one_case(pid)

        # INCREMENTAL SAVE: each case saved immediately
        _save_case(result)

        status = result.termination
        if status == TerminationState.SUCCESS.value:
            succeeded += 1
            expr_short = result.expression[:50] if result.expression else ""
            print(
                f"{status} ({result.wall_seconds:.1f}s) "
                f"epi={result.epistemic_state} "
                f"expr={expr_short}",
                flush=True,
            )
        elif result.abstained:
            print(
                f"{status} ({result.wall_seconds:.1f}s) "
                f"ABSTAIN: {result.abstain_reason}",
                flush=True,
            )
        else:
            print(
                f"{status} ({result.wall_seconds:.1f}s) "
                f"ERROR: {result.error_message[:80]}",
                flush=True,
            )

        manifest["cases"].append({
            "problem_id": pid,
            "status": status,
            "wall_seconds": result.wall_seconds,
        })

    manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["succeeded"] = succeeded
    manifest["total"] = total

    # Save manifest
    manifest_path = OUTDIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"N10 Phase-C: {succeeded}/{total} completed")
    print(f"Results: {OUTDIR}")
    print(f"{'='*60}")

    if succeeded == total:
        print("\nAll cases completed. Run with --validate then --freeze:")
        print(f"  python {__file__} --validate")
        print(f"  python {__file__} --freeze")


if __name__ == "__main__":
    main(sys.argv[1:])
