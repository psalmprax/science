#!/usr/bin/env python3
"""Run configured candidate builders and official verifiers, then seal a hash-bound audit bundle.

This runner does not implement or emulate any competition verifier. Each official verifier
must write its own JSON receipt at the configured per-run path. Missing/stale/invalid receipts
fail closed. Commands are argument arrays (never shell strings) and may use placeholders:
{repo_root}, {competition_root}, {run_dir}, {receipt_path}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

BASELINES = {
    "invsqrt": 2_076_774.233715282,
    "dqi": 24_439_463_583_672.93,
    "dfthcsos": 3_355_599_588.3155365,
}
EXPECTED_SHOTS = {"invsqrt": 200_540_928, "dqi": 102_400, "dfthcsos": 254_464}
PLACEHOLDER_MARKERS = ("PATH/TO/", "/ABSOLUTE/PATH/TO/", "REPLACE_WITH_", "TO_BE_CONFIGURED", "YOUR_", "<SET_")


def find_placeholders(value: Any, where: str = "plan") -> list[str]:
    """Reject an unconfigured template before creating any run directory or subprocess."""
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            found.extend(find_placeholders(v, f"{where}.{k}"))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(find_placeholders(v, f"{where}[{i}]"))
    elif isinstance(value, str):
        if any(marker in value for marker in PLACEHOLDER_MARKERS):
            found.append(f"{where}={value!r}")
    return found


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def value_at(obj: Any, dotted: str) -> Any:
    cur = obj
    for key in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(key)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if key not in cur:
                return None
            cur = cur[key]
        else:
            return None
    return cur


def check_comparison(actual: Any, rule: dict[str, Any]) -> tuple[bool, str]:
    op, expected = rule.get("op", "eq"), rule.get("value")
    if op == "eq": ok = actual == expected
    elif op == "ge": ok = isinstance(actual, (int, float)) and actual >= expected
    elif op == "gt": ok = isinstance(actual, (int, float)) and actual > expected
    elif op == "le": ok = isinstance(actual, (int, float)) and actual <= expected
    elif op == "lt": ok = isinstance(actual, (int, float)) and actual < expected
    elif op == "in": ok = actual in expected
    elif op == "truthy": ok = bool(actual)
    elif op == "finite_positive": ok = isinstance(actual, (int, float)) and math.isfinite(float(actual)) and actual > 0
    else: return False, f"unsupported check operator: {op}"
    return bool(ok), f"{actual!r} {op} {expected!r} -> {bool(ok)}"


def render_command(command: list[str], values: dict[str, str]) -> list[str]:
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a non-empty array of strings")
    return [part.format(**values) for part in command]


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace", check=False)
    finished = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {"argv": command, "cwd": str(cwd), "returncode": proc.returncode,
            "started_unix": started, "finished_unix": finished,
            "log_path": str(log_path), "log_sha256": sha256_file(log_path)}


def root_for_item(item: dict[str, Any], repo_root: Path, competition_root: Path) -> Path:
    root_name = item.get("root", "repo")
    if root_name == "repo":
        return repo_root
    if root_name == "competition":
        return competition_root
    raise ValueError(f"unsupported file root {root_name!r}; use 'repo' or 'competition'")


def verify_protected_files(repo_root: Path, items: list[dict[str, str]],
                           competition_root: Path | None = None) -> list[dict[str, Any]]:
    results = []
    competition_root = competition_root or repo_root
    for item in items:
        path = resolve(root_for_item(item, repo_root, competition_root), item["path"])
        expected = item["sha256"].lower()
        if not path.is_file():
            results.append({"path": str(path), "ok": False, "reason": "missing file", "expected_sha256": expected})
            continue
        actual = sha256_file(path)
        results.append({"path": str(path), "ok": actual == expected, "expected_sha256": expected, "actual_sha256": actual})
    return results



def snapshot_source_files(repo_root: Path, competition_root: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash exact source files used for baseline generation/verification; optionally enforce pinned hashes."""
    results: list[dict[str, Any]] = []
    for item in items:
        path = resolve(root_for_item(item, repo_root, competition_root), item["path"])
        row: dict[str, Any] = {"root": item.get("root", "repo"), "path": str(path), "required": True}
        if not path.is_file():
            row.update({"ok": False, "reason": "missing source file"})
        else:
            actual = sha256_file(path)
            expected = item.get("sha256")
            row.update({"ok": expected is None or actual.lower() == str(expected).lower(), "sha256": actual})
            if expected is not None:
                row["expected_sha256"] = str(expected).lower()
                if actual.lower() != str(expected).lower():
                    row["reason"] = "source hash mismatch"
        results.append(row)
    return results


def check_command_executables(plan: dict[str, Any]) -> list[str]:
    """Check the executable token of each builder/verifier command is resolvable."""
    errors: list[str] = []
    for component in plan.get("components", []):
        for field in ("candidate_builder_command", "official_verifier_command"):
            cmd = component.get(field)
            if not isinstance(cmd, list) or not cmd or not isinstance(cmd[0], str):
                errors.append(f"components[{component.get('name')}].{field} must be a non-empty argv list")
                continue
            exe = cmd[0]
            if Path(exe).is_absolute():
                if not Path(exe).is_file():
                    errors.append(f"executable does not exist: {exe}")
            elif shutil.which(exe) is None:
                errors.append(f"executable not found on PATH: {exe}")
    return errors


def run_plan(plan_path: Path, only: set[str] | None = None) -> int:
    plan_path = plan_path.resolve()
    plan = load_json(plan_path)
    if plan.get("schema_version") != 1:
        raise ValueError("plan schema_version must be 1")
    placeholders = find_placeholders(plan)
    if placeholders:
        raise ValueError("plan is still a template; resolve these placeholders first: " + "; ".join(placeholders[:12]))
    cmd_errors = check_command_executables(plan)
    if cmd_errors:
        raise ValueError("receipt plan preflight failed: " + "; ".join(cmd_errors))
    plan_dir = plan_path.parent
    repo_root = resolve(plan_dir, plan["repo_root"])
    competition_root = resolve(plan_dir, plan["competition_root"])
    output_root = resolve(repo_root, plan.get("output_root", "artifacts/quantum_compilation/official_receipts"))
    if not repo_root.is_dir(): raise FileNotFoundError(f"repo_root does not exist: {repo_root}")
    if not competition_root.is_dir(): raise FileNotFoundError(f"competition_root does not exist: {competition_root}")
    source_files_before = snapshot_source_files(repo_root, competition_root, plan.get("source_files", []))
    if any(not x["ok"] for x in source_files_before):
        raise FileNotFoundError("one or more official/source files are missing or have an unexpected hash")

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    values = {"repo_root": str(repo_root), "competition_root": str(competition_root), "run_dir": str(run_dir)}
    env = os.environ.copy()
    env.update({"APEDP_Q_REPO_ROOT": str(repo_root), "APEDP_Q_COMPETITION_ROOT": str(competition_root),
                "APEDP_Q_RECEIPT_RUN_DIR": str(run_dir), "PYTHONUNBUFFERED": "1"})
    started = time.time()

    before_protected = verify_protected_files(repo_root, plan.get("protected_files", []), competition_root)
    if any(not x["ok"] for x in before_protected):
        atomic_json(run_dir / "execution_manifest.json", {"status": "BLOCKED_INPUT_HASH_MISMATCH", "run_id": run_id,
                    "plan_path": str(plan_path), "protected_files_before": before_protected, "source_files_before": source_files_before})
        print(f"BLOCKED: one or more protected inputs are missing or mismatched; see {run_dir / 'execution_manifest.json'}", file=sys.stderr)
        return 2

    execution = {"schema_version": 1, "run_id": run_id, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "repo_root": str(repo_root), "competition_root": str(competition_root), "plan_sha256": sha256_file(plan_path),
                 "protected_files_before": before_protected, "source_files_before": source_files_before, "python": sys.version, "platform": platform.platform(),
                 "components": {}, "status": "RUNNING"}
    all_pass = True
    for component in plan.get("components", []):
        name = component.get("name")
        if only and name not in only:
            continue
        if name not in BASELINES:
            raise ValueError(f"unknown component {name!r}; expected one of {sorted(BASELINES)}")
        cdir = run_dir / name
        cdir.mkdir(parents=True)
        cinfo: dict[str, Any] = {"baseline_cost": BASELINES[name], "status": "RUNNING", "artifact_hashes": [], "checks": []}
        execution["components"][name] = cinfo
        try:
            if component.get("candidate_builder_command"):
                cmd = render_command(component["candidate_builder_command"], {**values, "component_dir": str(cdir), "receipt_path": str(cdir / "official_receipt.json")})
                r = run_command(cmd, repo_root, env, cdir / "build.log")
                cinfo["build"] = r
                if r["returncode"] != 0:
                    raise RuntimeError(f"candidate builder exited {r['returncode']}")
            else:
                raise RuntimeError("candidate_builder_command is missing; refusing to reuse an unbound artifact")

            artifact_rows = []
            artifact_values = {**values, "component_dir": str(cdir), "receipt_path": str(cdir / "official_receipt.json")}
            for raw in component.get("artifact_paths", []):
                resolved_raw = raw.format(**artifact_values)
                ap = resolve(repo_root, resolved_raw)
                if not ap.is_file():
                    artifact_rows.append({"path": str(ap), "ok": False, "reason": "missing artifact"})
                    continue
                artifact_rows.append({"path": str(ap), "ok": True, "size_bytes": ap.stat().st_size,
                                      "sha256": sha256_file(ap)})
            cinfo["artifact_hashes"] = artifact_rows
            if not artifact_rows or any(not a["ok"] for a in artifact_rows):
                raise RuntimeError("one or more expected candidate artifacts are missing")

            receipt_path = cdir / "official_receipt.json"
            values_component = {**values, "receipt_path": str(receipt_path), "component_dir": str(cdir)}
            cmd = render_command(component.get("official_verifier_command", []), values_component)
            vr = run_command(cmd, repo_root, env, cdir / "verify.log")
            cinfo["verifier"] = vr
            if vr["returncode"] != 0:
                raise RuntimeError(f"official verifier exited {vr['returncode']}")
            if not receipt_path.is_file():
                raise RuntimeError(f"official verifier did not write the configured receipt: {receipt_path}")
            receipt = load_json(receipt_path)
            cinfo["official_receipt_path"] = str(receipt_path)
            cinfo["official_receipt_sha256"] = sha256_file(receipt_path)
            cinfo["official_receipt"] = receipt

            # Universal receipt checks; benchmark-specific checks are plan-configured to avoid
            # pretending different competition schema versions share identical field names.
            checks = []
            status_path = component.get("receipt_status_field", "status")
            actual_status = value_at(receipt, status_path)
            checks.append({"field": status_path, "ok": actual_status == "PASS", "detail": f"status is {actual_status!r}"})
            score_path = component.get("receipt_score_field", "score")
            actual_score = value_at(receipt, score_path)
            score_ok = isinstance(actual_score, (int, float)) and math.isfinite(float(actual_score)) and actual_score > 0
            checks.append({"field": score_path, "ok": score_ok, "actual": actual_score})
            for rule in component.get("receipt_checks", []):
                actual = value_at(receipt, rule["field"])
                ok, detail = check_comparison(actual, rule)
                checks.append({"field": rule["field"], "ok": ok, "detail": detail})
            cinfo["checks"] = checks
            if not all(x["ok"] for x in checks):
                raise RuntimeError("official receipt failed one or more configured promotion checks")
            cinfo["score"] = float(actual_score)
            cinfo["improvement_vs_reference"] = BASELINES[name] / float(actual_score)
            cinfo["status"] = "PASS"
        except Exception as e:
            cinfo["status"] = "BLOCKED_OR_FAILED"
            cinfo["error"] = f"{type(e).__name__}: {e}"
            all_pass = False
            # Fail closed: no later benchmark starts after one stage fails unless plan says otherwise.
            if plan.get("stop_on_first_failure", True):
                break

    after_protected = verify_protected_files(repo_root, plan.get("protected_files", []), competition_root)
    source_files_after = snapshot_source_files(repo_root, competition_root, plan.get("source_files", []))
    source_hashes_unchanged = len(source_files_before) == len(source_files_after) and all(
        a.get("ok") and b.get("ok") and a.get("sha256") == b.get("sha256")
        for a, b in zip(source_files_before, source_files_after)
    )
    execution["source_files_after"] = source_files_after
    execution["source_hashes_unchanged"] = source_hashes_unchanged
    if not source_hashes_unchanged:
        all_pass = False
    execution["protected_files_after"] = after_protected
    if any(not x["ok"] for x in after_protected):
        all_pass = False
        execution["protected_inputs_unchanged"] = False
    else:
        execution["protected_inputs_unchanged"] = True
    execution["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    execution["duration_seconds"] = time.time() - started
    if only:
        selected_ok = all(execution["components"].get(n, {}).get("status") == "PASS" for n in only)
        execution["status"] = "SELECTED_COMPONENTS_PASS" if all_pass and selected_ok and execution["protected_inputs_unchanged"] and source_hashes_unchanged else "HOLD_OR_FAILED"
    else:
        required = {"invsqrt", "dqi", "dfthcsos"}
        configured = {c.get("name") for c in plan.get("components", [])}
        all_configured_and_passed = configured == required and all(
            execution["components"].get(n, {}).get("status") == "PASS" for n in required
        )
        execution["status"] = "ALL_COMPONENTS_PASS" if all_pass and all_configured_and_passed and execution["protected_inputs_unchanged"] and source_hashes_unchanged else "HOLD_OR_FAILED"

    passed_names = [n for n, x in execution["components"].items() if x.get("status") == "PASS"]
    if len(passed_names) == 3 and execution["protected_inputs_unchanged"]:
        ratios = [BASELINES[n] / execution["components"][n]["score"] for n in ("invsqrt", "dqi", "dfthcsos")]
        execution["portfolio"] = {"reference_costs": BASELINES, "improvement_factors": dict(zip(("invsqrt", "dqi", "dfthcsos"), ratios)),
                                   "product_improvement": math.prod(ratios), "geometric_mean_score": math.prod(ratios) ** (1 / 3),
                                   "promotion_status": "ELIGIBLE_FOR_LEDGER_REVIEW_NOT_AUTOMATIC_PROMOTION"}
    atomic_json(run_dir / "execution_manifest.json", execution)
    print(json.dumps({"status": execution["status"], "run_dir": str(run_dir),
                      "manifest": str(run_dir / "execution_manifest.json"),
                      "components": {n: c.get("status") for n, c in execution["components"].items()},
                      "portfolio": execution.get("portfolio")}, indent=2))
    return 0 if execution["status"] == "ALL_COMPONENTS_PASS" else 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True, help="JSON run-plan with explicit builder/verifier commands")
    p.add_argument("--only", nargs="+", choices=sorted(BASELINES), help="run selected benchmark components only")
    p.add_argument("--preflight-only", action="store_true", help="validate paths, hashes, and commands without building circuits")
    a = p.parse_args()
    try:
        if a.preflight_only:
            plan = load_json(a.plan.resolve())
            placeholders = find_placeholders(plan)
            if placeholders:
                print(json.dumps({"status": "BLOCKED_UNCONFIGURED_TEMPLATE", "placeholders": placeholders}, indent=2))
                return 2
            errors = check_command_executables(plan)
            plan_dir = a.plan.resolve().parent
            repo_root = resolve(plan_dir, plan["repo_root"])
            competition_root = resolve(plan_dir, plan["competition_root"])
            if not repo_root.is_dir(): errors.append(f"repo_root missing: {repo_root}")
            if not competition_root.is_dir(): errors.append(f"competition_root missing: {competition_root}")
            if repo_root.is_dir() and competition_root.is_dir():
                prot = verify_protected_files(repo_root, plan.get("protected_files", []), competition_root)
                sources = snapshot_source_files(repo_root, competition_root, plan.get("source_files", []))
                if any(not x["ok"] for x in prot): errors.append("one or more protected input hashes fail")
                if any(not x["ok"] for x in sources): errors.append("one or more source files missing or mismatched")
            else:
                prot, sources = [], []
            result = {"status": "READY_FOR_EXECUTION" if not errors else "BLOCKED_PREFLIGHT",
                      "repo_root": str(repo_root), "competition_root": str(competition_root),
                      "python": sys.version, "protected_file_checks": prot, "source_file_checks": sources, "errors": errors}
            print(json.dumps(result, indent=2))
            return 0 if not errors else 2
        return run_plan(a.plan, set(a.only) if a.only else None)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
