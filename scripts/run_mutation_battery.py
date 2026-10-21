r"""Targeted Mutation-Testing Battery (REG-MUT acceptance artifact).

Status: audit-only (scripts/ + written report; not a production module)

Response to the outstanding-requirements register (docs/SYSTEM_FLAWS_AUDIT.md,
REG-MUT): targeted mutation testing of the referee and leakage paths, with a
run report demonstrating each configured pin FAILS under its mutation.

Register targets and mutants:
  1. adjudicate anchor (Finding 14 re-anchor):
       MUT-A1  invert the holdout anchor (min -> max holdout_rmse) — the
               promotion head would anchor on the WORST certified rival.
       MUT-A2  drop the gate-4-certified guard on the re-anchor — a
               non-certified holdout head could take over the pair.
     Kills-only-if-preferee-discipline-holds pins: the Finding-14 anchor pins
     (test_reanchor_requires_gate4_certified_holdout_head,
     test_reanchored_pass_skips_bootstrap_confirmation).
  2. frozen-library term resolution (F8 fail-closed + F8 diagnostic detail):
       MUT-B1  defang the fail-closed refusal (unresolved dev-discovered term
               would be silently skipped, corrupting the frozen R^2 toward a
               flattering value — the original F8 defect class).
       MUT-B2  redact the drifted term names from the refusal diagnostic.
     Pins: TestF8FrozenLibraryTermAudit + TestF8ErrorDiagnosticDetail.
  3. hawkes standardization constants (Finding 10 leak class):
       MUT-C1  standardize the intensity against EVAL-frame mean/std instead
               of the fit-coupling constants (full-future leakage).
     Pin: test_hawkes_orderflow_dynamic_on_regular_bars.

Method, kept deliberately dependency-free (no third-party mutation framework):
each mutant is a single exact source substitution applied to the target file;
the configured pin subset runs in a FRESH interpreter; the original file
byte-identical restoration is verified after every mutant. A mutant counts as
KILLED only when its pin run exits non-zero (the pin fails) while the pin
passes on the pristine tree (pristine sanity run is checked first). The
process exits 0 iff every configured mutant is killed AND the pristine sanity
run passed — a survived mutant is a REG-MUT failure, never a soft warning.

Run:
    .venv/bin/python scripts/run_mutation_battery.py
Report:
    benchmarks/mutation_battery_report.json (machine-readable, committed)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

PIN_ROUND2 = ["tests/test_dyn_blind_audit_round2_remediation.py", "-k", "reanchor or F8"]
PIN_HAWKES = ["tests/test_audit_open_flaws_remediated.py::test_hawkes_orderflow_dynamic_on_regular_bars"]
# Sharper Finding-10 pin (added after MUT-C1 survived the identical-frames
# variant of the static pin — the battery did its job: it exposed a coverage
# gap, and the pin was sharpened, never weakened).
PIN_HAWKES_SHARP = ["tests/test_dyn_hawkes_standardization_pins.py"]
# 2026-10-21 external audit remediation pins: each MUT-D* mutant below must be
# killed (pin MUST fail under the mutation) by this suite.
PIN_ROUND3 = ["tests/test_dyn_external_audit_2026_10_21_pins.py"]

MUTANTS: list[dict] = [
    {
        "id": "MUT-A1",
        "target": "adjudicate anchor (Finding 14)",
        "file": "apedp_dyn/epistemic/adjudicate.py",
        "description": "invert holdout anchor min -> max (anchor on the WORST certified rival)",
        "find": "        ho_head = min(ho_scoring, key=lambda s: s.holdout_rmse)",
        "replace": "        ho_head = max(ho_scoring, key=lambda s: s.holdout_rmse)",
        "pins": [PIN_ROUND2],
        "pin_note": "Finding-14 anchor pins",
    },
    {
        "id": "MUT-A2",
        "target": "adjudicate anchor (Finding 14)",
        "file": "apedp_dyn/epistemic/adjudicate.py",
        "description": "drop the gate-4-certified guard on the holdout re-anchor",
        "find": "        if ho_head is not best and ho_head.provenance.get(\n                \"gate4_status\", {}).get(\"passed\", False):",
        "replace": "        if ho_head is not best:",
        "pins": [PIN_ROUND2],
        "pin_note": "test_reanchor_requires_gate4_certified_holdout_head",
    },
    {
        "id": "MUT-B1",
        "target": "frozen-library term resolution (F8 fail-closed)",
        "file": "apedp_dyn/epistemic/pde_discovery.py",
        "description": "defang the fail-closed refusal (unresolved term silently skipped)",
        "find": "    if missing:\n        raise ValueError(",
        "replace": "    if False and missing:\n        raise ValueError(",
        "pins": [PIN_ROUND2],
        "pin_note": "TestF8FrozenLibraryTermAudit",
    },
    {
        "id": "MUT-B2",
        "target": "frozen-library term resolution (F8 diagnostic detail)",
        "file": "apedp_dyn/epistemic/pde_discovery.py",
        "description": "redact the drifted term names from the refusal diagnostic",
        "find": "            + \", \".join(sorted(missing))",
        "replace": "            + \"(term names redacted)\"",
        "pins": [PIN_ROUND2],
        "pin_note": "TestF8ErrorDiagnosticDetail",
    },
    {
        "id": "MUT-C1",
        "target": "hawkes standardization constants (Finding 10)",
        "file": "apedp_dyn/proposers/hawkes_orderflow_proposer.py",
        "description": "standardize against eval-frame mean/std (full-future leak)",
        "find": "            if not binary_viable:\n                if fit_intens_std > 1e-8:\n                    return (intens - fit_intens_mean) / fit_intens_std",
        "replace": "            if not binary_viable:\n                if fit_intens_std > 1e-8:\n                    return (intens - intens.mean()) / intens.std()",
        "pins": [PIN_HAWKES, PIN_HAWKES_SHARP],
        "pin_note": "hawkes dynamic pin + sharp fit-constant standardization pin",
    },
    # ── 2026-10-21 external audit remediation pins (tests/
    # test_dyn_external_audit_2026_10_21_pins.py) ──
    {
        "id": "MUT-D1",
        "target": "S3 lock atomic take-over (FLAW-03)",
        "file": "apedp_dyn/epistemic/remote_storage.py",
        "description": "412 PreconditionFailed silently wins the lock (double-winner race back)",
        "find": "            if is_precondition_failed:\n                return token, False",
        "replace": "            if is_precondition_failed:\n                return token, True",
        "pins": [PIN_ROUND3],
        "pin_note": "test_flaw03_contested_lock_cannot_be_stolen",
    },
    {
        "id": "MUT-D2",
        "target": "S3 lock fallback put (FLAW-03 read-back ownership)",
        "file": "apedp_dyn/epistemic/remote_storage.py",
        "description": "parameter-rejection path skips the fallback put (acquire deadlocks)",
        "find": "            if not isinstance(exc, (ValueError, TypeError, AttributeError)):\n                if not self._parameter_rejected(exc):\n                    return token, False",
        "replace": "            if not isinstance(exc, (ValueError, TypeError, AttributeError)):\n                if not self._parameter_rejected(exc):\n                    return token, False\n            return token, False",
        "pins": [PIN_ROUND3],
        "pin_note": "test_flaw03_parameter_rejected_fallback_still_verifies_ownership",
    },
    {
        "id": "MUT-D3",
        "target": "CBF endpoint-headroom deficit split (FLAW-17)",
        "file": "apedp_dyn/robotics/cbf_certificate.py",
        "description": "revert to the least-change equal split (re-introduces the post-correction A1 violation)",
        "find": "            kap_i = np.minimum(_budget(vi_d, nrm[idx_i[bad]]), d)\n            kap_j = np.minimum(d - kap_i, _budget(-vj_d, nrm[idx_j[bad]]))",
        "replace": "            kap_i = d / 2.0\n            kap_j = d / 2.0",
        "pins": [PIN_ROUND3],
        "pin_note": "test_flaw17_cbf_final_projection_prevents_norm_violation",
    },
    {
        "id": "MUT-D4",
        "target": "re-anchor residual recovery + bootstrap (FLAW-08)",
        "file": "apedp_dyn/epistemic/adjudicate.py",
        "description": "drop the recovered paired series, force the point-estimate downgrade back",
        "find": "                resid_best = np.asarray(_anchor_resid, dtype=float)\n                resid_rival = np.asarray(_runner_resid, dtype=float)\n                bootstrap = True",
        "replace": "                resid_best = None\n                resid_rival = None\n                bootstrap = False",
        "pins": [PIN_ROUND3],
        "pin_note": "test_flaw08_re_anchor_runs_bootstrap_on_paired_residuals",
    },
]

_FROZEN_SNIPPETS: dict[str, str] = {}


def _pin_cmd(group: list[str]) -> list[str]:
    """Pin groups stored as (file, -k, selector) or full argv."""
    if group and group[0].startswith("tests/"):
        argv = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header"] + group
        return argv
    return [sys.executable, "-m", "pytest", "-q", "--no-header"] + group


def _run_and_capture(argv: list[str]) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(argv, cwd=str(_REPO), capture_output=True, text=True, timeout=1800)
    return {
        "argv": argv[1:],
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "tail": "\n".join((proc.stdout or "").strip().splitlines()[-5:]),
        "duration_s": round(time.perf_counter() - t0, 1),
    }


def _mutated_source(src: str, m: dict) -> tuple[str, int]:
    find = m["find"]
    n = src.count(find)
    if n != 1:
        raise ValueError(f"{m['id']}: anchor snippet occurs {n} times (must be exactly 1) in {m['file']}")
    return src.replace(find, m["replace"], 1), n


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    targets = sorted({m["file"] for m in MUTANTS})
    frozen = {f: Path(_REPO / f).read_text() for f in targets}
    report_entries: list[dict] = []
    try:
        # 0) pristine sanity run per pin group (a pin failing pristine means
        #    the battery is misconfigured, NOT a strong referee).
        sanity: dict[str, dict] = {}
        for group_key, group in (("round2", PIN_ROUND2), ("hawkes", PIN_HAWKES),
                                 ("hawkes_sharp", PIN_HAWKES_SHARP)):
            argv = _pin_cmd(group)
            sanity[group_key] = _run_and_capture(argv)
            print(f"pristine sanity [{group_key}]: passed={sanity[group_key]['passed']}", flush=True)
        if not all(s["passed"] for s in sanity.values()):
            for f_orig, content_orig in frozen.items():  # never leave drift behind
                Path(_REPO / f_orig).write_text(content_orig)
            Path(_REPO / "benchmarks" / "mutation_battery_report.json").write_text(json.dumps(
                {"timestamp_utc": started, "status": "CONFIG_ERROR",
                 "reason": "pristine pin run failed - battery misconfigured, NOT a referee failure",
                 "sanity": sanity, "mutants": "not run"}, indent=2))
            print("BATTERY ABORTED: pristine sanity failed (misconfigured pin group)", flush=True)
            return 2
        # 1) mutants: apply -> pin -> restore -> verify restoration
        for m in MUTANTS:
            path = Path(_REPO / m["file"])
            src = frozen[m["file"]]
            mutated, _occ = _mutated_source(src, m)
            path.write_text(mutated)
            results = []
            killed = False
            try:
                for group in m["pins"]:
                    argv = _pin_cmd(group)
                    res = _run_and_capture(argv)
                    results.append(res)
                    if not res["passed"]:
                        killed = True  # a failing pin under the mutant = killed
            finally:
                path.write_text(src)
                restored = path.read_text() == src
            entry = {
                "id": m["id"], "target": m["target"], "file": m["file"],
                "description": m["description"], "pin_note": m["pin_note"],
                "pin_commands": [r["argv"] for r in results],
                "killed": killed,
                "all_pins_yield": killed,
                "restoration_byte_identical": restored,
                "duration_s": sum(r["duration_s"] for r in results),
                "pin_tails": [{**{k: r[k] for k in ("returncode", "passed", "duration_s")}, "tail": r["tail"]} for r in results],
            }
            report_entries.append(entry)
            print(f"{m['id']} [{m['target']}]: killed={killed} restored={restored}", flush=True)
    finally:
        # hard invariant: never leave a mutant behind
        for f, content in frozen.items():
            p = Path(_REPO / f)
            if p.exists() and p.read_text() != content:
                p.write_text(content)

    all_killed = all(e["killed"] for e in report_entries)
    all_restored = all(e["restoration_byte_identical"] for e in report_entries)
    report = {
        "timestamp_utc": started,
        "prereg_target": "REG-MUT (docs/SYSTEM_FLAWS_AUDIT.md Outstanding requirements register)",
        "status": "ALL_KILLED" if (all_killed and all_restored) else "FAILED",
        "summary": {
            "n_mutants": len(report_entries),
            "n_killed": sum(1 for e in report_entries if e["killed"]),
            "all_pinned_mutants_killed": all_killed,
            "all_restorations_byte_identical": all_restored,
        },
        "sanity": sanity,
        "mutants": report_entries,
    }
    report_path = _REPO / "benchmarks" / "mutation_battery_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    n_alive = [e["id"] for e in report_entries if not e["killed"]]
    if n_alive:
        print(f"SURVIVED MUTANTS (REG-MUT failure): {n_alive}", flush=True)
    else:
        print("ALL CONFIGURED MUTANTS KILLED", flush=True)
    return 0 if (all_killed and all_restored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
