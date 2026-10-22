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
  4. 2026-10-21 external audit remediations (round 3, tests/
     test_dyn_external_audit_2026_10_21_pins.py):
       MUT-D1  S3 conditional-lock 412 silently wins (FLAW-03)
       MUT-D2  S3 parameter-rejection path skips the fallback put (FLAW-03)
       MUT-D3  CBF endpoint-headroom deficit split -> equal split (FLAW-17)
       MUT-D4  re-anchor residual recovery dropped (FLAW-08)
  5. 2026-10-21 rust-kernel remediations (cargo lib pins; cargo recompiles
     from the mutated source, so the pins see the mutation without touching
     the user's installed maturin extension):
       MUT-E1   Fornberg delta-sign defect reintroduced (FLAW-04)
       MUT-E2   uniformity-gate routing disabled (FLAW-04)
       MUT-E3   pole flag dropped, 2nd-order rollout branch (FLAW-06a;
                killed by the Bessel-J0 2nd-order pin)
       MUT-E3b  pole flag dropped, 1st-order rollout branch (FLAW-06a;
                killed by the max_diff_order=1 parabola pin)
       MUT-E4   O(1)-constant column back in the exact-RMSE ceiling scale
                (FLAW-12)
       MUT-E6   correlation-space eigensolve -> raw multi-stratum Gram
                (FLAW-15 v2)
  6. EHI pillar + dispatch surfaces (tests/test_dyn_ehi_phase5_pins.py,
     tests/test_dyn_discovery_dispatch_wiring.py):
       MUT-F1  selector fabrication guard disabled (competitive misfit could
               fabricate a decisive class route — flat-field pin)
       MUT-F2  dispatch conflict overridden instead of abstained (CONFLICT)
       MUT-F3  selector ambiguity demoted to a decisive ROUTED status
       MUT-F4  planned-action governance re-verification dropped
       MUT-F5  controller loops acquisition without the referee re-adjudication
       MUT-F6  per-family engine failure swallowed instead of recorded
  7. DE-discovery arm surfaces (tests/test_dyn_de_arm_p1_pins.py):
       MUT-G1  screen records empty exclusions (the Level-1 pass vanishes
               from the referee-facing artifact instead of ordering it)
       MUT-G2  parent ranking inverted (ascending rather than descending)
       MUT-G3  AD product rule replaced by the naive d(fg) = f'·g' rule
     Neutralized mutant NOT registered (honesty trail): MUT-E5 candidate,
     the absolute-tolerance gate revert — after the v2 well-scaled
     eigensolve (corr-space congruence + jack-of-all-strata spectral
     normalization) it grew no natural kill on any probe fixture (corr
     spectra sit at O(0.5) for both O(1) and O(1e6) amplitudes), i.e. the
     defect expression it targets is dissolved by the v2 fix; the
     corr-space solve itself is mutation-protected by MUT-E6.

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
# EHI pillar pins (selector fabrication guard, constraint states, information
# orders, fifth-pillar controller) and the dispatch-wiring pins: each MUT-F*
# pillar mutant below must be killed (pin MUST fail under the mutation) by
# these suites.
PIN_EHI_PILLARS = ["tests/test_dyn_ehi_phase5_pins.py"]
PIN_DISPATCH_WIRING = ["tests/test_dyn_discovery_dispatch_wiring.py"]
# DE-discovery arm pins (typed semantics, shared AD service, Level-1 screen
# with the mandatory dense escape hatch): each MUT-G* arm mutant below must
# be killed (pin MUST fail under the mutation) by this suite.
PIN_DE_ARM = ["tests/test_dyn_de_arm_p1_pins.py"]
# Rust-kernel mutants are pinned by cargo lib tests (cargo recompiles from the
# mutated source, so the pins see the mutation without touching the user's
# installed maturin extension).
CARGO_LIB_PIN = ["cargo", "test", "--lib", "-q", "--manifest-path", "rust/apedp-kernel/Cargo.toml"]

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
    # ── EHI pillar + dispatch mutants (tests/test_dyn_ehi_phase5_pins.py,
    # tests/test_dyn_discovery_dispatch_wiring.py) — the pillar surfaces" own
    # discipline, mutation-protected like the referee paths: ──
    {
        "id": "MUT-F1",
        "target": "selector fabrication guard (EHI Phase 1)",
        "file": "apedp_dyn/orchestration/model_class_selector.py",
        "description": "disable the guard that refuses decisive routing when the winner's defining diagnostic never fired (pure noise / flat fields would fabricate a class claim via competitors' misfit)",
        "find": "    if lacks_defining[top_cls]:",
        "replace": "    if False:",
        "pins": [PIN_EHI_PILLARS],
        "pin_note": "test_phase1_fabrication_guard_flat_field_grid",
    },
    {
        "id": "MUT-F2",
        "target": "dispatch conflict discipline (EHI dispatch wiring)",
        "file": "apedp_dyn/orchestration/dispatch.py",
        "description": "a decisive selector class the sieve masks gets dispatched anyway (override instead of a conflict abstention)",
        "find": "            chosen_engines=(), masked_engines=tuple(verdict.masked_engine_families),\n            status=\"CONFLICT\",\n            reason=(f\"selector decisively proposes",
        "replace": "            chosen_engines=(selector_family,), masked_engines=tuple(verdict.masked_engine_families),\n            status=\"CONFLICT\",\n            reason=(f\"selector decisively proposes",
        "pins": [PIN_DISPATCH_WIRING],
        "pin_note": "test_dispatch_conflict_is_abstention_not_override",
    },
    {
        "id": "MUT-F3",
        "target": "dispatch ambiguity visibility (EHI dispatch wiring)",
        "file": "apedp_dyn/orchestration/dispatch.py",
        "description": "selector ambiguity silently demoted to a decisive ROUTED status (hidden default route)",
        "find": "            status=(\"MULTIPLE_ROUTES\" if verdict.admissible_engine_families\n                    else \"UNROUTABLE\"),",
        "replace": "            status=\"ROUTED\",",
        "pins": [PIN_DISPATCH_WIRING],
        "pin_note": "test_dispatch_selector_ambiguity_keeps_admitted_families_live",
    },
    {
        "id": "MUT-F4",
        "target": "controller governance separation (EHI fifth pillar)",
        "file": "apedp_dyn/epistemic/research_controller.py",
        "description": "drop the planned-action governance re-verification (in-session referee tamper no longer aborts the run)",
        "find": "        if not self.verify_governance():\n            raise RuntimeError(",
        "replace": "        if False:\n            raise RuntimeError(",
        "pins": [PIN_EHI_PILLARS],
        "pin_note": "test_phase5_governance_state_never_shadow_adjudication",
    },
    {
        "id": "MUT-F5",
        "target": "controller self-adjudication ban (EHI fifth pillar)",
        "file": "apedp_dyn/epistemic/research_controller.py",
        "description": "continue looping instead of breaking after acquisition (the controller re-designs and re-executes its own orders without the external re-adjudication it exists to require)",
        "find": "                    # self-adjudicates its own order. The run ends here and\n                    # reports evidence-integrated-awaiting-external-verdict.\n                    break",
        "replace": "                    # self-adjudicates its own order. The run ends here and\n                    # reports evidence-integrated-awaiting-external-verdict.\n                    continue",
        "pins": [PIN_EHI_PILLARS],
        "pin_note": "test_phase5_controller_langmuir_loop_end_to_end (exactly one acquisition per session)",
    },
    {
        "id": "MUT-F6",
        "target": "dispatch failure honesty (EHI dispatch wiring)",
        "file": "apedp_dyn/orchestration/dispatch.py",
        "description": "swallow the per-family engine failure (a broken engine would be laundered into silence instead of a recorded error)",
        "find": "                errors[family] = f\"{type(exc).__name__}: {exc}\"",
        "replace": "                pass  # MUTANT: engine failure swallowed",
        "pins": [PIN_DISPATCH_WIRING],
        "pin_note": "test_dispatch_records_engine_failure_without_laundering",
    },
    # ── DE-discovery arm mutants (tests/test_dyn_de_arm_p1_pins.py) — the
    # arm's own discipline, mutation-protected like the referee paths: ──
    {
        "id": "MUT-G1",
        "target": "screen exclusion honesty (DE arm Level-1)",
        "file": "apedp_dyn/de_arm/dependency_screen.py",
        "description": "the screen records empty exclusions (the Level-1 pass stops ordering what it pruned, hiding the escape set from the referee)",
        "find": "            excluded=tuple(p for p, _ in scored[k:]),",
        "replace": "            excluded=(),",
        "pins": [PIN_DE_ARM],
        "pin_note": "test_screening_ranks_direct_parent_first_and_records_exclusions",
    },
    {
        "id": "MUT-G2",
        "target": "screen ranking direction (DE arm Level-1)",
        "file": "apedp_dyn/de_arm/dependency_screen.py",
        "description": "parent ranking inverted to ascending (the weakest predictor becomes the kept parent)",
        "find": "        scored.sort(key=lambda kv: (-kv[1], kv[0]))",
        "replace": "        scored.sort(key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0))",
        "pins": [PIN_DE_ARM],
        "pin_note": "test_screening_ranks_direct_parent_first_and_records_exclusions",
    },
    {
        "id": "MUT-G3",
        "target": "AD product rule (DE arm shared AD service)",
        "file": "apedp_dyn/de_arm/ad_service.py",
        "description": "naive differentiating rule (d(fg) = f'·g') replaces the exact product rule",
        "find": "        return Add(Mul(_diff(e.left, var), e.right), Mul(e.left, _diff(e.right, var)))",
        "replace": "        return Add(_diff(e.left, var), _diff(e.right, var))",
        "pins": [PIN_DE_ARM],
        "pin_note": "test_ad_service_exact_rules (multivar product/quotient expression)",
    },
    # ── 2026-10-21 external audit rust-kernel remediation pins (cargo lib) ──
    {
        "id": "MUT-E1",
        "target": "Fornberg weights recurrence (FLAW-04)",
        "file": "rust/apedp-kernel/src/rime/diff_pade.rs",
        "description": "reintroduce the delta-sign defect (weights stop annihilating constants)",
        "find": "                let term1 = (x_nodes[i] - x0) * c[k][i - 1][j];",
        "replace": "                let term1 = (x0 - x_nodes[i]) * c[k][i - 1][j];",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_nonuniform_grid_derivatives_use_fornberg_fallback",
    },
    {
        "id": "MUT-E2",
        "target": "uniform-grid gate (FLAW-04 routing)",
        "file": "rust/apedp-kernel/src/rime/diff_pade.rs",
        "description": "disable the uniformity gate (non-uniform grids hit the wrong 4th-order stencil)",
        "find": "    if !is_uniform_grid(x) {\n        return derivatives_nonuniform(x, y, 1);\n    }",
        "replace": "    if false {\n        return derivatives_nonuniform(x, y, 1);\n    }",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_nonuniform_grid_derivatives_use_fornberg_fallback",
    },
    {
        "id": "MUT-E3",
        "target": "pole mask on the delta-Pade rollout (FLAW-06a)",
        "file": "rust/apedp-kernel/src/rime/diff_pade.rs",
        "description": "drop the pole_hit flag at the 2nd-order branch (silently integrating through the singularity again)",
        "find": "            if s_val.abs() > 1e-10 {\n                (-p_val * y2_v - q_val * y1_v) / s_val\n            } else {\n                pole_hit.set(true);",  
        "replace": "            if s_val.abs() > 1e-10 {\n                (-p_val * y2_v - q_val * y1_v) / s_val\n            } else {\n                let _ = &pole_hit;",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_pole_masked_second_order_is_not_exact_recovery (pole_singularity_hit assert)",
    },
    {
        "id": "MUT-E3b",
        "target": "pole mask on the delta-Pade 1st-order branch (FLAW-06a)",
        "file": "rust/apedp-kernel/src/rime/diff_pade.rs",
        "description": "drop the pole_hit flag at the 1st-order branch (the branch the parabola fixture actually selects)",
        "find": "            if p_val.abs() > 1e-10 {\n                (-q_val * yv - r_val * yv * yv) / p_val\n            } else {\n                pole_hit.set(true);",
        "replace": "            if p_val.abs() > 1e-10 {\n                (-q_val * yv - r_val * yv * yv) / p_val\n            } else {\n                let _ = &pole_hit;",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_pole_masked_second_order_is_not_exact_recovery (pole_singularity_hit assert)",
    },
    {
        "id": "MUT-E4",
        "target": "exact-RMSE ceiling over active non-constant columns (FLAW-12)",
        "file": "rust/apedp-kernel/src/rime/implicit.rs",
        "description": "include the O(1) constant column again (scale-awareness destroyed)",
        "find": "        if names[j] == \"1\" || sparse_c[j] == 0.0 {\n            continue;\n        }",
        "replace": "        if false {\n            continue;\n        }",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_near_zero_scale_misfit_not_certified + test_large_scale_exact_relation_certifiable",
    },
    # MUT-E5 note (2026-10-21, honesty trail): an absolute-tolerance gate
        # reverted over the CORRELATION-space spectrum was registered as a
        # candidate mutant and GREW no kill on any natural fixture — the v2
        # well-scaled eigensolve removed the absolute/relative failure-mode
        # difference the v1 defect expressed (corr spectra probe: structurally
        # nonzero eigenvalues sit at O(0.5) for both O(1) and O(1e6) ODE
        # amplitudes; nothing natural lands in the <=1e-8 band). The v1
        # failure-mode itself is still mutation-protected by MUT-E6 (the
        # corr-space eigensolve itself). Registered mutations must kill; the
        # neutralized mutant is documented here instead of being shipped.
    {
        "id": "MUT-E6",
        "target": "correlation-space eigensolve (FLAW-15 v2)",
        "file": "rust/apedp-kernel/src/rime/lie_symmetry.rs",
        "description": "solve the RAW multi-stratum Gram again (extreme strata corrupt the nullspace)",
        "find": "            corr[i * k + j] = if div > 0.0 { gram[i * k + j] / div } else { 0.0 };",
        "replace": "            corr[i * k + j] = gram[i * k + j];",
        "pins": [CARGO_LIB_PIN],
        "pin_note": "test_scale_dominant_ode_keeps_genuine_symmetries",
    },
]

_FROZEN_SNIPPETS: dict[str, str] = {}


def _pin_cmd(group: list[str]) -> list[str]:
    """Pin groups stored as (file, -k, selector) for pytest, or as LITERAL
    argv (first token is an executable, e.g. cargo) run from the repo root."""
    if group and group[0].startswith("tests/"):
        argv = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", *group]
        return argv
    if group and group[0] == "cargo":
        return list(group)
    return [sys.executable, "-m", "pytest", "-q", "--no-header", *group]


def _run_and_capture(argv: list[str]) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(  # noqa: S603 - subprocess argv is fixed, non-user-controlled (battery pin groups are code constants)
        argv, cwd=str(_REPO), capture_output=True, text=True, timeout=1800)
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
                                 ("hawkes_sharp", PIN_HAWKES_SHARP),
                                 ("ehi_pillars", PIN_EHI_PILLARS),
                                 ("dispatch_wiring", PIN_DISPATCH_WIRING),
                                 ("de_arm", PIN_DE_ARM)):
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
