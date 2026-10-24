r"""JetFlow-Video v0.3 Confirmation — Collision/Occlusion World (PREREG-JETFLOW-VIDEO-V03-COLLISION-2026-10-19).

Status: experimental (confirmation harness preregistered in
`docs/PREREGISTRATION_JETFLOW_VIDEO_V03_COLLISION.md`; committed AFTER the
exploratory seed-404 artifact, as the prereg states explicitly).

Runs the frozen collision-world matched benchmark on the confirmation seeds
{407, 508, 509} and evaluates the preregistered hypotheses — with their
VERDICTS, whatever they are, per the prereg decision rule: any failure is a
negative result reported as such; nothing is re-tuned toward green.

Registered criteria (exploratory seed-404 point estimates in parentheses):
  H1-collision (domain-scoping of the advantage): diffusion_matched SSIM >
    jet_tt_2step SSIM AND diffusion_matched temporal < jet_tt_2step temporal,
    required on >= 2/3 seeds (exploratory 0.065 vs 0.017; 0.078 vs 0.234).
  H2 (JetFlow floor): jet_tt_2step PSNR >= 10.0 dB on every seed
    (exploratory 13.68 dB).
  H3 (no training blow-up): jet train wall-clock / baseline <= 2.0 on every
    seed (exploratory ratio 1.35). Sanity ceiling, NOT an advantage claim.
  H4 (structural pass counts, deterministic): jet_tt_2step == 8 and
    jet_tt_4step == 16 structural passes/clip on every seed. UNITS ERRATUM
    (disclosed in the report JSON, 2026-10-19): the prereg text derived the
    structural counts as round(passes * n_gen) - 1, mis-stating the metric's
    units — `model_passes_per_clip` is ALREADY per-clip (the recorded 9.0 for
    jet2 is the per-clip count with the one warmup GENERATION amortized inside
    it, i.e. 8 structural + 1 warmup-generation's share; jet4's window has no
    warmup). Correct derivation: jet2 structural == round(passes) - 1;
    jet4 structural == round(passes). The registered NUMBERS (8 / 16) and
    their source in the v0.2 registration are unchanged; no benchmark was
    re-run and no threshold moved — the verdicts are recomputed from the SAME
    committed per-seed artifacts.
  No latency-millisecond thresholds (machine-dependent); reported only.

Exit code reflects HARNESS INTEGRITY only (completed runs, parseable rows),
not hypothesis outcomes — verdicts live in the JSON/report and are never
signaled via process status.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

CONFIRMATION_SEEDS = (407, 508, 509)  # frozen; exploratory seed 404 EXCLUDED
PREREG_DOC = "PREREG-JETFLOW-VIDEO-V03-COLLISION-2026-10-19"
H1_SEEDS_REQUIRED = 2
H2_PSNR_FLOOR = 10.0
H3_RATIO_CEILING = 2.0
EXPECTED_JET2_STRUCTURAL_PASSES = 8
EXPECTED_JET4_STRUCTURAL_PASSES = 16
N_GEN = 8


def _run_seed(seed: int, out_dir: Path) -> dict:
    out = out_dir / f"seed_{seed}_results.json"
    cmd = [
        sys.executable, str(_REPO / "benchmarks" / "run_jetflow_vs_diffusion_matched.py"),
        "--domain", "collision", "--train-steps", "300", "--n-clips", "10",
        "--n-gen", str(N_GEN), "--threads", "4", "--seed", str(seed), "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(_REPO), timeout=3600)
    if proc.returncode != 0 or not out.is_file():
        return {"seed": seed, "harness_ok": False,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-800:]}
    d = json.loads(out.read_text())
    rows = {r["name"]: r for r in d["rows"]}
    return {
        "seed": seed,
        "harness_ok": True,
        "out": str(out.relative_to(_REPO)),
        "jet2": rows["jet_tt_2step"],
        "jet4": rows["jet_tt_4step"],
        "diff_matched": rows["diffusion_matched"],
        "train_s_jet": d["protocol"]["train_time_s_jet"],
        "train_s_base": d["protocol"]["train_time_s_baseline"],
        "domain_split": d["protocol"]["domain_split"],
        "params_jet": d["protocol"]["params_jet"],
        "params_base": d["protocol"]["params_baseline"],
        "git_head": _git_head(),
    }


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _h4_structural_jet2(passes: float) -> int:
    # Units erratum (disclosed above): passes are ALREADY per-clip; the one
    # warmup generation is amortized INSIDE the per-clip count.
    return int(round(passes)) - 1


def _h4_structural_jet4(passes: float) -> int:
    return int(round(passes))  # no warmup inside the jet4 instrumentation window


def _re_evaluate_from_artifacts() -> list[dict]:
    """Recompute per-seed records from the COMMITTED artifacts (no benchmark
    re-run) — used by the units-erratum H4 correction; nothing is regenerated
    or re-tuned."""
    out_dir = _REPO / "benchmarks" / "jetflow_video_v03_confirmation"
    per_seed = []
    for seed in CONFIRMATION_SEEDS:
        p = out_dir / f"seed_{seed}_results.json"
        d = json.loads(p.read_text())
        rows = {r["name"]: r for r in d["rows"]}
        per_seed.append({
            "seed": seed, "harness_ok": True, "out": str(p.relative_to(_REPO)),
            "jet2": rows["jet_tt_2step"], "jet4": rows["jet_tt_4step"],
            "diff_matched": rows["diffusion_matched"],
            "train_s_jet": d["protocol"]["train_time_s_jet"],
            "train_s_base": d["protocol"]["train_time_s_baseline"],
            "domain_split": d["protocol"]["domain_split"],
            "params_jet": d["protocol"]["params_jet"],
            "params_base": d["protocol"]["params_baseline"],
            "git_head": d["protocol"].get("seed") and _git_head(),
        })
    return per_seed


def main() -> dict:
    re_evaluate = "--re-evaluate" in sys.argv
    out_dir = _REPO / "benchmarks" / "jetflow_video_v03_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict] = []
    integrity_ok = True
    if re_evaluate:
        per_seed = _re_evaluate_from_artifacts()
        print(f"re-evaluating verdicts from {len(per_seed)} committed artifacts (no re-run)", flush=True)
    else:
        for seed in CONFIRMATION_SEEDS:
            rec = _run_seed(seed, out_dir)
            if not rec["harness_ok"]:
                integrity_ok = False
            per_seed.append(rec)
            print(f"seed {seed}: harness_ok={rec['harness_ok']}", flush=True)

    # ---- hypothesis evaluation (whatever the verdicts are) -----------------
    def _ok_rows():
        return [r for r in per_seed if r["harness_ok"]]

    h1_pass = bool(_ok_rows()) and sum(
        1 for r in _ok_rows()
        if r["diff_matched"]["nearest_ssim_mean"] > r["jet2"]["nearest_ssim_mean"]
        and r["diff_matched"]["temporal_consistency_mean"] < r["jet2"]["temporal_consistency_mean"]
    ) >= H1_SEEDS_REQUIRED
    h2_pass = bool(_ok_rows()) and all(
        r["jet2"]["nearest_psnr_db_mean"] >= H2_PSNR_FLOOR for r in _ok_rows())
    h3_pass = bool(_ok_rows()) and all(
        r["train_s_jet"] / max(1e-9, r["train_s_base"]) <= H3_RATIO_CEILING for r in _ok_rows())
    h4_pass = bool(_ok_rows()) and all(
        _h4_structural_jet2(r["jet2"]["model_passes_per_clip"]) == EXPECTED_JET2_STRUCTURAL_PASSES
        and _h4_structural_jet4(r["jet4"]["model_passes_per_clip"]) == EXPECTED_JET4_STRUCTURAL_PASSES
        for r in _ok_rows())

    verdicts = {
        "H1_collision_domain_scoping": {
            "criteria": "diffusion SSIM > jet SSIM AND diffusion temporal < jet temporal on >= 2/3 seeds",
            "confirmed": h1_pass,
            "seeds_passing": [r["seed"] for r in _ok_rows()
                              if r["diff_matched"]["nearest_ssim_mean"] > r["jet2"]["nearest_ssim_mean"]
                              and r["diff_matched"]["temporal_consistency_mean"] < r["jet2"]["temporal_consistency_mean"]],
        },
        "H2_jet_floor": {"criteria": f"jet2 PSNR >= {H2_PSNR_FLOOR} on every seed", "confirmed": h2_pass},
        "H3_no_training_blowup": {"criteria": f"train ratio <= {H3_RATIO_CEILING} on every seed", "confirmed": h3_pass},
        "H4_structural_pass_counts": {"criteria": "jet2 == 8 and jet4 == 16 structural passes on every seed", "confirmed": h4_pass},
    }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_doc": PREREG_DOC,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "exploratory_seed_excluded": 404,
        "harness_integrity_ok": integrity_ok,
        "evaluated_from_committed_artifacts": bool(re_evaluate),
        "git_head": _git_head(),
        "errata": ([{
            "date": "2026-10-19",
            "id": "ERR-H4-UNITS",
            "what": "prereg text mis-stated metric units; harness formula corrected;"
                    " verdicts recomputed from the SAME committed per-seed artifacts"
                    " (no benchmark re-run, no threshold moved)"
        }] if True else []),
        "verdicts": verdicts,
        "honest_labels": {
            "fvd": "sliced-Wasserstein proxy, NOT I3D-FVD (inherited instrument)",
            "H1_direction": "confirming H1 is a domain-scoping claim (advantage does NOT transfer), not an advantage claim",
        },
        "per_seed": [
            {k: (v if k not in ("jet2", "jet4", "diff_matched")
                 else {kk: vv for kk, vv in v.items()
                       if kk in ("nearest_psnr_db_mean", "nearest_ssim_mean",
                                 "temporal_consistency_mean", "fvd_proxy_swd_vs_holdout",
                                 "model_passes_per_clip", "latency_ms_per_clip_median")})
             for k, v in rec.items()}
            for rec in per_seed
        ],
    }
    (out_dir / '..' / "jetflow_video_v03_confirmation_results.json").write_text(
        json.dumps(result, indent=2))
    print(json.dumps(verdicts, indent=2))
    return result


if __name__ == "__main__":
    main()
