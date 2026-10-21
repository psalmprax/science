"""Registered pins for micro-sample adjudication fixes (defect-1/defect-2). audit-only.

Pins the outcome of `benchmarks/rime_microsample_adjudication_results.json`:
  - defect-1: `sound_monotonicity_v2` replaces the 1e-9 root-tolerance filter
    of the external `sound_continuous_monotonicity_proof`. The regression
    world is the counterexample that engine certified as monotonic:
    q(p) = p^3/3 - 2.5e-10 * p^2, whose derivative p*(p - 5e-10) is
    negative on (0, 5e-10) — the old filter dropped both positive roots.
  - defect-2: `classify_referee_decision` distinguishes ABSTAIN (empty
    packet) from REJECT (gates failed), so a vacuous `all([])` can never
    classify an abstention as a dynamics-inversion rejection.
  - Measured (previously prose) claims: Michaelis-Menten N=12 multi-scale
    origin LOOCV admissibility is 12/12 under the exact family certificate
    q'(p) = w*s/(s+p)^2 > 0; the Langmuir 8-scale trace pins RMSE 0.404810
    vs continuous 0.408720 and Wilcoxon W=22.0, p=0.3652.

Tolerance bands are set from the registered run (deterministic seeds); a
failure means the registered outcome moved and the study must be re-run
rather than patched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

_REPO = str(Path(__file__).resolve().parents[1])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmarks.run_rime_microsample_adjudication import (
    classify_referee_decision,
    certify_origin_constrained_rime,
    run_fold_level_rime_recovery,
    run_langmuir_1918,
    run_michaelis_menten,
    sound_monotonicity_v2,
)

_P = sp.Symbol("p", positive=True, real=True)


# ── defect-1: no absolute-tolerance root filtering ────────────────────────
def test_counterexample_with_micro_root_is_violated() -> None:
    """External engine certified this PROVEN_MONOTONIC; the fixed engine
    must return VIOLATED: dN/dp = p*(p - 5e-10) < 0 on (0, 5e-10)."""
    expr = _P**3 / 3 - 2.5e-10 * _P**2
    verdict, why = sound_monotonicity_v2(expr, _P)
    assert verdict == "VIOLATED", why


def test_origin_constrained_family_is_certified_by_engine_and_exact_proof() -> None:
    w, s = 39.33, 6.52
    v_gen, why_gen = sound_monotonicity_v2(w * (_P / s) / (1 + _P / s), _P)
    assert v_gen == "CERTIFIED_MONOTONIC", why_gen
    v_fam, why_fam = certify_origin_constrained_rime(w, s)
    assert v_fam == "CERTIFIED_MONOTONIC", why_fam


def test_family_certificate_rejects_negative_scale() -> None:
    verdict, _ = certify_origin_constrained_rime(w=-64.15, s=6.52)
    assert verdict == "VIOLATED"


def test_general_engine_matrix_agrees_with_external_verdicts() -> None:
    """External matrix cases where that engine did not mis-certify."""
    cases = [
        (-0.05 * _P**2 + 2.0 * _P + 1.0, "VIOLATED"),  # N = -0.1p + 2; root 20; N < 0 on (20, oo)
        (10.0 / (1 + 0.5 * _P), "VIOLATED"),           # constant negative numerator
        (1.0 / (_P - 2.0), "VIOLATED"),                # pole at p = 2
        (sp.sin(_P) + _P, "UNRESOLVED"),               # non-algebraic -> abstain, never sample
    ]
    for expr, expected in cases:
        verdict, _ = sound_monotonicity_v2(expr, _P)
        assert verdict == expected, f"case {expr}: got {verdict}"


def test_positive_quadratic_monotone_family_still_certifies() -> None:
    verdict, _ = sound_monotonicity_v2(0.05 * _P**2 + 2.0 * _P + 1.0, _P)
    assert verdict == "CERTIFIED_MONOTONIC"


# ── defect-2: empty packets cannot be classified as rejections ────────────
def _cap_gate(pr):
    return (pr["scale"] > 0), f"scale {pr['scale']:+.4f}"


def test_empty_packet_is_abstain_not_reject() -> None:
    decision, invariants, why = classify_referee_decision([], {"cap": _cap_gate})
    assert decision == "ABSTAIN_NO_HYPOTHESIS_GENERATED", why
    assert invariants == []


def test_negative_scale_packet_is_rejected() -> None:
    decision, _, _ = classify_referee_decision(
        [{"scale": -3.04}, {"scale": -69.65}], {"cap": _cap_gate}
    )
    assert decision == "REJECTED"


def test_positive_scale_packet_is_promoted() -> None:
    decision, _, why = classify_referee_decision([{"scale": 37.71}], {"cap": _cap_gate})
    assert decision == "PROMOTED", why


def test_mixed_packet_is_rejected_not_promoted() -> None:
    decision, _, _ = classify_referee_decision(
        [{"scale": 37.71}, {"scale": -3.04}], {"cap": _cap_gate}
    )
    assert decision == "REJECTED"


# ── registered-run pins ──────────────────────────────────────────────────
def test_michaelis_menten_12fold_unanimity_pinned() -> None:
    res = run_michaelis_menten()
    assert res["n"] == 12
    admissible = sum(bool(f["admissible"]) for f in res["folds"])
    assert admissible == 12, res["per_gate_passes"]
    assert res["unanimity"] is True
    assert 0.10 <= res["loocv_rmse"] <= 0.30, res["loocv_rmse"]


# ── fold-level RimeProposer recovery/abstention record ──────────────────
def test_fold_level_bank_restricted_pins() -> None:
    """RimeProposer inside each LOOCV fold over the Python-built sat_m bank:
    the v5-claimed per-fold result must hold -- RMSE 0.482436, selections
    sat_0.8 x1 + sat_0.7 x1 + sat_0.6 x2 + sat_0.66 x7, zero abstentions."""
    res = run_fold_level_rime_recovery()
    bank = res["bank_restricted"]
    assert bank["abstentions"] == 0
    assert abs(bank["loocv_rmse"] - 0.482436) < 2e-3, bank["loocv_rmse"]
    assert bank["recovery"]["sat_family_top_rank_folds"] == 11
    # per-fold anchor from the frozen run
    fold0 = bank["folds"][0]
    assert abs(fold0["error"] - (-0.728939)) < 1e-3
    assert abs(fold0["selected_scale_m"] - 0.8) < 1e-9
    counts = bank["selected_scale_m_counts"]
    assert counts.get("m=0.66", 0) == 7
    assert counts.get("m=0.6", 0) == 2
    assert counts.get("m=0.8", 0) == 1
    assert counts.get("m=0.7", 0) == 1
    # every fold must record BOTH outcomes explicitly (recovery/abstention)
    assert all("partial_gate_decision" in f for f in bank["folds"])


def test_fold_level_unrestricted_grammar_pins() -> None:
    """Unrestricted declared grammar (kernel-chosen scales): the kernel
    re-selects a saturation family in 11/11 folds autonomously; LOOCV RMSE
    0.629997 is the measured cost of kernel-scale freedom; zero abstentions."""
    res = run_fold_level_rime_recovery()
    gram = res["unrestricted_grammar"]
    assert gram["abstentions"] == 0
    assert abs(gram["loocv_rmse"] - 0.629997) < 5e-3, gram["loocv_rmse"]
    assert gram["recovery"]["sat_family_top_rank_folds"] == 11
    fams = {f["family"] for f in gram["folds"]}
    assert all(f.startswith("sat") for f in fams), fams


def test_fold_level_disclosed_limit_present() -> None:
    """The artifact must keep the two modes explicitly separated (the
    bank-restricted experiment is NOT unrestricted structural discovery)."""
    res = run_fold_level_rime_recovery()
    assert "NOT the same experiment" in res["disclosed_limit"]


def test_langmuir_8scale_trace_pinned() -> None:
    res = run_langmuir_1918()
    assert abs(res["loocv_rmse_8scale"] - 0.404810) < 1e-4
    assert abs(res["loocv_rmse_continuous"] - 0.408720) < 2e-3
    wilc = res["wilcoxon"]
    assert abs(wilc["W"] - 22.0) < 1e-6
    assert abs(wilc["p"] - 0.3652) < 5e-4
    assert res["unanimity_certified_by_exact_family_proof"] is True
