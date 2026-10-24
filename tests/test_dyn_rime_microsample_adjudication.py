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
    _NamespaceProposal,
    adjudicate_structure_aware,
    certify_origin_constrained_rime,
    classify_referee_decision,
    run_fold_level_rime_recovery,
    run_graph_validated_fold_referee,
    run_langmuir_1918,
    run_michaelis_menten,
    run_michaelis_menten_graph_validated,
    run_structure_aware_referee,
    sound_monotonicity_v2,
    validate_kernel_graph_structure,
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


# ── structure-aware referee (defect-3 patch): structure beats proxies ────
def test_counterfeit_quadratic_is_rejected_on_structure() -> None:
    """The external referee PROMOTEd this exact candidate (w>0, |b|=0.5,
    positive data correlation). The patched referee rejects it on declared
    structure: no saturating form, no finite positive limit."""
    verdict, reports, why = adjudicate_structure_aware([
        {"proposal": _NamespaceProposal("+5.0 * (sq(p)) +0.5", 5.0, 0.5), "s": None}
    ])
    assert verdict == "REJECTED", why
    assert "non-saturating structure" in reports[0]["rejected"]


def test_verified_sat_norm_discovery_is_promoted_with_disclosed_origin() -> None:
    """The tournament's verified winner (w=37.7147, b=3.97006) must be
    PROMOTED through the exact family certificate (monotonic for ANY
    intercept), with its origin deviation q(0)=b disclosed via
    origin_strict_satisfied=False — never gated by a |b| tolerance."""
    import numpy as np
    s_dev = float(np.std([34.0, 23.8, 17.3, 13.0, 9.5, 7.4, 6.1, 5.0, 4.0, 3.4, 2.8]))
    verdict, reports, why = adjudicate_structure_aware([
        {"proposal": _NamespaceProposal(
            "+37.7146770 * (sat_norm(p)) +3.9700646",
            37.71467702437663, 3.970064638094019), "s": s_dev}
    ])
    assert verdict == "PROMOTED", why
    rep = reports[0]
    assert all(rep["gates"].values())
    assert rep["origin_strict_satisfied"] is False
    assert abs(rep["q0"] - 3.9700646) < 1e-5
    assert abs(rep["q_inf"] - 41.6847) < 5e-3
    assert rep["monotonicity"] == "CERTIFIED_MONOTONIC"


def test_zero_intercept_baseline_is_strict_origin_promoted() -> None:
    verdict, reports, _ = adjudicate_structure_aware([
        {"proposal": _NamespaceProposal("+39.33 * (sat_norm(p)) +0", 39.33, 0.0), "s": 6.52}
    ])
    assert verdict == "PROMOTED"
    assert reports[0]["origin_strict_satisfied"] is True
    assert reports[0]["q0"] == 0.0


def test_negative_scale_saturation_is_inversion_rejected() -> None:
    verdict, reports, _ = adjudicate_structure_aware([
        {"proposal": _NamespaceProposal("-64.15 * (sat_norm(t)) +47.44", -64.15, 47.44),
         "s": 6.52}
    ])
    assert verdict == "REJECTED_QUALITATIVE_DYNAMICS_INVERSION"
    assert "capacity_positive" in reports[0]["rejected"]


def test_structure_aware_elevenfold_record_pinned() -> None:
    """Full frozen-path check: the 11 bank-restricted fold proposals pass the
    structure-aware gates 11/11 as verified structures, each with its origin
    deviation disclosed (0/11 satisfy strict q(0)=0) — the honest joint claim
    the reviewer asked to be observable from one artifact."""
    res = run_structure_aware_referee()
    cases = res["cases"]
    assert cases["counterfeit_quadratic"]["verdict"] == "REJECTED"
    assert cases["discovery_sat_norm_winner"]["verdict"] == "PROMOTED"
    assert cases["negative_scale_saturation"]["verdict"] == "REJECTED_QUALITATIVE_DYNAMICS_INVERSION"
    assert cases["zero_intercept_baseline"]["verdict"] == "PROMOTED"
    ef = res["elevenfold"]
    assert ef["verdict"] == "PROMOTED"
    assert ef["promoted"] == 11 and ef["total"] == 11
    assert ef["origin_strict_satisfied_count"] == 0


# ── graph-equation structural validation (reviewer milestone) ───────────
def test_graph_validated_folds_pinned() -> None:
    """The 11 bank-restricted folds through kernel-object referee: every fold
    validates on kernel metadata (graph_id/equation/single input) AND the
    executable affine-readout check (compute_score == w*col+b), then all
    gates; expected 11/11 PROMOTED with the affine readout exact."""
    res = run_graph_validated_fold_referee()
    assert res["abstentions"] == 0
    assert res["promoted"] == 11 and res["structure_validated_folds"] == 11
    assert res["max_affine_readout_dev"] < 1e-12, res["max_affine_readout_dev"]
    for f in res["folds"]:
        assert f["graph_checks"]["validated"] is True
        assert f["graph_id"] and f["kernel_equation"]
        assert f["verdict"] == "PROMOTED"
        assert f["origin_strict_satisfied"] is False  # b != 0 per fold


def test_kernel_graph_structure_tournament_winner_transform_mode() -> None:
    """Transform-mode validation: the full-grammar tournament winner declared
    as sat_norm(p) over base feature p must validate via kernel metadata and
    reproduce the fitted feature through _apply_transform with the fit-time
    scale, with the affine readout verified to machine precision."""
    import numpy as np
    import pandas as pd

    from apedp_dyn.proposers.rime_proposer import RimeProposer

    df = pd.DataFrame({
        "p": [34.0, 23.8, 17.3, 13.0, 9.5, 7.4, 6.1, 5.0, 4.0, 3.4, 2.8],
        "q": [33.0, 30.8, 28.2, 25.5, 23.9, 21.6, 19.0, 17.0, 15.1, 13.4, 12.0],
    })
    prop = RimeProposer(
        feature_cols=["p"], target_col="q", static_mode=True,
        algebraic_transforms=["sq", "ssqrt", "slog", "sat_norm"],
        min_rows=5, max_proposals=1, seed=42,
    ).fit_and_propose(df)[0]
    s_est = float(np.nanstd(df["p"].to_numpy(dtype=float)))
    checks = validate_kernel_graph_structure(
        prop, df_train=df, transform=("sat_norm", s_est)
    )
    assert checks["validated"] is True, checks
    assert checks["structure_source"] == "kernel_transform"
    assert checks["affine_readout_max_abs_dev"] < 1e-9


def test_graph_validation_rejects_tampered_affine_readout() -> None:
    """The validator must have teeth: metadata checks passing is NOT sufficient.
    A proposal whose compute_score deviates from the affine readout w*col+b
    fails exactly on affine_readout_verified while every metadata check holds."""
    import pandas as pd

    class Tampered:
        def __init__(self):
            # exact kernel-emitted shapes (bank mode): equation IS the column
            self.formula = "-3.03529 * (sat_0.66) +29.9478"
            self.declared_nominal_dependencies = ("sat_0.66",)
            self.parameters = {"scale": -3.03529, "intercept": 29.9478}
            self._meta = {
                "graph_id": "tampered_graph_id",
                "equation": "sat_0.66",
                "inputs_used_names": ["sat_0.66"],
                "inputs_used_idx": [0],
                "algebraic_transforms": None,
            }

        @property
        def metadata(self):
            return self._meta

        def compute_score(self, df):
            col = df["sat_0.66"].to_numpy(dtype=float)
            return self.parameters["scale"] * col + self.parameters["intercept"] + 1e-3

    df_train = pd.DataFrame({
        "p": [34.0, 23.8, 17.3, 13.0, 9.5, 7.4, 6.1, 5.0, 4.0, 3.4],
    })
    s_m = 0.66 * float(df_train["p"].std())
    df_train["sat_0.66"] = (df_train["p"] / s_m) / (1.0 + df_train["p"] / s_m)
    checks = validate_kernel_graph_structure(Tampered(), df_train=df_train)
    assert checks["graph_id_present"] and checks["equation_present"]
    assert checks["affine_readout_verified"] is False
    assert checks["validated"] is False
    assert checks["failed_check"] == "affine_readout_verified"


# ── Michaelis-Menten path through the graph-validated referee ───────────
def test_mm_graph_validated_folds_pinned() -> None:
    """The MM system through the SAME graph-validated referee: 12/12 folds
    promoted with 12/12 structure-validation (graph metadata + affine readout
    machine-exact), zero abstentions; actual-RIME LOOCV RMSE 0.1316 is the
    measured per-fold result and BEATS the b≡0 baseline (0.1740) here."""
    res = run_michaelis_menten_graph_validated()
    assert res["abstentions"] == 0
    assert res["promoted"] == 12 and res["structure_validated_folds"] == 12
    assert abs(res["loocv_rmse_actual_rime"] - 0.1316) < 2e-3, res["loocv_rmse_actual_rime"]
    assert res["graph_checks_max_affine_dev"] < 1e-12
    for f in res["folds"]:
        assert f["graph_id"] and f["kernel_equation"]
        assert f["verdict"] == "PROMOTED"
        assert all(f["gates"].values())
        assert f["origin_strict_satisfied"] is False  # b != 0 per fold
    # physical recovery: readout w estimates Vmax = 10 within 5%
    ws = [f["w"] for f in res["folds"]]
    assert abs(sum(ws) / len(ws) - 10.0) < 0.05 * 10.0, sum(ws) / len(ws)


def test_mm_full_tournament_adjudicated_pinned() -> None:
    """The existing 'MM: 6 proposals PROMOTED' table row is reproduced with
    measured semantics: exactly 6 bank proposals, all pass the structure-
    aware gates, saturation family at rank #1."""
    res = run_michaelis_menten_graph_validated()
    t = res["tournament"]
    assert t["n_proposals"] == 6
    assert t["referee_verdict"] == "PROMOTED"
    assert t["admissible"] == 6
    assert t["top_rank_family"] == "sat"


def test_langmuir_8scale_trace_pinned() -> None:
    res = run_langmuir_1918()
    assert abs(res["loocv_rmse_8scale"] - 0.404810) < 1e-4
    assert abs(res["loocv_rmse_continuous"] - 0.408720) < 2e-3
    wilc = res["wilcoxon"]
    assert abs(wilc["W"] - 22.0) < 1e-6
    assert abs(wilc["p"] - 0.3652) < 5e-4
    assert res["unanimity_certified_by_exact_family_proof"] is True
