"""QD replacement-term generation run (design + executable harness).

Goal: beat refit-Brus on held-out data with a physically admissible,
structurally novel equation -- or confirm Brus with teeth.

Phase A (data, curated separately -- see QD_DATA_SPEC below): multi-material
TEM-anchored QD sizing tables. Phase B (this file): GP run driving
``hybrid_discover`` with ambiguity-group-derived template priors, judged by
``bakeoff`` with refit-Brus as an explicit bar competitor plus physics
admissibility post-checks. Nothing here fabricates data: the only
non-tabulated point is the exact bulk limit dE -> 0 as R -> infinity,
labeled as a boundary condition wherever used.

Verdict classes: GENERATION (wins + admissible + novel), CONFIRMATION
(Brus-equivalent wins), INCONCLUSIVE (all else).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BULK_EG = {"CdSe": 1.74, "CdS": 2.42, "CdTe": 1.50}  # eV, textbook bulk gaps
REQUIRED_COLUMNS = ("material", "D_nm", "lambda_peak_nm", "source",
                    "ligand", "solvent", "T_K")

QD_DATA_SPEC = """
Phase A data requirements (curate before the real run):
  SOURCES: Yu et al. 2003 Chem.Mater. (CdS/CdSe/CdTe sizing curves);
    Peng et al. 2001 JACS 123,183 (CdSe kinetics); Murray/Norris/Bawendi
    1993 JACS (CdSe); Rogach et al. aqueous CdTe sets.
  FIELDS: material, D_nm (TEM mean), D_std_nm, lambda_peak_nm, Eg_eV,
    source, ligand, solvent, T_K. Rows without TEM diameters excluded;
    ligand/solvent shifts flagged, not silently pooled.
  COVERAGE: R in [0.7, 5.0] nm radius, N >= 30 across >= 3 chemistries.
  SPLITS: train on CdSe+CdS (+ their bulk anchors), hold out CdTe entirely
    (different chemistry = out-of-family test). Bulk anchor per material
    from BULK_EG at R=100 nm.
  GO/NO-GO for running Phase B: N_train >= 20, Rmax/Rmin >= 4 per material,
    else the [1,6]-nm unidentifiability result says don't bother.
"""


def validate_qd_table(rows: list[dict]) -> dict:
    """Validate curated QD rows against the Phase A schema (no stats faked).

    Returns {'n': int, 'materials': [...], 'errors': [...]}; errors empty
    means the table is runnable.
    """
    errors: list = []
    mats: set = set()
    for i, row in enumerate(rows):
        for col in REQUIRED_COLUMNS:
            if col not in row:
                errors.append(f"row {i}: missing '{col}'")
        try:
            d = float(row.get("D_nm", "nan"))
            lam = float(row.get("lambda_peak_nm", "nan"))
            if not (0.5 < d < 30.0) or not (300.0 < lam < 900.0):
                errors.append(f"row {i}: out-of-range D/lambda")
            mats.add(str(row.get("material", "?")))
        except (TypeError, ValueError):
            errors.append(f"row {i}: non-numeric D/lambda")
    return {"n": len(rows), "materials": sorted(mats), "errors": errors}


def yu_cdse_rows() -> list[dict]:
    """Yu et al. 2003 CdSe sizing curve, densely sampled (derived, labeled).

    Stand-in until Phase A curation lands; every row is traceable to the
    published polynomial, not an independent measurement.
    """
    lam = np.arange(440, 615, 15, dtype=float)
    D = 1.6122e-9 * lam**4 - 2.6575e-6 * lam**3 + 1.6242e-3 * lam**2 - 0.4277 * lam + 41.57
    return [{"material": "CdSe", "D_nm": float(d), "lambda_peak_nm": float(lam_nm),
             "source": "Yu2003-sizing-curve(derived)", "ligand": "TOPO",
             "solvent": "toluene", "T_K": 298.0} for d, lam_nm in zip(D, lam, strict=False)]


def to_arrays(rows: list[dict], with_anchor: bool = True):
    """Rows -> (R, dE); bulk anchor appended per material when requested."""
    R_list: list[float] = []
    dE_list: list[float] = []
    for row in rows:
        r = float(row["D_nm"]) / 2.0
        R_list.append(r)
        dE_list.append(1240.0 / float(row["lambda_peak_nm"]) - BULK_EG[row["material"]])
    R = np.asarray(R_list, dtype=float)
    dE = np.asarray(dE_list, dtype=float)
    if with_anchor:
        R = np.append(R, [100.0] * len({row["material"] for row in rows}))
        dE = np.append(dE, [0.0] * len({row["material"] for row in rows}))
    return R, dE


def ambiguity_prior_templates(R: np.ndarray, dE: np.ndarray) -> tuple[list, list]:
    """One simplest-representative prior per ambiguity group, mapped to R/inv_R.

    Mapping: 'Rinv' -> 'inv_R', 'Rinv2' -> '(inv_R**2)'. Shortest member wins
    (fewest structural ops); exact-duplicate aliases already removed by the
    degeneracy filter, so each prior stands for a distinguishable direction.
    """
    from apedp.discovery import sparse_select_terms

    out = sparse_select_terms({"Rinv": 1 / R, "Rinv2": 1 / R**2}, dE,
                              ambiguity_corr=0.9)
    priors = []
    for group in out.get("ambiguity_groups", []):
        rep = min(group, key=lambda t: (len(t), t))
        priors.append(rep.replace("Rinv2", "(inv_R**2)").replace("Rinv", "inv_R"))
    # Deduplicate while preserving order.
    seen, unique = set(), []
    for prior in priors:
        if prior not in seen:
            seen.add(prior)
            unique.append(prior)
    return unique, out.get("ambiguity_groups", [])


def refit_brus_form(y_train) -> Callable[..., Any]:
    """Bakeoff extra_form factory: OLS [1/R^2, 1/R] refit -- the bar to beat.

    Takes y_train from the enclosing scope because the bakeoff protocol
    passes only x dicts to form functions.
    """
    import numpy as np

    ytr = np.asarray(y_train, float).ravel()

    def _fn(train, test):
        A = np.column_stack([1 / np.asarray(train["R"]) ** 2,
                             1 / np.asarray(train["R"])])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        B = np.column_stack([1 / np.asarray(test["R"]) ** 2,
                             1 / np.asarray(test["R"])])
        return B @ coef, {"coef": [round(float(v), 4) for v in coef]}

    return _fn


def admissibility(expr_str: str) -> dict:
    """Physics admissibility post-checks on a log-spaced R grid (pure fn)."""
    import sympy as sp

    grid = np.logspace(np.log10(0.5), 2.0, 200)
    try:
        fn = sp.lambdify(sp.Symbol("R"), sp.sympify(expr_str), "numpy")
        vals = np.asarray(fn(grid), dtype=float)
    except (TypeError, ValueError, ArithmeticError, KeyError) as e:
        return {"status": "FAIL", "reason": f"unevaluable: {e}"[:120]}
    if not bool(np.all(np.isfinite(vals))):
        return {"status": "FAIL", "reason": "non-finite predictions in [0.5,100]nm"}
    checks: dict[str, Any] = {
        "positive": bool(np.all(vals > 0)),
        "monotone_decreasing": bool(np.all(np.diff(vals) < 0)),
        "bulk_limit_eV": round(float(vals[-1]), 5),
    }
    checks["bulk_limit_ok"] = abs(checks["bulk_limit_eV"]) < 0.05
    # Coulomb-sign report (info only -- never a steering constraint).
    symptoms = []
    try:
        poly = sp.Poly(sp.sympify(expr_str), sp.Symbol("R"))
        for monom, coeff in poly.as_dict().items():
            if monom[0] == -1 and float(coeff) > 0:
                symptoms.append(f"positive 1/R coefficient {float(coeff):.4f}")
    except (TypeError, ValueError, ArithmeticError, KeyError, AttributeError,
            sp.PolynomialError) as exc:
        # sp.PolynomialError descends from plain Exception, not from any of
        # the builtins above, so it escaped this guard and propagated out of
        # admissibility(). That matters because the very priors this module
        # builds are non-polynomial: ambiguity_prior_templates emits inv_R and
        # inv_R**2, and "1/R" is not a polynomial in R, so sp.Poly raises on
        # the module's own intended input. An info-only diagnostic must never
        # be the thing that fails the run.
        logger.debug("Coulomb-sign inspection skipped for %r: %s", expr_str, exc)
    checks["coulomb_sign_notes"] = symptoms
    checks["status"] = ("PASS" if checks["positive"] and checks["monotone_decreasing"]
                        and checks["bulk_limit_ok"] else "FAIL")
    return checks


def run_qd_generation(R_tr, dE_tr, R_te, dE_te, *, generations=8,
                      pop_size=25, seed=42) -> dict:
    """Execute one generation run; returns the verdict dict."""
    from apedp.discovery import check_data_adequacy
    from apedp.hybrid import hybrid_discover
    from apedp.scientific_layers import bakeoff

    gate = check_data_adequacy({"R": np.asarray(R_tr, dtype=float)},
                               np.asarray(dE_tr, dtype=float))
    if gate["verdict"] != "OK":
        return {"expression": "", "verdict": "INCONCLUSIVE",
                "reason": "data gate: " + gate["verdict"] + ": "
                          + "; ".join(gate.get("reasons", [])),
                "adequacy": gate}

    priors, groups = ambiguity_prior_templates(R_tr, dE_tr)
    res = hybrid_discover({"R": np.asarray(R_tr),
                           "inv_R": 1 / np.asarray(R_tr)},
                          np.asarray(dE_tr), variables=["R", "inv_R"],
                          template_priors=priors, generations=generations,
                          pop_size=pop_size, seed=seed)
    expr = res.get("expression", "")
    verdict: dict = {"expression": expr, "priors": priors,
                     "n_ambiguity_groups": len(groups),
                     "probes": res.get("probes", [])}
    if not expr:
        verdict["verdict"] = "INCONCLUSIVE"
        verdict["reason"] = "GP returned no expression"
        return verdict
    import sympy as sp

    from apedp.gated_assurance import safe_parse_expr

    base = ["R"]
    # Normalize GP terminals to bare R: inv_R -> 1/R, so every downstream
    # judge (bakeoff, admissibility, novelty) evaluates the same function.
    ev = safe_parse_expr(expr)
    if sp.Symbol("inv_R") in ev.free_symbols:
        ev = ev.subs(sp.Symbol("inv_R"), 1 / sp.Symbol("R"))
    expr_ev = str(ev)

    def _cand(d):
        cols = [np.asarray(d[v], float).ravel() for v in base]
        fn = sp.lambdify(base, safe_parse_expr(expr_ev), "numpy")
        return np.asarray(fn(*cols), float).ravel()

    bo = bakeoff({"R": np.asarray(R_tr)}, np.asarray(dE_tr),
                 {"R": np.asarray(R_te)}, np.asarray(dE_te), _cand,
                 extra_forms={"refit-Brus": refit_brus_form(dE_tr)})
    rows = {row["form"]: row for row in bo["rows"] if not row.get("failed")}
    cand_mae = rows.get("candidate", {}).get("mae")
    brus_mae = rows.get("refit-Brus", {}).get("mae")
    verdict["bakeoff"] = {k: v.get("mae") for k, v in rows.items()}
    verdict["admissibility"] = admissibility(expr_ev)
    if cand_mae is None or brus_mae is None:
        verdict.update(verdict="INCONCLUSIVE", reason="bakeoff evaluation failed")
        return verdict
    gap = brus_mae - cand_mae
    verdict["mae_gap_vs_brus_eV"] = round(gap, 5)
    # Novelty: Brus-equivalent iff candidate predictions track refit-Brus
    # predictions almost exactly on a wide grid.
    grid = np.logspace(np.log10(0.5), 2.0, 200)
    try:
        fn = sp.lambdify(base, safe_parse_expr(expr_ev), "numpy")
        cpred = np.asarray(fn(grid), float).ravel()
        A = np.column_stack([1 / grid**2, 1 / grid, np.ones_like(grid)])
        coef, *_ = np.linalg.lstsq(A, cpred, rcond=None)
        bpred = A @ coef
        resid = float(np.sqrt(np.mean((cpred - bpred) ** 2))
                      / (np.sqrt(np.mean(cpred**2)) + 1e-300))
    except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
        logger.warning("Brus-equivalence residual unavailable; nan substituted: %s", exc)
        resid = float("nan")
    verdict["brus_equivalence_resid"] = resid
    novel = bool(np.isfinite(resid)) and resid > 0.01
    admissible = verdict["admissibility"]["status"] == "PASS"
    if gap >= 0.01 and admissible and novel:
        verdict["verdict"] = "GENERATION"
    elif not novel and admissible:
        verdict["verdict"] = "CONFIRMATION"
    else:
        verdict["verdict"] = "INCONCLUSIVE"
        verdict["reason"] = (f"gap={gap:.4f}eV admissible={admissible} "
                             f"novel={novel}")
    return verdict


if __name__ == "__main__":
    print("=== QD Generation Run (dry-run on Yu CdSe: small-R train) ===")
    rows = yu_cdse_rows()
    print("schema check:", validate_qd_table(rows))
    R, dE = to_arrays(rows)
    order = np.argsort(R)
    R, dE = R[order], dE[order]
    # Train = small-R half (+ anchor last element stays in train), test = large-R.
    cut = len(R) // 2
    R_tr, dE_tr = R[:cut], dE[:cut]
    R_te, dE_te = R[cut:-1], dE[cut:-1]
    print(f"train n={len(R_tr)} (R {R_tr.min():.2f}-{R_tr.max():.2f}nm), "
          f"test n={len(R_te)}")
    out = run_qd_generation(R_tr, dE_tr, R_te, dE_te)
    print("thin-train VERDICT:", out["verdict"], out.get("reason", ""))
    assert out["verdict"] == "INCONCLUSIVE" and "gate" in out.get("reason", "")
    # Adequate-data path still executes end to end (GP governed as before).
    out2 = run_qd_generation(R, dE, R_te, dE_te)
    print("full-train expression:", (out2["expression"] or "")[:80])
    print("full-train VERDICT:", out2["verdict"], out2.get("reason", ""))
