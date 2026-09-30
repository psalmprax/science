"""Hybrid discovery orchestrator: detectors shrink, GP enumerates, metrics govern.

Pipeline per call (all steps measured and reported, none silent):
  1. Probe: separability + composition-family detection on (x, y).
  2. Configure: recursion fast-path when separable; else transform terminals
     (log/exp/inverse of inputs) suggested by the detected family.
  3. Enumerate: bounded GP evolution with all mutation flags on.
  4. Govern: bakeoff vs standard-form library + domain audit + A/B report.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Failures a best-effort stage may legitimately hit: a missing optional
#: dependency, malformed input, or a numeric/operational error raised by the
#: discovery stack. Anything outside this set is a defect and must propagate
#: instead of being recorded as a failed stage (mirrors
#: ``gated_assurance._PROBE_ERRORS``; see docs/INDEPENDENT_REVIEW_2026-09-23.md
#: §4.2).
_PROBE_ERRORS = (RuntimeError, TypeError, ValueError, ArithmeticError,
                 KeyError, ImportError)


def hybrid_multistart(x_data: dict, y_data, variables: list,
                      target_unit: str = "dimensionless",
                      template_priors: list | None = None,
                      seeds: tuple = (42, 7, 123),
                      generations: int = 10, pop_size: int = 25) -> dict:
    """Multi-restart GP: K independent evolutions, deduped + ranked.

    Each seed runs :func:`hybrid_discover`; winners are canonicalized by
    sympy simplification and ranked by test-set MAE from their bakeoffs
    (missing bakeoff = ranked last). Reports whether restarts agree
    (same canonical winner) -- disagreement is the measurable signature
    of local-optima trapping. Deterministic given seeds.
    """
    import numpy as np

    from apedp.gated_assurance import safe_parse_expr

    runs = []
    for seed in seeds:
        try:
            out = hybrid_discover(
                {k: np.asarray(v, float) for k, v in x_data.items()},
                np.asarray(y_data, float), variables,
                target_unit=target_unit, template_priors=template_priors,
                generations=generations, pop_size=pop_size, seed=seed)
        except _PROBE_ERRORS as e:
            runs.append({"seed": seed, "error": str(e)[:120]})
            continue
        expr = out.get("expression", "")
        try:
            canon = str(safe_parse_expr(expr).simplify()) if expr else ""
        except (TypeError, ValueError, ArithmeticError) as exc:
            logger.debug("Canonicalization failed for seed %s; using raw form: %s", seed, exc)
            canon = expr
        mae = float("inf")
        try:
            rows = (out.get("bakeoff") or {}).get("rows", [])
            cand: dict[str, Any] = next(
                (r for r in rows if r.get("form") == "candidate"), {})
            if cand.get("mae") is not None and np.isfinite(cand["mae"]):
                mae = float(cand["mae"])
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            logger.warning(
                "Bake-off MAE unavailable for seed %s; candidate scored as inf: %s", seed, exc
            )
        runs.append({"seed": seed, "expression": expr,
                     "canonical": canon, "test_mae": mae})
    ranked, seen = [], set()
    for run in sorted([r for r in runs if "error" not in r],
                      key=lambda r: r["test_mae"]):
        if run["canonical"] and run["canonical"] not in seen:
            seen.add(run["canonical"])
            ranked.append(run)
    return {"runs": runs, "ranked": ranked,
            "agreement": len(ranked) > 0 and all(
                r["canonical"] == ranked[0]["canonical"] for r in ranked),
            "winner": ranked[0] if ranked else None}


def hybrid_discover(x_data: dict, y_data, variables: list,
                    target_unit: str = "dimensionless",
                    template_priors: list | None = None,
                    generations: int = 15, pop_size: int = 40,
                    seed: int = 42, shuffle: bool = False) -> dict:
    import numpy as np

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import (
        MultiVariableSRE,
        detect_composition_family,
        detect_multiplicative_split,
        sparse_select_terms,
    )
    from apedp.scientific_layers import (
        bakeoff,
        candidate_report,
        domain_audit,
    )

    x = {k: np.asarray(v, float).ravel() for k, v in x_data.items()}
    y = np.asarray(y_data, float).ravel()
    notes: list = []

    # 1. Probe.
    split = detect_multiplicative_split(x, y)
    family = detect_composition_family(x, y)
    notes.append(f"separable={bool(split)} family={family.get('family')}")
    # Sparse pre-filter: active library terms become seed priors so the
    # GP starts from data-selected structure, not just generic templates.
    sparse = sparse_select_terms(
        dict(x.items()), y)
    sparse_priors = [t for t in sparse.get("terms", [])
                     if t not in ("1",)]
    if sparse_priors:
        notes.append(f"sparse priors: {sparse_priors} "
                     f"(library {sparse.get('n_library')})")

    # 2. Configure transform terminals from probes (ablation-reportable).
    anku = dict(x)
    extra = []
    if family.get("family") in ("exponential", "saturating-exponential"):
        v = family.get("variable")
        if v in anku:
            anku[f"log_{v}"] = np.log(np.abs(anku[v]) + 1e-300)
            extra.append(f"log_{v}")
            notes.append(f"terminal added: log_{v} ({family['family']})")
    if family.get("family") == "rational":
        v = family.get("variable")
        if v in anku:
            anku[f"inv_{v}"] = 1.0 / np.where(anku[v] == 0, np.nan, anku[v])
            extra.append(f"inv_{v}")
            notes.append(f"terminal added: inv_{v} (rational)")
    # Family-routed seed templates: the detected family contributes generic
    # templates (constants unfitted), so the search starts in the right
    # neighborhood without being handed the answer. Reported in probes.
    fam = family.get("family")
    fam_templates = {
        "exponential": ["exp({v})", "exp(-{v})"],
        "arrhenius": ["exp(-1/{v})", "exp(1/{v})"],
        "saturating-exponential": ["1-exp(-{v})", "exp(-{v})"],
        "rational": ["1/{v}", "1/(1+{v})"],
    }
    routed = []
    if fam in fam_templates and family.get("variable") in anku:
        v = family["variable"]
        routed = [t.format(v=v) for t in fam_templates[fam]]
        notes.append(f"family-routed templates: {routed} ({fam})")

    # 3. Enumerate (recursion fast-path lives inside run_discovery).
    engine = DimensionalAssuranceEngine()
    for v in list(anku):
        engine.register_variable(v, "dimensionless")
    var_list = variables + extra
    sre = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=engine),
        variables=var_list, target_unit_str=target_unit,
        template_priors=(list(template_priors or []) + routed
                           + sparse_priors), seed=seed,
        allow_power_mutation=True, allow_rational_mutation=True,
        allow_compose_mutation=True, allow_recursive_reduction=True)
    n = len(y)
    if shuffle:
        # Seeded permutation first: the default sequential split is an
        # extrapolation test on sorted inputs and an arbitrary slice
        # otherwise -- callers must choose consciously. Recorded in probes.
        perm = np.random.default_rng(seed).permutation(n)
        anku = {k: v[perm] for k, v in anku.items()}
        y = y[perm]
        notes.append("split: shuffled 70/30 (seeded)")
    else:
        notes.append("split: sequential first-70% train "
                     "(extrapolation test on sorted inputs)")
    cut = max(1, int(0.7 * n))
    xtr = {k: v[:cut] for k, v in anku.items()}
    xte = {k: v[cut:] for k, v in anku.items()}
    ytr, yte = y[:cut], y[cut:]
    best, mse, hist = sre.run_discovery(xtr, ytr, generations=generations,
                                        pop_size=pop_size, seed=seed)

    # 4. Govern.
    import sympy as sp

    from apedp.gated_assurance import safe_parse_expr
    expr_str = str(best) if best is not None else ""
    used = sorted(map(str, safe_parse_expr(expr_str).free_symbols)) \
        if expr_str else []
    base_vars = [v for v in used if v in xte]

    def _cand(d):
        cols = [np.asarray(d[v], float).ravel() for v in base_vars]
        fn = sp.lambdify(base_vars, safe_parse_expr(expr_str), "numpy")
        return np.asarray(fn(*cols), float).ravel()

    out: dict = {"expression": expr_str, "train_mse": mse,
                 "reduction": bool(hist and "reduction" in hist[0]),
                 "probes": notes}
    if best is None or not base_vars:
        out.update({"bakeoff": None, "domain": None, "report": None})
        return out
    try:
        bo = bakeoff({k: xtr[k] for k in base_vars}, ytr,
                     {k: xte[k] for k in base_vars}, yte, _cand)
        top = bo["rows"][0] if bo["rows"] else {}
        cand_row: dict = next((r for r in bo["rows"]
                               if r["form"] == "candidate"), {})
        da = domain_audit(expr_str,
                          {k: (float(np.min(xte[k])), float(np.max(xte[k])))
                           for k in base_vars},
                          {"output_min": 0.0}
                          if bool(np.all(yte >= 0)) else None)
        rep = candidate_report(
            "hybrid-cand", expr_str, "bakeoff-library",
            {k: (float(np.min(xte[k])), float(np.max(xte[k])))
             for k in base_vars}, len(yte),
            cand_row.get("mae", float("nan")),
            top.get("mae", float("nan")), "PASS", da,
            "train/test split", "none",
            "not established",
            ab_p=cand_row.get("vs_best_p"),
            ab_verdict=cand_row.get("vs_best"))
        out.update({"bakeoff": bo, "domain": da, "report": rep})
        # Decision layer (additive): physics marked declared-not-measured
        # because this path never runs dimensional validation — the details
        # record that, so no backend can mistake it for measured evidence.
        try:
            import math as _math

            from apedp.decision import (
                CandidateClass,
                CandidateEvidence,
                DecisionEngine,
                EvidenceStrength,
                RuleBasedBackend,
                ValidationResult,
            )
            mae = cand_row.get("mae", float("nan"))
            evidence = CandidateEvidence(
                candidate_id="hybrid-cand",
                candidate_class=CandidateClass.CANDIDATE,
                mathematical=ValidationResult(True),
                physics=ValidationResult(
                    True, details={"basis": "declared-not-measured"}),
                numerical=ValidationResult(
                    bool(_math.isfinite(mae or float("nan"))),
                    score=None if not _math.isfinite(mae or float("nan"))
                    else mae),
                empirical=ValidationResult(False),
                evidence_strength=EvidenceStrength.LOW,
                independent_datasets=0, benchmark_datasets=0,
                domain_valid=da.get("status") != "FAIL",
                evidence_ids=("hybrid-bakeoff", "hybrid-domain"))
            out["decision"] = DecisionEngine(
                RuleBasedBackend()).evaluate(evidence).to_dict()
        except (ImportError, TypeError, ValueError, KeyError, AttributeError,
                NotImplementedError) as e:
            out["decision"] = {"error": str(e)[:200]}
    except _PROBE_ERRORS as e:
        out.update({"bakeoff": None, "domain": None, "report": None,
                    "decision": None, "govern_error": str(e)[:200]})
    return out
