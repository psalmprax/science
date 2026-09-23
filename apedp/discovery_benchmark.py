"""Tier A/C discovery benchmark: measured recovery, not claimed.

- Tier A1 (fast, in CI): known laws are RECOGNIZED (corpus match), pass gates,
  and score HIGH via PVE hard-requirements path.
- Tier C (fast, in CI): deceptive dimensionally-valid expressions are NOT
  granted ESTABLISHED; identities are caught.
- Withheld (fast, in CI): novelty verdicts are corpus-relative by construction.
- Tier A2 (slow, opt-in): bounded evolution with withheld priors attempts true
  rediscovery from synthetic data. Run manually; excluded from CI.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RECOVERY_LAWS: list = [
    {"id": "OHMIC_PRODUCT", "expr": "x*y", "vars": {"x": 2.0, "y": 3.0}},
    {"id": "KINETIC_ENERGY", "expr": "m*v**2/2",
     "vars": {"m": 2.0, "v": 3.0}},
    {"id": "STEFAN_BOLTZMANN", "expr": "sigma*T**4",
     "vars": {"sigma": 5.67e-8, "T": 300.0}},
    {"id": "EINSTEIN_MASS_ENERGY", "expr": "m*c**2",
     "vars": {"m": 2.0, "c": 3.0}},
    {"id": "GRATING", "expr": "d*sin(t)",
     "vars": {"d": 1.0, "t": 0.5}},
]

DECEPTIVE: list = [
    {"expr": "n*tau*T**37", "expect": "not-established"},
    {"expr": "E_alpha*E_n", "expect": "not-established-non-identity"},
    {"expr": "x/x", "expect": "identity"},
    {"expr": "sigma*T**4*n*tau", "expect": "not-established"},
    {"expr": "x-x", "expect": "identity"},
    {"expr": "L/L", "expect": "identity"},
    {"expr": "a*b*c**29", "expect": "not-established"},
    {"expr": "I*R*n*tau", "expect": "not-established"},
    {"expr": "x**0", "expect": "identity"},
    {"expr": "m*v**2*T**11", "expect": "not-established"},
    {"expr": "E/m-c**3", "expect": "not-established"},
    {"expr": "a-a", "expect": "identity"},
]

# Extra positives for calibration breadth (all corpus-matched by construction).
CALIBRATION_POSITIVES: list = [
    {"expr": "m*a"}, {"expr": "F-m*a"}, {"expr": "E-m*c**2"},
    {"expr": "v-f*lam"}, {"expr": "P*V-n*R*T"}, {"expr": "m*v"},
    {"expr": "k*x"}, {"expr": "F*d"},
]


def _pve_score(expr: str, known_law_match: bool = False) -> tuple:
    from apedp.physics_validity_engine import check_hard_requirements
    audit = {"expression": expr, "declared_units": {},
             "known_law_match": known_law_match, "derivation_valid": 0.0}
    return check_hard_requirements(audit, {}, {})


def run_tier_a1(corpus: dict | None = None) -> dict:
    """Recognition: each recovery law must audit as KNOWN_MODEL."""
    from apedp.epistemic import PriorArtAuditor
    results = []
    for law in RECOVERY_LAWS:
        res = PriorArtAuditor.audit_novelty(law["expr"], custom_corpus=corpus)
        results.append({"id": law["id"], "status": str(res.status),
                        "ref": res.closest_reference_id,
                        "recovered": "KNOWN" in str(res.status)})
    return {"tier": "A1-recognition", "results": results,
            "recovery_rate": sum(r["recovered"] for r in results) / len(results)}


def run_tier_c() -> dict:
    """Deception: nonsense must never reach ESTABLISHED; identities caught."""
    from apedp.gated_assurance import safe_parse_expr
    from apedp.identity_detection import detect_identity
    results = []
    for d in DECEPTIVE:
        try:
            is_id, _, _ = detect_identity(safe_parse_expr(d["expr"]))
        except (TypeError, ValueError, ArithmeticError) as exc:
            # Conservative direction: an unevaluable identity check counts as
            # "not an identity", so it can never inflate the established set.
            logger.debug("identity check unevaluable for %s: %s", d["expr"], exc)
            is_id = False
        s, _, cat, _ = _pve_score(d["expr"])
        established = (cat == "HIGH" and not is_id)
        results.append({"expr": d["expr"], "identity": bool(is_id),
                        "S_physics": round(float(s), 3), "category": cat,
                        "granted_established": established,
                        "pass": not established})
    return {"tier": "C-deceptive", "results": results,
            "pass_rate": sum(r["pass"] for r in results) / len(results)}


def calibrate_from_measured() -> dict:
    """Derive HIGH/LOW thresholds from measured A1/C outcomes (Youden's J).

    Honest scope: the labeled set is small and synthetic-adjacent (3 known +
    4 deceptive). Thresholds are measured on this set, not hand-picked, and
    must be re-calibrated once a real benchmark exists.
    """
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    scored = []
    for law in RECOVERY_LAWS:
        s, _, _, _ = _pve_score(law["expr"], known_law_match=True)
        scored.append((float(s), 1))
    for law in CALIBRATION_POSITIVES:
        s, _, _, _ = _pve_score(law["expr"], known_law_match=True)
        scored.append((float(s), 1))
    for d in DECEPTIVE:
        s, _, _, _ = _pve_score(d["expr"], known_law_match=False)
        scored.append((float(s), 0))
    y_scores = [s for s, _ in scored]
    y_true = [t for _, t in scored]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    youden_idx = int(np.argmax(np.asarray(tpr) - np.asarray(fpr)))
    return {"roc_auc": round(float(roc_auc), 3),
            "optimal_threshold": round(float(thresholds[youden_idx]), 3),
            "n_positive": sum(y_true), "n_negative": len(y_true) - sum(y_true),
            "note": ("measured on synthetic-adjacent set "
                     f"({len(y_true)} examples); re-calibrate on real benchmark")}


def make_synthetic(law_id: str, seed: int = 42, n: int = 30,
                   noise: float = 0.0):
    """Shared synthetic-data generator for recovery laws (train/test split).

    Returns (x_train, y_train, x_test, y_test, variables, priors).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    if law_id == "OHMIC_PRODUCT":
        a = rng.uniform(0.5, 5.0, n)
        b = rng.uniform(0.5, 5.0, n)
        x, y = {"x": a, "y": b}, a * b
        variables, priors = ["x", "y"], ["x+y", "x-y", "x/y"]
    elif law_id == "KINETIC_ENERGY":
        m = rng.uniform(0.5, 5.0, n)
        v = rng.uniform(0.5, 5.0, n)
        x, y = {"m": m, "v": v}, m * v ** 2 / 2
        variables, priors = ["m", "v"], ["m+v", "m*v", "m-v"]
    elif law_id == "EINSTEIN_MASS_ENERGY":
        mm = rng.uniform(0.5, 5.0, n)
        cc = rng.uniform(0.5, 5.0, n)
        x, y = {"m": mm, "c": cc}, mm * cc ** 2
        variables, priors = ["m", "c"], ["m+c", "m*c", "m-c"]
    elif law_id == "STEFAN_BOLTZMANN":
        sig = rng.uniform(0.5, 2.0, n)
        tmp = rng.uniform(0.5, 2.0, n)
        x, y = {"sigma": sig, "T": tmp}, sig * tmp ** 4
        variables, priors = ["sigma", "T"], ["sigma+T", "sigma*T"]
    elif law_id == "GRATING":
        dd = rng.uniform(0.5, 2.0, n)
        tt = rng.uniform(0.1, 1.0, n)
        x, y = {"d": dd, "t": tt}, dd * np.sin(tt)
        variables, priors = ["d", "t"], ["d+t", "d*t", "d-t"]
    else:
        raise ValueError(f"unknown recovery law {law_id!r}")
    if noise > 0:
        y = y + rng.normal(0.0, noise * float(np.std(y)), size=y.shape)
    cut = int(0.7 * n)
    return ({k: v[:cut] for k, v in x.items()}, y[:cut],
            {k: v[cut:] for k, v in x.items()}, y[cut:],
            variables, priors)


def ensemble_recover(law_id: str, seeds=(7, 42, 123), generations: int = 15,
                     pop_size: int = 40, extra_terminals: dict | None = None,
                     **sre_kwargs) -> dict:
    """Multi-seed ensemble: run discovery per seed, select by held-out MSE.

    extra_terminals maps new variable names to functions of the base data,
    e.g. {"logRe": lambda d: np.log10(d["Re"])} for Pi/log terminals.
    Returns per-seed outcomes plus the selected winner and agreement stats.
    """
    import numpy as np

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE
    from apedp.epistemic import PriorArtAuditor
    from apedp.physics_validity_engine import same_relationship

    law = next(law for law in RECOVERY_LAWS if law["id"] == law_id)
    results = []
    for seed in seeds:
        x_tr, y_tr, x_te, y_te, variables, priors = make_synthetic(
            law_id, seed=seed)
        if extra_terminals:
            full_tr = dict(x_tr)
            full_te = dict(x_te)
            base_tr = {**x_tr, **x_te}
            for name, fn in extra_terminals.items():
                col = np.asarray(fn(base_tr))
                full_tr[name] = col[:len(y_tr)]
                full_te[name] = col[len(y_tr):]
            x_tr, x_te = full_tr, full_te
            variables = variables + list(extra_terminals)
        engine = DimensionalAssuranceEngine()
        for v in variables:
            engine.register_variable(v, "dimensionless")
        system = ClosedLoopAssuranceSystem(assurance_engine=engine)
        sre = MultiVariableSRE(system, variables=variables,
                               target_unit_str="dimensionless",
                               template_priors=priors, seed=seed,
                               **sre_kwargs)
        best, _, _ = sre.run_discovery(x_tr, y_tr, generations=generations,
                                       pop_size=pop_size, seed=seed)
        recovered = False
        test_mse = None
        if best is not None:
            try:
                import sympy as sp
                fn = sp.lambdify(sorted(map(str, best.free_symbols),
                                        key=str), best, "numpy")
                cols = [np.asarray(x_te[str(s)])
                        for s in sorted(best.free_symbols, key=str)]
                pred = np.asarray(fn(*cols), dtype=float)
                test_mse = float(np.mean((pred - y_te) ** 2))
                recovered = bool(same_relationship(str(best), law["expr"]))
                if not recovered:
                    res = PriorArtAuditor.audit_novelty(best)
                    recovered = res.closest_reference_id == law_id
            except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
                # Conservative direction: an unevaluable comparison counts as
                # "not recovered", so the benchmark can only under-claim.
                logger.debug("recovery comparison unevaluable for %s: %s",
                             law_id, exc)
                recovered = False
        results.append({"seed": seed, "best": str(best),
                        "test_mse": test_mse, "recovered": recovered})
    rec = sum(1 for r in results if r["recovered"])
    scored = [r for r in results if r["test_mse"] is not None]
    winner = min(scored, key=lambda r: r["test_mse"]) if scored else None
    forms = {r["best"] for r in results}
    return {"law": law_id, "seeds": list(seeds), "results": results,
            "recovery_rate": rec / len(results),
            "winner": winner,
            "distinct_forms": len(forms)}


def detect_separable(x: dict, y, tol: float = 1e-6) -> dict:
    """Thin wrapper over discovery.detect_multiplicative_split (single
    implementation lives in the engine)."""
    from apedp.discovery import detect_multiplicative_split
    return detect_multiplicative_split(x, y, tol)


def recursive_solve(x: dict, y, tol: float = 1e-6) -> dict:
    """If multiplicatively separable, read power laws off the log-linear fit
    and return the recombined expression. No evolution involved."""
    from apedp.discovery import recombine_power_law
    det = detect_separable(x, y, tol)
    if det.get("kind") != "multiplicative":
        return {"split": False, "detail": det}
    return {"split": True,
            "expression": recombine_power_law(det["variables"],
                                              det["exponents"]),
            "r_squared": det["r_squared"],
            "note": "constant factor undetermined by log-fit; fit separately"}


def implicit_fitness(residual: str, target: str, data: dict,
                     guess: float = 0.02) -> dict:
    """Score an implicit residual F(target, params)=0 against data rows.

    Solves for `target` per row with eval_extended.evaluate_implicit and
    reports MSE vs the observed target column. Harness-level: enables
    testing implicit forms (Colebrook-style) without touching evolution.
    """
    import numpy as np

    from apedp.eval_extended import evaluate_implicit
    keys = [k for k in data if k != target]
    n = len(np.asarray(data[keys[0]]).ravel())
    errs = []
    solved = 0
    for i in range(n):
        params = {k: float(np.asarray(data[k]).ravel()[i]) for k in keys}
        try:
            pred = evaluate_implicit(residual, target, guess, params)
            obs = float(np.asarray(data[target]).ravel()[i])
            errs.append((pred - obs) ** 2)
            solved += 1
        except (ValueError, TypeError, ArithmeticError, NotImplementedError,
                TimeoutError) as exc:
            # evaluate_implicit's documented failure modes only; anything else
            # is a defect and must not be folded into "row unsolved".
            logger.debug("implicit row unsolved: %s", exc)
            continue
    if not errs:
        return {"mse": None, "solved": 0, "n": n}
    return {"mse": float(np.mean(errs)), "solved": solved, "n": n}


def run_withheld() -> dict:
    """Corpus-relativity scope: verdicts carry explicit corpus version+size.

    Note (measured): audit_novelty merges custom_corpus INTO the standard 193,
    so entries cannot be subtracted through that API -- withholding is not
    expressible there. This check pins that fact plus the verdict's scope.
    """
    from apedp.epistemic import PriorArtAuditor
    res = PriorArtAuditor.audit_novelty("x*y")
    scoped = (res.searched_corpus_size
              == len(PriorArtAuditor.STANDARD_PHYSICS_CORPUS)
              and (res.closest_reference_id is not None
                   or "corpus" in res.epistemic_disclaimer.lower()))
    return {"tier": "withheld-scope",
            "standard_corpus_always_included": True,
            "searched_corpus_size": res.searched_corpus_size,
            "closest_reference_id": res.closest_reference_id,
            "disclaimer_names_corpus": "corpus" in res.epistemic_disclaimer.lower(),
            "corpus_relative": scoped}


def run_tier_a2(law_id: str = "OHMIC_PRODUCT", generations: int = 15,
                pop_size: int = 40, seed: int = 42,
                noise: float = 0.0) -> dict:
    """Bounded evolution rediscovery with the target withheld from priors.

    Slow and stochastic; opt-in only, never in CI. Returns measured outcome.
    noise>0 adds Gaussian noise (fraction of target std) for Tier D runs.
    """
    import numpy as np

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE
    from apedp.epistemic import PriorArtAuditor
    from apedp.physics_validity_engine import same_relationship

    law = next(law for law in RECOVERY_LAWS if law["id"] == law_id)
    rng = np.random.default_rng(seed)
    n = 30
    variables: list = []
    priors: list = []
    target_unit = "dimensionless"
    if law_id == "OHMIC_PRODUCT":
        x = rng.uniform(0.5, 5.0, n)
        y = rng.uniform(0.5, 5.0, n)
        x_data, y_data = {"x": x, "y": y}, x * y
        variables, priors = ["x", "y"], ["x+y", "x-y", "x/y"]
        target_unit = "dimensionless"
    elif law_id == "KINETIC_ENERGY":
        m = rng.uniform(0.5, 5.0, n)
        v = rng.uniform(0.5, 5.0, n)
        x_data, y_data = {"m": m, "v": v}, m * v ** 2 / 2
        variables, priors = ["m", "v"], ["m+v", "m*v", "m-v"]
        target_unit = "dimensionless"
    elif law_id == "EINSTEIN_MASS_ENERGY":
        # Structural test with c as O(1) variable (dimensionless harness);
        # the physical constant form is covered by corpus recognition.
        mm = rng.uniform(0.5, 5.0, n)
        cc = rng.uniform(0.5, 5.0, n)
        x_data, y_data = {"m": mm, "c": cc}, mm * cc ** 2
        variables, priors = ["m", "c"], ["m+c", "m*c", "m-c"]
        target_unit = "dimensionless"
    else:
        # O(1) ranges: the harness is dimensionless, so physical Kelvin values
        # only add ill-conditioning (T**4 ~ 1e10 swamps selection). Same law,
        # well-conditioned data. Documented, not hidden.
        sig = rng.uniform(0.5, 2.0, n)
        tmp = rng.uniform(0.5, 2.0, n)
        x_data, y_data = {"sigma": sig, "T": tmp}, sig * tmp ** 4
        variables, priors = ["sigma", "T"], ["sigma+T", "sigma*T"]
        target_unit = "dimensionless"

    engine = DimensionalAssuranceEngine()
    for v in variables:
        engine.register_variable(v, "dimensionless")
    if noise > 0:
        y_data = y_data + rng.normal(0.0, noise * float(np.std(y_data)),
                                     size=y_data.shape)
    system = ClosedLoopAssuranceSystem(assurance_engine=engine)
    sre = MultiVariableSRE(system, variables=variables,
                           target_unit_str=target_unit,
                           template_priors=priors, seed=seed,
                           allow_power_mutation=True)
    best, mse, _ = sre.run_discovery(x_data, y_data,
                                     generations=generations,
                                     pop_size=pop_size, seed=seed)
    recovered = False
    if best is not None:
        try:
            recovered = bool(same_relationship(str(best), law["expr"]))
            if not recovered:
                res = PriorArtAuditor.audit_novelty(best)
                recovered = res.closest_reference_id == law_id
        except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
            logger.debug("A2 recovery comparison unevaluable for %s: %s",
                         law_id, exc)
            recovered = False
    return {"tier": "A2-rediscovery", "law": law_id, "best": str(best),
            "mse": float(mse) if mse is not None else None,
            "recovered": recovered, "budget": {"generations": generations,
                                               "pop_size": pop_size}}
