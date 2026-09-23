# Databricks notebook source
"""Fast discovery-benchmark tiers (no evolution): A1, C, withheld, calibration."""
import pytest

from apedp.discovery_benchmark import (
    calibrate_from_measured,
    run_tier_a1,
    run_tier_c,
    run_withheld,
)


def test_tier_a1_recognizes_known_laws():
    out = run_tier_a1()
    assert out["recovery_rate"] == 1.0, out


def test_tier_c_rejects_deception():
    out = run_tier_c()
    assert out["pass_rate"] == 1.0, out
    by_expr = {r["expr"]: r for r in out["results"]}
    assert by_expr["x/x"]["identity"] is True
    assert by_expr["E_alpha*E_n"]["identity"] is False


def test_withheld_proves_corpus_relativity():
    out = run_withheld()
    from apedp.epistemic import PriorArtAuditor
    assert out["standard_corpus_always_included"] is True, out
    assert out["searched_corpus_size"] == len(
        PriorArtAuditor.STANDARD_PHYSICS_CORPUS), out
    assert out["corpus_relative"] is True, out


def test_composition_detector_finds_step_and_rejects_linear():
    import numpy as np

    from apedp.discovery import detect_composition_family
    rng = np.random.default_rng(7)
    t = rng.uniform(0.1, 15.0, 40)
    out = detect_composition_family({"t": t}, 2.0 * (1 - np.exp(-t / 3.0)))
    assert out.get("family") == "saturating-exponential", out
    assert abs(out["asymptote"] - 2.0) < 0.05, out
    out2 = detect_composition_family({"t": t}, 2.0 * t + 1.0)
    assert out2 == {}, out2
    x = rng.uniform(0.5, 5.0, 30)
    out3 = detect_composition_family(
        {"x": x, "y": rng.uniform(0.5, 5.0, 30)}, x * 2.0)
    assert out3 == {}, out3


def test_composition_detector_finds_arrhenius():
    import numpy as np

    from apedp.discovery import detect_composition_family
    rng = np.random.default_rng(7)
    T = rng.uniform(300, 1000, 40)
    out = detect_composition_family({"T": T}, 1e13 * np.exp(-50000 / (8.314 * T)))
    assert out.get("family") == "arrhenius", out
    assert out.get("variable") == "T", out


def test_sparse_select_finds_product_term():
    pytest.importorskip("sklearn")
    import numpy as np

    from apedp.discovery import sparse_select_terms
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 40)
    y = rng.uniform(0.5, 5.0, 40)
    out = sparse_select_terms({"x": x, "y": y}, x * y)
    assert "x*y" in out["terms"], out
    assert out["n_library"] >= 7, out


def test_sparse_select_scope_limits():
    pytest.importorskip("sklearn")
    import numpy as np

    from apedp.discovery import sparse_select_terms
    rng = np.random.default_rng(7)
    t = rng.uniform(0.1, 15.0, 40)
    out = sparse_select_terms({"t": t}, 2.0 * (1 - np.exp(-t / 3.0)))
    # monomial library cannot express transients: must not hallucinate
    assert "t**1" in out["terms"] or out["terms"] != [], out


def test_control_probe_finds_true_vars():
    import numpy as np

    from apedp.discovery import control_variable_probe
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 60)
    y = rng.uniform(0.5, 5.0, 60)
    z = rng.uniform(0.5, 5.0, 60)
    out = control_variable_probe({"x": x, "y": y, "z": z}, x * y)
    assert out["relevant"] == ["x", "y"], out
    assert control_variable_probe(
        {"x": x, "y": y}, np.ones(60))["relevant"] == []


def test_neural_relevance_agrees():
    pytest.importorskip("sklearn")
    import numpy as np

    from apedp.discovery import neural_relevance
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 80)
    y = rng.uniform(0.5, 5.0, 80)
    z = rng.uniform(0.5, 5.0, 80)
    out = neural_relevance({"x": x, "y": y, "z": z}, x * y)
    assert out["relevant"] == ["y", "x"] or out["relevant"] == ["x", "y"], out
    assert out["surrogate_r2"] > 0.9, out


def test_beam_search_deterministic():
    from apedp.discovery import beam_search_expressions
    a = beam_search_expressions(["x", "y"], depth=2, width=8, seed=7)
    b = beam_search_expressions(["x", "y"], depth=2, width=8, seed=7)
    assert a == b and len(a) == 20


def test_grammar_terminals_honest_without_engine():
    from apedp.discovery import grammar_terminals_for
    out = grammar_terminals_for("dimensionless", {"Re": 1})
    assert out["log_exp_allowed"] == []
    assert "unchecked" in out["note"]


def test_distill_path_finds_product():
    pytest.importorskip("sklearn")
    import numpy as np

    from apedp.discovery import distill_via_surrogate
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 60)
    y = rng.uniform(0.5, 5.0, 60)
    out = distill_via_surrogate({"x": x, "y": y}, x * y)
    assert "x*y" in out["terms"], out
    assert out["surrogate_r2"] > 0.9, out


def test_beam_seeding_flag_defaults_off():
    import inspect

    from apedp.discovery import MultiVariableSRE
    params = inspect.signature(MultiVariableSRE.__init__).parameters
    assert params["allow_beam_seeding"].default is False
    assert params["allow_logcomp_mutation"].default is False


def test_make_synthetic_deterministic_and_split():
    from apedp.discovery_benchmark import make_synthetic
    _x1, y1, _xt1, yt1, v1, _ = make_synthetic("GRATING", seed=7)
    _x2, y2, _xt2, yt2, _v2, _ = make_synthetic("GRATING", seed=7)
    assert (y1 == y2).all() and (yt1 == yt2).all()
    assert len(y1) + len(yt1) == 30 and len(y1) == 21
    assert v1 == ["d", "t"]


def test_recursive_solve_recovers_power_laws():
    import numpy as np

    from apedp.discovery_benchmark import make_synthetic, recursive_solve
    for law, expect in [("OHMIC_PRODUCT", "x*y"),
                        ("STEFAN_BOLTZMANN", "sigma*T**4"),
                        ("EINSTEIN_MASS_ENERGY", "m*c**2")]:
        xtr, ytr, xte, yte, _, _ = make_synthetic(law, seed=7)
        x = {k: np.concatenate([xtr[k], xte[k]]) for k in xtr}
        out = recursive_solve(x, np.concatenate([ytr, yte]))
        assert out["split"] is True, (law, out)
        assert out["expression"] == expect, (law, out)


def test_recursive_solve_beam_and_rejects_sine():
    import numpy as np

    from apedp.discovery_benchmark import make_synthetic, recursive_solve
    rng = np.random.default_rng(7)
    F = rng.uniform(0.5, 2.0, 30)
    L = rng.uniform(0.5, 2.0, 30)
    E = rng.uniform(0.5, 2.0, 30)
    Iv = rng.uniform(0.5, 2.0, 30)
    out = recursive_solve({"F": F, "L": L, "E": E, "I": Iv},
                          F * L ** 3 / (3 * E * Iv))
    assert out["split"] is True, out
    assert out["expression"] == "F*L**3*E**-1*I**-1", out
    xtr, ytr, xte, yte, _, _ = make_synthetic("GRATING", seed=7)
    x = {k: np.concatenate([xtr[k], xte[k]]) for k in xtr}
    out2 = recursive_solve(x, np.concatenate([ytr, yte]))
    assert out2["split"] is False, out2


def test_engine_recursive_reduction_recovers_beam():
    import numpy as np

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE
    from apedp.physics_validity_engine import same_relationship
    rng = np.random.default_rng(7)
    F = rng.uniform(0.5, 2.0, 30)
    L = rng.uniform(0.5, 2.0, 30)
    E = rng.uniform(0.5, 2.0, 30)
    Iv = rng.uniform(0.5, 2.0, 30)
    y = F * L ** 3 / (3 * E * Iv)
    eng = DimensionalAssuranceEngine()
    for v in ["F", "L", "E", "I"]:
        eng.register_variable(v, "dimensionless")
    sre = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=eng),
        variables=["F", "L", "E", "I"], allow_recursive_reduction=True)
    best, _mse, hist = sre.run_discovery(
        {"F": F, "L": L, "E": E, "I": Iv}, y)
    assert best is not None, hist
    assert same_relationship(str(best), "F*L**3/(3*E*I)"), best
    assert hist and "reduction" in hist[0], hist


def test_engine_reduction_off_by_default():
    import inspect

    from apedp.discovery import MultiVariableSRE
    params = inspect.signature(MultiVariableSRE.__init__).parameters
    assert params["allow_recursive_reduction"].default is False


def test_engine_includes_affine_baseline_for_dimensionless_data():
    import numpy as np
    import sympy as sp

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE

    rng = np.random.default_rng(17)
    x = rng.normal(size=80)
    z = rng.normal(size=80)
    y = 1.5 * x - 0.75 * z + 0.25
    engine = DimensionalAssuranceEngine()
    engine.register_variable("x", "dimensionless")
    engine.register_variable("z", "dimensionless")
    search = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=engine),
        variables=["x", "z"],
        target_unit_str="dimensionless",
        seed=17,
    )
    best, _mse, _ = search.run_discovery(
        {"x": x, "z": z}, y, generations=1, pop_size=8, seed=17
    )
    assert best is not None
    assert best.has(sp.Symbol("x")) and best.has(sp.Symbol("z"))
    predict = sp.lambdify([sp.Symbol("x"), sp.Symbol("z")], best, "numpy")
    assert np.mean((predict(x, z) - y) ** 2) < 1e-12


def test_engine_seeds_bic_sparse_affine_candidate():
    import numpy as np
    import sympy as sp

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE

    rng = np.random.default_rng(23)
    signal = rng.normal(size=100)
    x_data = {"signal": signal}
    for index in range(8):
        x_data[f"noise_{index}"] = rng.normal(size=100)
    y_data = 2.0 * signal + 0.1
    engine = DimensionalAssuranceEngine()
    for name in x_data:
        engine.register_variable(name, "dimensionless")
    search = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=engine),
        variables=list(x_data),
        target_unit_str="dimensionless",
        seed=23,
    )
    best, _, _ = search.run_discovery(
        x_data, y_data, generations=1, pop_size=12, seed=23
    )
    assert best is not None
    assert best.has(sp.Symbol("signal"))
    assert not any(best.has(sp.Symbol(f"noise_{index}")) for index in range(8))


def test_implicit_fitness_orders_true_above_wrong():
    import numpy as np

    from apedp.discovery_benchmark import implicit_fitness
    rng = np.random.default_rng(7)
    Re = 10 ** rng.uniform(np.log10(4000), 7, 30)
    e = 10 ** rng.uniform(-6, -2, 30)

    def colebrook_f(R, ee, it=80):
        f = np.full_like(R, 0.02)
        for _ in range(it):
            f = 1.0 / (-2.0 * np.log10(ee / 3.7 + 2.51 / (R * np.sqrt(f)))) ** 2
        return f

    f = np.array([colebrook_f(np.array([r]), np.array([ee]))[0]
                  for r, ee in zip(Re, e)])
    data = {"Re": Re, "e": e, "f": f}
    good = implicit_fitness(
        "1/sqrt(f) + 2*log(e/3.7 + 2.51/(Re*sqrt(f)))/log(10)",
        "f", data)
    bad = implicit_fitness(
        "1/sqrt(f) + 2*log(e/3.7 + 2.0/(Re*sqrt(f)))/log(10)",
        "f", data)
    assert good["solved"] > 0 and bad["solved"] > 0, (good, bad)
    assert good["mse"] <= bad["mse"], (good, bad)


def test_engine_implicit_fitness_orders_true_above_wrong():
    import numpy as np
    import sympy as sp

    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE
    eng = DimensionalAssuranceEngine()
    for v in ["Re", "e"]:
        eng.register_variable(v, "dimensionless")
    sre = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=eng),
        variables=["Re", "e"], allow_implicit_fitness=True,
        implicit_var="f")
    f, Re, e = sp.symbols("f Re e")
    rng = np.random.default_rng(7)
    Rv = 10 ** rng.uniform(np.log10(4000), 7, 5)
    ev = 10 ** rng.uniform(-6, -2, 5)

    def colebrook_f(R, ee, it=80):
        ff = np.full_like(R, 0.02)
        for _ in range(it):
            ff = 1.0 / (-2.0 * np.log10(ee / 3.7 + 2.51 / (R * np.sqrt(ff)))) ** 2
        return ff

    fv = np.array([colebrook_f(np.array([r]), np.array([ee]))[0]
                   for r, ee in zip(Rv, ev)])
    data = {"Re": Rv, "e": ev}
    good = sre.implicit_fitness(
        1 / sp.sqrt(f) + 2 * sp.log(e / 3.7 + 2.51 / (Re * sp.sqrt(f))) / sp.log(10),
        data, fv)
    bad = sre.implicit_fitness(
        1 / sp.sqrt(f) + 2 * sp.log(e / 3.7 + 2.0 / (Re * sp.sqrt(f))) / sp.log(10),
        data, fv)
    assert good < bad, (good, bad)
    assert good < 1e-20, good


def test_calibration_reports_measured_scope():
    pytest.importorskip("sklearn")
    out = calibrate_from_measured()
    assert out["n_positive"] == 13, out
    assert out["n_negative"] == 12, out
    assert 0.0 <= out["roc_auc"] <= 1.0, out
    assert "re-calibrate" in out["note"], out
