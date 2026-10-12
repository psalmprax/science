"""Unit tests for CompetitorAbstentionWrapper and Dimensional Constant Synthesis."""
import sympy as sp

from apedp.dimensional_layer import audit_dimensional_admissibility
from apedp.pipeline import DiscoveryPipeline
from benchmarks.abstention_wrapper import CompetitorAbstentionWrapper
from benchmarks.tournament_runner import load_problem_public_data


def test_audit_dimensional_admissibility_positive():
    """Verify that physical formulas with dimensional constants pass admissibility."""
    var_units = {"a": "astronomical_unit", "T": "year", "r": "meter"}

    # Kepler: T = a**1.5
    res = audit_dimensional_admissibility(sp.sympify("a**1.5"), var_units, "year")
    assert res["admissible"] is True
    assert res["status"] == "VALID"

    # Inverse square: F = 10.0 / r**2
    res2 = audit_dimensional_admissibility(sp.sympify("10.0 / r**2"), var_units, "newton")
    assert res2["admissible"] is True
    assert res2["status"] == "VALID"

    # Affine model: 1.8*a - 0.5
    res3 = audit_dimensional_admissibility(sp.sympify("1.8*a - 0.5"), var_units, "year")
    assert res3["admissible"] is True
    assert res3["status"] == "VALID"


def test_audit_dimensional_admissibility_negative():
    """Verify that uncompensable physical violations are decisively rejected."""
    var_units = {"x": "meter", "t": "second", "r": "meter"}

    # Transcendental applied to bare dimensioned symbol
    res_sin = audit_dimensional_admissibility(sp.sympify("sin(r)"), var_units, "joule")
    assert res_sin["admissible"] is False
    assert res_sin["status"] == "INVALID"

    # Incompatible addition of bare dimensioned symbols
    res_add = audit_dimensional_admissibility(sp.sympify("x + t"), var_units, "meter")
    assert res_add["admissible"] is False
    assert res_add["status"] == "INVALID"

    # Exponent with dimensions
    res_pow = audit_dimensional_admissibility(sp.sympify("x**t"), var_units, "meter")
    assert res_pow["admissible"] is False
    assert res_pow["status"] == "INVALID"


def test_pipeline_validate_dimensions_with_synthesis():
    """Verify DiscoveryPipeline.validate_dimensions handles dimensional constant synthesis."""
    # Fixed path under the system temp dir: harmless in a test process, and the
    # pipeline only uses it as a persistence root.
    pipeline = DiscoveryPipeline("/tmp/test_pipeline_val_dim")  # noqa: S108
    pipeline.engine.register_variable("a", "astronomical_unit")
    pipeline.engine.register_variable("T", "year")

    # Kepler expression
    expr = sp.sympify("a**1.5")
    val_res = pipeline.validate_dimensions(expr, "year", allow_constant_synthesis=True)
    assert val_res["dimensionally_consistent"] is True
    assert val_res.get("synthesis_applied") is True


def test_competitor_abstention_wrapper_execution(tmp_path):
    """Verify CompetitorAbstentionWrapper executes end-to-end on P01."""
    wrapper = CompetitorAbstentionWrapper(output_base_dir=tmp_path / "wrapper_test")
    p1_data = load_problem_public_data("P01")

    res = wrapper.evaluate_with_decision_layer(
        problem_id="P01",
        public_data=p1_data,
        competitor_candidates=["a**1.5", "1.8*a - 0.5"],
        competitor_name="test_competitor",
        include_neural_adversary=True,
    )

    assert res["problem_id"] == "P01"
    assert res["wrapped_competitor"] == "test_competitor"
    assert res["epistemic_state"] in ["UNIQUE_SUPPORT", "MULTI_HYPOTHESIS"]
    assert res["top_expression"] == "a**1.5"
    assert res["dimensional_status"] == "VALID"
    assert res["near_ood_mse"] < 1e-10
