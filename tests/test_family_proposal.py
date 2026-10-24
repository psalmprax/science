# Databricks notebook source
import numpy as np
import sympy as sp

from apedp.family_proposal import (
    discover_with_proposed_family,
    select_univariate_family,
)
from apedp.turnover_models import select_log_scale_turnover


def test_selects_shifted_power_for_offset_decay():
    x = np.arange(1, 80, dtype=float)
    y = 30.0 / (x + 0.75) ** 1.3 + 0.2
    candidate = select_univariate_family(x, y, variable="time")
    assert candidate.family == "shifted-power"
    assert candidate.mse < 1e-3


def test_recovers_exponential_decay_with_a_low_error_candidate():
    x = np.linspace(0.0, 10.0, 80)
    y = 2.5 * np.exp(-0.7 * x)
    candidate = select_univariate_family(x, y, variable="time")
    predict = sp.lambdify(sp.Symbol("time"), candidate.expression, "numpy")
    assert np.mean((predict(x) - y) ** 2) < 1e-3


def test_turnover_selection_detects_nanocrystal_peak():
    from apedp.ingestion import NanocrystallineMicrocompressionBenchmark

    grain_size, strength, _, _ = NanocrystallineMicrocompressionBenchmark.get_dataset()
    candidate = select_log_scale_turnover(grain_size.ravel() / 1e-9, strength / 1e6, variable="grain_nm")
    assert candidate is not None
    assert 8.0 < candidate.peak_x < 20.0


def test_discovery_wrapper_records_training_only_family_proposal():
    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.discovery import MultiVariableSRE

    x = np.arange(1, 40, dtype=float)
    y = 20.0 / (x + 0.5) ** 1.1 + 0.2
    engine = DimensionalAssuranceEngine()
    engine.register_variable("time", "dimensionless")
    search = MultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=engine),
        independent_vars=["time"],
        target_unit_str="dimensionless",
        seed=17,
    )
    expression, _, history = discover_with_proposed_family(
        search, {"time": x}, y, generations=1, pop_size=10, seed=17
    )
    assert expression is not None
    assert history[0]["proposed_family"] == "shifted-power"


def test_pipeline_uses_family_guided_discovery_class():
    from apedp.family_guided_discovery import FamilyGuidedMultiVariableSRE
    from apedp.pipeline import MultiVariableSRE as PipelineSRE

    assert PipelineSRE is FamilyGuidedMultiVariableSRE


# COMMAND ----------

def test_validation_selector_uses_untouched_holdout():
    from apedp.family_proposal import select_univariate_family_validated

    x = np.arange(1, 100, dtype=float)
    y = 25.0 / (x + 0.7) ** 1.25 + 0.3
    candidate = select_univariate_family_validated(
        x[:70], y[:70], x[70:], y[70:], variable="time"
    )
    assert candidate is not None
    assert candidate.family == "shifted-power"


# COMMAND ----------

def test_family_guided_engine_records_validation_selection():
    from apedp.core import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine
    from apedp.family_guided_discovery import FamilyGuidedMultiVariableSRE

    x = np.arange(1, 50, dtype=float)
    y = 18.0 / (x + 0.6) ** 1.2 + 0.2
    engine = DimensionalAssuranceEngine()
    engine.register_variable("time", "dimensionless")
    search = FamilyGuidedMultiVariableSRE(
        ClosedLoopAssuranceSystem(assurance_engine=engine),
        independent_vars=["time"],
        target_unit_str="dimensionless",
        seed=19,
    )
    expression, _, history = search.run_discovery(
        {"time": x}, y, generations=1, pop_size=10, seed=19
    )
    assert expression is not None
    assert history[0]["proposal_selection"] == "trailing_validation"
    assert history[0]["proposal_validation_rows"] >= 3
