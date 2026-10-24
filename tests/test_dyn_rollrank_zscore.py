"""rollrank/rollzscore pins (doc §82): causal evaluation, dimensionless output
from dimensional input, no roll-of-roll, H-family expressibility through the
real generation frontier + engine on the frozen BTC panel.
"""

import numpy as np
import pytest

from apedp_dyn.discovery_l2.ast_generator import (
    RollingOpNode,
    VariableNode,
    generate_dimensionally_typed_ast,
)
from apedp_dyn.discovery_l2.dimensional_types import Dimension


def _var(name="V", **kw):
    return VariableNode(name, Dimension(**kw) if kw else Dimension())


def test_rollrank_causal_bounded():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 200))  # non-monotonic: rank must vary
    r = RollingOpNode("rollrank", _var(), window=10).evaluate({"V": x})
    assert bool(np.isnan(r[:9]).all())
    assert bool(((r[9:] >= 0) & (r[9:] <= 1)).all())
    assert float(np.std(r[9:])) > 0.05  # genuinely discriminating, not constant


def test_rollzscore_math_and_causality():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    z = RollingOpNode("rollzscore", _var(), window=40).evaluate({"V": x})
    assert bool(np.isnan(z[:39]).all())
    w = x[100:140]
    assert abs(z[139] - (x[139] - w.mean()) / w.std()) < 1e-9


def test_dimensionless_output_from_dimensional_input():
    dim_vol = Dimension(m=1)  # volume-like dimension, whatever the registry uses
    v = VariableNode("V", dim_vol)
    assert RollingOpNode("rollrank", v).dimension.is_dimensionless()
    assert RollingOpNode("rollzscore", v).dimension.is_dimensionless()


def test_no_roll_of_roll():
    with pytest.raises(ValueError):
        RollingOpNode("rollrank", RollingOpNode("rollmean", _var()))


def test_expanding_zscore_matches_frozen_construction():
    """H-A fidelity: expanding_zscore must equal grammar_space.causal_zscore."""
    from apedp_dyn.discovery_l2.ast_generator import MIN_ZSCORE_HISTORY
    from apedp_dyn.sandbox.grammar_space import causal_zscore

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    node = RollingOpNode("expanding_zscore", _var(), window=MIN_ZSCORE_HISTORY)
    z = node.evaluate({"V": x})
    zp = causal_zscore(x, MIN_ZSCORE_HISTORY)
    assert bool(np.isnan(z[: MIN_ZSCORE_HISTORY - 1]).all())
    m = np.isfinite(zp)
    assert float(np.nanmax(np.abs(z[m] - zp[m]))) < 1e-12


def test_expanding_zscore_causal_no_future_leak():
    from apedp_dyn.discovery_l2.ast_generator import MIN_ZSCORE_HISTORY

    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 300)
    node = RollingOpNode("expanding_zscore", _var(), window=MIN_ZSCORE_HISTORY)
    z = node.evaluate({"V": x})
    x2 = x.copy()
    x2[200:] += 100.0  # perturb only the future
    z2 = node.evaluate({"V": x2})
    assert float(np.nanmax(np.abs(z[:200] - z2[:200]))) == 0.0
    assert z[250] != z2[250]  # but it does react after the perturbation


def test_expanding_zscore_handles_nan_history():
    from apedp_dyn.discovery_l2.ast_generator import MIN_ZSCORE_HISTORY
    from apedp_dyn.sandbox.grammar_space import causal_zscore

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    x[10:15] = np.nan
    node = RollingOpNode("expanding_zscore", _var(), window=MIN_ZSCORE_HISTORY)
    z = node.evaluate({"V": x})
    zp = causal_zscore(x, MIN_ZSCORE_HISTORY)
    m = np.isfinite(zp)
    assert float(np.nanmax(np.abs(z[m] - zp[m]))) < 1e-12


def test_generation_expresses_h_family():
    nodes = generate_dimensionally_typed_ast(
        variables={"V": Dimension(), "r": Dimension()},
        operators=("+", "-", "*", "/", "abs", "log", "rollmean", "rollrank", "rollzscore"),
        max_depth=3, max_complexity=8,
    )
    strs = [n.to_str() for n in nodes]
    # H-RV witness: bare-leaf rollrank collapses on monotonic probe grids
    # (constant-1.0 fingerprint) so the frontier carries it on abs/log
    # children instead; on strictly positive real volume abs(V)==V exactly.
    assert any("rollrank40(abs(V))" in s for s in strs), "H-RV form missing"
    assert any("rollzscore40" in s for s in strs), "H-A components missing"
    assert any("abs(r)" in s for s in strs), "H-P form missing"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
