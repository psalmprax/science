"""Hybrid orchestrator: probe -> configure -> enumerate -> govern."""
from apedp.hybrid import hybrid_discover


def test_hybrid_ohmic_end_to_end():
    import numpy as np
    rng = np.random.default_rng(7)
    x = rng.uniform(0.5, 5.0, 30)
    y = rng.uniform(0.5, 5.0, 30)
    out = hybrid_discover({"x": x, "y": y}, x * y, ["x", "y"],
                          template_priors=["x+y"],
                          generations=5, pop_size=20, seed=7)
    assert out["reduction"] is True, out
    assert "x*y" in out["expression"].replace(" ", ""), out
    assert out["report"] is not None, out
    for field in ("REFERENCE:", "CANDIDATE:", "PHYSICAL DOMAIN:",
                  "EXPERIMENTAL VALIDATION:", "LITERATURE NOVELTY:"):
        assert field in out["report"], field
    assert out["domain"]["status"] == "PASS", out["domain"]


def test_hybrid_nonseparable_falls_through():
    import numpy as np

    from apedp.hybrid import hybrid_discover
    rng = np.random.default_rng(7)
    t = rng.uniform(0.1, 3.0, 12)
    out = hybrid_discover({"t": t}, 2.0 * t + 1.0, ["t"],
                          template_priors=["t"],
                          generations=2, pop_size=10, seed=7)
    assert out["reduction"] is False, out
    assert out["expression"], out
