"""The tensor-train equation engine with a fitted readout (apedp_mmlm/tensor_train_equations.py)."""
from __future__ import annotations

import numpy as np
import sympy as sp

from apedp_mmlm import equation_language as el
from apedp_mmlm.tensor_train_equations import TensorTrainEquationModel

a, b, c = sp.symbols("a b c")
TREES = [el.to_tree(e) for e in (a * b, a / b, a * b**2, a * b / c, sp.sqrt(a / b), a / b**2, a * b * c, a**2 * b,
                                 a * b / c**2, sp.sqrt(a * b), a / (b * c), a * b**2 / c)]


def test_the_fitted_readout_beats_a_uniform_code_where_the_counted_one_does_not():
    held_out = el.to_tree(a**2 * b / c)
    uniform = el.EquationLanguageModel("uniform").bits(held_out)
    fitted = TensorTrainEquationModel().fit(TREES).bits(held_out)
    counted = el.EquationLanguageModel("tensor_train").fit(TREES).bits(held_out)
    assert np.isfinite(fitted) and fitted < uniform
    assert fitted < counted


def test_training_is_deterministic():
    one = TensorTrainEquationModel().fit(TREES).bits(TREES[0])
    assert one == TensorTrainEquationModel().fit(TREES).bits(TREES[0])
