"""Regression pins for discovery-L2 generation bounds and basis screening.

Covers two measured failure modes on the quadratic-drag zoo problem
(step-forced; train velocity never negative):

- ``_canonical_key_and_validity`` ran SymPy ``cancel``/``trigsimp`` on every
  generated node; one trigger-mass expression could stall generation
  indefinitely (>300 s observed). Expressions above the op budget must take
  the ``to_str()`` fallback, while small expressions keep their normalized
  monic SymPy key.
- Basis screening ranked every term by raw target correlation, which the
  step's pre-step dead band tilts toward drive-carrying mimics, pushing the
  drag family below the basis cutoff. The screen now reserves slots for
  univariate shape terms (MultiTermL2Engine Step 2); the end-to-end guard
  for that is ``tests/test_dyn_quadratic_drag_discovery.py``.
"""
import sympy as sp

from apedp_dyn.discovery_l2.ast_generator import (
    BinaryOpNode,
    ConstantNode,
    PowNode,
    UnaryOpNode,
    VariableNode,
    _canonical_key_and_validity,
    collect_variable_names,
)
from apedp_dyn.discovery_l2.dimensional_types import DIMENSIONLESS


def _x(name: str = "x") -> VariableNode:
    return VariableNode(name, DIMENSIONLESS)


class TestCanonicalizationBound:
    def test_small_expression_takes_sympy_normal_form(self):
        ok, key = _canonical_key_and_validity(PowNode(_x(), 3))
        assert ok is True
        assert key == "x**3"

    def test_coefficient_scale_is_normalized_away(self):
        tiny = BinaryOpNode("*", ConstantNode(2.0), PowNode(_x(), 2))
        large = BinaryOpNode("*", ConstantNode(5.0), PowNode(_x(), 2))
        assert (_canonical_key_and_validity(tiny)[1]
                == _canonical_key_and_validity(large)[1])

    def test_equivalent_trig_forms_share_one_key(self):
        left = BinaryOpNode("*", ConstantNode(2.0),
                            BinaryOpNode("*", UnaryOpNode("sin", _x()),
                                         UnaryOpNode("cos", _x())))
        right = BinaryOpNode("*", ConstantNode(2.0),
                             BinaryOpNode("*", UnaryOpNode("cos", _x()),
                                          UnaryOpNode("sin", _x())))
        ok_a, key_a = _canonical_key_and_validity(left)
        ok_b, key_b = _canonical_key_and_validity(right)
        assert ok_a and ok_b
        assert key_a == key_b

    def test_large_expression_skips_sympy_and_falls_back_to_to_str(self):
        node: object = _x()
        for _ in range(26):
            node = UnaryOpNode("sin", node)
        assert sp.count_ops(node.to_sympy(), visual=False) > 24
        ok, key = _canonical_key_and_validity(node)
        assert ok is True
        assert key == node.to_str()


class TestCollectVariableNames:
    def test_drag_shape_is_univariate(self):
        drag = BinaryOpNode("*", UnaryOpNode("abs", _x("v")), _x("v"))
        assert collect_variable_names(drag) == frozenset({"v"})

    def test_composite_reports_every_variable(self):
        comp = BinaryOpNode("*", _x("F"), UnaryOpNode("sin", _x("x")))
        assert collect_variable_names(comp) == frozenset({"F", "x"})

    def test_constants_carry_no_variables(self):
        assert collect_variable_names(ConstantNode(3.0)) == frozenset()
