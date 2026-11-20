"""Laws from outside the repository, read without executing anything they contain.

Status: experimental

``odebench()`` returns the right-hand sides of ODEBench (d'Ascoli et al., the
benchmark of the ODEFormer paper): 63 dynamical systems, most from Strogatz's
*Nonlinear Dynamics and Chaos*, each with published parameter values. The file is
fetched once from the ODEFormer repository, cached in the ignored ``data/mmlm/``
and refused if its SHA-256 differs from the pin. It is a Python source file; it
is parsed with :mod:`ast` and never imported or evaluated, and each equation
string is converted to a SymPy expression by a small whitelist walker
(:func:`parse_expression`), not by ``sympify``.

Every component of a system is one law. Parameters ``c_i`` are replaced by the
first published set of values, so they become numeric constants exactly as the
constants in SRBench's formulas do.
"""
from __future__ import annotations

import ast
import hashlib
import operator
import urllib.request
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "mmlm"
ODEBENCH_URL = "https://raw.githubusercontent.com/sdascoli/odeformer/main/odeformer/odebench/strogatz_equations.py"
ODEBENCH_SHA256 = "06bbb9dae2886a82f0a1d4b0cd062d063241bcfed56c01ef2a3a01d863bcf8b4"
_FUNCTIONS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
              "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh, "abs": sp.Abs, "Abs": sp.Abs, "cot": sp.cot,
              "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def parse_expression(text: str, values: dict[str, float] | None = None) -> sp.Expr:
    """A SymPy expression from arithmetic text: numbers, names, ``+ - * / **`` and whitelisted functions only.

    ``^`` is read as a power, as the ODEBench strings write it.
    """
    values = values or {}

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
        if isinstance(node, ast.Name):
            if node.id in values:
                value = values[node.id]
                return sp.Integer(int(value)) if float(value).is_integer() else sp.Float(value)
            if node.id == "pi":
                return sp.pi
            return sp.Symbol(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.Call) and not node.keywords:
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in _FUNCTIONS:
                return _FUNCTIONS[name](*(walk(arg) for arg in node.args))
        raise ValueError(f"not an arithmetic expression: {ast.dump(node)[:80]}")

    return walk(ast.parse(text.strip().replace("^", "**"), mode="eval"))


def _number(node: ast.AST) -> float:
    return float(parse_expression(ast.unparse(node)))


def _odebench_source() -> str:
    path = CACHE / "odebench_strogatz_equations.py"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ODEBENCH_URL, timeout=120) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ODEBENCH_SHA256:
        raise ValueError(f"{path} does not match its pinned SHA-256")
    return data.decode("utf-8")


def odebench_entries(source: str) -> list[dict[str, Any]]:
    """The ``equations`` list of the ODEBench file as (id, dim, component texts, first constants)."""
    out: list[dict[str, Any]] = []
    for statement in ast.parse(source).body:
        if not (isinstance(statement, ast.Assign) and getattr(statement.targets[0], "id", "") == "equations"):
            continue
        assert isinstance(statement.value, ast.List)
        for item in statement.value.elts:
            assert isinstance(item, ast.Dict)
            fields = {key.value: value for key, value in zip(item.keys, item.values, strict=True) if isinstance(key, ast.Constant)}
            consts = fields["consts"]
            assert isinstance(consts, ast.List) and isinstance(consts.elts[0], ast.List)
            out.append({"id": int(_number(fields["id"])), "dim": int(_number(fields["dim"])),
                        "components": [part.strip() for part in ast.literal_eval(fields["eq"]).split("|")],
                        "consts": [_number(c) for c in consts.elts[0].elts]})
    return out


def odebench() -> list[dict[str, Any]]:
    """One law per component: name, SymPy expression in ``x0, x1, ...`` (only the variables it uses), text."""
    laws = []
    for entry in odebench_entries(_odebench_source()):
        values = {f"c_{i}": value for i, value in enumerate(entry["consts"])}
        for k, text in enumerate(entry["components"]):
            expr = parse_expression(text, values)
            symbols = sorted(expr.free_symbols, key=lambda s: int(str(s).split("_")[1]))
            expr = expr.xreplace({s: sp.Symbol(f"x{i}") for i, s in enumerate(symbols)})
            laws.append({"name": f"odebench_{entry['id']:02d}_{k}", "expr": expr, "n_vars": len(symbols), "text": text})
    return laws
