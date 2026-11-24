"""Equality saturation over the expressions of the law grammar: showing two expressions equal by rewrites.

Status: experimental

An e-graph holds many expressions at once, with every class of nodes known to be
equal merged into one. Rewrite rules (commutativity, associativity,
distribution, powers, ``exp`` of a sum, constant folding) add equal forms until
nothing new appears or a size limit is reached. Two expressions are **shown
equal** when they end in the same class: the rewrites that merged them are a
proof, under the rules used. When they do not, nothing is concluded: the rules
are incomplete and the search is bounded, so ``equal`` answers ``True`` or
``None``, never ``False``.

The rules are sound for real numbers where both sides are defined. The rules
that need positive arguments (``log`` of a product, a power of a power with a
non-integer exponent) are used only with ``positive=True``, which the law
grammar's domain (inputs on (1, 5)) allows for its inputs but not for every
sub-expression, so it is off by default.

``simplest`` extracts the smallest member of a class: a canonical short form,
used to count two ways of writing one building block as one.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import sympy as sp

FUNCTIONS = ("exp", "log", "sin", "cos", "tanh", "tan")
NODE_LIMIT = 4000
ITERATIONS = 30

Node = tuple[str, Any, tuple[int, ...]]           # (operation, payload, children)


class Unsupported(ValueError):
    """An expression with a part this grammar does not write."""


def _number(value: float) -> float:
    return float(f"{value:.12g}")


def term_of(expr: sp.Expr) -> tuple[Any, ...]:
    """A sympy expression as a nested tuple: ``("add", a, b)``, ``("var", "x0")``, ``("num", 2.0)``."""
    if expr.is_Symbol:
        return ("var", str(expr))
    if expr.is_Number:
        value = float(expr)
        if not math.isfinite(value):
            raise Unsupported(f"not a finite number: {expr}")
        return ("num", _number(value))
    if expr.is_Add or expr.is_Mul:
        op = "add" if expr.is_Add else "mul"
        args = [term_of(arg) for arg in expr.args]
        out = args[0]
        for arg in args[1:]:
            out = (op, out, arg)
        return out
    if expr.is_Pow:
        return ("pow", term_of(expr.base), term_of(expr.exp))
    name = type(expr).__name__
    if name in FUNCTIONS and len(expr.args) == 1:
        return (name, term_of(expr.args[0]))
    raise Unsupported(f"not in the grammar: {name}")


def sympy_of(term: tuple[Any, ...]) -> sp.Expr:
    op = term[0]
    if op == "var":
        return sp.Symbol(term[1])
    if op == "num":
        value = term[1]
        return sp.Integer(int(value)) if float(value).is_integer() else sp.Float(value)
    if op == "add":
        return sympy_of(term[1]) + sympy_of(term[2])
    if op == "mul":
        return sympy_of(term[1]) * sympy_of(term[2])
    if op == "pow":
        return sympy_of(term[1]) ** sympy_of(term[2])
    return getattr(sp, op)(sympy_of(term[1]))


@dataclass
class Saturation:
    """How a saturation ended."""

    iterations: int
    nodes: int
    classes: int
    saturated: bool              #: no rule added anything new
    limited: bool                #: stopped by the node limit


class EGraph:
    """Classes of equal expressions, with congruence kept by ``rebuild``."""

    def __init__(self, *, positive: bool = False, node_limit: int = NODE_LIMIT) -> None:
        self.parent: list[int] = []
        self.classes: dict[int, set[Node]] = {}
        self.memo: dict[Node, int] = {}
        self.positive = positive
        self.node_limit = int(node_limit)

    # ------------------------------------------------------------------ basics
    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def _canon(self, node: Node) -> Node:
        return (node[0], node[1], tuple(self.find(c) for c in node[2]))

    def add_node(self, node: Node) -> int:
        node = self._canon(node)
        if node in self.memo:
            return self.find(self.memo[node])
        cid = len(self.parent)
        self.parent.append(cid)
        self.classes[cid] = {node}
        self.memo[node] = cid
        return cid

    def add(self, term: tuple[Any, ...]) -> int:
        """Add a nested-tuple term; a leaf ``("class", id)`` stands for a class already present."""
        op = term[0]
        if op == "class":
            return self.find(int(term[1]))
        if op == "var":
            return self.add_node(("var", term[1], ()))
        if op == "num":
            return self.add_node(("num", _number(float(term[1])), ()))
        return self.add_node((op, None, tuple(self.add(child) for child in term[1:])))

    def add_expr(self, expr: sp.Expr) -> int:
        return self.add(term_of(expr))

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if len(self.classes[a]) < len(self.classes[b]):
            a, b = b, a
        self.parent[b] = a
        self.classes[a] |= self.classes.pop(b)
        return True

    def rebuild(self) -> None:
        """Restore congruence: nodes with equal operations on equal classes are one class."""
        while True:
            memo: dict[Node, int] = {}
            merged = False
            for cid in list(self.classes):
                if cid not in self.classes:
                    continue
                root = self.find(cid)
                nodes = {self._canon(n) for n in self.classes[root]}
                self.classes[root] = nodes
                for node in tuple(nodes):
                    other = memo.get(node)
                    if other is None:
                        memo[node] = root
                    elif self.find(other) != self.find(root):
                        merged |= self.union(other, root)
            self.memo = {n: self.find(c) for n, c in memo.items()}
            if not merged:
                return

    def size(self) -> int:
        return sum(len(nodes) for nodes in self.classes.values())

    def nodes_of(self, cid: int) -> set[Node]:
        return self.classes[self.find(cid)]

    def constant(self, cid: int) -> float | None:
        for node in self.nodes_of(cid):
            if node[0] == "num":
                return float(node[1])
        return None

    # ------------------------------------------------------------------ rules
    def _open(self, cid: int) -> set[Node]:
        """The nodes of a class the rules may look inside: none if the class is a number.

        A class equal to a number holds every way of writing it that was met (``a - a``,
        ``0 * b``); rewriting those further adds nodes without end and shows nothing new.
        """
        nodes = self.nodes_of(cid)
        return set() if any(node[0] == "num" for node in nodes) else nodes

    def _matches(self) -> Iterator[tuple[int, tuple[Any, ...]]]:
        """Every (class, term equal to it) the rules give for the current graph."""
        C = ("class",)
        for cid in list(self.classes):
            for op, _payload, ch in list(self.classes.get(cid, ())):
                if op in ("add", "mul"):
                    a, b = ch
                    yield cid, (op, (*C, b), (*C, a))                                      # commutativity
                    for op2, _p2, ch2 in self._open(a):
                        if op2 == op:                                                      # (x op y) op b = x op (y op b)
                            yield cid, (op, (*C, ch2[0]), (op, (*C, ch2[1]), (*C, b)))
                    ka, kb = self.constant(a), self.constant(b)
                    if ka is not None and kb is not None:
                        value = ka + kb if op == "add" else ka * kb
                        if math.isfinite(value):
                            yield cid, ("num", value)                                      # constant folding
                    unit = 0.0 if op == "add" else 1.0
                    if kb == unit:
                        yield cid, (*C, a)                                                 # a + 0 = a, a * 1 = a
                    if op == "mul" and kb == 0.0:
                        yield cid, ("num", 0.0)
                    if op == "add" and self.find(a) == self.find(b):
                        yield cid, ("mul", ("num", 2.0), (*C, a))
                    if op == "mul":
                        if self.find(a) == self.find(b):
                            yield cid, ("pow", (*C, a), ("num", 2.0))
                        for op2, _p2, ch2 in self._open(b):
                            if op2 == "add":                                               # a (x + y) = a x + a y
                                yield cid, ("add", ("mul", (*C, a), (*C, ch2[0])), ("mul", (*C, a), (*C, ch2[1])))
                            if op2 == "pow" and self.find(ch2[0]) == self.find(a):
                                n = self.constant(ch2[1])
                                if n is not None:                                          # a a^n = a^(n+1)
                                    yield cid, ("pow", (*C, a), ("num", n + 1.0))
                        for op2, _p2, ch2 in self._open(a):
                            for op3, _p3, ch3 in self._open(b):
                                if op2 == op3 == "pow" and self.find(ch2[0]) == self.find(ch3[0]):
                                    m, n = self.constant(ch2[1]), self.constant(ch3[1])
                                    if m is not None and n is not None:                    # a^m a^n = a^(m+n)
                                        yield cid, ("pow", (*C, ch2[0]), ("num", m + n))
                                if op2 == op3 == "exp":                                    # e^x e^y = e^(x+y)
                                    yield cid, ("exp", ("add", (*C, ch2[0]), (*C, ch3[0])))
                    if op == "add":
                        for op2, _p2, ch2 in self._open(a):
                            for op3, _p3, ch3 in self._open(b):
                                if op2 == op3 == "mul" and self.find(ch2[0]) == self.find(ch3[0]):
                                    yield cid, ("mul", (*C, ch2[0]), ("add", (*C, ch2[1]), (*C, ch3[1])))
                                if op3 == "mul" and self.constant(ch3[0]) == -1.0 and self.find(ch3[1]) == self.find(a):
                                    yield cid, ("num", 0.0)                                # a - a = 0
                elif op == "pow":
                    base, exponent = ch
                    n = self.constant(exponent)
                    kb = self.constant(base)
                    if n == 1.0:
                        yield cid, (*C, base)
                    if n in (2.0, 3.0):                                                    # a^n = a a^(n-1)
                        yield cid, ("mul", (*C, base), ("pow", (*C, base), ("num", n - 1.0)))
                    if n == 0.0:
                        yield cid, ("num", 1.0)
                    if kb is not None and n is not None and (kb > 0 or float(n).is_integer()):
                        try:
                            value = kb ** n
                        except (OverflowError, ZeroDivisionError):
                            value = math.nan
                        if isinstance(value, float) and math.isfinite(value):
                            yield cid, ("num", value)
                    if n is not None and (float(n).is_integer() or self.positive):
                        for op2, _p2, ch2 in self._open(base):
                            if op2 == "mul":                                               # (a b)^n = a^n b^n
                                yield cid, ("mul", ("pow", (*C, ch2[0]), ("num", n)), ("pow", (*C, ch2[1]), ("num", n)))
                            if op2 == "pow":
                                m = self.constant(ch2[1])
                                if m is not None and ((float(m).is_integer() and float(n).is_integer()) or self.positive):
                                    yield cid, ("pow", (*C, ch2[0]), ("num", m * n))
                elif op == "exp":
                    (arg,) = ch
                    for op2, _p2, ch2 in self._open(arg):
                        if op2 == "add":
                            yield cid, ("mul", ("exp", (*C, ch2[0])), ("exp", (*C, ch2[1])))
                    if self.constant(arg) == 0.0:
                        yield cid, ("num", 1.0)
                elif op == "log" and self.positive:
                    (arg,) = ch
                    for op2, _p2, ch2 in self._open(arg):
                        if op2 == "mul":
                            yield cid, ("add", ("log", (*C, ch2[0])), ("log", (*C, ch2[1])))
                        if op2 == "pow":
                            yield cid, ("mul", (*C, ch2[1]), ("log", (*C, ch2[0])))

    def saturate(self, iterations: int = ITERATIONS) -> Saturation:
        """Apply every rule until nothing changes, the iterations run out or the node limit is passed."""
        done = 0
        for done in range(1, iterations + 1):
            found = list(self._matches())
            before = (self.size(), len(self.classes))
            for cid, term in found:
                if self.size() > self.node_limit:
                    break
                self.union(cid, self.add(term))
            self.rebuild()
            if (self.size(), len(self.classes)) == before:
                return Saturation(done, self.size(), len(self.classes), True, False)
            if self.size() > self.node_limit:
                return Saturation(done, self.size(), len(self.classes), False, True)
        return Saturation(done, self.size(), len(self.classes), False, False)

    # ------------------------------------------------------------- extraction
    def simplest(self, cid: int) -> tuple[Any, ...]:
        """The member of a class with the fewest nodes."""
        best: dict[int, tuple[int, tuple[Any, ...]]] = {}
        changed = True
        while changed:
            changed = False
            for root, nodes in self.classes.items():
                for op, payload, ch in nodes:
                    if any(self.find(c) not in best for c in ch):
                        continue
                    term: tuple[Any, ...]
                    if op in ("var", "num"):
                        cost, term = 1, (op, payload)
                    else:
                        cost = 1 + sum(best[self.find(c)][0] for c in ch)
                        term = (op, *(best[self.find(c)][1] for c in ch))
                    if root not in best or cost < best[root][0]:
                        best[root] = (cost, term)
                        changed = True
        return best[self.find(cid)][1]


def equal(a: sp.Expr, b: sp.Expr, *, positive: bool = False, iterations: int = ITERATIONS,
          node_limit: int = NODE_LIMIT) -> dict[str, Any]:
    """Whether rewrites show ``a`` equal to ``b``: ``{"equal": True}`` with how, or ``{"equal": None}``. Never ``False``."""
    graph = EGraph(positive=positive, node_limit=node_limit)
    try:
        left, right = graph.add_expr(sp.sympify(a)), graph.add_expr(sp.sympify(b))
    except Unsupported as exc:
        return {"equal": None, "reason": str(exc)}
    if graph.find(left) == graph.find(right):
        return {"equal": True, "iterations": 0, "nodes": graph.size()}
    status = Saturation(0, graph.size(), len(graph.classes), False, False)
    for step in range(1, iterations + 1):
        status = graph.saturate(1)
        if graph.find(left) == graph.find(right):
            return {"equal": True, "iterations": step, "nodes": status.nodes}
        if status.saturated or status.limited:
            break
    why = "saturated without merging them" if status.saturated else (
        "node limit reached" if status.limited else "iteration limit reached")
    return {"equal": None, "reason": why, "nodes": status.nodes}


def simplest(expr: sp.Expr, *, positive: bool = False, iterations: int = 4, node_limit: int = NODE_LIMIT) -> sp.Expr:
    """The smallest form of ``expr`` the rewrites reach (the expression itself if nothing smaller is found)."""
    graph = EGraph(positive=positive, node_limit=node_limit)
    try:
        root = graph.add_expr(sp.sympify(expr))
    except Unsupported:
        return sp.sympify(expr)
    graph.saturate(iterations)
    return sympy_of(graph.simplest(root))
