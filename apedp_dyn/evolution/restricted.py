"""The subset of Python a candidate function may be written in, checked before it runs.

Status: experimental

A coding model writes real code, and real code can do anything. A candidate is
therefore accepted only if its syntax tree stays inside an allowlist: function
definitions, arithmetic, comparisons, loops, comprehensions, calls, and
attribute access by a name from a fixed list (array methods and the parts of
``numpy`` and ``math`` that compute). Everything else is refused before any of
it runs:

* no ``import``, class, decorator, ``global``, ``del``, ``yield``, ``finally``
  (a ``return`` inside one would swallow the jail's time limit), and no
  ``with`` except ``np.errstate``;
* no name or attribute that starts with an underscore, so no ``__class__``,
  ``__globals__`` or ``__import__``;
* no attribute outside the list, so no ``np.load``, ``.tofile``, ``.format``
  (whose replacement fields can walk attributes), ``.gi_frame``; and an
  attribute may be read, never assigned;
* a name that is read must be defined in the candidate, or be one of a few
  harmless builtins, ``np`` or ``math``;
* ``except`` must name the exception; the names it may use do not include the
  one the jail raises when time is up.

Type annotations are not checked, because they are never evaluated: the jail
compiles a candidate with deferred annotations (``COMPILE_FLAGS``).

This is the first of two barriers; the second is the process the code runs in
(``jail``). The check is by syntax: it does not prove a candidate terminates
or uses little memory, which the jail's limits handle.

**Limits.** An allowlist is as safe as its entries. Each attribute here was
chosen because it computes and neither reads nor writes anything outside the
arrays it is given; adding one is a security decision, not a convenience.
"""
from __future__ import annotations
import __future__ as _future

import ast
from collections.abc import Iterator

MAX_CHARS = 20_000
MAX_NODES = 6_000
#: how the jail has to compile a candidate: annotations are kept as text and never evaluated
COMPILE_FLAGS = _future.annotations.compiler_flag

#: builtins a candidate may call; the jail defines exactly these (``print`` as a function that does nothing)
ALLOWED_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance", "len",
    "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "str", "sum",
    "tuple", "zip", "True", "False", "None",
})
#: exceptions a candidate may catch or raise
ALLOWED_EXCEPTIONS = frozenset({
    "ArithmeticError", "AssertionError", "Exception", "FloatingPointError", "IndexError", "KeyError", "MemoryError",
    "OverflowError", "RecursionError", "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError",
})
ALLOWED_MODULES = frozenset({"np", "math"})
#: attribute names a candidate may read, on whatever object
ALLOWED_ATTRIBUTES = frozenset({
    # numpy: making and shaping arrays
    "array", "asarray", "zeros", "ones", "full", "empty", "zeros_like", "ones_like", "full_like", "arange",
    "linspace", "geomspace", "logspace", "eye", "column_stack", "vstack", "hstack", "concatenate", "stack",
    "meshgrid", "reshape", "ravel", "flatten", "transpose", "T", "shape", "size", "ndim", "dtype", "astype", "copy",
    "tolist", "item", "float64", "squeeze", "expand_dims", "atleast_1d", "atleast_2d", "newaxis", "repeat", "tile",
    "take", "delete", "diag", "triu", "tril", "triu_indices", "tril_indices", "outer", "dot", "matmul", "einsum",
    "cross", "fill",
    # numpy: elementwise
    "abs", "absolute", "sign", "sqrt", "cbrt", "exp", "expm1", "log", "log1p", "log2", "log10", "sin", "cos", "tan",
    "tanh", "sinh", "cosh", "arctan", "arctan2", "arcsin", "arccos", "power", "square", "reciprocal", "maximum",
    "minimum", "clip", "where", "isfinite", "isnan", "isinf", "nan_to_num", "floor", "ceil", "round", "hypot",
    "add", "subtract", "multiply", "divide", "negative", "mod", "logical_and", "logical_or", "logical_not",
    # numpy: reductions and order
    "sum", "prod", "mean", "std", "var", "median", "quantile", "percentile", "min", "max", "amin", "amax", "argmin",
    "argmax", "argsort", "sort", "unique", "cumsum", "cumprod", "diff", "gradient", "interp", "all", "any",
    "count_nonzero", "nonzero", "flatnonzero", "argwhere", "nanmean", "nanstd", "nanmedian", "nanmin", "nanmax",
    "nansum", "corrcoef", "cov", "average", "ptp", "isclose", "allclose", "array_equal", "isscalar", "searchsorted",
    "polyfit", "polyval",
    # numpy: constants and linear algebra
    "inf", "nan", "pi", "e", "finfo", "eps", "tiny", "linalg", "lstsq", "solve", "pinv", "inv", "norm", "svd",
    "eigh", "qr", "det", "matrix_rank", "cond",
    # numpy: seeded random numbers
    "random", "default_rng", "normal", "uniform", "standard_normal", "choice", "permutation", "integers", "shuffle",
    # numpy: silencing floating-point warnings
    "errstate",
    # math
    "isqrt", "fabs", "factorial", "comb", "gcd", "erf", "erfc", "gamma", "lgamma", "degrees", "radians", "tau",
    "copysign", "fsum", "trunc",
    # lists, dicts, sets, strings
    "append", "extend", "insert", "pop", "remove", "clear", "index", "count", "reverse", "keys", "values", "items",
    "get", "setdefault", "update", "discard", "union", "intersection", "difference", "join", "strip", "lstrip",
    "rstrip", "replace", "split", "startswith", "endswith", "lower", "upper", "real", "imag",
})

_STATEMENTS = (ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While, ast.If,
               ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.Try, ast.With, ast.Assert, ast.Raise)
_EXPRESSIONS = (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp, ast.Call, ast.Attribute, ast.Subscript,
                ast.Slice, ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp,
                ast.SetComp, ast.GeneratorExp, ast.Starred, ast.NamedExpr, ast.JoinedStr, ast.FormattedValue,
                ast.Lambda)
_STRUCTURE = (ast.Module, ast.arguments, ast.arg, ast.keyword, ast.comprehension, ast.ExceptHandler, ast.withitem,
              ast.Load, ast.Store, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)
_ALLOWED = (*_STATEMENTS, *_EXPRESSIONS, *_STRUCTURE)
_NEVER_EVALUATED = ("annotation", "returns")


class RejectedCodeError(ValueError):
    """The candidate is outside the allowed subset; the message says where and why."""


def _where(node: ast.AST) -> str:
    return f"line {getattr(node, 'lineno', '?')}"


def _nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Every node that can be evaluated: the tree without its type annotations."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        for name, value in ast.iter_fields(node):
            if name in _NEVER_EVALUATED:
                continue
            if isinstance(value, ast.AST):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(item for item in value if isinstance(item, ast.AST))


def _bound(node: ast.AST) -> str | None:
    """The name this node binds, if it binds one."""
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return node.id
    if isinstance(node, ast.ExceptHandler):
        return node.name
    return None


def _exception_names(node: ast.expr | None) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Tuple) and all(isinstance(e, ast.Name) for e in node.elts):
        return [e.id for e in node.elts if isinstance(e, ast.Name)]
    return None


def _check_node(node: ast.AST, defined: set[str]) -> None:
    if not isinstance(node, _ALLOWED):
        raise RejectedCodeError(f"{_where(node)}: {type(node).__name__} is not allowed")
    name = _bound(node)
    if name is not None and (name.startswith("_") or name in ALLOWED_MODULES):
        reason = "starts with an underscore" if name.startswith("_") else "may not be redefined"
        raise RejectedCodeError(f"{_where(node)}: the name {name!r} {reason}")
    if isinstance(node, ast.FunctionDef):
        if node.decorator_list or getattr(node, "type_params", None):
            raise RejectedCodeError(f"{_where(node)}: decorators and type parameters are not allowed")
        if node.args.vararg or node.args.kwarg:
            raise RejectedCodeError(f"{_where(node)}: *args and **kwargs parameters are not allowed")
    elif isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise RejectedCodeError(f"{_where(node)}: the name {node.id!r} starts with an underscore")
        if isinstance(node.ctx, ast.Load) and node.id not in defined and node.id not in ALLOWED_BUILTINS \
                and node.id not in ALLOWED_MODULES and node.id not in ALLOWED_EXCEPTIONS:
            raise RejectedCodeError(f"{_where(node)}: the name {node.id!r} is not defined here and is not allowed")
    elif isinstance(node, ast.Attribute):
        if not isinstance(node.ctx, ast.Load):
            raise RejectedCodeError(f"{_where(node)}: an attribute may be read, not assigned ({node.attr!r})")
        if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRIBUTES:
            raise RejectedCodeError(f"{_where(node)}: the attribute {node.attr!r} is not allowed")
    elif isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        raise RejectedCodeError(f"{_where(node)}: bytes are not allowed")
    elif isinstance(node, ast.Try) and node.finalbody:
        raise RejectedCodeError(f"{_where(node)}: finally is not allowed")
    elif isinstance(node, ast.ExceptHandler):
        caught = _exception_names(node.type)
        if caught is None or any(c not in ALLOWED_EXCEPTIONS for c in caught):
            raise RejectedCodeError(f"{_where(node)}: except has to name exceptions from {sorted(ALLOWED_EXCEPTIONS)}")
    elif isinstance(node, ast.With):
        for item in node.items:
            call = item.context_expr
            silencing = (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                         and call.func.attr == "errstate" and isinstance(call.func.value, ast.Name)
                         and call.func.value.id == "np")
            if not silencing or item.optional_vars is not None:
                raise RejectedCodeError(f"{_where(node)}: with is allowed for np.errstate(...) only")


def check(code: str, function: str, parameters: int | None = None) -> ast.Module:
    """Parse ``code`` and refuse it unless it stays inside the subset and defines ``function``.

    ``parameters`` is how many positional parameters ``function`` has to take. Returns the tree.
    Nothing of ``code`` is compiled or run here.
    """
    if not isinstance(code, str) or len(code) > MAX_CHARS:
        raise RejectedCodeError(f"the code has to be text of at most {MAX_CHARS} characters")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise RejectedCodeError(f"not Python: {type(exc).__name__}: {exc}"[:200]) from None
    nodes = list(_nodes(tree))
    if len(nodes) > MAX_NODES:
        raise RejectedCodeError(f"the code has {len(nodes)} syntax nodes; at most {MAX_NODES} are read")
    defined = {name for name in map(_bound, nodes) if name}
    for node in nodes:
        _check_node(node, defined)
    for statement in tree.body:
        docstring = isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) \
            and isinstance(statement.value.value, str)
        if not isinstance(statement, (ast.FunctionDef, ast.Assign, ast.AnnAssign)) and not docstring:
            raise RejectedCodeError(f"{_where(statement)}: only function definitions and assignments may stand at the top")
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    if len(found) != 1:
        raise RejectedCodeError(f"the code has to define exactly one function named {function!r} at the top")
    if parameters is not None and (len(found[0].args.args) != parameters or found[0].args.kwonlyargs
                                   or found[0].args.posonlyargs):
        raise RejectedCodeError(f"{function!r} has to take exactly {parameters} positional parameters")
    return tree


def describe() -> str:
    """The rules, in words a model can follow."""
    return (
        "The code is checked before it runs and refused if it leaves this subset of Python: function definitions "
        "(nested ones and lambda too), assignments, if/for/while, comprehensions, try/except with a named exception "
        "(use `except Exception` for numpy's errors; no finally), raise, assert, arithmetic, comparisons, calls, "
        "f-strings, and `with np.errstate(...)`. No import (so no scipy and no itertools: write the loops), no "
        "class, decorator, global, del or yield. No name or attribute that starts with an underscore. `np` (numpy) "
        "and `math` are already there; from them and from arrays, lists, dicts and strings only computing "
        "attributes are allowed (for example np.linalg.lstsq, np.column_stack, np.where, x.sum(), x.T, list.append, "
        "str.join), not file or system access, not np.ndarray and not str.format. Type annotations are ignored. "
        "print does nothing. Builtins: " + ", ".join(sorted(ALLOWED_BUILTINS - {"True", "False", "None"})) + "."
    )
