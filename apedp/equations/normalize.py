"""LaTeX cleanup and LaTeX->SymPy normalization with quarantine."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_OR_SEP = "|||"


def split_alternatives(raw: str) -> list:
    """Split 'A or B or C' display-math into separate candidate strings."""
    s = raw or ""
    s = re.sub(r"(\\quad|\\qquad|\s)*\{?\\text\{\s*or\s*\}\}?(\s|\\quad|\\qquad)*",
               _OR_SEP, s)
    parts = s.split(_OR_SEP)
    return [p.strip(" .,;") for p in parts if p.strip(" .,;")]


def clean_latex(raw: str) -> str:
    s = (raw or "").strip()
    s = s.strip("$")
    s = re.sub(r"\{\\displaystyle\s*", "{", s)
    s = re.sub(r"\\displaystyle\s*", "", s)
    s = re.sub(r"\\begin\{[^}]*\}", " ", s)
    s = re.sub(r"\\end\{[^}]*\}", " ", s)
    s = re.sub(r"\\text\{[^}]*\}", " ", s)
    s = re.sub(r"\\(quad|qquad|;|,|!| )", " ", s)
    s = re.sub(r"\\(left|right|big|bigg|Big|Bigg)\s*([(\[{|)\]}|.])", r"\2", s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\boldsymbol\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\{([A-Za-z]+)\}\s*\\over\s*\{([A-Za-z0-9_^ ]+)\}",
               r"(\1)/(\2)", s)
    for _g, _a in (("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma"),
                   ("delta", "delta"), ("epsilon", "epsilon"),
                   ("varepsilon", "epsilon"), ("zeta", "zeta"),
                   ("eta", "eta"), ("theta", "theta"), ("vartheta", "theta"),
                   ("kappa", "kappa"), ("lambda", "lam"), ("mu", "mu"),
                   ("nu", "nu"), ("xi", "xi"), ("pi", "pi"),
                   ("rho", "rho"), ("sigma", "sigma"), ("tau", "tau"),
                   ("phi", "phi"), ("varphi", "phi"), ("chi", "chi"),
                   ("psi", "psi"), ("omega", "omega"),
                   ("nabla", "nabla"), ("partial", "d"),
                   ("infty", "oo"), ("hbar", "h_bar")):
        s = re.sub(r"\\" + _g + r"(?![A-Za-z])", _a, s)
    s = re.sub(r"\\(hat|vec|bar|dot|tilde|overline)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*\[([^]]*)\]\s*\{([^}]*)\}", r"(\2)**(1/(\1))", s)
    s = re.sub(r"\\(frac|dfrac)\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\2)/(\3)", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\,", "")
    for fn in ("sin", "cos", "tan", "log", "ln", "exp", "sqrt"):
        s = re.sub(r"\\" + fn + r"(?![A-Za-z])", fn, s)
    # Trig powers: sin^2 x -> sin(x)**2.
    s = re.sub(r"\b(sin|cos|tan)\^(\d+)\s+([A-Za-z_][A-Za-z0-9_]*)",
               r"\1(\3)**\2", s)
    s = s.replace("{", "").replace("}", "")
    # Implicit products of single capitals: GV->G*V, IR->I*R.
    # Assumption: adjacent uppercase letters are separate symbols, not one
    # multi-letter name. Documented limitation; failures quarantine honestly.
    s = re.sub(r"(?<=[A-Z])(?=[A-Z])", "*", s)
    # Digit-letter products are unambiguous: 2gh->2*g*h.
    s = re.sub(r"(?<=[0-9])(?=[A-Za-z_])", "*", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;")


# Bare "I" is electric current in harvested electronics equations, but the
# shared safe_parse_expr reserves I for the imaginary unit. Scope the override
# to this pipeline only so core behavior is unchanged.
def physics_symbols():
    import sympy as sp
    return {"I": sp.Symbol("I")}


def is_tensor_notation(cleaned: str) -> bool:
    """Detect index/tensor markup (c_pqrs, T_ij) for explicit quarantine.

    Single-digit state labels (P_1, V_2) are legitimate textbook usage and
    pass through. Multi-character subscripts (c_pqrs) or swarms of indexed
    symbols indicate tensor/index calculus outside this pipeline's scope.
    """
    syms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    indexed = [s for s in syms if "_" in s]
    multi_sub = [s for s in indexed if len(s.split("_", 1)[1]) > 1]
    return len(multi_sub) >= 2 or len(indexed) >= 6


def latex_to_sympy_str(cleaned: str) -> tuple:
    """Return (ok, expr_str, notes). Quarantines unparseable input.

    Equations with '=' normalize to residual form (lhs - rhs) so that
    F=ma, a=F/m, ma-F=0 all enter canonicalization as comparable residuals.
    """
    # Lone tokens are fragments, not equations — quarantine before parsing so
    # that x/x (has operator, simplifies to 1) still reaches identity detection.
    if not re.search(r"[=+\-*/^]", cleaned):
        return False, cleaned, "quarantine: fragment (lone symbol/number)"
    if is_tensor_notation(cleaned):
        return False, cleaned, "quarantine: tensor/index notation out of scope"
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore
        expr = parse_latex(cleaned)
        return True, str(expr), "latex-parse"
    except (ImportError, TypeError, ValueError, ArithmeticError,
            AttributeError) as exc:
        logger.debug("LaTeX parse unavailable for %r; falling back to sympy: %s", cleaned, exc)
    try:
        from apedp.gated_assurance import safe_parse_expr
        syms = physics_symbols()
        if "=" in cleaned:
            lhs, rhs = cleaned.split("=", 1)
            expr = safe_parse_expr(lhs, syms) - safe_parse_expr(rhs, syms)
            if expr.is_Symbol:
                # Degenerate residual (e.g. t=0 initial condition): a point
                # condition, not a relationship between variables.
                return False, cleaned, "quarantine: degenerate residual"
            return True, str(expr), "equation-residual"
        expr = safe_parse_expr(cleaned, syms)
        return True, str(expr), "sympy-direct"
    except (ImportError, TypeError, ValueError, ArithmeticError,
            AttributeError, KeyError) as e2:
        return False, cleaned, f"quarantine: {e2}"
