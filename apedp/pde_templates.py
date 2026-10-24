"""APEDP PDE templates: recognize named PDE/ODE shapes as opaque classes.

Algebraic prior art lives in ``PriorArtAuditor``; anything with
``Derivative``/``Integral`` lands here. Templates match on derivative-order
structure per undefined function (``T(x, t)``, ``u(x, t)``), never on
coefficient values, so any diffusion-shaped residual is ``DIFFUSION_1D``
regardless of its diffusivity symbol. Term-wise dimensional checks stay with
the caller; this module only names the shape.
"""
from __future__ import annotations

import sympy as sp

TEMPLATES: dict[str, str] = {
    "DIFFUSION_1D": "u_t = D*u_xx (+ source): first time + second space derivative of one field",
    "HEAT_1D": "alias of DIFFUSION_1D for temperature",
    "WAVE_1D": "u_tt = c^2*u_xx: second time + second space derivative of one field",
    "LAPLACE_2D": "u_xx + u_yy = 0: second derivatives in two space vars, no source",
    "POISSON_2D": "u_xx + u_yy = source: Laplacian plus a source term",
    "CONTINUITY_1D": "rho_t + (flux)_x = 0: first time + first space derivative",
    "EULER_MOMENTUM_1D": "u_t + u*u_x + p_x/rho = 0: first-order advection + pressure, no viscosity",
    "NS_X_1D": "u_t + ... = -p_x/rho + nu*u_xx: adds a second space derivative to Euler",
    "SCHRODINGER_TD_1D": "I*hbar*psi_t + hbar^2/2m*psi_xx - V*psi = 0: first time + second space + imaginary unit",
    "MAXWELL_FARADAY_1D": "E_x + B_t = 0: first space of E plus first time of B",
    "MAXWELL_AMPERE_1D": "B_x - mu*eps*E_t = 0: first space of B plus first time of E",
}


def _deriv_records(expr: sp.Expr) -> dict[str, list[dict[str, int]]]:
    """Map undefined-function name -> list of {var: order} per Derivative node."""
    rec: dict[str, list[dict[str, int]]] = {}
    for d in expr.atoms(sp.Derivative):
        base = d.args[0] if d.args else None
        try:
            fname = base.func.__name__  # type: ignore[union-attr]
        except (AttributeError, TypeError):
            fname = "?"
        orders: dict[str, int] = {}
        rest = list(d.args[1:])
        i = 0
        while i < len(rest):
            v = rest[i]
            if getattr(v, "is_Symbol", False):
                if i + 1 < len(rest) and getattr(rest[i + 1], "is_Integer", False):
                    orders[str(v)] = orders.get(str(v), 0) + int(rest[i + 1])
                    i += 2
                else:
                    orders[str(v)] = orders.get(str(v), 0) + 1
                    i += 1
            else:
                # Tuple forms: python (x, 2) or sympy Tuple(x, 2).
                try:
                    items = tuple(v)
                except TypeError:
                    items = ()
                if len(items) == 2:
                    try:
                        orders[str(items[0])] = orders.get(str(items[0]), 0) + int(items[1])
                    except (TypeError, ValueError):
                        pass
                    i += 1
                else:
                    i += 1
        rec.setdefault(str(fname), []).append(orders)
    return rec


def _orders_for(rec: dict[str, list[dict[str, int]]], fname: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for orders in rec.get(fname, []):
        for var, order in orders.items():
            merged[var] = max(merged.get(var, 0), order)
    return merged


def recognize_pde(expr: sp.Expr) -> list[str]:
    """Return matched template names (empty when the shape is unknown)."""
    rec = _deriv_records(expr)
    if not rec:
        return []
    hits: list[str] = []
    funcs = list(rec)
    has_i = bool(expr.has(sp.I))

    # Wave before diffusion (second time implies wave, not diffusion).
    wave_fields = [f for f in funcs if _orders_for(rec, f).get("t", 0) >= 2 and _orders_for(rec, f).get("x", 0) >= 2]
    if wave_fields:
        hits.append("WAVE_1D")
    diff_fields = [
        f for f in funcs
        if _orders_for(rec, f).get("t", 0) == 1 and _orders_for(rec, f).get("x", 0) >= 2
    ]
    if diff_fields and not wave_fields:
        hits.extend(["DIFFUSION_1D", "HEAT_1D"])
    laplace_fields = [f for f in funcs if _orders_for(rec, f).get("x", 0) >= 2 and _orders_for(rec, f).get("y", 0) >= 2]
    if laplace_fields:
        hits.append("LAPLACE_2D")
        # Heuristic: more than the two pure second-derivative terms present.
        top = expr if expr.is_Add else sp.Add(expr)
        if len(top.args) > 2:
            hits.append("POISSON_2D")
    # Continuity: first time somewhere + first space somewhere.
    any_t1 = any(_orders_for(rec, f).get("t", 0) >= 1 for f in funcs)
    any_x1 = any(
        _orders_for(rec, f).get("x", 0) >= 1
        for f in funcs
        if _orders_for(rec, f).get("x", 0) == 1
    )
    if any_t1 and any_x1 and not diff_fields and not wave_fields and not laplace_fields:
        hits.append("CONTINUITY_1D")
    # Euler vs Navier-Stokes on velocity-like fields.
    for f in funcs:
        orders = _orders_for(rec, f)
        if (orders.get("t", 0) >= 1 and orders.get("x", 0) >= 2
                and (f.lower() in ("u", "v", "w", "vel") or f in funcs[:1])
                and "NS_X_1D" not in hits):
            # Only claim viscosity when a pressure gradient co-occurs.
            p_like = [g for g in funcs if g.lower() in ("p", "press", "pressure")]
            if p_like and any(_orders_for(rec, g).get("x", 0) >= 1 for g in p_like):
                hits.append("NS_X_1D")
    # Euler: first-order advection + pressure gradient, no second derivatives anywhere.
    if not any(v >= 2 for f in funcs for v in _orders_for(rec, f).values()):
        p_like = [g for g in funcs if g.lower() in ("p", "press", "pressure")]
        if p_like and any(_orders_for(rec, g).get("x", 0) >= 1 for g in p_like) and any_t1:
            hits.append("EULER_MOMENTUM_1D")
    # Schrodinger: psi_t + psi_xx + imaginary unit.
    for f in funcs:
        if f.lower() in ("psi", "phi", "wavefn"):
            orders = _orders_for(rec, f)
            if orders.get("t", 0) >= 1 and orders.get("x", 0) >= 2 and has_i:
                hits.append("SCHRODINGER_TD_1D")
    # Maxwell 1D pairs.
    for e_name in ("E", "e", "Ex"):
        for b_name in ("B", "b", "By"):
            if e_name in rec and b_name in rec:
                e_o, b_o = _orders_for(rec, e_name), _orders_for(rec, b_name)
                if e_o.get("x", 0) >= 1 and b_o.get("t", 0) >= 1:
                    hits.append("MAXWELL_FARADAY_1D")
                if b_o.get("x", 0) >= 1 and e_o.get("t", 0) >= 1:
                    hits.append("MAXWELL_AMPERE_1D")
    # Schrodinger fallback on any field when I is present with t1 + xx.
    if has_i and "SCHRODINGER_TD_1D" not in hits:
        for f in funcs:
            orders = _orders_for(rec, f)
            if orders.get("t", 0) == 1 and orders.get("x", 0) >= 2:
                hits.append("SCHRODINGER_TD_1D")
                break
    seen: list[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return seen
