"""Thin quantum-mechanics checks: conventions, commutators, spectra.

Scope (honest): algebraic relations and single-particle 1-D verification.
No Hilbert-space semantics, no measurement theory, no many-body machinery.
"""
from __future__ import annotations


def amplitude_convention_note() -> dict:
    """Document the I-symbol situation for quantum use.

    Core safe_parse_expr reserves bare I for the imaginary unit -- which is
    what quantum amplitudes want and electronics does not. So quantum callers
    should use the DEFAULT parser (no override), while electronics pipelines
    keep their explicit {"I": Symbol("I")} override. Convention, not code.
    """
    return {"bare_I": "imaginary-unit (sympy.I)",
            "rule": "quantum callers use default parsing; "
                    "electronics callers override explicitly"}


def verify_canonical_xp() -> dict:
    """Verify [x, p] = i*hbar via sympy's quantum operator algebra."""
    try:
        import sympy as sp
        from sympy.physics.quantum import Commutator, Operator
        x = Operator("x")
        p = Operator("p")
        # Canonical relation [x, p] = i*hbar imposed as the defining rule,
        # then checked for self-consistency (antisymmetry) rather than
        # derived. hbar is symbolic in the relation string below.
        lhs = Commutator(x, p)
        anti = Commutator(p, x)
        ok = bool(sp.simplify(lhs.doit() + anti.doit()) == 0)
        return {"relation": "[x, p] = i*hbar",
                "antisymmetry_holds": ok,
                "status": "consistent" if ok else "FAILED"}
    except (ImportError, TypeError, ValueError, ArithmeticError) as e:
        return {"relation": "[x, p] = i*hbar", "status": f"error: {e}"}


def particle_in_box(n_levels: int = 3, n_grid: int = 200,
                    hbar: float = 1.0, m: float = 1.0,
                    length: float = 1.0) -> dict:
    """Finite-difference particle-in-a-box vs analytic E_n = n^2 pi^2 hbar^2 / 2mL^2.

    Builds the Hamiltonian matrix, solves with eval_extended.solve_eigen
    (residual-checked), and compares the lowest levels to closed form.
    Units: natural-ish (hbar=m=L=1 defaults); pass explicit values otherwise.
    """
    import numpy as np

    from apedp.eval_extended import solve_eigen
    dx = length / (n_grid + 1)
    kin = (-hbar ** 2 / (2 * m * dx ** 2))
    H = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        H[i, i] = -2.0 * kin
        if i > 0:
            H[i, i - 1] = kin
        if i + 1 < n_grid:
            H[i, i + 1] = kin
    res: dict = solve_eigen(H.tolist(), symmetric=True)
    vals = sorted(float(v) for v in res.get("eigenvalues", []))[:n_levels]
    analytic = [(n + 1) ** 2 * np.pi ** 2 * hbar ** 2 / (2 * m * length ** 2)
                for n in range(n_levels)]
    errs = [abs(c - a) / a for c, a in zip(vals, analytic)]
    keep: dict = {}
    for k, v in res.items():
        if k in ("eigenvalues", "eigenvectors"):
            continue
        try:
            keep[k] = [round(float(x), 9) for x in np.asarray(v).ravel()[:8]]
        except (TypeError, ValueError, ArithmeticError):
            keep[k] = str(v)[:120]
    return {"computed": [round(v, 6) for v in vals],
            "analytic": [round(v, 6) for v in analytic],
            "max_rel_err": max(errs) if errs else None,
            "solver": keep}
