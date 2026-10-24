"""Report fidelity: verdicts derive from measured rows, never from prose.

What was wrong. The benchmark report template hardcoded its gate-verdict
labels (`**PASSED** (Exact Recovery)`, `**REFUSED**`) while the JSON it
claimed to report carried its own machine verdict per row
(`referee_whiteness_pass`). For assisted RIME the two disagreed: the JSON
says `false` (|rho1| = 0.9691 on ~1e-8 roundoff residuals trips the whiteness
rule) while the report prints PASSED. The physics point behind the hardcode
is fair -- autocorrelation of numerical noise next to R2 = 1.000000 should
not refuse an exact recovery -- but a hardcoded label is not a rule.

The fix. `verdict_for_row` implements the rule with EXPLICIT caller
arguments (AGENTS.md 1.3): pass by gate (extrapolation R2 above tau AND
|rho1| below max), or pass by the exact-recovery bypass (extrapolation RMSE
below eps AND R2 within delta of 1). The report template renders this
function's output verbatim, and `test_dyn_landauer_report_fidelity.py`
recomputes every verdict and numeric cell from the JSON and asserts the
markdown matches. A hand-edit to any verdict breaks the build.
"""
from __future__ import annotations

from typing import Any


def verdict_for_row(
    r2_extrapolation: float,
    rho1: float,
    rmse_extrapolation: float,
    *,
    tau_extrapolation: float = 0.99,
    rho_max: float = 0.5,
    eps_exact_rmse: float = 1e-6,
    delta_exact_r2: float = 1e-9,
) -> tuple[str, str]:
    """(verdict, reason) derived from a measured row. No prose, no exceptions.

    Order is load-bearing: the exact-recovery bypass is checked FIRST, because
    residuals at the numerical floor have meaningless autocorrelation and must
    not be refused by a whiteness rule built for structural misfit.
    """
    try:
        r2 = float(r2_extrapolation)
    except (TypeError, ValueError):
        return "REFUSED", "extrapolation R2 not a number"
    try:
        rmse = float(rmse_extrapolation)
    except (TypeError, ValueError):
        rmse = float("inf")
    try:
        rho = float(rho1)
        rho_finite = rho == rho  # NaN-safe without importing numpy
    except (TypeError, ValueError):
        rho, rho_finite = float("nan"), False

    if rmse <= eps_exact_rmse and r2 >= 1.0 - delta_exact_r2:
        return ("PASSED",
                f"exact recovery by rule (RMSE {rmse:.2e} <= {eps_exact_rmse:.0e}, "
                f"R2 {r2:.9f} within {delta_exact_r2:.0e} of 1)")
    if not rho_finite:
        return "REFUSED", "residual autocorrelation undefined and no exact recovery"
    if r2 >= tau_extrapolation and abs(rho) < rho_max:
        return ("PASSED",
                f"gate pass (R2 {r2:.4f} >= {tau_extrapolation}, |rho1| {abs(rho):.4f} < {rho_max})")
    if r2 < tau_extrapolation:
        return "REFUSED", f"extrapolation collapse (R2 {r2:.4f} < {tau_extrapolation})"
    return "REFUSED", f"residual structure (|rho1| {abs(rho):.4f} >= {rho_max})"


def render_verdict_cell(verdict: str, reason: str) -> str:
    """Markdown table cell for a verdict. The reason travels with the label."""
    return f"**{verdict}** ({reason})"


def render_float(value: Any, precision: int) -> str:
    """Backtick-quoted numeric cell. NaN renders as n/a with its cause stated."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "`n/a`"
    if v != v:
        return "`n/a (exact-zero residuals)`"
    return f"`{v:.{precision}f}`"


def check_row_against_report(
    row: dict[str, Any],
    md_row: str,
    *,
    numeric_cells: tuple[tuple[str, int], ...] = (
        ("r2_train", 4), ("r2_extrapolation", 4), ("rmse_extrapolation", 4), ("rho1", 4),
    ),
    **verdict_kwargs: Any,
) -> list[str]:
    """Recompute every cell of one markdown table row from its JSON row.

    Returns a list of mismatch descriptions (empty = faithful). Numeric cells
    must match to half the printed precision; the verdict cell must equal
    `render_verdict_cell(*verdict_for_row(...))` exactly.
    """
    problems: list[str] = []
    for key, prec in numeric_cells:
        if key not in row:
            problems.append(f"row lacks key {key!r}")
            continue
        try:
            actual = float(row[key])
        except (TypeError, ValueError):
            problems.append(f"row[{key!r}] not numeric: {row[key]!r}")
            continue
        if actual != actual:
            if "`n/a" not in md_row:
                problems.append(f"row[{key!r}] is NaN but md shows a number")
            continue
        want = f"`{actual:.{prec}f}`"
        if want not in md_row:
            problems.append(f"row[{key!r}]: md lacks {want} (json {actual!r})")
    verdict, reason = verdict_for_row(
        row.get("r2_extrapolation", float("nan")),
        row.get("rho1", float("nan")),
        row.get("rmse_extrapolation", float("inf")),
        **verdict_kwargs,
    )
    want_cell = render_verdict_cell(verdict, reason)
    if want_cell not in md_row:
        problems.append(f"verdict cell mismatch: want {want_cell!r}")
    return problems
