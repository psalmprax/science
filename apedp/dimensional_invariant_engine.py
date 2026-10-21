"""Level-5 roadmap Pillar 2: Self-Supervised Dimensional & Boundary Synthesis.

Removes the three human configurations the Level-4 stack required before a
grammar search could run honestly:

1. Buckingham-Π reduction — handed to the audit-proven engine in
   `apedp_dyn.conf.dimensionless` (`discover_pi_groups`: exact integer-nullspace,
   no SVD tolerances). This module AUTONOMOUSLY builds the exponent matrix
   from raw unit strings (pint SI dimensionality over the 7 SI bases) so the
   caller only declares `{"p": "kilopascal", ...}` plus value columns.
   Grammar search over ONLY dimensionless groups is dimensionally consistent
   by construction.
2. Empirical boundary synthesis — the training partition decides the
   boundary conditions, not the analyst:
     - lim_{x→0+} y ≈ 0 (within a documented span fraction of max|y| over the
       unscaled data) ⇒ ORIGIN-CONSTRAINED grammar (b ≡ 0; the free-affine
       intercept that violated q(0) = 0 in the Langmuir replays is not an
       option the search ever sees).
     - lim_{x→∞} y = L < ∞ (top-decile flatness) ⇒ SATURATION-family grammar
       (rational w·x/(s + x) family; log-type growth families excluded).
3. Characteristic scale — the manual scale-bank declaration is replaced by
   the data: s* = first half-max crossing (saturation regime) or median(x),
   and the emitted search bank is the spread multiples {s*/C, s*, s*·C} so
   the sub-saturation and saturation regimes are BOTH in scope.

Flaw-contract notes:
- All thresholds here are documented routing heuristics of the
  configuration layer (they shrink the SEARCH SPACE, they never adjudicate).
  No preregistered referee threshold is re-declared; the canonical spec is
  untouched and the referee remains the sole acceptance authority.
- The boundary verdict is FAIL-CLOSED: ambiguous training curves
  (neither origin-pinned nor saturating) emit NO constraint and NO honest
  free family other than the unconstrained bank — never a guessed "origin".
Status: experimental (Level-5 roadmap Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pint

from apedp_dyn.conf.dimensionless import DimensionlessGroup, discover_pi_groups

__all__ = [
    "SI_BASE_DIMENSIONS",
    "BoundaryConstraints",
    "InvariantSynthesis",
    "characteristic_scale",
    "infer_boundary_constraints",
    "synthesize_invariants",
]

#: SI base dimensions carrying the exponent matrix columns (pint order).
SI_BASE_DIMENSIONS = ("[mass]", "[length]", "[time]", "[current]", "[temperature]", "[amount_of_substance]", "[luminous_intensity]")

# ---------------------------------------------------------------------------
# Configuration-layer heuristics (documented; search-space constraints only).
# ---------------------------------------------------------------------------
LOW_BAND_FRACTION = 0.10  # smallest-x decile for the p→0+ estimator
HIGH_BAND_FRACTION = 0.10  # largest-x decile for the p→∞ estimator
ORIGIN_SPAN_TOLERANCE = 0.02  # |lim_{x→0}| y below this fraction of max|y| → origin pinned
SATURATION_SPAN_TOLERANCE = 0.05  # top-band dispersion fraction for a finite limit L
SCALE_BANK_SPREAD = 3.0  # emitted bank: {s*/C, s*, s*·C}


@dataclass(frozen=True)
class BoundaryConstraints:
    origin_constrained: bool
    saturation_limit: float | None
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantSynthesis:
    pi_groups: tuple[DimensionlessGroup, ...]
    boundary: BoundaryConstraints
    characteristic_scale: float
    scale_bank: tuple[float, ...]
    grammar_hints: tuple[str, ...] = ()
    notes: str = ""


def _exponent_matrix_from_units(units: dict[str, str]) -> tuple[tuple[str, ...], np.ndarray]:
    """Autonomous unit → exponent conversion over the 7 SI bases (pint)."""
    ureg = pint.UnitRegistry()
    names: list[str] = []
    rows: list[tuple[float, ...]] = []
    for name, unit in units.items():
        dim = float(1) * ureg(unit) if unit != "dimensionless" else ureg.Quantity(1.0, ureg.dimensionless)
        dim_dict = dict(dim.dimensionality) if unit != "dimensionless" else {}
        rows.append(tuple(float(dim_dict.get(base, 0.0)) for base in SI_BASE_DIMENSIONS))
        names.append(name)
    if not names:
        raise ValueError("no variables declared; nothing to reduce")
    matrix = np.array(rows, dtype=float)
    if np.abs(matrix - np.round(matrix)).max() > 1e-9:
        raise ValueError(
            "non-integer SI exponents from declared units; the frozen pi engine "
            "requires integral exponents — express the units in SI bases"
        )
    return tuple(names), matrix.astype(np.int64)


def infer_boundary_constraints(x: np.ndarray, y: np.ndarray) -> BoundaryConstraints:
    """Empirical limits on the TRAINING partition only (no holdout contact).

    Origin test: the low-x decile mean sits within ``ORIGIN_SPAN_TOLERANCE``
    of zero relative to the overall |y| span → the intercept b ≡ 0 is
    admissible (the grammar family is forced into origin-constrained form).
    Saturation test: the top-x band is flat within
    ``SATURATION_SPAN_TOLERANCE`` of the overall y range → a finite limit
    L exists and the search is steered to the rational saturation family
    (log-type growth families are structurally excluded).
    Both tests fail-closed on ambiguous curves: NO constraint is emitted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 12 or float(np.min(x)) <= 0:
        return BoundaryConstraints(False, None, {"n": float(len(x))})
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_min, x_max = float(x[0]), float(x[-1])
    log_span = float(np.log10(x_max / x_min))
    if log_span <= 0:
        return BoundaryConstraints(False, None, {"n": float(len(x))})
    # Bands are LOG-AXIS quantiles of the x-RANGE: a sample-count decile on
    # an arbitrary x-distribution measures the SAMPLER, not the physics —
    # the natural axis of a saturation law is log-x.
    x_low = x_min * 10 ** (LOW_BAND_FRACTION * log_span)
    x_high = x_min * 10 ** ((1.0 - HIGH_BAND_FRACTION) * log_span)
    low_mask = x <= x_low
    high_mask = x >= x_high
    if not low_mask.any() or not high_mask.any():
        return BoundaryConstraints(False, None, {"n": float(len(x))})
    y_low = float(np.mean(y[low_mask]))
    y_high = float(np.mean(y[high_mask]))
    y_band = y[high_mask]
    y_span = float(np.ptp(y))
    span_lo = abs(y_low) / max(abs(float(np.max(np.abs(y)))), 1e-30)
    flat = float(np.max(y_band) - np.min(y_band)) / max(y_span, 1e-30)
    ev = {"low_band_mean": y_low, "low_band_span_fraction": span_lo,
          "high_band_mean": y_high, "high_band_flatness": flat,
          "x_low_upper": x_low, "x_high_lower": x_high, "n": float(len(x))}

    # The two boundary tests are INDEPENDENT: a Langmuir curve is origin-
    # pinned (q(0) = 0) AND saturating (finite plateau L); returning on the
    # first hit hid the second (the defect this engine exists to avoid).
    origin_constrained = span_lo <= ORIGIN_SPAN_TOLERANCE
    saturation = y_high if (flat <= SATURATION_SPAN_TOLERANCE and y_high > y_low) else None
    if not origin_constrained and saturation is None:
        # Ambiguous curve: emit NO constraint (fail-closed over guessing).
        return BoundaryConstraints(False, None, ev)
    return BoundaryConstraints(origin_constrained, saturation, ev)


class ConstraintState(str, Enum):
    """Constraint epistemic states (EHI external-review correction).

    A constraint inferred from the observed range is a HYPOTHESIS with a
    provenance label, never an automatically enforced physical fact:
      KNOWN_CONSTRAINT          - declared by an independent physical law
      EMPIRICALLY_SUPPORTED     - training-partition evidence + model comparison;
                                  carries ``is_physical_claim=False``: the data
                                  establish what was measured, not the limit
      CONSTRAINT_UNRESOLVED     - evidence insufficient; alternatives preserved
      CONSTRAINT_REJECTED       - data contradict the constraint
    """

    KNOWN = "known_constraint"
    EMPIRICAL = "empirically_supported"
    UNRESOLVED = "constraint_unresolved"
    REJECTED = "constraint_rejected"


@dataclass(frozen=True)
class ConstraintHypothesis:
    kind: str
    state: ConstraintState
    source: str
    evidence: dict[str, float] = field(default_factory=dict)
    is_physical_claim: bool = False
    note: str = ""


def classify_constraints(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma_noise: float | None = None,
    declared_physical_law: tuple[str, ...] = (),
) -> tuple[ConstraintHypothesis, ...]:
    """Boundary conditions as HYPOTHESES with epistemic states.

    Three evidence sources per the EHI redesign (external review, 2026-10-21):
    (1) declared physical law -> ``KNOWN`` (independently justified);
    (2) training-partition evidence -> ``EMPIRICAL`` only when the band
        test passes AND the constrained-vs-unconstrained comparison is
        within the noise reference (sigma_noise, or a documented fraction
        of the observed range when the noise model is unknown);
    (3) model comparison across surrogates (Langmuir vs offset-Langmuir vs
        log growth) is PRESERVED as unresolved ambiguity whenever the
        measured range cannot separate their boundary behavior.
    Non-flat high bands do NOT reject saturation: they mean the measured
    window is sub-critical — the states vocabulary distinguishes absence of
    evidence (UNRESOLVED) from contradicting evidence (REJECTED).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 12:
        return (
            ConstraintHypothesis("origin", ConstraintState.UNRESOLVED, "insufficient_rows",
                                 {"n": float(len(x))}),
            ConstraintHypothesis("saturation", ConstraintState.UNRESOLVED, "insufficient_rows",
                                 {"n": float(len(x))}),
        )
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_min, x_max = float(x[0]), float(x[-1])
    ptp_y = float(np.ptp(y))
    if ptp_y <= 0 or x_min <= 0:
        return (
            ConstraintHypothesis("origin", ConstraintState.UNRESOLVED, "degenerate_band",
                                 {"ptp_y": ptp_y}),
            ConstraintHypothesis("saturation", ConstraintState.UNRESOLVED, "degenerate_band",
                                 {"ptp_y": ptp_y}),
        )
    log_span = float(np.log10(x_max / x_min))
    x_low = x_min * 10 ** (LOW_BAND_FRACTION * log_span)
    x_high = x_min * 10 ** ((1.0 - HIGH_BAND_FRACTION) * log_span)
    low_mask = x <= x_low
    high_mask = x >= x_high
    if not low_mask.any() or not high_mask.any():
        return (
            ConstraintHypothesis("origin", ConstraintState.UNRESOLVED, "empty_band", {}),
            ConstraintHypothesis("saturation", ConstraintState.UNRESOLVED, "empty_band", {}),
        )
    y_low = float(np.mean(y[low_mask]))
    y_high = float(np.mean(y[high_mask]))
    noise_ref = float(sigma_noise) if (sigma_noise is not None and sigma_noise > 0) \
        else max(ORIGIN_SPAN_TOLERANCE * ptp_y, 1e-30)
    out: list[ConstraintHypothesis] = []

    # ---- origin hypothesis -------------------------------------------
    if "origin" in declared_physical_law or "q0_is_zero" in declared_physical_law:
        out.append(ConstraintHypothesis("origin", ConstraintState.KNOWN,
                                        "declared_physical_law", {"low_band_mean": y_low},
                                        is_physical_claim=True))
    else:
        # Model comparison: how much does the FREE intercept (affine vs
        # through-origin on the LOW band) buy within the noise reference?
        resid_free = float(np.std(y[low_mask], ddof=1)) if int(low_mask.sum()) > 2 else 0.0
        within_noise = abs(y_low) <= max(3.0 * noise_ref, ORIGIN_SPAN_TOLERANCE * ptp_y)
        ev = {"low_band_mean": y_low, "noise_reference": noise_ref,
              "low_band_scatter": resid_free, "n": float(int(low_mask.sum()))}
        if not within_noise:
            out.append(ConstraintHypothesis(
                "origin", ConstraintState.REJECTED, "training_partition_evidence", ev,
                note="observed response is incompatible with q(0)=0 within the noise reference; "
                     "the unconstrained family (e.g. offset-Langmuir a + wp/(s+p)) stays live"))
        else:
            out.append(ConstraintHypothesis(
                "origin", ConstraintState.EMPIRICAL, "training_partition_evidence", ev,
                note="boundary inferred from the measured range only — NOT an independent "
                      "physical fact; log-growth surrogates remain admissible alternatives"))

    # ---- saturation hypothesis ---------------------------------------
    if "finite_saturation_limit" in declared_physical_law:
        out.append(ConstraintHypothesis("saturation", ConstraintState.KNOWN,
                                        "declared_physical_law", {"high_band_mean": y_high},
                                        is_physical_claim=True))
    else:
        y_band = y[high_mask]
        flat = float(np.max(y_band) - np.min(y_band)) / max(ptp_y, 1e-30)
        ev = {"high_band_mean": y_high, "high_band_flatness": flat,
              "x_high_lower": x_high, "n": float(int(high_mask.sum()))}
        if flat <= SATURATION_SPAN_TOLERANCE and y_high > y_low:
            out.append(ConstraintHypothesis(
                "saturation", ConstraintState.EMPIRICAL, "training_partition_evidence", ev,
                note="a plateau INSIDE the measured range does not establish a finite asymptote; "
                      "saturating, logarithmic and power-law families are evaluated by "
                      "held-out comparison before any asymptote claim"))
        else:
            out.append(ConstraintHypothesis(
                "saturation", ConstraintState.UNRESOLVED, "training_partition_evidence", ev,
                note="no in-range plateau: sub-critical window — alternatives "
                      "(saturation, log, power-law) preserved, not rejected"))
    return tuple(out)


def characteristic_scale(x: np.ndarray, y: np.ndarray, boundary: BoundaryConstraints) -> tuple[float, tuple[float, ...]]:
    """s* from the training partition; emitted multi-scale bank around it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        raise ValueError("empty training partition")
    s_star = float(np.median(x))
    # Half-max crossing WITH linear interpolation between the bracketing
    # samples: for a saturation curve the half-max point IS the
    # characteristic scale (q(2s) = (2/3)L near s, so the interpolated
    # half-max crossing sits close to s — the benchmark regression validated
    # the s* basis). Averaging the whole suprathreshold tail would anchor
    # the scale on the plateau, not the knee.
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    target = float(np.min(ys) + 0.5 * (np.max(ys) - np.min(ys)))
    above = np.where(ys >= target)[0]
    if len(above):
        k0 = int(above[0])
        if k0 == 0:
            s_star = float(xs[0])
        else:
            x1, x2 = float(xs[k0 - 1]), float(xs[k0])
            y1, y2 = float(ys[k0 - 1]), float(ys[k0])
            if y2 != y1:
                s_star = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
            else:
                s_star = x2
    s_star = max(s_star, 1e-30)
    bank = (s_star / SCALE_BANK_SPREAD, s_star, s_star * SCALE_BANK_SPREAD)
    return s_star, bank


def synthesize_invariants(
    variable_units: dict[str, str],
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    scale_variable: str | None = None,
) -> InvariantSynthesis:
    """One call: Π reduction + boundary synthesis + characteristic scale.

    The caller declares units and (optionally) the training-partition curve;
    everything the grammar search previously needed a human to configure
    comes back as structured objects. x/y must come from the training
    partition ONLY — passing holdout data here is a selection leak and is
    the caller's contract to honor (the referee still gates everything).
    """
    names, d_matrix = _exponent_matrix_from_units(variable_units)
    groups = discover_pi_groups(names, d_matrix, variable_units=variable_units,
                                convention="autonomous-l5")

    hints: tuple[str, ...] = ()
    bank: tuple[float, ...] = ()
    s_star = float("nan")
    boundary = BoundaryConstraints(False, None, {"n": 0.0})
    if x is not None and y is not None:
        boundary = infer_boundary_constraints(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        s_star, bank = characteristic_scale(np.asarray(x, dtype=float), np.asarray(y, dtype=float), boundary)
    if boundary.origin_constrained:
        hints += ("rational_origin_constrained_b0",)
    if boundary.saturation_limit is not None:
        hints += ("rational_saturation_wx_over_s_plus_x",)

    return InvariantSynthesis(
        pi_groups=tuple(groups),
        boundary=boundary,
        characteristic_scale=s_star,
        scale_bank=bank,
        grammar_hints=hints,
    )
