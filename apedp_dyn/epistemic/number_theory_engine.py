"""Discrete and algebraic number theory discovery engine.

Exposes native Rust routines for:
1. Exact rational coefficient reconstruction (Farey sequences / Continued Fractions).
2. Integer relation discovery (PSLQ / lattice null-space search for physical conservation laws).
3. Discrete modular lattice discovery (quantized spatial / energy step identification).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

try:
    import apedp_kernel
    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False


@dataclass(frozen=True)
class RationalConstant:
    num: int
    den: int
    value: float

    def __str__(self) -> str:
        if self.den == 1:
            return str(self.num)
        return f"{self.num}/{self.den}"


@dataclass(frozen=True)
class IntegerRelation:
    coefficients: list[int]
    residual: float
    formula: str
    is_exact: bool


@dataclass(frozen=True)
class ModularLattice:
    spacing: float
    offset: float
    residual_rmse: float
    integer_indices: list[int]
    formula: str


def recover_rational_constant(
    val: float,
    max_denom: int = 1000,
    eps: float = 1e-4,
) -> RationalConstant | None:
    """Recover irreducible rational p/q matching float val within tolerance."""
    if not _KERNEL_AVAILABLE:
        return None
    raw = apedp_kernel.rime_number_theory_farey_json(float(val), int(max_denom), float(eps))
    data = json.loads(raw)
    if not data:
        return None
    num = data["num"]
    den = data["den"]
    return RationalConstant(num=num, den=den, value=num / den)


def discover_integer_relation(
    values: Sequence[float],
    labels: Sequence[str] | None = None,
    max_coeff: int = 5,
    eps: float = 1e-3,
) -> IntegerRelation | None:
    """Find non-trivial integer coefficients m such that sum m_i * x_i ≈ 0."""
    if not _KERNEL_AVAILABLE or len(values) < 2:
        return None
    lbls = list(labels) if labels is not None else None
    raw = apedp_kernel.rime_number_theory_integer_relation_json(
        [float(v) for v in values],
        int(max_coeff),
        float(eps),
        lbls,
    )
    data = json.loads(raw)
    if not data:
        return None
    return IntegerRelation(
        coefficients=data["coefficients"],
        residual=data["residual"],
        formula=data["formula"],
        is_exact=data["is_exact"],
    )


def discover_modular_lattice(
    points: Sequence[float],
    min_spacing: float = 0.01,
    max_spacing: float = 100.0,
    max_rmse: float = 1e-3,
) -> ModularLattice | None:
    """Detect if 1D points fall on a periodic lattice x_i = offset + k_i * spacing."""
    if not _KERNEL_AVAILABLE or len(points) < 4:
        return None
    raw = apedp_kernel.rime_number_theory_modular_lattice_json(
        [float(p) for p in points],
        float(min_spacing),
        float(max_spacing),
        float(max_rmse),
    )
    data = json.loads(raw)
    if not data:
        return None
    return ModularLattice(
        spacing=data["spacing"],
        offset=data["offset"],
        residual_rmse=data["residual_rmse"],
        integer_indices=data["integer_indices"],
        formula=data["formula"],
    )
