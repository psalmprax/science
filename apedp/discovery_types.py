"""Strongly-typed result objects for APEDP discovery modules.

Replaces loose untyped dictionaries (e.g. {"terms": [], "note": ...}) with structured,
well-typed dataclasses while maintaining full backwards compatibility via dict-like
accessors (__getitem__, get, keys, items, to_dict, __iter__, __len__).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any


class DictCompatibleResult:
    """Base mixin providing dictionary-like access to dataclasses for backwards compatibility."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload, no-any-return]

    def keys(self) -> list[str]:
        return list(self.to_dict().keys())

    def values(self) -> list[Any]:
        return list(self.to_dict().values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self.to_dict().items())


@dataclass(frozen=True)
class BicSelectionResult(DictCompatibleResult):
    """Result of forward stepwise BIC feature selection."""

    terms: list[str] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    bic: float = float("inf")
    note: str | None = None
    mdl_score: float | None = None


@dataclass(frozen=True)
class DataAdequacyResult(DictCompatibleResult):
    """Result of preliminary data adequacy and identifiability audit."""

    verdict: str  # "OK", "THIN", "INSUFFICIENT"
    n: int
    n_vars: int
    range_ratios: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SparseSelectionResult(DictCompatibleResult):
    """Result of sparse term selection with collinearity and stability filtering."""

    terms: list[str] = field(default_factory=list)
    intercept: float = 0.0
    bic: float = float("inf")
    selector: str = "bic_forward"
    n_library: int = 0
    n_dropped: int = 0
    n_library_raw: int = 0
    dropped: list[str] = field(default_factory=list)
    dropped_collinear: list[dict[str, Any]] = field(default_factory=list)
    ambiguity_groups: list[list[str]] = field(default_factory=list)
    ambiguity_corr: float | None = None
    stability_rate: float = 1.0
    adequacy: Any = None
    alpha: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class PowerLawResult(DictCompatibleResult):
    """Result of single-variable or multi-variable power-law detection."""

    family: str = "power-law"
    variable: str = ""
    exponent: float = 0.0
    r_squared: float = 0.0
    snapped: str | None = None
    confident: bool = False


@dataclass(frozen=True)
class CompositionFamilyResult(DictCompatibleResult):
    """Result of composition and transition family detection."""

    family: str
    inner_expr: str = ""
    outer_func: str = ""
    r_squared: float = 0.0
    priors: list[Any] = field(default_factory=list)
    note: str | None = None


@dataclass(frozen=True)
class CrossFamilyResult(DictCompatibleResult):
    """Result of cross-family comparison and selection."""

    best: dict[str, Any] | None = None
    all_candidates: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def family(self) -> str | None:
        return self.best.get("family") if self.best else None

    @property
    def variable(self) -> str | None:
        return self.best.get("variable") if self.best else None

    @property
    def expression(self) -> str | None:
        return self.best.get("expression") if self.best else None

    @property
    def bic(self) -> float:
        return float(self.best.get("bic", float("inf"))) if self.best else float("inf")

    @property
    def r_squared(self) -> float:
        return float(self.best.get("r_squared", 0.0)) if self.best else 0.0

    @property
    def num_params(self) -> int:
        return int(self.best.get("num_params", 0)) if self.best else 0


@dataclass(frozen=True)
class NeuralRelevanceResult(DictCompatibleResult):
    """Result of neural surrogate feature relevance screening."""

    relevant: list[str] = field(default_factory=list)
    surrogate_r2: float = 0.0
    converged: bool = False
    n_iter: int = 0
    importances: dict[str, float] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class SurrogateDistillationResult(DictCompatibleResult):
    """Result of surrogate dense sampling and symbolic distillation."""

    terms: list[str] = field(default_factory=list)
    surrogate_r2: float = 0.0
    converged: bool = True
    n_iter: int = 0
    dense_r2: float = 0.0
    dropped: list[str] = field(default_factory=list)
    dropped_collinear: list[dict[str, Any]] = field(default_factory=list)
    ambiguity_groups: list[list[str]] = field(default_factory=list)
    stability_rate: float = 1.0
    note: str | None = None
